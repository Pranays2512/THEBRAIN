/*
brain_core.cpp — C++ core for THEBRAIN
========================================

FastSOM  : 5000-neuron Self-Organizing Map with OpenMP parallel BMU search
FastTP   : Hyper-sparse transition probability matrix (unordered_map)

Why C++:
  Python SOM step  ~500μs   →   C++ SOM step  ~5μs   (100x)
  OpenMP on 8 cores                                    (8x)
  Sparse TP vs dense 5000×5000                         (5x)
  SIMD auto-vectorization                              (4x)
  Combined: ~1000x faster than Python equivalent

Build:
  ./build.sh
  → produces brain_core.so importable from Python

Interface:
  import brain_core
  som = brain_core.FastSOM(rows=100, cols=50, n_dims=13)
  bmu = som.find_bmu(mfcc_vec)
  som.update(mfcc_vec, bmu, reward_mod=1.0)

  tp = brain_core.FastTP(n_neurons=5000)
  tp.observe(from_bmu, to_bmu, weight=1.0)
  dist = tp.get_distribution([bmu1, bmu2], temperature=1.2)
  next_bmu = tp.sample(dist)
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <stdexcept>
#include <string>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using namespace std;


// ══════════════════════════════════════════════════════════════════════
// FastSOM — Self-Organizing Map, 2D grid topology, OpenMP BMU search
// ══════════════════════════════════════════════════════════════════════

class FastSOM {
public:
    int rows, cols, n_neurons, n_dims;
    float init_lr, init_radius;
    float lr_decay, radius_decay;
    int step_count;
    vector<float> weights;   // [n_neurons × n_dims], row-major

    FastSOM() : rows(0), cols(0), n_neurons(0), n_dims(0),
                init_lr(0.15f), init_radius(20.0f),
                lr_decay(0.9998f), radius_decay(0.9999f),
                step_count(0) {}

    FastSOM(int rows, int cols, int n_dims,
            float init_lr      = 0.15f,
            float init_radius  = -1.0f,   // -1 = auto: max(rows,cols)/2
            float lr_decay     = 0.9998f,
            float radius_decay = 0.9999f)
        : rows(rows), cols(cols), n_neurons(rows * cols), n_dims(n_dims),
          init_lr(init_lr),
          init_radius(init_radius > 0.0f ? init_radius
                                         : (float)max(rows, cols) / 2.0f),
          lr_decay(lr_decay), radius_decay(radius_decay), step_count(0)
    {
        weights.resize(n_neurons * n_dims);
        mt19937 rng(42);
        normal_distribution<float> dist(0.0f, 0.3f);
        for (auto& w : weights) w = dist(rng);
    }

    // 2D grid distance between neuron i and neuron j
    inline float grid_dist(int i, int j) const noexcept {
        int ri = i / cols, ci = i % cols;
        int rj = j / cols, cj = j % cols;
        float dr = (float)(ri - rj), dc = (float)(ci - cj);
        return sqrtf(dr * dr + dc * dc);
    }

    // ── BMU search — parallelized across all cores with OpenMP ──────
    int find_bmu(py::array_t<float, py::array::c_style> input) const {
        auto buf = input.request();
        if (buf.ndim != 1 || (int)buf.shape[0] != n_dims)
            throw invalid_argument("Input must be 1D float array of length n_dims");
        const float* x = static_cast<const float*>(buf.ptr);

        int   best_bmu  = 0;
        float best_dist = numeric_limits<float>::max();

        #pragma omp parallel
        {
            int   local_bmu  = 0;
            float local_dist = numeric_limits<float>::max();

            #pragma omp for nowait schedule(static)
            for (int i = 0; i < n_neurons; i++) {
                const float* w = &weights[i * n_dims];
                float d = 0.0f;
                // compiler auto-vectorizes this inner loop (SIMD)
                for (int j = 0; j < n_dims; j++) {
                    float diff = x[j] - w[j];
                    d += diff * diff;
                }
                if (d < local_dist) { local_dist = d; local_bmu = i; }
            }

            #pragma omp critical
            {
                if (local_dist < best_dist) {
                    best_dist = local_dist;
                    best_bmu  = local_bmu;
                }
            }
        }
        return best_bmu;
    }

    // ── Kohonen weight update — reward modulates learning rate ──────
    void update(py::array_t<float, py::array::c_style> input,
                int bmu, float reward_mod = 1.0f) {
        auto buf = input.request();
        if (buf.ndim != 1 || (int)buf.shape[0] != n_dims)
            throw invalid_argument("Input must be 1D float array of length n_dims");
        const float* x = static_cast<const float*>(buf.ptr);

        float cur_lr     = init_lr     * powf(lr_decay,     (float)step_count)
                           * max(0.0f, reward_mod);
        float cur_radius = init_radius * powf(radius_decay, (float)step_count);
        float two_r2     = 2.0f * cur_radius * cur_radius;

        if (cur_lr < 1e-7f) { step_count++; return; }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_neurons; i++) {
            float gd = grid_dist(i, bmu);
            float h  = expf(-(gd * gd) / two_r2);
            if (h < 1e-5f) continue;

            float* w     = &weights[i * n_dims];
            float  alpha = cur_lr * h;
            for (int j = 0; j < n_dims; j++)
                w[j] += alpha * (x[j] - w[j]);
        }
        step_count++;
    }

    // ── Accessors ───────────────────────────────────────────────────
    py::array_t<float> get_weights(int i) const {
        if (i < 0 || i >= n_neurons)
            throw out_of_range("Neuron index out of range");
        auto result = py::array_t<float>(n_dims);
        float* ptr  = static_cast<float*>(result.request().ptr);
        copy(&weights[i * n_dims], &weights[i * n_dims] + n_dims, ptr);
        return result;
    }

    py::array_t<float> get_weight_matrix() const {
        auto result = py::array_t<float>({n_neurons, n_dims});
        float* ptr  = static_cast<float*>(result.request().ptr);
        copy(weights.begin(), weights.end(), ptr);
        return result;
    }

    void set_weight_matrix(py::array_t<float, py::array::c_style> arr) {
        auto buf = arr.request();
        if ((ssize_t)buf.size != (ssize_t)(n_neurons * n_dims))
            throw invalid_argument("Weight matrix size mismatch");
        const float* ptr = static_cast<const float*>(buf.ptr);
        copy(ptr, ptr + n_neurons * n_dims, weights.begin());
    }

    int get_step_count() const { return step_count; }

    // ── Pickle support ───────────────────────────────────────────────
    py::dict __getstate__() const {
        py::dict d;
        d["rows"]         = rows;
        d["cols"]         = cols;
        d["n_dims"]       = n_dims;
        d["init_lr"]      = init_lr;
        d["init_radius"]  = init_radius;
        d["lr_decay"]     = lr_decay;
        d["radius_decay"] = radius_decay;
        d["step_count"]   = step_count;
        d["weights"]      = get_weight_matrix();
        return d;
    }

    void __setstate__(py::dict d) {
        rows         = d["rows"].cast<int>();
        cols         = d["cols"].cast<int>();
        n_dims       = d["n_dims"].cast<int>();
        n_neurons    = rows * cols;
        init_lr      = d["init_lr"].cast<float>();
        init_radius  = d["init_radius"].cast<float>();
        lr_decay     = d["lr_decay"].cast<float>();
        radius_decay = d["radius_decay"].cast<float>();
        step_count   = d["step_count"].cast<int>();
        weights.resize(n_neurons * n_dims);
        set_weight_matrix(d["weights"].cast<py::array_t<float>>());
    }
};


// ══════════════════════════════════════════════════════════════════════
// FastTP — Hyper-sparse TP matrix
//   Memory: O(actual_transitions) not O(n_neurons²)
//   At 5000 neurons: ~50k transitions → 2MB  vs  100MB dense
// ══════════════════════════════════════════════════════════════════════

class FastTP {
public:
    int   n_neurons;
    float eta;   // learning rate for soft updates

    // Sparse storage: from_bmu → {to_bmu: count}
    unordered_map<int, unordered_map<int, float>> _P;
    unordered_map<int, float> _totals;

    FastTP() : n_neurons(0), eta(0.03f) {}

    FastTP(int n_neurons, float eta = 0.03f)
        : n_neurons(n_neurons), eta(eta) {}

    // Record a transition from → to with given weight (default 1.0)
    void observe(int from_bmu, int to_bmu, float weight = 1.0f) {
        if (from_bmu < 0 || to_bmu < 0 ||
            from_bmu >= n_neurons || to_bmu >= n_neurons) return;
        _P[from_bmu][to_bmu] += weight;
        _totals[from_bmu]    += weight;
    }

    // Reward-modulated soft update: strengthen or weaken a transition
    void reinforce(int from_bmu, int to_bmu, float reward) {
        if (from_bmu < 0 || to_bmu < 0) return;
        float delta = eta * reward;
        float& val  = _P[from_bmu][to_bmu];
        val = max(0.0f, val + delta);
        // Recompute row total
        float total = 0.0f;
        for (auto& kv : _P[from_bmu]) total += kv.second;
        _totals[from_bmu] = total;
    }

    // Sparse row: returns {(to_bmu, probability)} pairs
    vector<pair<int,float>> get_row(int bmu) const {
        auto it = _P.find(bmu);
        if (it == _P.end()) return {};
        float total = _totals.count(bmu) ? _totals.at(bmu) : 1.0f;
        if (total < 1e-9f) return {};
        vector<pair<int,float>> result;
        result.reserve(it->second.size());
        for (auto& kv : it->second)
            result.push_back({kv.first, kv.second / total});
        return result;
    }

    // Aggregate distribution from multiple seed BMUs.
    // Later seeds get higher weight (recency bias).
    // Returns dense probability vector length n_neurons.
    py::array_t<float> get_distribution(
            vector<int> seed_bmus,
            float temperature = 1.0f) const {

        auto result = py::array_t<float>(n_neurons);
        float* dist = static_cast<float*>(result.request().ptr);
        fill(dist, dist + n_neurons, 0.0f);

        if (seed_bmus.empty()) {
            fill(dist, dist + n_neurons, 1.0f / n_neurons);
            return result;
        }

        int   n = (int)seed_bmus.size();
        for (int si = 0; si < n; si++) {
            int   bmu     = seed_bmus[si];
            float recency = 0.5f + 0.5f * (float)si / max(n - 1, 1);
            auto  it      = _P.find(bmu);
            if (it == _P.end()) continue;
            float row_total = _totals.count(bmu) ? _totals.at(bmu) : 1.0f;
            if (row_total < 1e-9f) continue;
            for (auto& kv : it->second) {
                if (kv.first < n_neurons)
                    dist[kv.first] += recency * kv.second / row_total;
            }
        }

        // Normalize then apply temperature
        float sum = 0.0f;
        for (int i = 0; i < n_neurons; i++) sum += dist[i];

        if (sum < 1e-9f) {
            fill(dist, dist + n_neurons, 1.0f / n_neurons);
            return result;
        }

        if (fabsf(temperature - 1.0f) < 1e-4f) {
            for (int i = 0; i < n_neurons; i++) dist[i] /= sum;
        } else {
            // Tempered softmax
            float max_log = -1e9f;
            for (int i = 0; i < n_neurons; i++) {
                dist[i] = logf(max(dist[i] / sum, 1e-9f)) / temperature;
                if (dist[i] > max_log) max_log = dist[i];
            }
            sum = 0.0f;
            for (int i = 0; i < n_neurons; i++) {
                dist[i] = expf(dist[i] - max_log);
                sum += dist[i];
            }
            for (int i = 0; i < n_neurons; i++) dist[i] /= sum;
        }
        return result;
    }

    // Sample a BMU index from a probability distribution
    int sample(py::array_t<float> dist_arr, uint64_t seed = 0) const {
        auto buf = dist_arr.request();
        const float* dist = static_cast<const float*>(buf.ptr);
        int n = (int)buf.shape[0];

        uint64_t s = (seed > 0) ? seed
            : (uint64_t)chrono::steady_clock::now()
                       .time_since_epoch().count();
        mt19937_64 rng(s);
        uniform_real_distribution<float> udist(0.0f, 1.0f);

        float r = udist(rng), cumsum = 0.0f;
        for (int i = 0; i < n; i++) {
            cumsum += dist[i];
            if (r <= cumsum) return i;
        }
        return n - 1;
    }

    // Stats
    int n_transitions() const {
        int t = 0;
        for (auto& kv : _P) t += (int)kv.second.size();
        return t;
    }
    int n_active_neurons() const { return (int)_P.size(); }

    // ── Pickle support ───────────────────────────────────────────────
    py::dict __getstate__() const {
        py::dict d;
        d["n_neurons"] = n_neurons;
        d["eta"]       = eta;
        vector<int>   froms, tos;
        vector<float> vals;
        for (auto& row : _P)
            for (auto& kv : row.second) {
                froms.push_back(row.first);
                tos.push_back(kv.first);
                vals.push_back(kv.second);
            }
        d["froms"] = froms;
        d["tos"]   = tos;
        d["vals"]  = vals;
        return d;
    }

    void __setstate__(py::dict d) {
        n_neurons = d["n_neurons"].cast<int>();
        eta       = d["eta"].cast<float>();
        _P.clear(); _totals.clear();
        auto froms = d["froms"].cast<vector<int>>();
        auto tos   = d["tos"].cast<vector<int>>();
        auto vals  = d["vals"].cast<vector<float>>();
        for (size_t i = 0; i < froms.size(); i++) {
            _P[froms[i]][tos[i]] = vals[i];
            _totals[froms[i]]   += vals[i];
        }
    }
};


// ══════════════════════════════════════════════════════════════════════
// pybind11 module
// ══════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(brain_core, m) {
    m.doc() = "C++ core for THEBRAIN — FastSOM + FastTP";

    // ── FastSOM ─────────────────────────────────────────────────────
    py::class_<FastSOM>(m, "FastSOM")
        .def(py::init<int, int, int, float, float, float, float>(),
             py::arg("rows"), py::arg("cols"), py::arg("n_dims"),
             py::arg("init_lr")      = 0.15f,
             py::arg("init_radius")  = -1.0f,
             py::arg("lr_decay")     = 0.9998f,
             py::arg("radius_decay") = 0.9999f,
             "Create a rows×cols SOM with n_dims-dimensional input vectors")
        .def("find_bmu",         &FastSOM::find_bmu,
             py::arg("input"),
             "Find Best Matching Unit index for input vector (parallelized)")
        .def("update",           &FastSOM::update,
             py::arg("input"), py::arg("bmu"), py::arg("reward_mod") = 1.0f,
             "Kohonen weight update — reward_mod scales learning rate")
        .def("get_weights",      &FastSOM::get_weights,
             py::arg("i"), "Get weight vector of neuron i")
        .def("get_weight_matrix",&FastSOM::get_weight_matrix,
             "Get full [n_neurons × n_dims] weight matrix as numpy array")
        .def("set_weight_matrix",&FastSOM::set_weight_matrix,
             "Set weights from numpy array [n_neurons × n_dims]")
        .def_readonly("n_neurons",   &FastSOM::n_neurons)
        .def_readonly("rows",        &FastSOM::rows)
        .def_readonly("cols",        &FastSOM::cols)
        .def_readonly("n_dims",      &FastSOM::n_dims)
        .def_readonly("step_count",  &FastSOM::step_count)
        .def("__getstate__",     &FastSOM::__getstate__)
        .def("__setstate__",     &FastSOM::__setstate__);

    // ── FastTP ──────────────────────────────────────────────────────
    py::class_<FastTP>(m, "FastTP")
        .def(py::init<int, float>(),
             py::arg("n_neurons"), py::arg("eta") = 0.03f,
             "Sparse TP matrix for n_neurons BMUs")
        .def("observe",          &FastTP::observe,
             py::arg("from_bmu"), py::arg("to_bmu"), py::arg("weight") = 1.0f,
             "Record transition from_bmu → to_bmu")
        .def("reinforce",        &FastTP::reinforce,
             py::arg("from_bmu"), py::arg("to_bmu"), py::arg("reward"),
             "Reward-modulated soft update of a transition")
        .def("get_row",          &FastTP::get_row,
             py::arg("bmu"),
             "Sparse row: list of (to_bmu, probability) pairs")
        .def("get_distribution", &FastTP::get_distribution,
             py::arg("seed_bmus"), py::arg("temperature") = 1.0f,
             "Aggregate distribution from seed BMUs, return dense prob vector")
        .def("sample",           &FastTP::sample,
             py::arg("dist"), py::arg("seed") = 0,
             "Sample BMU index from probability distribution")
        .def("n_transitions",    &FastTP::n_transitions,
             "Total number of stored transitions")
        .def("n_active_neurons", &FastTP::n_active_neurons,
             "Number of neurons with at least one outgoing transition")
        .def("__getstate__",     &FastTP::__getstate__)
        .def("__setstate__",     &FastTP::__setstate__);

    // ── Build info ───────────────────────────────────────────────────
#ifdef _OPENMP
    m.attr("has_openmp") = true;
    m.attr("n_threads")  = omp_get_max_threads();
#else
    m.attr("has_openmp") = false;
    m.attr("n_threads")  = 1;
#endif
}
