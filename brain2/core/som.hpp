#pragma once
/*
 * som.hpp — Self-Organizing Map, Component 1 of Brain v2
 *
 * General-purpose SOM. Not tied to any world or task.
 * Input: any float vector of n_dims.
 * Learns via unsupervised competitive learning + neighbourhood update.
 *
 * Thread safety:
 *   find_bmu(), activation_map()  — read-only, safe to call concurrently
 *   update()                      — write, mutex-protected
 *
 * SIMD: relies on clang/gcc auto-vectorization with -O3 -ffast-math.
 *       Produces NEON on ARM (Apple Silicon), AVX2 on x86 automatically.
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <mutex>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <memory>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace brain2 {

class SOM {
public:
    int rows, cols, n_neurons, n_dims;

private:
    std::vector<float>           weights_;
    float                        lr_, radius_, lr_decay_, radius_decay_;
    int                          step_;
    std::unique_ptr<std::mutex>  update_mtx_;

    // L2 squared distance — simple loop, auto-vectorized by compiler
    inline float l2sq(const float* __restrict__ a,
                      const float* __restrict__ b) const noexcept {
        float s = 0.f;
        for (int i = 0; i < n_dims; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

public:
    SOM() : rows(0), cols(0), n_neurons(0), n_dims(0),
            lr_(0), radius_(0), lr_decay_(0), radius_decay_(0), step_(0),
            update_mtx_(std::make_unique<std::mutex>()) {}

    SOM(int rows, int cols, int n_dims,
        float init_lr      = 0.15f,
        float lr_decay     = 0.9998f,
        float radius_decay = 0.9999f,
        unsigned seed      = 42)
        : rows(rows), cols(cols), n_neurons(rows * cols), n_dims(n_dims),
          lr_(init_lr),
          radius_(float(std::max(rows, cols)) / 2.f),
          lr_decay_(lr_decay), radius_decay_(radius_decay), step_(0),
          update_mtx_(std::make_unique<std::mutex>())
    {
        weights_.resize(size_t(n_neurons) * n_dims);
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.f, 0.3f);
        for (auto& w : weights_) w = dist(rng);
    }

    // Move-only (mutex is not copyable)
    SOM(SOM&&)            = default;
    SOM& operator=(SOM&&) = default;
    SOM(const SOM&)       = delete;
    SOM& operator=(const SOM&) = delete;

    // Find best matching unit — parallel read, no lock
    int find_bmu(const std::vector<float>& input) const {
        if ((int)input.size() != n_dims)
            throw std::invalid_argument("SOM::find_bmu: input dim mismatch");
        const float* inp = input.data();
        float best_d = std::numeric_limits<float>::max();
        int   best_i = 0;

#ifdef USE_OPENMP
        #pragma omp parallel
        {
            float ld = std::numeric_limits<float>::max();
            int   li = 0;
            #pragma omp for nowait schedule(static)
            for (int i = 0; i < n_neurons; i++) {
                float d = l2sq(inp, weights_.data() + size_t(i) * n_dims);
                if (d < ld) { ld = d; li = i; }
            }
            #pragma omp critical
            { if (ld < best_d) { best_d = ld; best_i = li; } }
        }
#else
        for (int i = 0; i < n_neurons; i++) {
            float d = l2sq(inp, weights_.data() + size_t(i) * n_dims);
            if (d < best_d) { best_d = d; best_i = i; }
        }
#endif
        return best_i;
    }

    // Full activation map: inverted normalized distance per neuron
    std::vector<float> activation_map(const std::vector<float>& input) const {
        if ((int)input.size() != n_dims)
            throw std::invalid_argument("SOM::activation_map: input dim mismatch");
        const float* inp = input.data();
        std::vector<float> dists(n_neurons, 0.f);

#ifdef USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_neurons; i++)
            dists[i] = l2sq(inp, weights_.data() + size_t(i) * n_dims);

        float mn = *std::min_element(dists.begin(), dists.end());
        float mx = *std::max_element(dists.begin(), dists.end());
        float rng = mx - mn;
        if (rng > 0.f)
            for (auto& d : dists) d = 1.f - (d - mn) / rng;
        else
            std::fill(dists.begin(), dists.end(), 1.f);
        return dists;
    }

    // 2D grid distance between neurons i and j
    inline float grid_dist(int i, int j) const noexcept {
        float dr = float(i / cols) - float(j / cols);
        float dc = float(i % cols) - float(j % cols);
        return std::sqrt(dr * dr + dc * dc);
    }

    // Update weights toward input — mutex-protected
    void update(const std::vector<float>& input, int bmu,
                float reward_mod = 1.f) {
        std::lock_guard<std::mutex> lock(*update_mtx_);
        const float* inp = input.data();
        float eff_lr = lr_ * std::max(0.01f, std::min(reward_mod, 3.f));
        float r2     = radius_ * radius_ * 2.f;

#ifdef USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n_neurons; i++) {
            float d = grid_dist(i, bmu);
            float h = std::exp(-d * d / r2);
            if (h < 1e-4f) continue;
            float* w  = weights_.data() + size_t(i) * n_dims;
            float  sc = eff_lr * h;
            for (int j = 0; j < n_dims; j++)
                w[j] += sc * (inp[j] - w[j]);
        }
        lr_     *= lr_decay_;
        radius_ *= radius_decay_;
        step_++;
    }

    // Get neuron weight vector (copy — safe across threads)
    std::vector<float> neuron_weights(int i) const {
        if (i < 0 || i >= n_neurons)
            throw std::out_of_range("SOM::neuron_weights: out of range");
        auto b = weights_.begin() + size_t(i) * n_dims;
        return std::vector<float>(b, b + n_dims);
    }

    int   step()   const noexcept { return step_;   }
    float lr()     const noexcept { return lr_;     }
    float radius() const noexcept { return radius_; }

    // Binary save/load
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("SOM::save: cannot open " + path);
        f.write((const char*)&rows,          sizeof(int));
        f.write((const char*)&cols,          sizeof(int));
        f.write((const char*)&n_dims,        sizeof(int));
        f.write((const char*)&lr_,           sizeof(float));
        f.write((const char*)&radius_,       sizeof(float));
        f.write((const char*)&lr_decay_,     sizeof(float));
        f.write((const char*)&radius_decay_, sizeof(float));
        f.write((const char*)&step_,         sizeof(int));
        f.write((const char*)weights_.data(),
                (std::streamsize)(weights_.size() * sizeof(float)));
    }

    static SOM load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("SOM::load: cannot open " + path);
        SOM s;
        f.read((char*)&s.rows,          sizeof(int));
        f.read((char*)&s.cols,          sizeof(int));
        f.read((char*)&s.n_dims,        sizeof(int));
        f.read((char*)&s.lr_,           sizeof(float));
        f.read((char*)&s.radius_,       sizeof(float));
        f.read((char*)&s.lr_decay_,     sizeof(float));
        f.read((char*)&s.radius_decay_, sizeof(float));
        f.read((char*)&s.step_,         sizeof(int));
        s.n_neurons = s.rows * s.cols;
        s.weights_.resize(size_t(s.n_neurons) * s.n_dims);
        f.read((char*)s.weights_.data(),
               (std::streamsize)(s.weights_.size() * sizeof(float)));
        return s;
    }
};

} // namespace brain2
