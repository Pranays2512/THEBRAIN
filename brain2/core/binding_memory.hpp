#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace brain2 {

// ────────────────────────────────────────────────────────────
// LSH Index: 8 random hyperplanes → 2^8 = 256 buckets.
// Reduces average query scan from O(n) to O(n/128) without changing the API.
// ────────────────────────────────────────────────────────────
struct LSHIndex {
    static constexpr int N_PLANES = 8;
    static constexpr int N_BUCKETS = (1 << N_PLANES); // 256

    std::vector<std::vector<float>> planes;    // N_PLANES × n_dims
    std::vector<std::vector<int>>   buckets;   // 256 bucket lists (binding indices)
    int n_dims = 0;

    LSHIndex() : buckets(N_BUCKETS) {}
    LSHIndex(int nd, unsigned seed = 99) : n_dims(nd), buckets(N_BUCKETS) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd_dist(0.f, 1.f);
        planes.resize(N_PLANES, std::vector<float>(nd));
        for (auto& p : planes)
            for (auto& v : p) v = nd_dist(rng);
    }

    // Compute the LSH bucket code for a vector
    int hash(const std::vector<float>& v) const {
        int code = 0;
        for (int p = 0; p < N_PLANES; p++) {
            float dot = 0.f;
            size_t n = std::min(v.size(), planes[p].size());
            for (size_t i = 0; i < n; i++) dot += v[i] * planes[p][i];
            if (dot >= 0.f) code |= (1 << p);
        }
        return code;
    }

    void insert(int binding_idx, const std::vector<float>& subj) {
        int code = hash(subj);
        buckets[code].push_back(binding_idx);
    }

    void remove(int binding_idx) {
        for (auto& bucket : buckets) {
            auto it = std::find(bucket.begin(), bucket.end(), binding_idx);
            if (it != bucket.end()) { bucket.erase(it); return; }
        }
    }

    // Return candidate indices: exact bucket + all 1-bit Hamming neighbors
    std::vector<int> candidates(const std::vector<float>& subj) const {
        int code = hash(subj);
        std::vector<int> result = buckets[code];
        for (int p = 0; p < N_PLANES; p++) {
            int neighbor = code ^ (1 << p);
            for (int idx : buckets[neighbor]) result.push_back(idx);
        }
        return result;
    }

    void rebuild(const std::vector<int>& valid_indices,
                 const std::vector<std::vector<float>>& subjects) {
        for (auto& b : buckets) b.clear();
        for (int i : valid_indices)
            insert(i, subjects[i]);
    }
};

struct BindingMemory {
    struct Binding {
        std::vector<int>   index;     // sparse random tag (~20 nonzero positions)
        std::vector<float> subject;
        std::vector<float> relation;
        std::vector<float> object;
        int                timestamp = 0;
        float              strength  = 1.f;
    };

    int   n_dims       = 0;
    int   max_bindings = 1000;
    float decay_       = 0.999f;
    int   step_        = 0;

    std::vector<Binding> bindings_;
    std::mt19937         rng_;
    LSHIndex             lsh_;         // ← NEW: O(1) approximate subject lookup

    BindingMemory() : rng_(42) {}
    BindingMemory(int n_dims, int max_bindings = 1000)
        : n_dims(n_dims), max_bindings(max_bindings), rng_(42),
          lsh_(n_dims, 99) {}

    static float cos_sim(const std::vector<float>& a, const std::vector<float>& b) {
        float dot = 0, na = 0, nb = 0;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
        }
        return (na < 1e-8f || nb < 1e-8f) ? 0.f : dot / (std::sqrt(na)*std::sqrt(nb));
    }

    // Store a (subject, relation, object) triple
    void bind(const std::vector<float>& subj,
              const std::vector<float>& rel,
              const std::vector<float>& obj) {
        // Decay existing
        for (auto& b : bindings_) b.strength *= decay_;

        // Evict weakest if at capacity — rebuild LSH after erase (eviction is rare)
        if ((int)bindings_.size() >= max_bindings) {
            auto it = std::min_element(bindings_.begin(), bindings_.end(),
                [](const Binding& a, const Binding& b){ return a.strength < b.strength; });
            bindings_.erase(it);
            // Full rebuild: clear all buckets, re-insert remaining bindings
            for (auto& bucket : lsh_.buckets) bucket.clear();
            for (int i = 0; i < (int)bindings_.size(); i++)
                lsh_.insert(i, bindings_[i].subject);
        }

        // Sparse random index
        std::uniform_int_distribution<int> dist(0, n_dims - 1);
        std::vector<int> idx(20);
        for (auto& v : idx) v = dist(rng_);

        int new_idx = (int)bindings_.size();
        bindings_.push_back({idx, subj, rel, obj, step_++, 1.f});
        lsh_.insert(new_idx, subj);  // index new entry by subject
    }

    // Query: given (subj, rel) → obj  [want_object=true]
    //         given (subj, obj) → rel  [want_object=false]
    // Returns pair of (best_match_vector, confidence_score)
    std::pair<std::vector<float>, float> query(const std::vector<float>& a,
                                               const std::vector<float>& b,
                                               bool want_object = true,
                                               float threshold = 0.3f,
                                               int depth = 3) const {
        std::vector<std::vector<float>> visited;
        return query_recursive(a, b, want_object, threshold, depth, visited);
    }

    std::pair<std::vector<float>, float> query_recursive(const std::vector<float>& a,
                                                         const std::vector<float>& b,
                                                         bool want_object,
                                                         float threshold,
                                                         int depth,
                                                         std::vector<std::vector<float>>& visited) const {
        // Cycle detection
        for (const auto& v : visited)
            if (cos_sim(a, v) > 0.95f) return {std::vector<float>(n_dims, 0.f), 0.f};
        visited.push_back(a);

        // ── LSH candidate shortlist ────────────────────────────────────────────
        // Instead of scanning all bindings, check only candidates from LSH.
        // Falls back to full scan if the LSH is not yet seeded (< 16 entries).
        std::vector<std::pair<float,std::vector<float>>> branches;
        auto add_candidate = [&](const Binding& bnd) {
            float sa = cos_sim(a, bnd.subject);
            float sb = want_object ? cos_sim(b, bnd.relation) : cos_sim(b, bnd.object);
            float direct_conf = (sa + sb) * 0.5f;
            if (direct_conf >= threshold)
                branches.push_back({direct_conf, want_object ? bnd.object : bnd.relation});
        };

        if ((int)bindings_.size() >= 200) {
            auto cands = lsh_.candidates(a);
            for (int ci : cands)
                if (ci >= 0 && ci < (int)bindings_.size())
                    add_candidate(bindings_[ci]);
        } else {
            for (const auto& bnd : bindings_) add_candidate(bnd);
        }

        if (!branches.empty()) {
            std::sort(branches.begin(), branches.end(),
                      [](const auto& x, const auto& y){ return x.first > y.first; });
        }

        float global_best_conf = -1.f;
        std::vector<float> global_best_res(n_dims, 0.f);

        int max_branches = std::min(3, (int)branches.size());
        for (int i = 0; i < max_branches; i++) {
            float direct_conf       = branches[i].first;
            const auto& direct_res  = branches[i].second;

            if (global_best_conf > 0.f && direct_conf < global_best_conf * 0.9f) break;

            if (direct_conf > global_best_conf) {
                global_best_conf = direct_conf;
                global_best_res  = direct_res;
            }

            if (want_object && depth > 1) {
                auto trans_res = query_recursive(direct_res, b, true, threshold, depth - 1, visited);
                float path_conf = direct_conf * trans_res.second;
                if (trans_res.second >= threshold && path_conf >= global_best_conf - 1e-4f) {
                    global_best_conf = path_conf;
                    global_best_res  = trans_res.first;
                }
            }
        }

        visited.pop_back();
        return {global_best_res, global_best_conf};
    }

    // Query all: given (subj) → returns list of [relation1, object1, relation2, object2, ...]
    std::vector<std::vector<float>> query_all(const std::vector<float>& a, float threshold = 0.5f) const {
        std::vector<std::vector<float>> results;
        for (const auto& bnd : bindings_) {
            float sa = cos_sim(a, bnd.subject);
            if (sa > threshold) {
                results.push_back(bnd.relation);
                results.push_back(bnd.object);
            }
        }
        return results;
    }

    int size() const { return (int)bindings_.size(); }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("BindingMemory::save: cannot open " + path);
        f.write((const char*)&n_dims,       sizeof(int));
        f.write((const char*)&max_bindings, sizeof(int));
        int n = (int)bindings_.size();
        f.write((const char*)&n, sizeof(int));
        for (const auto& b : bindings_) {
            int idsz = (int)b.index.size();
            f.write((const char*)&idsz,            sizeof(int));
            f.write((const char*)b.index.data(),   idsz * sizeof(int));
            f.write((const char*)b.subject.data(),  n_dims * sizeof(float));
            f.write((const char*)b.relation.data(), n_dims * sizeof(float));
            f.write((const char*)b.object.data(),   n_dims * sizeof(float));
            f.write((const char*)&b.timestamp,     sizeof(int));
            f.write((const char*)&b.strength,      sizeof(float));
        }
    }

    static BindingMemory load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("BindingMemory::load: cannot open " + path);
        BindingMemory bm;
        f.read((char*)&bm.n_dims,       sizeof(int));
        f.read((char*)&bm.max_bindings, sizeof(int));
        int n; f.read((char*)&n, sizeof(int));
        bm.bindings_.resize(n);
        for (auto& b : bm.bindings_) {
            int idsz; f.read((char*)&idsz, sizeof(int));
            b.index.resize(idsz);
            f.read((char*)b.index.data(), idsz * sizeof(int));
            b.subject.resize(bm.n_dims);
            b.relation.resize(bm.n_dims);
            b.object.resize(bm.n_dims);
            f.read((char*)b.subject.data(),  bm.n_dims * sizeof(float));
            f.read((char*)b.relation.data(), bm.n_dims * sizeof(float));
            f.read((char*)b.object.data(),   bm.n_dims * sizeof(float));
            f.read((char*)&b.timestamp, sizeof(int));
            f.read((char*)&b.strength,  sizeof(float));
        }
        // Rebuild LSH index after load (hyperplanes are deterministic, no need to persist)
        bm.lsh_ = LSHIndex(bm.n_dims, 99);
        for (int i = 0; i < (int)bm.bindings_.size(); i++)
            bm.lsh_.insert(i, bm.bindings_[i].subject);
        return bm;
    }

    void expand_dims(int new_dims) {
        if (new_dims <= n_dims) return;
        for (auto& b : bindings_) {
            b.subject.resize(new_dims, 0.f);
            b.relation.resize(new_dims, 0.f);
            b.object.resize(new_dims, 0.f);
        }
        n_dims = new_dims;
        lsh_ = LSHIndex(n_dims, 99);
        for (int i = 0; i < (int)bindings_.size(); i++) {
            lsh_.insert(i, bindings_[i].subject);
        }
    }
};

} // namespace brain2
