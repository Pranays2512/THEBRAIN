#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace brain2 {

struct BindingMemory {
    struct Binding {
        std::vector<int>   index;     // sparse random tag (~20 nonzero positions)
        std::vector<float> subject;
        std::vector<float> relation;
        std::vector<float> object;
        int                timestamp = 0;
        float              strength  = 1.f;
    };

    int n_dims       = 0;
    int max_bindings = 1000;
    float decay_     = 0.999f;
    int   step_      = 0;

    std::vector<Binding> bindings_;
    std::mt19937         rng_;

    BindingMemory() : rng_(42) {}
    BindingMemory(int n_dims, int max_bindings = 1000)
        : n_dims(n_dims), max_bindings(max_bindings), rng_(42) {}

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

        // Evict weakest if at capacity
        if ((int)bindings_.size() >= max_bindings) {
            auto it = std::min_element(bindings_.begin(), bindings_.end(),
                [](const Binding& a, const Binding& b){ return a.strength < b.strength; });
            bindings_.erase(it);
        }

        // Sparse random index
        std::uniform_int_distribution<int> dist(0, n_dims - 1);
        std::vector<int> idx(20);
        for (auto& v : idx) v = dist(rng_);

        bindings_.push_back({idx, subj, rel, obj, step_++, 1.f});
    }

    // Query: given (subj, rel) → obj  [want_object=true]
    //         given (subj, obj) → rel  [want_object=false]
    // Returns pair of (best_match_vector, confidence_score)
    std::pair<std::vector<float>, float> query(const std::vector<float>& a,
                                               const std::vector<float>& b,
                                               bool want_object = true,
                                               float threshold = 0.3f) const {
        float best = -1.f;
        const Binding* best_bnd = nullptr;
        for (const auto& bnd : bindings_) {
            float sa = cos_sim(a, bnd.subject);
            float sb = want_object ? cos_sim(b, bnd.relation) : cos_sim(b, bnd.object);
            float s  = (sa + sb) * 0.5f; 
            if (s > best) { best = s; best_bnd = &bnd; }
        }
        if (!best_bnd || best < threshold) return {std::vector<float>(n_dims, 0.f), best};
        return {want_object ? best_bnd->object : best_bnd->relation, best};
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
        return bm;
    }
};

} // namespace brain2
