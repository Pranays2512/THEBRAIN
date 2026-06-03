#pragma once
#include "basal_ganglia.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace brain2 {

struct ProceduralMemory {
    struct Procedure {
        std::string        name;
        std::vector<float> trigger_embedding;
        std::vector<Op>    steps;
        float              success_rate = 1.f;
        int                use_count    = 0;
    };

    int n_dims         = 0;
    int max_procedures = 200;
    std::vector<Procedure> procedures_;

    ProceduralMemory() = default;
    explicit ProceduralMemory(int n_dims) : n_dims(n_dims) {}

    static float cos_sim(const std::vector<float>& a, const std::vector<float>& b) {
        float dot = 0, na = 0, nb = 0;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; i++) {
            dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
        }
        return (na < 1e-8f || nb < 1e-8f) ? 0.f : dot / (std::sqrt(na) * std::sqrt(nb));
    }

    // Store a successful op-chain given the context that triggered it
    void consolidate(const std::vector<Op>& ops,
                     const std::vector<float>& context,
                     const std::string& name = "") {
        // Update existing procedure if context is very similar
        for (auto& p : procedures_) {
            if (cos_sim(p.trigger_embedding, context) > 0.9f) {
                p.steps = ops;  // overwrite the sequence with the new one
                p.success_rate = 0.9f * p.success_rate + 0.1f * 1.0f;
                p.use_count++;
                return;
            }
        }
        // Evict least-used if at capacity
        if ((int)procedures_.size() >= max_procedures) {
            auto it = std::min_element(procedures_.begin(), procedures_.end(),
                [](const Procedure& a, const Procedure& b){
                    return a.use_count < b.use_count; });
            procedures_.erase(it);
        }
        Procedure p;
        p.name              = name.empty()
                            ? ("proc_" + std::to_string(procedures_.size()))
                            : name;
        p.trigger_embedding = context;
        p.steps             = ops;
        p.success_rate      = 1.f;
        p.use_count         = 1;
        procedures_.push_back(std::move(p));
    }

    // Retrieve the best-matching procedure for current context (nullptr if none)
    Procedure* retrieve(const std::vector<float>& context) {
        float best = 0.4f;  // minimum similarity threshold
        Procedure* best_p = nullptr;
        for (auto& p : procedures_) {
            float s = cos_sim(p.trigger_embedding, context) * p.success_rate;
            if (s > best) { best = s; best_p = &p; }
        }
        if (best_p) best_p->use_count++;
        return best_p;
    }

    int size() const { return (int)procedures_.size(); }

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("ProceduralMemory::save: cannot open " + path);
        f.write((const char*)&n_dims, sizeof(int));
        int n = (int)procedures_.size();
        f.write((const char*)&n, sizeof(int));
        for (const auto& p : procedures_) {
            int ns = (int)p.name.size();
            f.write((const char*)&ns, sizeof(int));
            f.write(p.name.data(), ns);
            f.write((const char*)p.trigger_embedding.data(), n_dims * sizeof(float));
            int st = (int)p.steps.size();
            f.write((const char*)&st, sizeof(int));
            f.write((const char*)p.steps.data(), st * sizeof(Op));
            f.write((const char*)&p.success_rate, sizeof(float));
            f.write((const char*)&p.use_count,    sizeof(int));
        }
    }

    static ProceduralMemory load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("ProceduralMemory::load: cannot open " + path);
        ProceduralMemory pm;
        f.read((char*)&pm.n_dims, sizeof(int));
        int n; f.read((char*)&n, sizeof(int));
        pm.procedures_.resize(n);
        for (auto& p : pm.procedures_) {
            int ns; f.read((char*)&ns, sizeof(int));
            p.name.resize(ns); f.read(&p.name[0], ns);
            p.trigger_embedding.resize(pm.n_dims);
            f.read((char*)p.trigger_embedding.data(), pm.n_dims * sizeof(float));
            int st; f.read((char*)&st, sizeof(int));
            p.steps.resize(st);
            f.read((char*)p.steps.data(), st * sizeof(Op));
            f.read((char*)&p.success_rate, sizeof(float));
            f.read((char*)&p.use_count,    sizeof(int));
        }
        return pm;
    }
};

} // namespace brain2
