#pragma once
#include "binding_memory.hpp"
#include <vector>
#include <cmath>

namespace brain2 {

struct AnalogyEngine {
    BindingMemory* bm = nullptr;
    
    AnalogyEngine() = default;
    AnalogyEngine(BindingMemory* binding) : bm(binding) {}

    // Structure Mapping
    // Given a novel situation (subject, relation), we find an analogous memory where the 
    // structural relationship matches, and the entities share abstract similarity in context.
    std::vector<float> structure_map(const std::vector<float>& subj, 
                                     const std::vector<float>& rel, 
                                     const std::vector<float>& context) {
        if (!bm || bm->bindings_.empty()) return std::vector<float>(subj.size(), 0.f);

        float best_score = -1.f;
        std::vector<float> best_obj(subj.size(), 0.f);

        for (const auto& b : bm->bindings_) {
            // 1. Structural match: Does the memory share the same core relation?
            float rel_sim = BindingMemory::cos_sim(rel, b.relation);
            
            // 2. Abstract entity match: Does the memory's subject share features with our novel subject
            // or the current abstract context (e.g. both are "dangerous" or "elements")
            float ctx_sim = BindingMemory::cos_sim(context, b.subject);
            float subj_sim = BindingMemory::cos_sim(subj, b.subject);
            
            // The magic of analogy: We don't need subj_sim to be 1.0 (exact match).
            // We just need a strong structural match (rel_sim) and some abstract conceptual overlap (ctx_sim).
            float score = rel_sim * 0.6f + std::max(ctx_sim, subj_sim) * 0.4f;
            
            if (score > best_score) {
                best_score = score;
                best_obj = b.object;
            }
        }
        return best_obj;
    }
};

} // namespace brain2
