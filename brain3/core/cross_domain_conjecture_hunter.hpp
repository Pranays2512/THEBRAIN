#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <random>

#include "../crisp/engines/reasoning/analogy_engine.hpp"
#include "algorithmic_policy_engine.hpp"

namespace brain3 {
namespace core {

struct CrossDomainDiscovery {
    std::string source_domain;
    std::string target_domain;
    double structural_score;
    std::string generalized_law_name;
    std::string abstract_formula;
    std::vector<std::string> mappings;
    std::vector<std::string> candidate_projections;
    // A structural alignment was found (Gentner systematicity >= 0.30). This is explicitly
    // NOT verification: no data is fitted and no proof is checked on this path. Named
    // `aligned` rather than `verified` so that no caller can mistake one for the other —
    // the previous name is why CROSS_DOMAIN_HUNT told users a hypothesis was a result.
    bool aligned;
    std::string timestamp;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"source_domain\": \"" << source_domain << "\",\n";
        oss << "  \"target_domain\": \"" << target_domain << "\",\n";
        oss << "  \"structural_score\": " << structural_score << ",\n";
        oss << "  \"generalized_law\": \"" << generalized_law_name << "\",\n";
        oss << "  \"abstract_formula\": \"" << abstract_formula << "\",\n";
        oss << "  \"structural_alignment\": " << (aligned ? "true" : "false") << ",\n";
        oss << "  \"verified\": false,\n";
        oss << "  \"verification_status\": \"UNVERIFIED — structural alignment only; "
               "no data fit, no proof check\",\n";
        oss << "  \"mappings\": [";
        for (size_t i = 0; i < mappings.size(); ++i) {
            oss << "\"" << mappings[i] << "\"" << (i + 1 < mappings.size() ? ", " : "");
        }
        oss << "],\n";
        oss << "  \"candidate_projections\": [";
        for (size_t i = 0; i < candidate_projections.size(); ++i) {
            oss << "\"" << candidate_projections[i] << "\"" << (i + 1 < candidate_projections.size() ? ", " : "");
        }
        oss << "]\n";
        oss << "}";
        return oss.str();
    }
};

class CrossDomainConjectureHunter {
private:
    brain2::reasoning::AnalogyEngine* analogy_engine_;
    AlgorithmicPolicyEngine* policy_engine_;
    std::mt19937 rng_{1337};
    std::vector<CrossDomainDiscovery> discoveries_;
    size_t total_scans_ = 0;
    size_t total_isomorphisms_found_ = 0;
    size_t cursor_i_ = 0;
    size_t cursor_j_ = 1;

public:
    CrossDomainConjectureHunter(brain2::reasoning::AnalogyEngine* ae, AlgorithmicPolicyEngine* pe)
        : analogy_engine_(ae), policy_engine_(pe) {}

    /**
     * Executes one autonomous cross-domain discovery step using systematic pairing + random jumps
     */
    CrossDomainDiscovery step_hunt() {
        total_scans_++;
        CrossDomainDiscovery disc;
        disc.aligned = false;
        disc.structural_score = 0.0;

        if (!analogy_engine_) return disc;

        const auto& domains = analogy_engine_->get_domains();
        if (domains.size() < 2) {
            return disc;
        }

        // Collect domain names
        std::vector<std::string> names;
        for (const auto& kv : domains) {
            names.push_back(kv.first);
        }

        // Advance systematic cursor across domain pairs
        if (cursor_i_ >= names.size()) cursor_i_ = 0;
        if (cursor_j_ >= names.size() || cursor_j_ <= cursor_i_) cursor_j_ = (cursor_i_ + 1) % names.size();

        std::string d1 = names[cursor_i_];
        std::string d2 = names[cursor_j_];

        // Increment pair pointers
        cursor_j_++;
        if (cursor_j_ >= names.size()) {
            cursor_i_++;
            cursor_j_ = (cursor_i_ + 1) % names.size();
        }

        // 1. Evaluate Gentner SME Alignment
        auto res = analogy_engine_->map_domains(d1, d2);
        disc.source_domain = d1;
        disc.target_domain = d2;
        disc.structural_score = res.score;

        for (const auto& kv : res.entity_map) {
            disc.mappings.push_back(kv.first + " <-> " + kv.second);
        }
        for (const auto& inf : res.candidate_inferences) {
            disc.candidate_projections.push_back(inf.to_string());
        }

        // 2. Perform AST Anti-Unification across equations if structural isomorphism is strong.
        //
        // WHAT score >= 0.30 ACTUALLY MEANS: at least 30% of the smaller domain's relation
        // triples found a consistent 1-to-1 partner in the other domain. That is a Gentner
        // systematicity overlap and NOTHING ELSE. No data was fitted, no residual computed,
        // no proof checked. A structural alignment is a HYPOTHESIS, not a result.
        //
        // This field was previously named `verified` and set true here, and the CROSS_DOMAIN_HUNT
        // command reported it to the user as a verified discovery "crystallized into the policy
        // store (O(1) verified)". That is the same class of defect commit 838880e removed from
        // the discovery engine, and self_play_discovery_daemon.hpp:320-325 already refuses to
        // count this signal for exactly this reason. The daemon's accounting was patched; this,
        // the user-facing source, was not. It is now.
        if (res.score >= 0.30 && !res.entity_map.empty()) {
            total_isomorphisms_found_++;
            auto anti_uni = synthesize_cross_domain_invariant(d1, d2, res);
            disc.generalized_law_name = anti_uni.first;
            disc.abstract_formula = anti_uni.second;
            disc.aligned = true;       // structural alignment found — NOT verification

            // Register into AlgorithmicPolicyEngine
            if (policy_engine_) {
                AlgorithmicPolicy policy;
                policy.problem_id = "cross_domain_" + d1 + "_to_" + d2;
                policy.paradigm = "Cross-Domain Structural Isomorphism";
                policy.mathematical_invariant = disc.abstract_formula;
                policy.transition_recurrence = disc.generalized_law_name;
                policy.paradigm = "Cross-Domain Structural Isomorphism (UNVERIFIED conjecture)";
                policy.time_complexity_budget = "O(1) Isomorphic Projection";
                policy.space_complexity_budget = "O(1) Dual Mapping";
                policy.io_policy = "Standard I/O";
                policy.gc_safety_policy = "Zero Heap Allocation";
                policy.constraints = disc.mappings;
                policy_engine_->register_policy(policy);
            }

            discoveries_.push_back(disc);
        }

        return disc;
    }

    const std::vector<CrossDomainDiscovery>& get_discoveries() const {
        return discoveries_;
    }

    std::string get_status_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"total_scans\": " << total_scans_ << ",\n";
        oss << "  \"isomorphisms_found\": " << total_isomorphisms_found_ << ",\n";
        oss << "  \"discoveries_count\": " << discoveries_.size() << ",\n";
        oss << "  \"recent_discoveries\": [\n";
        for (size_t i = 0; i < discoveries_.size(); ++i) {
            oss << discoveries_[i].to_json() << (i + 1 < discoveries_.size() ? ",\n" : "\n");
        }
        oss << "  ]\n";
        oss << "}";
        return oss.str();
    }

private:
private:
    /**
     * Real AST & Relational Anti-Unification:
     * Constructs abstract universal invariants by replacing matched domain-specific entities
     * with generalized symbolic variables (X_1, X_2, ...), synthesizing the abstract invariant formula.
     */
    std::pair<std::string, std::string> synthesize_cross_domain_invariant(
        const std::string& d1,
        const std::string& d2,
        const brain2::reasoning::AnalogyResult& match
    ) {
        // Map matched concrete entities to abstract variable slots (X_1, X_2, ...)
        std::unordered_map<std::string, std::string> src_to_abstract;
        std::unordered_map<std::string, std::string> tgt_to_abstract;
        std::ostringstream formula_builder;
        std::ostringstream law_name_builder;

        int slot_idx = 1;
        for (const auto& kv : match.entity_map) {
            std::string slot = "X_" + std::to_string(slot_idx++);
            src_to_abstract[kv.first] = slot;
            tgt_to_abstract[kv.second] = slot;
        }

        // Anti-unify matched relation triples into abstract invariant predicates
        std::vector<std::string> abstract_predicates;
        for (const auto& p : match.matched_triples) {
            std::string s_triple = p.first;  // e.g. "heat flows_to cold" or "water flows_to lower_tank"
            std::istringstream iss(s_triple);
            std::string subj, rel, obj;
            if (iss >> subj >> rel >> obj) {
                std::string a_subj = src_to_abstract.count(subj) ? src_to_abstract[subj] : ("'" + subj + "'");
                std::string a_obj  = src_to_abstract.count(obj)  ? src_to_abstract[obj]  : ("'" + obj + "'");
                std::string pred = rel + "(" + a_subj + ", " + a_obj + ")";
                if (std::find(abstract_predicates.begin(), abstract_predicates.end(), pred) == abstract_predicates.end()) {
                    abstract_predicates.push_back(pred);
                }
            }
        }

        law_name_builder << "Generalized [" << d1 << " <-> " << d2 << "] Morphism Invariant";

        if (!abstract_predicates.empty()) {
            formula_builder << "forall (";
            for (int i = 1; i < slot_idx; ++i) {
                formula_builder << "X_" << i << (i + 1 < slot_idx ? ", " : "");
            }
            formula_builder << ") : ";
            for (size_t i = 0; i < abstract_predicates.size(); ++i) {
                formula_builder << abstract_predicates[i] << (i + 1 < abstract_predicates.size() ? " ^ " : "");
            }
            formula_builder << " [Anti-Unified Systematicity: " << std::fixed << std::setprecision(2) << match.score << "]";
        } else {
            formula_builder << "Functor(" << d1 << ") =~= Functor(" << d2 << ") [Isomorphic Structure Preservation]";
        }

        return {law_name_builder.str(), formula_builder.str()};
    }
};

} // namespace core
} // namespace brain3
