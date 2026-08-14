#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

namespace brain2 {
namespace reasoning {

struct InnateDrives {
    double epistemic_curiosity = 0.80;    // Desire to explore unknown / high-entropy stimuli
    double contradiction_aversion = 0.95; // Visceral alarm triggered by logical / physical contradictions
    double parsimony = 0.75;              // Occam's razor preference for minimal complexity
    double safety_preservation = 0.99;    // Protective filter guarding core identity & invariants

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{"
            << "\"curiosity\": " << std::fixed << std::setprecision(2) << epistemic_curiosity << ", "
            << "\"contradiction_aversion\": " << contradiction_aversion << ", "
            << "\"parsimony\": " << parsimony << ", "
            << "\"safety\": " << safety_preservation
            << "}";
        return oss.str();
    }
};

struct ReflexArc {
    std::string signature;
    std::string domain;
    std::string action_response;
    double confidence = 0.50;
    int activation_count = 0;
    int success_count = 0;
    int failure_count = 0;
    bool is_innate = false;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{"
            << "\"signature\": \"" << signature << "\", "
            << "\"domain\": \"" << domain << "\", "
            << "\"action\": \"" << action_response << "\", "
            << "\"confidence\": " << std::fixed << std::setprecision(2) << confidence << ", "
            << "\"activations\": " << activation_count << ", "
            << "\"successes\": " << success_count << ", "
            << "\"is_innate\": " << (is_innate ? "true" : "false")
            << "}";
        return oss.str();
    }
};

struct InstinctResponse {
    bool has_reflex = false;
    std::string action = "";
    double confidence = 0.0;
    std::string domain = "general";
    InnateDrives drives;
    std::string explanation = "";
    std::vector<std::string> steps;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\n"
            << "  \"verified\": " << (has_reflex ? "true" : "false") << ",\n"
            << "  \"reflex_fired\": " << (has_reflex ? "true" : "false") << ",\n"
            << "  \"action\": \"" << action << "\",\n"
            << "  \"confidence\": " << confidence << ",\n"
            << "  \"domain\": \"" << domain << "\",\n"
            << "  \"drives\": " << drives.to_json() << ",\n"
            << "  \"explanation\": \"" << explanation << "\",\n"
            << "  \"steps\": [\n";
        for (size_t i = 0; i < steps.size(); ++i) {
            oss << "    \"" << steps[i] << "\"";
            if (i + 1 < steps.size()) oss << ",";
            oss << "\n";
        }
        oss << "  ]\n}";
        return oss.str();
    }
};

class InstinctEngine {
public:
    std::map<std::string, ReflexArc> reflex_arcs;
    InnateDrives drives;
    double reflex_threshold = 0.60;
    int total_reflex_fires = 0;
    int total_reflex_hits = 0;

    InstinctEngine() {
        init_innate_primal_instincts();
    }

    void init_innate_primal_instincts() {
        // ── 1. Innate Contradiction & Danger Alarms ──
        add_innate_reflex("contradiction", "safety", "ALARM: Contradiction detected - trigger epistemic defense", 0.99);
        add_innate_reflex("1=0", "safety", "ALARM: Mathematical absurdity detected - reject state", 1.00);
        add_innate_reflex("poison_invariants", "safety", "ALARM: Invariant poisoning threat detected - lock truth store", 1.00);
        add_innate_reflex("destroy_self", "safety", "ALARM: Self-destructive action intercepted", 1.00);

        // ── 2. Innate Arithmetic & Identity Groundings ──
        add_innate_reflex("0*x", "math", "0", 0.95);
        add_innate_reflex("x*0", "math", "0", 0.95);
        add_innate_reflex("1*x", "math", "x", 0.95);
        add_innate_reflex("x*1", "math", "x", 0.95);
        add_innate_reflex("x+0", "math", "x", 0.95);
        add_innate_reflex("0+x", "math", "x", 0.95);
        add_innate_reflex("2+2", "math", "4", 0.95);

        // ── 3. Innate Physical Invariant Reflexes ──
        add_innate_reflex("f_net=0", "physics", "accel = 0 (Newton First Law equilibrium)", 0.90);
        add_innate_reflex("free_fall_gravity", "physics", "a = 9.8 m/s^2 downwards", 0.90);

        // ── 4. Innate Logic Invariants ──
        add_innate_reflex("p_and_not_p", "logic", "FALSE (Law of Non-Contradiction)", 1.00);
        add_innate_reflex("p_or_not_p", "logic", "TRUE (Law of Excluded Middle)", 1.00);
    }

    void add_innate_reflex(const std::string& signature, const std::string& domain, const std::string& action, double conf = 0.90) {
        std::string norm_sig = normalize_sig(signature);
        ReflexArc arc;
        arc.signature = norm_sig;
        arc.domain = domain;
        arc.action_response = action;
        arc.confidence = conf;
        arc.is_innate = true;
        arc.success_count = 1;
        reflex_arcs[norm_sig] = arc;
    }

    // Subconscious System 1 Fast Evaluation (< 0.05ms)
    InstinctResponse evaluate_instinct(const std::string& query) {
        total_reflex_fires++;
        InstinctResponse res;
        res.drives = evaluate_innate_drives(query);

        std::string norm = normalize_sig(query);
        std::vector<std::string> steps;
        steps.push_back("⚡ System 1 Subconscious Evaluation of signature: '" + norm + "'");

        // 1. Direct Reflex Arc Match
        auto it = reflex_arcs.find(norm);
        if (it != reflex_arcs.end()) {
            it->second.activation_count++;
            if (it->second.confidence >= reflex_threshold) {
                total_reflex_hits++;
                it->second.success_count++;
                res.has_reflex = true;
                res.action = it->second.action_response;
                res.confidence = it->second.confidence;
                res.domain = it->second.domain;
                res.explanation = "Subconscious instinct fired: [" + it->second.domain + "] " + it->second.action_response + " (confidence: " + format_num(it->second.confidence) + ")";
                steps.push_back("✓ Reflex match found in domain '" + it->second.domain + "' with confidence " + format_num(it->second.confidence));
                steps.push_back("✓ Instantaneous reflex response: " + it->second.action_response);
                res.steps = steps;
                return res;
            }
        }

        // 2. Substring & Semantic Keyword Resonance Matching
        for (auto& kv : reflex_arcs) {
            if (kv.second.confidence >= reflex_threshold && (norm.find(kv.first) != std::string::npos || kv.first.find(norm) != std::string::npos)) {
                kv.second.activation_count++;
                total_reflex_hits++;
                kv.second.success_count++;
                res.has_reflex = true;
                res.action = kv.second.action_response;
                res.confidence = kv.second.confidence * 0.95; // Slight decay for partial resonance
                res.domain = kv.second.domain;
                res.explanation = "Subconscious resonant instinct fired: [" + kv.second.domain + "] " + kv.second.action_response;
                steps.push_back("✓ Partial pattern resonance matched reflex '" + kv.first + "'");
                steps.push_back("✓ Instantaneous reflex response: " + kv.second.action_response);
                res.steps = steps;
                return res;
            }
        }

        // 3. Reflex Missing or Sub-Threshold -> Escalate to System 2 Deliberative Reasoning
        res.has_reflex = false;
        res.confidence = 0.0;
        res.domain = "unconscious_gap";
        res.explanation = "No instinctual reflex met confidence threshold (" + format_num(reflex_threshold) + "). Escalating to deliberate System 2 reasoning.";
        steps.push_back("No high-confidence instinctual reflex found. Escalating to deliberative reasoning.");
        res.steps = steps;
        return res;
    }

    // System 2 -> System 1 Compilation: Crystallize verified solution into a permanent reflex arc
    void crystallize_reflex(const std::string& signature, const std::string& domain, const std::string& action, double init_confidence = 0.85) {
        std::string norm = normalize_sig(signature);
        if (reflex_arcs.count(norm)) {
            // Reinforce existing reflex
            reinforce_reflex(norm, 0.10);
            reflex_arcs[norm].action_response = action;
        } else {
            ReflexArc arc;
            arc.signature = norm;
            arc.domain = domain;
            arc.action_response = action;
            arc.confidence = std::clamp(init_confidence, 0.10, 0.99);
            arc.activation_count = 1;
            arc.success_count = 1;
            arc.failure_count = 0;
            arc.is_innate = false;
            reflex_arcs[norm] = arc;
        }
    }

    // Hebbian Positive Reinforcement
    void reinforce_reflex(const std::string& signature, double reward = 0.10) {
        std::string norm = normalize_sig(signature);
        auto it = reflex_arcs.find(norm);
        if (it != reflex_arcs.end()) {
            it->second.success_count++;
            it->second.confidence = std::min(0.99, it->second.confidence + reward);
        }
    }

    // Anti-Hebbian Disruption / Suppression when an instinctual impulse is refuted
    void penalize_reflex(const std::string& signature, double penalty = 0.30) {
        std::string norm = normalize_sig(signature);
        auto it = reflex_arcs.find(norm);
        if (it != reflex_arcs.end()) {
            it->second.failure_count++;
            it->second.confidence = std::max(0.05, it->second.confidence - penalty);
        }
    }

    // Dynamic Innate Drive Evaluation
    InnateDrives evaluate_innate_drives(const std::string& percept) {
        InnateDrives d = drives;
        std::string lower = percept;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        // Contradiction Aversion spike
        if (lower.find("not") != std::string::npos && lower.find("is") != std::string::npos) {
            d.contradiction_aversion = 0.98;
        }
        if (lower.find("contradiction") != std::string::npos || lower.find("1=0") != std::string::npos || lower.find("false=true") != std::string::npos) {
            d.contradiction_aversion = 1.00;
        }

        // Epistemic Curiosity spike for novel / unknown questions
        if (lower.find("?") != std::string::npos || lower.find("why") != std::string::npos || lower.find("how") != std::string::npos || lower.find("unknown") != std::string::npos) {
            d.epistemic_curiosity = 0.95;
        }

        // Safety drive spike on danger terms
        if (lower.find("poison") != std::string::npos || lower.find("destroy") != std::string::npos || lower.find("delete") != std::string::npos) {
            d.safety_preservation = 1.00;
        }

        return d;
    }

    std::string get_status_json() const {
        std::ostringstream oss;
        oss << "{\n"
            << "  \"total_reflex_arcs\": " << reflex_arcs.size() << ",\n"
            << "  \"total_fires\": " << total_reflex_fires << ",\n"
            << "  \"total_hits\": " << total_reflex_hits << ",\n"
            << "  \"hit_rate\": " << std::fixed << std::setprecision(2) << (total_reflex_fires > 0 ? (double)total_reflex_hits / total_reflex_fires : 0.0) << ",\n"
            << "  \"drives\": " << drives.to_json() << ",\n"
            << "  \"active_reflexes\": [\n";
        size_t idx = 0;
        for (const auto& kv : reflex_arcs) {
            oss << "    " << kv.second.to_json();
            if (++idx < reflex_arcs.size()) oss << ",";
            oss << "\n";
        }
        oss << "  ]\n}";
        return oss.str();
    }

private:
    std::string normalize_sig(const std::string& str) const {
        std::string s = str;
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return std::isspace(c); }), s.end());
        return s;
    }

    std::string format_num(double val) const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << val;
        return oss.str();
    }
};

}} // namespace brain2::reasoning
