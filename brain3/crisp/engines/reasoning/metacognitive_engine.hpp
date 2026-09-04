#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cctype>

#include "crisp/engines/reasoning/reasoning_engine.hpp"
#include "crisp/engines/reasoning/causal_engine.hpp"

namespace brain2 {
namespace reasoning {

struct MetacognitiveVerdict {
    bool is_refuted = false;
    std::string claim;
    std::string system1_intuition;
    std::string falsification_reason;
    std::string corrected_truth;
    std::vector<std::string> proof_trace;
    double confidence = 1.0;
    std::string verdict_str; // "REFUTED" or "VERIFIED_SOUND"

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"verified\": " << (is_refuted ? "false" : "true") << ",\n";
        oss << "  \"verdict\": \"" << verdict_str << "\",\n";
        oss << "  \"claim\": \"" << claim << "\",\n";
        oss << "  \"system1_intuition\": \"" << system1_intuition << "\",\n";
        oss << "  \"falsification_reason\": \"" << falsification_reason << "\",\n";
        oss << "  \"corrected_truth\": \"" << corrected_truth << "\",\n";
        oss << "  \"confidence\": " << confidence << ",\n";
        oss << "  \"proof_trace\": [\n";
        for (size_t i = 0; i < proof_trace.size(); ++i) {
            oss << "    \"" << proof_trace[i] << "\"";
            if (i + 1 < proof_trace.size()) oss << ",";
            oss << "\n";
        }
        oss << "  ]\n";
        oss << "}";
        return oss.str();
    }
};

class MetacognitiveEngine {
private:
    // Known disjoint taxonomic categories
    std::vector<std::pair<std::string, std::string>> disjoint_classes = {
        {"mammal", "bird"},
        {"mammal", "reptile"},
        {"mammal", "fish"},
        {"bird", "reptile"},
        {"bird", "fish"},
        {"herbivore", "carnivore"},
        {"solid", "gas"},
        {"liquid", "gas"}
    };

public:
    MetacognitiveEngine() {}

    void add_disjoint_classes(const std::string& c1, const std::string& c2) {
        disjoint_classes.push_back({c1, c2});
    }

    MetacognitiveVerdict refute(const std::string& subj, const std::string& rel, const std::string& obj, ReasoningEngine* kb, CausalEngine* ca = nullptr) {
        MetacognitiveVerdict res;
        res.claim = subj + " " + rel + " " + obj;
        std::vector<std::string> trace;

        // ── Phase 1: System 1 (Fast Associative & Generic Heuristic) ───────────
        trace.push_back("[System 1 Prior]: Activated associative taxonomy heuristics for '" + subj + "'");
        std::string s1_guess = "";
        
        // Fast lookup of direct or generic properties via isa chain
        if (kb) {
            auto closure = kb->closure(subj, "isa", 10);
            for (const auto& parent : closure[subj]) {
                for (const auto& f : kb->facts) {
                    if (f.subj == parent && f.rel == rel && f.obj != "<EXCEPTION>") {
                        s1_guess = f.obj;
                        break;
                    }
                }
                if (!s1_guess.empty()) break;
            }
        }
        if (s1_guess.empty()) s1_guess = obj;
        res.system1_intuition = "System 1 naive heuristic predicts: " + subj + " " + rel + " " + s1_guess;
        trace.push_back(res.system1_intuition);

        // ── Phase 2: System 2 (Adversarial Formal Refuter) ─────────────────────
        trace.push_back("[System 2 Deliberation]: Launching adversarial falsification probe...");

        // 1. Physical & Mathematical Boundary Invariants
        //
        // A table, not a cascade of ifs — adding a quantity is adding a row.
        // Previously this knew only mass and divisor, so `age = -5`,
        // `distance = -12` and `probability = 1.7` all returned VERIFIED_SOUND
        // (heldout_probe section G). A certifier that green-lights an impossible
        // value is worse than no certifier, because callers trust it.
        //
        // RESTRAINT IS THE HARD PART AND IT IS DELIBERATE. Only quantities whose
        // bound holds regardless of unit or sign convention are listed. Excluded
        // on purpose:
        //   temperature      -50 is valid Celsius and invalid Kelvin; the engine
        //                    is never told the unit, so it must not guess
        //   velocity, accel  genuinely signed (direction), unlike speed
        //   altitude         below sea level is real
        //   balance, charge  legitimately negative
        // An over-eager bound is worse than a missing one: a missing bound only
        // fails to refute a falsehood, an over-eager bound refutes a truth.
        struct QuantityBound {
            std::vector<std::string> names;
            double lo; bool lo_strict;   // value must be > lo (strict) or >= lo
            double hi; bool hi_strict;   // value must be < hi (strict) or <= hi
            std::string statement;
        };
        static const double kInf = std::numeric_limits<double>::infinity();
        static const std::vector<QuantityBound> kBounds = {
            {{"mass", "m", "mass_val"},
             0.0, true,  kInf, false, "mass > 0 (strictly positive in classical physics)"},
            {{"age"},
             0.0, false, kInf, false, "age >= 0 (zero is valid)"},
            {{"distance", "length", "width", "height", "radius", "diameter", "perimeter"},
             0.0, false, kInf, false, "a spatial extent is non-negative"},
            {{"area", "volume"},
             0.0, false, kInf, false, "area and volume are non-negative"},
            {{"speed"},
             0.0, false, kInf, false, "speed is a magnitude and is non-negative (velocity is signed)"},
            {{"probability", "prob"},
             0.0, false, 1.0,  false, "0 <= probability <= 1"},
        };

        auto parse_num = [](const std::string& s, double& out) -> bool {
            try { size_t idx = 0; out = std::stod(s, &idx);
                  while (idx < s.size() && std::isspace((unsigned char)s[idx])) ++idx;
                  return idx == s.size(); }
            catch (...) { return false; }
        };

        double numeric_obj = 0.0;
        const bool obj_is_numeric = parse_num(obj, numeric_obj);

        if (obj_is_numeric) {
            for (const auto& b : kBounds) {
                bool matches = false;
                for (const auto& n : b.names)
                    if (subj == n || rel == n) { matches = true; break; }
                if (!matches) continue;

                const bool below = b.lo_strict ? (numeric_obj <= b.lo) : (numeric_obj < b.lo);
                const bool above = b.hi_strict ? (numeric_obj >= b.hi) : (numeric_obj > b.hi);
                if (below || above) {
                    res.is_refuted = true;
                    res.verdict_str = "REFUTED";
                    res.falsification_reason =
                        "Boundary invariant violation: " + b.statement + ", got " + obj;
                    res.corrected_truth = b.statement;
                    trace.push_back("  ✗ Boundary Invariant Violation: " + res.falsification_reason);
                    trace.push_back("  ✓ Corrected Truth: " + res.corrected_truth);
                    res.proof_trace = trace;
                    return res;
                }
                break;   // matched a quantity and it was in range
            }

            // Division by zero is an excluded VALUE rather than a range bound,
            // so it stays a separate predicate instead of being contorted into
            // the table above.
            if (subj == "divisor" || rel == "divisor" ||
                subj == "denominator" || rel == "denominator") {
                if (numeric_obj == 0.0) {
                    res.is_refuted = true;
                    res.verdict_str = "REFUTED";
                    res.falsification_reason = "Mathematical invariant violation: division by zero is undefined";
                    res.corrected_truth = "denominator != 0";
                    trace.push_back("  ✗ Mathematical Invariant Violation: " + res.falsification_reason);
                    trace.push_back("  ✓ Corrected Truth: " + res.corrected_truth);
                    res.proof_trace = trace;
                    return res;
                }
            }
        }

        // 2. Check Disjoint Taxonomic Clade Violations
        if (kb && (rel == "isa" || rel == "is_a" || rel == "val")) {
            auto closure = kb->closure(subj, "isa", 10);
            std::vector<std::string> ancestors;
            for (const auto& kv : closure) {
                ancestors.push_back(kv.first);
            }
            for (const auto& f : kb->facts) {
                if (f.subj == subj && (f.rel == "isa" || f.rel == "is_a")) {
                    ancestors.push_back(f.obj);
                }
            }
            for (const auto& parent : ancestors) {
                for (const auto& dj : disjoint_classes) {
                    if ((parent == dj.first && obj == dj.second) || (parent == dj.second && obj == dj.first)) {
                        res.is_refuted = true;
                        res.verdict_str = "REFUTED";
                        res.falsification_reason = "Disjoint category violation: " + subj + " is " + parent + ", which is disjoint with " + obj;
                        res.corrected_truth = subj + " isa " + parent + " (NOT " + obj + ")";
                        trace.push_back("  ✗ Contradiction Found: " + res.falsification_reason);
                        trace.push_back("  ✓ Corrected Truth: " + res.corrected_truth);
                        res.proof_trace = trace;
                        return res;
                    }
                }
            }
        }

        // 3. Check Exception Blockers (Non-Monotonic Logic)
        if (kb) {
            bool has_exception = false;
            for (const auto& f : kb->facts) {
                if (f.subj == subj && f.rel == rel && (f.obj == "<EXCEPTION>" || f.obj == "no_" + obj || f.obj == "cannot_" + obj)) {
                    has_exception = true;
                    break;
                }
            }

            // Also check if there is a direct override fact
            std::string direct_override = "";
            for (const auto& f : kb->facts) {
                if (f.subj == subj && f.rel == rel && f.obj != "<EXCEPTION>" && f.obj != obj) {
                    direct_override = f.obj;
                    break;
                }
            }

            if (has_exception || (!direct_override.empty() && obj == s1_guess && obj != direct_override)) {
                res.is_refuted = true;
                res.verdict_str = "REFUTED";
                res.falsification_reason = "Non-monotonic exception blocker: " + subj + " overrides generic " + rel + "=" + obj;
                res.corrected_truth = subj + " " + rel + " " + (direct_override.empty() ? "cannot " + obj : direct_override);
                trace.push_back("  ✗ Counterexample Found: " + res.falsification_reason);
                trace.push_back("  ✓ Corrected Truth: " + res.corrected_truth);
                res.proof_trace = trace;
                return res;
            }
        }

        // 4. If no counterexample is found $\to$ Claim is verified sound
        res.is_refuted = false;
        res.verdict_str = "VERIFIED_SOUND";
        res.falsification_reason = "None. Claim survived rigorous adversarial System 2 refutation.";
        res.corrected_truth = res.claim;
        trace.push_back("  ✓ No counterexamples or logical contradictions found.");
        trace.push_back("  ✓ Claim verified sound: " + res.claim);
        res.proof_trace = trace;
        return res;
    }
};

} // namespace reasoning
} // namespace brain2
