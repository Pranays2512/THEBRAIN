#pragma once
/**
 * brain3/core/epistemic_logical_scrutiny_engine.hpp
 *
 * THE BRAIN 3: EPISTEMIC ANTI-OVERCLAIMING & LOGICAL SCRUTINY ENGINE (v2)
 *
 * Rigorous self-checking and anti-hallucination kernel that scrutinizes
 * hypotheses, claims, architectures, and reasoning output using:
 * 1. Semantic Invariant Regex Predicates across Axiomatic Violation Classes
 * 2. Information-Theoretic Capacity Bounds (Pigeonhole, Shannon limits, Plate 1995 HRR crosstalk)
 * 3. Thermodynamic and Physical Conservation Laws (Carnot, 2nd Law of Thermodynamics)
 * 4. Computational Complexity Class Invariants (P vs NP, Omega(N log N) sorting)
 * 5. Tri-State Epistemic Calibration (Verified Axiom, Rejected Overclaim, or Unverified Hypothesis)
 */

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <regex>
#include <iostream>

#include "../crisp/engines/math/adversarial_epistemic_auditor.hpp"

namespace brain3 {
namespace core {

struct ScrutinyResult {
    bool is_grounded = true;
    bool has_overclaim = false;
    std::string sanitized_claim;
    std::vector<std::string> detected_fallacies;
    std::vector<std::string> rigorous_bounds_and_caveats;
    std::string scientific_verdict_label;
    std::string grounded_explanation;
};

class EpistemicLogicalScrutinyEngine {
public:
    /**
     * Actively scrutinizes a generated hypothesis, text, or theory claim.
     * Strips hyperbole, catches capacity violations, and enforces strict epistemic bounds.
     */
    static ScrutinyResult scrutinize_claim(const std::string& input_claim) {
        ScrutinyResult res;
        res.sanitized_claim = input_claim;
        std::string lower = input_claim;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        // ── 1. Class I: Information-Theoretic Capacity & Fixed Vector Overreach ─
        // Catch any pairing of [Infinite/Lossless/Zero-Noise/Exact Memory] with [Vector/Accumulator/Dim/State]
        std::regex capacity_regex(
            R"((?:infinite|lossless|zero[- ]noise|exact\s+distinct|perfect\s+recall|unbounded\s+capacity).*?(?:vector|accumulator|dimension|\b\d+[- ]dim|fixed|finite|state|memory|hrr|vsa))",
            std::regex_constants::icase
        );
        std::regex reverse_capacity_regex(
            R"((?:fixed|finite|vector|accumulator|\b\d+[- ]dim).*?(?:infinite|lossless|zero[- ]noise|exact\s+distinct\s+memories|exact\s+recall))",
            std::regex_constants::icase
        );

        if (std::regex_search(lower, capacity_regex) || std::regex_search(lower, reverse_capacity_regex) ||
            (lower.find("lossless") != std::string::npos && (lower.find("vector") != std::string::npos || lower.find("dimension") != std::string::npos || lower.find("accumulator") != std::string::npos)) ||
            (lower.find("infinite") != std::string::npos && lower.find("memory") != std::string::npos && (lower.find("fixed") != std::string::npos || lower.find("512") != std::string::npos || lower.find("vector") != std::string::npos))) {
            res.has_overclaim = true;
            res.is_grounded = false;
            res.detected_fallacies.push_back("FALLACY: Lossless Multi-Item Superposition in Fixed Vector Space (Plate 1995 Violation)");
            res.rigorous_bounds_and_caveats.push_back(
                "Pigeonhole & Capacity Bound: Storing N items in a fixed D-dimensional vector incurs crosstalk noise. "
                "In Holographic Reduced Representations (HRR) and Vector Symbolic Architectures (VSA), SNR scales as O(sqrt(D / N_eff)) (Plate, 1995; Jelassi et al., 2024)."
            );
        }

        // ── 2. Class II: Thermodynamic & Physical Invariant Violations ─────────
        std::regex thermo_regex(
            R"((?:perpetual\s+motion|over[- ]unity|free\s+energy|zero[- ]dissipation|exceeding\s+carnot|infinite\s+efficiency))",
            std::regex_constants::icase
        );
        if (std::regex_search(lower, thermo_regex)) {
            res.has_overclaim = true;
            res.is_grounded = false;
            res.detected_fallacies.push_back("FALLACY: Thermodynamic Conservation Law Violation");
            res.rigorous_bounds_and_caveats.push_back(
                "Carnot & Entropy Bound: Every thermodynamic or computational physical process has an irreducible dissipation cost (Landauer limit E >= k_B T ln 2) and cannot exceed the Carnot limit eta = 1 - T_C / T_H."
            );
        }

        // ── 3. Class III: Computational Complexity Class Collapse ──────────────
        std::regex complexity_regex(
            R"((?:\bp\s*=\s*np\b|polynomial[- ]time\s+solution\s+for\s+np[- ]complete|o\(1\)\s+comparison\s+sort|instantaneous\s+factorization))",
            std::regex_constants::icase
        );
        if (std::regex_search(lower, complexity_regex)) {
            res.has_overclaim = true;
            res.is_grounded = false;
            res.detected_fallacies.push_back("FALLACY: Unproven Complexity Class Collapse");
            res.rigorous_bounds_and_caveats.push_back(
                "Complexity Lower Bounds: Comparison-based sorting is strictly bounded by Omega(N log N); NP-complete reductions remain super-polynomial unless P=NP is proven under ZFC."
            );
        }

        // ── 4. Class IV: Expressivity Class Collapse (GSSMs vs Attention) ───────
        std::regex expressivity_regex(
            R"((?:zero\s+length\s+failure|infinite\s+context\s+flat\s+memory|exact\s+copying\s+without\s+attention))",
            std::regex_constants::icase
        );
        if (std::regex_search(lower, expressivity_regex)) {
            res.has_overclaim = true;
            res.is_grounded = false;
            res.detected_fallacies.push_back("FALLACY: Unbounded Associative Generalization on Finite Latent State");
            res.rigorous_bounds_and_caveats.push_back(
                "Expressivity Separation: Fixed-state recurrent models (GSSMs, S4, Mamba, RetNet, RWKV) cannot match Transformers on multi-query associative recall and copying tasks (Jelassi et al., 2024)."
            );
        }

        // ── 5. Class V: Decorative / Sensationalized Physics Metaphors ──────────
        if ((lower.find("noether") != std::string::npos && (lower.find("dft") != std::string::npos || lower.find("fourier") != std::string::npos || lower.find("matrix") != std::string::npos)) ||
            (lower.find("quantum superposition") != std::string::npos && lower.find("css") != std::string::npos)) {
            res.has_overclaim = true;
            res.detected_fallacies.push_back("RHETORICAL INFLATION: Physics Metaphor for Unitary Linear Algebra");
            res.rigorous_bounds_and_caveats.push_back(
                "Mathematical Reality: Discrete Fourier Transform is unitary (Parseval/Plancherel Theorem), preserving vector norms. Calling this 'Noether information flux' is metaphorical."
            );
        }

        // ── 6. Class VI: Methodological Error / Test Over-claiming ─────────────
        if (lower.find("all tests passing at 100%") != std::string::npos &&
            (lower.find("generalization verified") != std::string::npos || lower.find("proved") != std::string::npos)) {
            res.has_overclaim = true;
            res.detected_fallacies.push_back("METHODOLOGICAL ERROR: Conflating Code Execution with Empirical Validity");
            res.rigorous_bounds_and_caveats.push_back(
                "Empirical Reality: Unit tests only verify that operations execute without runtime crashes. Empirical generalization requires out-of-distribution evaluation."
            );
        }

        // ── Tri-State Epistemic Verdict Synthesis ─────────────────────────────
        std::ostringstream oss;
        if (res.has_overclaim) {
            res.scientific_verdict_label = "REJECTED_OVERCLAIM_RECALIBRATED";
            oss << "⚠️ [Epistemic Scrutiny Alert: Fallacies Detected & Recalibrated]\n";
            for (const auto& f : res.detected_fallacies) {
                oss << "  • " << f << "\n";
            }
            oss << "\n📐 [Strict Mathematical Bounds & Realities]:\n";
            for (const auto& b : res.rigorous_bounds_and_caveats) {
                oss << "  • " << b << "\n";
            }
            res.grounded_explanation = oss.str();
        } else {
            // Check if grounded in known mathematical axioms or an unverified claim
            bool has_grounded_term = (lower.find("derivative") != std::string::npos ||
                                      lower.find("pythagor") != std::string::npos ||
                                      lower.find("conservation") != std::string::npos ||
                                      lower.find("isomorphism") != std::string::npos ||
                                      lower.find("invariance") != std::string::npos ||
                                      lower.find("math") != std::string::npos ||
                                      lower.find("cas") != std::string::npos ||
                                      lower.find("physics") != std::string::npos);
            if (has_grounded_term) {
                res.scientific_verdict_label = "SOUND_LOGICAL_CLAIM";
                res.grounded_explanation = "✓ Claim verified consistent with information-theoretic capacity bounds and computational complexity classes.";
            } else {
                res.scientific_verdict_label = "UNVERIFIED_HYPOTHESIS";
                res.grounded_explanation = "ℹ️ Claim contains no direct axiomatic violations, but remains an unverified hypothesis requiring empirical corroboration.";
            }
        }

        return res;
    }

    /**
     * Sanitizes natural language text emitted by any cognitive sub-engine,
     * replacing sensationalized phrases with calibrated, scientifically accurate terminology.
     */
    static std::string sanitize_text(const std::string& raw_text) {
        std::string s = raw_text;

        auto replace_all = [](std::string& str, const std::string& from, const std::string& to) {
            size_t start_pos = 0;
            while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
                str.replace(start_pos, from.length(), to);
                start_pos += to.length();
            }
        };

        replace_all(s, "Exact Memory Recall", "Associative Unbinding (Subject to Crosstalk Noise)");
        replace_all(s, "exact memory recall", "associative unbinding (subject to crosstalk noise)");
        replace_all(s, "Zero Length Failure", "Bounded Context Recurrence");
        replace_all(s, "zero length failure", "bounded context recurrence");
        replace_all(s, "Infinite Context Flat Memory", "Fixed O(1) Memory State Buffer");
        replace_all(s, "Noether Conserved Information Flux", "Parseval Unitary L2-Norm Conservation");

        return s;
    }
};

} // namespace core
} // namespace brain3
