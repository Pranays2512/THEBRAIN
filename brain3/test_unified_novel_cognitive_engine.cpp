/**
 * brain3/test_unified_novel_cognitive_engine.cpp
 *
 * Comprehensive Test Suite for UnifiedNovelCognitiveEngine
 * Verifies unified execution across Abduction, CAS, Frontier Problems,
 * Epistemic Scrutiny, Ancient Alignment, and STAMLAT Hamiltonian Flow.
 */

#include <iostream>
#include <cassert>
#include <chrono>
#include <vector>
#include <string>

#include "crisp/engines/unified_novel_cognitive_engine.hpp"

using namespace thebrain::unified;

void run_test(const std::string& name, bool condition) {
    if (condition) {
        std::cout << "  [PASS] " << name << "\n";
    } else {
        std::cerr << "  [FAIL] " << name << "\n";
        std::exit(1);
    }
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "THE BRAIN 3 — UNIFIED NOVEL COGNITIVE ENGINE VERIFICATION SUITE\n";
    std::cout << "================================================================================\n\n";

    auto start_all = std::chrono::high_resolution_clock::now();
    UnifiedNovelCognitiveEngine engine;

    // Test 1: Abductive Latent Variable Discovery (Beta Decay -> Neutrino)
    std::cout << "[Section 1: Abductive Latent Variable Discovery]\n";
    {
        auto report = engine.solve_anomaly("missing_beta_decay_momentum", 300);
        run_test("Report marked as success", report.success);
        run_test("Discovered latent entity (neutrino)", report.discovered_latent_entity.find("neutrino") != std::string::npos);
        run_test("Low residual error (< 0.10)", report.residual_error < 0.10);
        run_test("Passed Epistemic Audit", report.passed_epistemic_audit);
        run_test("STAMLAT Converged", report.stamlat_converged);
        std::cout << "    -> " << report.summary_text << "\n\n";
    }

    // Test 2: Formal CAS Novel Theory Synthesis (Navier-Stokes Fisher Curvature)
    std::cout << "[Section 2: Formal CAS Scientific Theory Synthesis]\n";
    {
        auto report = engine.synthesize_theory("navier_stokes_fisher");
        run_test("Theory synthesis success", report.success);
        run_test("Mathematical formulation non-empty", !report.mathematical_formulation.empty());
        run_test("CAS Derivation Trace generated", !report.cas_derivation_trace.empty());
        run_test("Passed Epistemic Audit", report.passed_epistemic_audit);
        run_test("High Novelty Score (> 0.80)", report.novelty_score > 0.80);
        std::cout << "    -> " << report.summary_text << "\n\n";
    }

    // Test 3: Frontier Problem Attack (Collatz 2-Adic Lyapunov Invariant)
    std::cout << "[Section 3: Millennium & Frontier Open Problem Attack]\n";
    {
        auto report = engine.attack_frontier_problem("collatz_conjecture");
        run_test("Conjecture attack success", report.success);
        run_test("Generated formal invariant equation", !report.mathematical_formulation.empty());
        run_test("Proof trace generated", !report.audited_invariants.empty());
        std::cout << "    -> " << report.summary_text << "\n\n";
    }

    // Test 4: Ancient-Modern Epistemic Alignment
    std::cout << "[Section 4: Ancient-Modern Epistemic Grounding]\n";
    {
        auto alignments = engine.alignment()->find_alignments("nyaya");
        run_test("Nyaya alignments found", !alignments.empty());
        run_test("High systematicity score (> 0.8)", alignments.front().systematicity_score >= 0.8);
        std::cout << "    -> Aligned: " << alignments.front().ancient_concept 
                  << " with " << alignments.front().modern_concept 
                  << " (Score: " << alignments.front().systematicity_score << ")\n\n";
    }

    // Test 5: STAMLAT Phase-Space Hamiltonian Evolution
    std::cout << "[Section 5: STAMLAT Spinor Geometric Flow]\n";
    {
        std::vector<std::vector<float>> dummy_seq(3, std::vector<float>(16, 0.3f));
        auto states = engine.stamlat()->forward_sequence(dummy_seq, true);
        run_test("Symplectic sequence computed", states.size() == 3);
        double energy = UnifiedNovelCognitiveEngine::compute_hamiltonian_energy(states.front());
        run_test("Hamiltonian energy non-negative", energy >= 0.0);
        std::cout << "    -> Sequence Length: " << states.size() << " | Phase Hamiltonian Energy: " << energy << "\n\n";
    }

    // Test 6: Master Autonomous Dispatch (solve_or_synthesize)
    std::cout << "[Section 6: Master Autonomous Dispatcher]\n";
    {
        auto rep1 = engine.solve_or_synthesize("riemann_hypothesis");
        run_test("Auto-routed to Frontier Engine", rep1.mode_executed == "FRONTIER_MILLENNIUM_ATTACK");

        auto rep2 = engine.solve_or_synthesize("hubble_tension");
        run_test("Auto-routed to CAS Theory Generator", rep2.mode_executed == "FORMAL_CAS_THEORY_SYNTHESIS");

        auto rep3 = engine.solve_or_synthesize("flat_galactic_rotation");
        run_test("Auto-routed to Abductive Engine", rep3.mode_executed == "ABDUCTIVE_LATENT_SYNTHESIS");

        std::cout << "    -> rep1: " << rep1.summary_text << "\n";
        std::cout << "    -> rep2: " << rep2.summary_text << "\n";
        std::cout << "    -> rep3: " << rep3.summary_text << "\n\n";
    }

    auto end_all = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_all - start_all).count();

    std::cout << "================================================================================\n";
    std::cout << "ALL UNIFIED COGNITIVE ENGINE TESTS PASSED (100.0%) in " << total_ms << " ms\n";
    std::cout << "================================================================================\n";

    return 0;
}
