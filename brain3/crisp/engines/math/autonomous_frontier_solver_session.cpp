/**
 * brain3/crisp/engines/math/autonomous_frontier_solver_session.cpp
 *
 * Driver and execution harness for The Brain's Autonomous Frontier Solver Session.
 * Runs The Rocket pipeline against:
 * 1. Collatz Conjecture
 * 2. Erdős-Straus Open Mordell Classes
 * 3. 3D Navier-Stokes Regularity
 * 4. Goldbach Conjecture
 */

#include "autonomous_frontier_solver_session.hpp"
#include <iostream>
#include <iomanip>

using namespace thebrain::frontier_solver;

void print_investigation(const SolverInvestigationReport& rep) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "🏛️ OPEN PROBLEM INVESTIGATION: " << rep.problem_name << "\n";
    std::cout << "   Context          : " << rep.historical_context << "\n";
    std::cout << "   Epistemic Status : " << rep.final_epistemic_status << "\n";
    std::cout << "   Execution Time   : " << std::fixed << std::setprecision(4) << rep.execution_time_ms << " ms\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "💡 THE ROCKET GENERATOR PROPOSAL:\n";
    std::cout << "   " << rep.generator_proposal << "\n\n";
    std::cout << "💥 SMT BREAKER FALSIFICATION TEST:\n";
    std::cout << "   " << rep.smt_breaker_result << "\n\n";
    std::cout << "📜 FORMAL PROVER DEDUCTIONS:\n";
    std::cout << "   " << rep.formal_prover_result << "\n\n";
    std::cout << "🛡️ ADVERSARIAL SKEPTIC AUDITOR VERDICT:\n";
    std::cout << "   " << rep.adversarial_auditor_verdict << "\n\n";
    std::cout << "🌟 WHAT WAS GENUINELY DISCOVERED / PROVEN:\n";
    std::cout << "   " << rep.what_was_discovered_or_proven << "\n\n";
    std::cout << "⚠️ EXACT OPEN MATHEMATICAL GAP REMAINING:\n";
    std::cout << "   " << rep.what_remains_open << "\n";
}

int main() {
    std::cout << "\n🧠 ==========================================================================\n";
    std::cout << "   THE BRAIN — AUTONOMOUS FRONTIER MATHEMATICAL SOLVER SESSION\n";
    std::cout << "   Executing Generator -> SMT Breaker -> Formal Prover -> Adversarial Auditor\n";
    std::cout << "==========================================================================\n";

    FrontierSolverSession session;

    // 1. Collatz Conjecture
    auto rep1 = session.investigate_collatz();
    print_investigation(rep1);

    // 2. Erdős-Straus Mordell Classes
    auto rep2 = session.investigate_erdos_straus();
    print_investigation(rep2);

    // 3. 3D Navier-Stokes
    auto rep3 = session.investigate_navier_stokes();
    print_investigation(rep3);

    // 4. Goldbach Conjecture
    auto rep4 = session.investigate_goldbach();
    print_investigation(rep4);

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "🏁 FRONTIER SOLVER SESSION COMPLETE: RIGOROUS SCIENTIFIC RESULTS GENERATED.\n";
    std::cout << std::string(80, '=') << "\n\n";

    return 0;
}
