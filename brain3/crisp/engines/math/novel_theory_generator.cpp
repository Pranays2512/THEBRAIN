/**
 * brain3/crisp/engines/math/novel_theory_generator.cpp
 *
 * Driver and execution harness for The Brain's Autonomous Novel Theory Synthesis Engine.
 */

#include "novel_theory_generator.hpp"
#include <iostream>
#include <iomanip>

using namespace thebrain::novel_theory;

void print_novel_theory(const NovelTheoryPackage& pkg, int index) {
    std::cout << "\n==========================================================================\n";
    std::cout << "💡 NOVEL THEORY " << index << ": " << pkg.theory_name << "\n";
    std::cout << "==========================================================================\n";
    std::cout << "• Origin Field       : " << pkg.primary_domain << "\n";
    std::cout << "• Application Target : " << pkg.target_domain << "\n";
    std::cout << "• Synthesis Time     : " << std::fixed << std::setprecision(2) << pkg.generation_time_ms << " ms\n\n";

    std::cout << "🚨 Unsolved Scientific Crisis:\n";
    std::cout << "  " << pkg.unsolved_anomaly_or_crisis << "\n\n";

    std::cout << "🧬 Invented Latent Concept / Mechanism:\n";
    std::cout << "  " << pkg.invented_latent_entity_or_mechanism << "\n\n";

    std::cout << "🌉 Cross-Domain Duality Bridge:\n";
    std::cout << "  " << pkg.cross_domain_isomorphism_mapping << "\n\n";

    std::cout << "📐 Mathematical Formulation (Exact Equations):\n";
    std::cout << "  " << pkg.mathematical_formulation_equation << "\n\n";

    std::cout << "🔬 Exact CAS Deductive Proof / Verification:\n";
    std::cout << "  " << pkg.exact_cas_deduction_result << "\n\n";

    std::cout << "🔮 Falsifiable & Testable Predictions:\n";
    for (const auto& pred : pkg.falsifiable_testable_predictions) {
        std::cout << "  " << pred << "\n";
    }

    std::cout << "\n⚖️ Epistemic Audit Status: [" << pkg.epistemic_audit_verdict << "]\n";
}

int main() {
    std::cout << "\n🧠 ==========================================================================\n";
    std::cout << "   THE BRAIN — AUTONOMOUS NOVEL SCIENTIFIC & MATHEMATICAL THEORY SYNTHESIS\n";
    std::cout << "   Powered by Abductive MCTS, Cross-Domain Bridges & Exact CAS Verification\n";
    std::cout << "==========================================================================\n";

    NovelTheoryGenerator generator;

    int idx = 1;
    auto t1 = generator.synthesize_fluid_information_entropy_theory();
    print_novel_theory(t1, idx++);

    auto t2 = generator.synthesize_non_hermitian_topological_memory_theory();
    print_novel_theory(t2, idx++);

    auto t3 = generator.synthesize_holographic_island_hubble_tension_theory();
    print_novel_theory(t3, idx++);

    std::cout << "\n==========================================================================\n";
    std::cout << "🏁 AUTONOMOUS THEORY SYNTHESIS COMPLETE: 3 NOVEL THEORIES GENERATED & AUDITED\n";
    std::cout << "==========================================================================\n\n";

    return 0;
}
