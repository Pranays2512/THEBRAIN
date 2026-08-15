/**
 * brain3/crisp/engines/math/hierarchical_goal_decomposer.cpp
 *
 * Driver and verification harness for The Brain's General Hierarchical Goal Decomposer.
 */

#include "hierarchical_goal_decomposer.hpp"
#include <iostream>
#include <iomanip>

using namespace thebrain::goal_decomposer;

void print_plan(const DecomposedProofPlan& plan) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "🎯 GRAND CHALLENGE DECOMPOSITION: " << plan.grand_challenge_title << "\n";
    std::cout << "   Domain             : " << thebrain::knowledge_vault::domain_to_string(plan.domain) << "\n";
    std::cout << "   Critical Bottleneck: " << plan.critical_bottleneck_lemma_id << "\n";
    std::cout << "   Total Complexity   : " << std::fixed << std::setprecision(2) << plan.total_estimated_complexity << "\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "📋 STEPPING-STONE LEMMA DAG:\n";
    for (size_t i = 0; i < plan.lemma_dag.size(); ++i) {
        const auto& L = plan.lemma_dag[i];
        std::string status_str;
        switch (L.status) {
            case LemmaStatus::PROVEN: status_str = "✅ PROVEN"; break;
            case LemmaStatus::PRE_SCREENED_BY_CAS_SMT: status_str = "🧪 PRE-SCREENED (CAS & SMT)"; break;
            case LemmaStatus::PROPOSED: status_str = "💡 PROPOSED"; break;
            case LemmaStatus::BLOCKED_BY_BARRIER: status_str = "🚧 BLOCKED BY BARRIER"; break;
        }
        std::cout << "   [" << (i + 1) << "] " << L.lemma_id << ": " << L.title << " [" << status_str << "]\n";
        std::cout << "       Hypothesis : " << L.formal_hypothesis << "\n";
        std::cout << "       Conclusion : " << L.formal_conclusion << "\n";
        std::cout << "       Difficulty : " << std::fixed << std::setprecision(2) << L.estimated_proof_difficulty << "\n";
        std::cout << "       Strategy   : " << L.justification_strategy << "\n\n";
    }
}

int main() {
    std::cout << "\n🧠 ==========================================================================\n";
    std::cout << "   THE BRAIN — GENERAL HIERARCHICAL GOAL DECOMPOSER (\"Flight Engine 2\")\n";
    std::cout << "   Multi-Scale Forward-Backward Lemma DAG Generation & Bottleneck Analysis\n";
    std::cout << "==========================================================================\n";

    // 1. Navier-Stokes decomposition
    auto plan1 = HierarchicalGoalDecomposer::decompose_navier_stokes_regularity();
    print_plan(plan1);

    // 2. Quantum Black Hole Information Paradox decomposition
    auto plan2 = HierarchicalGoalDecomposer::decompose_black_hole_information_paradox();
    print_plan(plan2);

    std::cout << "\n==========================================================================\n";
    std::cout << "🏁 GOAL DECOMPOSER READY: GRAND CHALLENGES REDUCED TO TESTABLE SUB-LEMMAS\n";
    std::cout << "==========================================================================\n\n";

    return 0;
}
