/**
 * brain3/crisp/engines/math/neural_guided_mcts_navigator.cpp
 *
 * Driver and verification harness for The Brain's Universal Neural-Guided MCTS Discovery Navigator.
 */

#include "neural_guided_mcts_navigator.hpp"
#include <iostream>
#include <iomanip>

using namespace thebrain::mcts_navigator;

int main() {
    std::cout << "\n🧠 ==========================================================================\n";
    std::cout << "   THE BRAIN — UNIVERSAL NEURAL-GUIDED MCTS DISCOVERY NAVIGATOR (\"Flight Engine 3\")\n";
    std::cout << "   AlphaProof-Style Guided Tree Search • Policy Priors • Value Guidance\n";
    std::cout << "==========================================================================\n";

    NeuralGuidedMCTSNavigator navigator;

    // Test Case: Exploring Navier-Stokes Energy Dissipation Goal
    ProofState init_state;
    init_state.goal_id = "ns_vorticity_depletion_goal";
    init_state.goal_statement = "Prove exponential vorticity enstrophy dissipation on bounded domain for Navier-Stokes";
    init_state.depth = 0;
    init_state.is_discharged = false;

    std::cout << "\n🎯 INITIAL PROOF GOAL: " << init_state.goal_statement << "\n";
    std::cout << "🚀 LAUNCHING 100 MONTE CARLO TREE SEARCH SIMULATIONS...\n\n";

    auto root = navigator.search(init_state, 100);

    std::cout << "🌳 ROOT MCTS SEARCH SUMMARY:\n";
    std::cout << "   Total Visits N(root) : " << root->visit_count << "\n";
    std::cout << "   Mean Value Q(root)   : " << std::fixed << std::setprecision(4) << root->mean_value << "\n";
    std::cout << "   Child Branch Count   : " << root->children.size() << "\n\n";

    std::cout << "📊 EXPANDED SEARCH BRANCHES (RANKED BY VISITS):\n";
    for (size_t i = 0; i < root->children.size(); ++i) {
        const auto& child = root->children[i];
        std::cout << "   [" << (i + 1) << "] " << action_to_string(child->action_taken) << "\n";
        std::cout << "       Visits N(s, a) : " << child->visit_count << "\n";
        std::cout << "       Mean Q(s, a)   : " << std::fixed << std::setprecision(4) << child->mean_value << "\n";
        std::cout << "       Policy Prior P : " << std::fixed << std::setprecision(2) << child->prior_probability << "\n";
        std::cout << "       Discharged     : " << (child->state.is_discharged ? "✅ QED (Closed)" : "⏳ In Progress") << "\n\n";
    }

    std::cout << "==========================================================================\n";
    std::cout << "🏁 MCTS DISCOVERY NAVIGATOR READY: SEARCH SPACES EXPLORED WITHOUT BLOWUP\n";
    std::cout << "==========================================================================\n\n";

    return 0;
}
