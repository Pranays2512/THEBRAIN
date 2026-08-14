#include <iostream>
#include <string>
#include <vector>
#include "fuzzy/engines/synthesis/unified_proposer.hpp"

using namespace brain3::engines::synthesis;

int main() {
    std::cout << "===========================================\n";
    std::cout << "  BRAIN 3: ADAPTIVE RECURRENT INTUITION\n";
    std::cout << "===========================================\n\n";

    UnifiedProposer proposer;
    
    // We will simulate 3 types of problems: Math, Code, Conjecture.
    Problem math_prob;
    math_prob.type = "equation";
    math_prob.data_str = "2x + 4 = 10";
    
    Problem code_prob;
    code_prob.type = "synthesize";
    code_prob.data_str = "def hello(): print('world')";
    
    Problem conj_prob;
    conj_prob.type = "conjecture";
    conj_prob.data_str = "Test F=ma";

    // In a real scenario, the policies do actual work. Here, they just return true if it's the right type.
    // We modify the mock policies in UnifiedProposer slightly for the demo to succeed only on their own domains.
    proposer.policies[0].solve_fn = [](const Problem& p) { return p.type == "equation"; };
    proposer.policies[1].solve_fn = [](const Problem& p) { return p.type == "synthesize"; };
    proposer.policies[2].solve_fn = [](const Problem& p) { return p.type == "conjecture"; };

    std::cout << "--- PHASE 1: UNTRAINED (Guessing randomly) ---\n";
    // At first, weights are random (0.01 initialization), so it will guess.
    // It will likely fail and fallback, which triggers learning (backward pass).
    
    for (int epoch = 1; epoch <= 50; ++epoch) {
        if (epoch <= 3) {
            std::cout << "\n[Epoch " << epoch << "]\n";
            std::cout << "Problem 1 (Math):\n";
            proposer.solve(math_prob);
            
            std::cout << "\nProblem 2 (Code):\n";
            proposer.solve(code_prob);
            
            std::cout << "\nProblem 3 (Conjecture):\n";
            proposer.solve(conj_prob);
        } else {
            // Train silently
            proposer.solve(math_prob);
            proposer.solve(code_prob);
            proposer.solve(conj_prob);
        }
    }
    
    std::cout << "\n--- PHASE 2: TRAINED (Perfect Intuition) ---\n";
    // After 3 epochs, the weights should have adapted perfectly to the 12-feature context.
    std::cout << "\nProblem 1 (Math):\n";
    proposer.solve(math_prob);
    
    std::cout << "\nProblem 2 (Code):\n";
    proposer.solve(code_prob);

    return 0;
}
