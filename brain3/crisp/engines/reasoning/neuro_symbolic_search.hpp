#pragma once
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include "crisp/engines/reasoning/tree_reason.hpp"
#include "crisp/engines/reasoning/monte_carlo_tree.hpp"

namespace brain2 {
namespace reasoning {

// Wrapper to start MCTS from an arbitrary state
template<typename State, typename Hash = std::hash<State>>
class SubMCTSProblem : public SearchProblem<State, Hash> {
private:
    const SearchProblem<State, Hash>& base_problem;
    State start_state;
public:
    SubMCTSProblem(const SearchProblem<State, Hash>& base, State start) 
        : base_problem(base), start_state(start) {}
        
    State initial() const override { return start_state; }
    bool is_goal(const State& s) const override { return base_problem.is_goal(s); }
    std::vector<std::tuple<std::string, State, double>> moves(const State& s) const override { return base_problem.moves(s); }
    double novelty(const State& s) const override { return base_problem.novelty(s); }
};

// Neuro-Symbolic Problem: A* Search with MCTS as the Heuristic
template<typename State, typename Hash = std::hash<State>>
class NeuroSymbolicProblem : public SearchProblem<State, Hash> {
private:
    const SearchProblem<State, Hash>& base_problem;
    mutable std::map<State, double> memoization_table; // Temporary Transposition Table
    int mcts_iterations;
    int max_rollout_depth;

public:
    NeuroSymbolicProblem(const SearchProblem<State, Hash>& base, int mcts_iters = 10, int rollout_depth = 5) 
        : base_problem(base), mcts_iterations(mcts_iters), max_rollout_depth(rollout_depth) {}

    ~NeuroSymbolicProblem() {
        std::cout << "[Neuro-Symbolic] Destroying temporary memoization table (" << memoization_table.size() << " cached states freed).\n";
    }

    State initial() const override { return base_problem.initial(); }
    bool is_goal(const State& s) const override { return base_problem.is_goal(s); }
    std::vector<std::tuple<std::string, State, double>> moves(const State& s) const override {
        return base_problem.moves(s);
    }
    double novelty(const State& s) const override { return base_problem.novelty(s); }

    // THE NEURO-SYMBOLIC HEURISTIC
    double heuristic(const State& s) const override {
        if (memoization_table.count(s)) {
            return memoization_table[s]; // Cache hit (Extremely fast)
        }

        // Cache miss: Run a micro-MCTS simulation to "intuit" the distance to the goal
        SubMCTSProblem<State, Hash> sub_prob(base_problem, s);
        MonteCarloConfig cfg;
        cfg.iterations = mcts_iterations;
        cfg.rollout_depth = max_rollout_depth;
        cfg.goal_reward = 100.0;
        cfg.novelty_weight = 1.0;
        
        auto result = solve_mcts(sub_prob, cfg);
        
        // Convert MCTS reward to an A* Cost. 
        // If MCTS found the goal perfectly, cost is 0.
        // Otherwise, we use a high cost, inversely proportional to how close MCTS felt it got.
        double h_val = 50.0; // Base high cost
        if (result.solved) {
            h_val = result.path.size(); // If it knows the path, the heuristic is the exact path length!
        }
        
        memoization_table[s] = h_val; // Save to temporary table
        return h_val;
    }
};

template<typename State, typename Hash = std::hash<State>>
SearchResult<State, Hash> solve_neuro_symbolic(const SearchProblem<State, Hash>& problem, int mcts_iters = 10) {
    // 1. Wrap the problem in the Neuro-Symbolic heuristic
    NeuroSymbolicProblem<State, Hash> ns_prob(problem, mcts_iters);
    
    // 2. Run A* search (which will now seamlessly use MCTS under the hood)
    return solve_astar(ns_prob, 5000); // Strict node limit for A*
}

} // namespace reasoning
} // namespace brain2
