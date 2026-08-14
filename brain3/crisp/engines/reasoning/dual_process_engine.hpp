#pragma once
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <memory>
#include <functional>

#include "crisp/engines/reasoning/tree_reason.hpp"

namespace brain2 {
namespace reasoning {

// ── Dual Process Solver ───────────────────────────────────────────────────────
// Solves problems using a tiered approach:
// 1. Memory: Look up cached exact solutions.
// 2. Reflex: Greedy policy rollout (no search).
// 3. Deliberation: Full A* search.
// All solutions found by tier 2 or 3 are cached into tier 1.

template<typename State>
struct DualResult {
    bool found;
    std::vector<std::pair<std::string, State>> path;
    std::string tier;
};

template<typename State, typename Hash = std::hash<State>>
class DualProcessSolver {
private:
    std::map<State, std::vector<std::pair<std::string, State>>> cache;
    std::map<std::string, int> stats = {
        {"memory", 0}, {"reflex", 0}, {"deliberation", 0}, {"unsolved", 0}
    };
    int max_len;

public:
    DualProcessSolver(int max_len = 6) : max_len(max_len) {}

    // A mock reflex rollout for now (since policy models differ per domain)
    std::vector<std::pair<std::string, State>> _reflex(const SearchProblem<State>& prob) {
        State current = prob.initial();
        std::vector<std::pair<std::string, State>> path;
        for (int i = 0; i < max_len; ++i) {
            if (prob.is_goal(current)) return path;
            auto moves = prob.moves(current);
            if (moves.empty()) break;
            // Greedily pick the first/best move (in a real system, guided by policy net)
            auto best_move = moves[0];
            path.push_back({std::get<0>(best_move), std::get<1>(best_move)});
            current = std::get<1>(best_move);
        }
        if (prob.is_goal(current)) return path;
        return {};
    }

    DualResult<State> solve(const SearchProblem<State>& prob, int max_nodes = 200000) {
        State init = prob.initial();
        if (cache.count(init)) {
            stats["memory"]++;
            return {true, cache[init], "memory"};
        }

        auto reflex_path = _reflex(prob);
        if (!reflex_path.empty() || prob.is_goal(init)) { // reflex succeeded
            cache[init] = reflex_path;
            stats["reflex"]++;
            return {true, reflex_path, "reflex"};
        }

        auto search_res = solve_astar(prob, max_nodes);
        if (search_res.solved) {
            cache[init] = search_res.path;
            stats["deliberation"]++;
            return {true, search_res.path, "deliberation"};
        }

        stats["unsolved"]++;
        return {false, {}, "deliberation"};
    }

    const std::map<std::string, int>& get_stats() const { return stats; }
};

} // namespace reasoning
} // namespace brain2
