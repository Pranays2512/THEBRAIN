#pragma once
#include "crisp/engines/reasoning/tree_reason.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace brain2 {
namespace reasoning {

struct MonteCarloConfig {
    int iterations = 1000;
    int rollout_depth = 8;
    double exploration = 1.41421356237;
    double goal_reward = 10.0;
    double novelty_weight = 1.0;
    unsigned seed = 7;
    bool stop_on_goal = true;
};

template<typename State, typename Hash = std::hash<State>>
struct MonteCarloResult {
    bool solved = false;
    std::vector<std::pair<std::string, State>> path;
    double reward = 0.0;
    int simulations = 0;
    int nodes_expanded = 0;
};

template<typename State, typename Hash = std::hash<State>>
MonteCarloResult<State, Hash> solve_mcts(const SearchProblem<State, Hash>& problem,
                                         MonteCarloConfig cfg = {}) {
    struct Edge {
        std::string label;
        State state;
        double cost;
    };

    struct Node {
        State state;
        int parent = -1;
        std::string label_from_parent;
        std::vector<Edge> untried;
        std::vector<int> children;
        int visits = 0;
        double value = 0.0;
    };

    auto node_path = [](const std::vector<Node>& nodes, int id) {
        std::vector<std::pair<std::string, State>> out;
        while (id >= 0 && nodes[id].parent >= 0) {
            out.push_back({nodes[id].label_from_parent, nodes[id].state});
            id = nodes[id].parent;
        }
        std::reverse(out.begin(), out.end());
        return out;
    };

    auto state_reward = [&](const State& s) {
        double reward = cfg.novelty_weight * problem.novelty(s);
        if (problem.is_goal(s)) reward += cfg.goal_reward;
        double h = problem.heuristic(s);
        if (std::isfinite(h) && h > 0.0) reward += 1.0 / (1.0 + h);
        return reward;
    };

    std::mt19937 rng(cfg.seed);
    std::vector<Node> nodes;
    State root = problem.initial();
    nodes.push_back({root, -1, "", {}, {}, 0, 0.0});
    for (const auto& [label, nxt, cost] : problem.moves(root)) {
        nodes[0].untried.push_back({label, nxt, cost});
    }

    MonteCarloResult<State, Hash> best;
    best.reward = state_reward(root);
    if (problem.is_goal(root)) {
        best.solved = true;
        return best;
    }

    for (int iter = 0; iter < cfg.iterations; ++iter) {
        best.simulations = iter + 1;
        int current = 0;

        while (nodes[current].untried.empty() && !nodes[current].children.empty()) {
            double best_score = -std::numeric_limits<double>::infinity();
            int best_child = nodes[current].children.front();
            double parent_log = std::log(std::max(1, nodes[current].visits));

            for (int child_id : nodes[current].children) {
                const auto& child = nodes[child_id];
                double mean = child.visits == 0 ? 0.0 : child.value / child.visits;
                double explore = cfg.exploration * std::sqrt(parent_log / (child.visits + 1.0));
                double score = mean + explore + cfg.novelty_weight * problem.novelty(child.state);
                if (score > best_score) {
                    best_score = score;
                    best_child = child_id;
                }
            }
            current = best_child;
        }

        if (!nodes[current].untried.empty()) {
            std::uniform_int_distribution<int> pick(0, static_cast<int>(nodes[current].untried.size()) - 1);
            int edge_index = pick(rng);
            Edge edge = nodes[current].untried[edge_index];
            nodes[current].untried.erase(nodes[current].untried.begin() + edge_index);

            Node child{edge.state, current, edge.label, {}, {}, 0, 0.0};
            for (const auto& [label, nxt, cost] : problem.moves(edge.state)) {
                child.untried.push_back({label, nxt, cost});
            }

            nodes.push_back(child);
            int child_id = static_cast<int>(nodes.size()) - 1;
            nodes[current].children.push_back(child_id);
            current = child_id;
        }

        std::vector<std::pair<std::string, State>> rollout_path = node_path(nodes, current);
        State sim_state = nodes[current].state;
        double reward = state_reward(sim_state);
        bool solved = problem.is_goal(sim_state);

        for (int depth = 0; depth < cfg.rollout_depth && !solved; ++depth) {
            auto moves = problem.moves(sim_state);
            if (moves.empty()) break;

            std::vector<double> weights;
            weights.reserve(moves.size());
            for (const auto& [label, nxt, cost] : moves) {
                weights.push_back(0.001 + std::max(0.0, problem.novelty(nxt)) + (problem.is_goal(nxt) ? cfg.goal_reward : 0.0));
            }

            std::discrete_distribution<int> pick(weights.begin(), weights.end());
            int move_id = pick(rng);
            const auto& [label, nxt, cost] = moves[move_id];
            sim_state = nxt;
            rollout_path.push_back({label, sim_state});

            reward = std::max(reward, state_reward(sim_state));
            solved = problem.is_goal(sim_state);
        }

        bool improves_best = best.path.empty() || reward > best.reward || (solved && !best.solved);
        if (improves_best) {
            best.solved = solved;
            best.path = rollout_path;
            best.reward = reward;
            best.simulations = iter + 1;
            best.nodes_expanded = static_cast<int>(nodes.size());
        }

        int back = current;
        while (back >= 0) {
            nodes[back].visits++;
            nodes[back].value += reward;
            back = nodes[back].parent;
        }

        if (solved && cfg.stop_on_goal) break;
    }

    best.nodes_expanded = static_cast<int>(nodes.size());
    return best;
}

} // namespace reasoning
} // namespace brain2
