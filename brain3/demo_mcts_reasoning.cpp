#include "crisp/engines/reasoning/monte_carlo_tree.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace brain2::reasoning;

struct IdeaState {
    std::vector<std::string> concepts;

    bool operator<(const IdeaState& other) const {
        return concepts < other.concepts;
    }
};

class NovelIdeaSearch : public SearchProblem<IdeaState> {
public:
    IdeaState initial() const override {
        return {{"memory"}};
    }

    bool is_goal(const IdeaState& s) const override {
        return has(s, "memory") && has(s, "crystal") && has(s, "prediction") && has(s, "error");
    }

    double heuristic(const IdeaState& s) const override {
        double missing = 0.0;
        for (const auto& target : targets) {
            if (!has(s, target)) missing += 1.0;
        }
        return missing;
    }

    double novelty(const IdeaState& s) const override {
        double score = 0.0;
        score += 0.25 * unique_domains(s);
        if (has(s, "memory") && has(s, "crystal")) score += 1.5;
        if (has(s, "prediction") && has(s, "error")) score += 1.0;
        if (has(s, "memory") && has(s, "prediction")) score += 0.8;
        if (has(s, "crystal") && has(s, "error")) score += 0.7;
        for (const auto& concept : s.concepts) {
            if (concept != "memory" && concept != "crystal" && concept != "prediction" && concept != "error") {
                score -= 0.7;
            }
        }
        return score;
    }

    std::vector<std::tuple<std::string, IdeaState, double>> moves(const IdeaState& s) const override {
        std::vector<std::tuple<std::string, IdeaState, double>> out;
        for (const auto& candidate : candidates) {
            if (has(s, candidate)) continue;
            auto next = s;
            next.concepts.push_back(candidate);
            std::sort(next.concepts.begin(), next.concepts.end());
            out.push_back({"blend in " + candidate, next, 1.0});
        }
        return out;
    }

private:
    const std::vector<std::string> candidates{
        "crystal", "prediction", "error", "river", "market", "syntax", "gravity", "echo"
    };
    const std::vector<std::string> targets{"memory", "crystal", "prediction", "error"};

    bool has(const IdeaState& s, const std::string& concept) const {
        return std::find(s.concepts.begin(), s.concepts.end(), concept) != s.concepts.end();
    }

    int unique_domains(const IdeaState& s) const {
        std::set<std::string> domains;
        for (const auto& concept : s.concepts) {
            if (concept == "memory" || concept == "prediction" || concept == "error") {
                domains.insert("cognition");
            } else if (concept == "crystal" || concept == "gravity" || concept == "river") {
                domains.insert("physical");
            } else {
                domains.insert("symbolic");
            }
        }
        return static_cast<int>(domains.size());
    }
};

std::string render(const IdeaState& s) {
    std::ostringstream out;
    for (size_t i = 0; i < s.concepts.size(); ++i) {
        if (i) out << " + ";
        out << s.concepts[i];
    }
    return out.str();
}

int main() {
    NovelIdeaSearch problem;
    MonteCarloConfig cfg;
    cfg.iterations = 500;
    cfg.rollout_depth = 5;
    cfg.seed = 19;
    cfg.novelty_weight = 2.0;
    cfg.goal_reward = 20.0;
    cfg.stop_on_goal = false;

    auto result = solve_mcts(problem, cfg);

    std::cout << "Task: structurally search for a novel concept blend\n";
    std::cout << "Start: " << render(problem.initial()) << "\n\n";
    for (size_t i = 0; i < result.path.size(); ++i) {
        std::cout << i + 1 << ". " << result.path[i].first
                  << " -> " << render(result.path[i].second) << "\n";
    }
    std::cout << "\nSolved: " << (result.solved ? "yes" : "no") << "\n";
    std::cout << "Reward: " << result.reward << "\n";
    std::cout << "Simulations: " << result.simulations << "\n";
    std::cout << "Expanded tree nodes: " << result.nodes_expanded << "\n";

    if (result.solved && !result.path.empty()) {
        std::cout << "\nCandidate invention: predictive memory crystal\n";
        std::cout << "Interpretation: a stable memory structure that preserves prediction-error traces for later reasoning.\n";
    }

    return result.solved ? 0 : 1;
}
