#include "crisp/engines/reasoning/monte_carlo_tree.hpp"
#include "crisp/engines/reasoning/tree_reason.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace brain2::reasoning;

struct Sample2D {
    double q;
    double p;
};

struct Term {
    std::string name;
    std::vector<double> values;
    double cost;
    bool coupled;
};

struct EnergyState {
    uint32_t mask = 0;
    std::string expr = "0";
    std::vector<double> values;
    int depth = 0;

    bool operator<(const EnergyState& other) const {
        return mask < other.mask;
    }
};

class GraduateEnergyProblem : public SearchProblem<EnergyState> {
public:
    GraduateEnergyProblem() {
        for (const auto& s : samples) {
            target_values.push_back(target(s.q, s.p));
        }

        add_term("2*q^2", [](double q, double) { return 2.0 * q * q; }, 1.1, false);
        add_term("2*q*p", [](double q, double p) { return 2.0 * q * p; }, 1.2, true);
        add_term("p^2", [](double, double p) { return p * p; }, 1.0, false);
        add_term("1", [](double, double) { return 1.0; }, 0.6, false);
        add_term("q^2", [](double q, double) { return q * q; }, 0.9, false);
        add_term("(q+p)^2", [](double q, double p) { return (q + p) * (q + p); }, 1.8, true);
        add_term("q", [](double q, double) { return q; }, 1.4, false);
        add_term("p", [](double, double p) { return p; }, 1.4, false);
        add_term("-q*p", [](double q, double p) { return -q * p; }, 1.5, true);
    }

    EnergyState initial() const override {
        return {0, "0", std::vector<double>(samples.size(), 0.0), 0};
    }

    bool is_goal(const EnergyState& s) const override {
        return !known_masks.count(s.mask) && matches_target(s.values);
    }

    double heuristic(const EnergyState& s) const override {
        return mean_abs_error(s.values) + 0.03 * std::max(0, s.depth - 4);
    }

    double novelty(const EnergyState& s) const override {
        double score = known_masks.count(s.mask) ? -4.0 : 0.5;
        score += 2.5 / (1.0 + mean_abs_error(s.values));
        score += 0.35 * s.depth;
        if (uses_coupled_term(s.mask)) score += 1.7;
        if (is_positive_definite_shape(s.mask)) score += 1.2;
        if (s.depth > 5) score -= 2.0;
        return score;
    }

    std::vector<std::tuple<std::string, EnergyState, double>> moves(const EnergyState& s) const override {
        std::vector<std::tuple<std::string, EnergyState, double>> out;
        if (s.depth >= 5) return out;

        for (size_t i = 0; i < terms.size(); ++i) {
            uint32_t bit = 1u << i;
            if (s.mask & bit) continue;

            EnergyState next;
            next.mask = s.mask | bit;
            next.expr = append_expr(s.expr, terms[i].name);
            next.depth = s.depth + 1;
            next.values.reserve(s.values.size());
            for (size_t j = 0; j < s.values.size(); ++j) {
                next.values.push_back(s.values[j] + terms[i].values[j]);
            }

            double cost = terms[i].cost;
            if (known_masks.count(next.mask)) cost += 2.5;
            out.push_back({"add " + terms[i].name, next, cost});
        }

        return out;
    }

    std::string examples() const {
        std::ostringstream out;
        for (size_t i = 0; i < samples.size(); ++i) {
            if (i) out << ", ";
            out << "V(" << samples[i].q << "," << samples[i].p << ")=" << target_values[i];
        }
        return out.str();
    }

    std::string hidden_formula() const {
        return "V(q,p) = q^2 + (q+p)^2 + 1 = 2q^2 + 2qp + p^2 + 1";
    }

private:
    std::vector<Sample2D> samples{{-2, 1}, {-1, -1}, {0, 2}, {1, 0}, {2, -1}, {3, 2}};
    std::vector<double> target_values;
    std::vector<Term> terms;
    std::set<uint32_t> known_masks{
        1u << 4,                 // q^2
        1u << 2,                 // p^2
        1u << 5,                 // (q+p)^2
        (1u << 4) | (1u << 2),   // q^2 + p^2
        (1u << 6) | (1u << 7),   // q + p
    };

    template<typename Fn>
    void add_term(const std::string& name, Fn fn, double cost, bool coupled) {
        std::vector<double> values;
        values.reserve(samples.size());
        for (const auto& s : samples) {
            values.push_back(fn(s.q, s.p));
        }
        terms.push_back({name, values, cost, coupled});
    }

    static double target(double q, double p) {
        return q * q + (q + p) * (q + p) + 1.0;
    }

    bool matches_target(const std::vector<double>& values) const {
        for (size_t i = 0; i < values.size() && i < target_values.size(); ++i) {
            if (std::abs(values[i] - target_values[i]) > 1e-9) return false;
        }
        return values.size() == target_values.size();
    }

    double mean_abs_error(const std::vector<double>& values) const {
        double err = 0.0;
        for (size_t i = 0; i < values.size() && i < target_values.size(); ++i) {
            err += std::abs(values[i] - target_values[i]);
        }
        return err / static_cast<double>(target_values.size());
    }

    bool uses_coupled_term(uint32_t mask) const {
        for (size_t i = 0; i < terms.size(); ++i) {
            if ((mask & (1u << i)) && terms[i].coupled) return true;
        }
        return false;
    }

    bool is_positive_definite_shape(uint32_t mask) const {
        bool has_q2 = mask & ((1u << 0) | (1u << 4) | (1u << 5));
        bool has_p2 = mask & ((1u << 2) | (1u << 5));
        bool has_bias = mask & (1u << 3);
        return has_q2 && has_p2 && has_bias;
    }

    std::string append_expr(const std::string& expr, const std::string& term) const {
        if (expr == "0") return term;
        return "(" + expr + " + " + term + ")";
    }
};

void print_astar(const SearchResult<EnergyState>& result) {
    std::cout << "A* verified construction\n";
    std::cout << "  solved: " << (result.solved ? "yes" : "no") << "\n";
    std::cout << "  nodes expanded: " << result.nodes_expanded << "\n";
    std::cout << "  path cost: " << result.cost << "\n";
    for (size_t i = 0; i < result.path.size(); ++i) {
        std::cout << "  " << i + 1 << ". " << result.path[i].first
                  << " -> " << result.path[i].second.expr << "\n";
    }
    if (!result.path.empty()) {
        std::cout << "  formula: " << result.path.back().second.expr << "\n";
    }
    std::cout << "\n";
}

void print_mcts(const MonteCarloResult<EnergyState>& result) {
    std::cout << "MCTS novelty construction\n";
    std::cout << "  solved: " << (result.solved ? "yes" : "no") << "\n";
    std::cout << "  simulations: " << result.simulations << "\n";
    std::cout << "  tree nodes: " << result.nodes_expanded << "\n";
    std::cout << "  reward: " << result.reward << "\n";
    for (size_t i = 0; i < result.path.size(); ++i) {
        std::cout << "  " << i + 1 << ". " << result.path[i].first
                  << " -> " << result.path[i].second.expr << "\n";
    }
    if (!result.path.empty()) {
        std::cout << "  formula: " << result.path.back().second.expr << "\n";
    }
    std::cout << "\n";
}

int main() {
    GraduateEnergyProblem problem;

    std::cout << "Graduate-level formula invention task\n";
    std::cout << "Domain: coupled quadratic energy / Lyapunov candidate\n";
    std::cout << "Known memory: q^2, p^2, (q+p)^2, q^2+p^2, q+p\n";
    std::cout << "Samples: " << problem.examples() << "\n";
    std::cout << "Hidden verifier formula: " << problem.hidden_formula() << "\n\n";

    auto astar = solve_astar(problem, 100000);
    print_astar(astar);

    MonteCarloConfig cfg;
    cfg.iterations = 2000;
    cfg.rollout_depth = 5;
    cfg.seed = 31;
    cfg.goal_reward = 40.0;
    cfg.novelty_weight = 2.0;
    cfg.stop_on_goal = false;
    auto mcts = solve_mcts(problem, cfg);
    print_mcts(mcts);

    return astar.solved && mcts.solved ? 0 : 1;
}
