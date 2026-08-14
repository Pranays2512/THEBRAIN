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

struct IntegrationTerm {
    std::string formula;
    std::vector<double> derivative_values;
    double cost;
    bool exp_poly;
    bool correction;
};

struct IntegralState {
    uint32_t mask = 0;
    std::string formula = "0";
    std::vector<double> derivative_values;
    int depth = 0;

    bool operator<(const IntegralState& other) const {
        return mask < other.mask;
    }
};

class IntegrationByPartsProblem : public SearchProblem<IntegralState> {
public:
    IntegrationByPartsProblem() {
        for (double x : xs) {
            integrand_values.push_back(integrand(x));
        }

        add_term("e^x*x^2", [](double x) { return std::exp(x) * (x * x + 2.0 * x); }, 1.0, true, false);
        add_term("-2*e^x*x", [](double x) { return -2.0 * std::exp(x) * (x + 1.0); }, 1.1, true, true);
        add_term("2*e^x", [](double x) { return 2.0 * std::exp(x); }, 0.8, true, true);
        add_term("e^x*x", [](double x) { return std::exp(x) * (x + 1.0); }, 1.0, true, false);
        add_term("-e^x*x", [](double x) { return -std::exp(x) * (x + 1.0); }, 1.1, true, true);
        add_term("e^x", [](double x) { return std::exp(x); }, 0.7, true, false);
        add_term("-e^x", [](double x) { return -std::exp(x); }, 0.8, true, true);
        add_term("x^3/3", [](double x) { return x * x; }, 1.3, false, false);
        add_term("x^2", [](double x) { return 2.0 * x; }, 1.4, false, false);
        add_term("sin(x)", [](double x) { return std::cos(x); }, 1.6, false, false);
    }

    IntegralState initial() const override {
        return {0, "0", std::vector<double>(xs.size(), 0.0), 0};
    }

    bool is_goal(const IntegralState& s) const override {
        return !known_masks.count(s.mask) && matches_integrand(s.derivative_values);
    }

    double heuristic(const IntegralState& s) const override {
        return mean_abs_error(s.derivative_values) + 0.04 * std::max(0, s.depth - 3);
    }

    double novelty(const IntegralState& s) const override {
        double score = known_masks.count(s.mask) ? -4.0 : 0.5;
        score += 3.0 / (1.0 + mean_abs_error(s.derivative_values));
        score += 0.15 * s.depth;
        if (uses_exp_poly(s.mask)) score += 1.2;
        if (uses_correction_chain(s.mask)) score += 2.0;
        if (uses_correction_chain(s.mask) && s.depth == 3) score += 1.5;
        if (has_cancelling_pair(s.mask)) score -= 3.0;
        if (s.depth > 3) score -= 1.5 * (s.depth - 3);
        return score;
    }

    std::vector<std::tuple<std::string, IntegralState, double>> moves(const IntegralState& s) const override {
        std::vector<std::tuple<std::string, IntegralState, double>> out;
        if (s.depth >= 5) return out;

        for (size_t i = 0; i < terms.size(); ++i) {
            uint32_t bit = 1u << i;
            if (s.mask & bit) continue;

            IntegralState next;
            next.mask = s.mask | bit;
            next.formula = append_formula(s.formula, terms[i].formula);
            next.depth = s.depth + 1;
            next.derivative_values.reserve(s.derivative_values.size());
            for (size_t j = 0; j < s.derivative_values.size(); ++j) {
                next.derivative_values.push_back(s.derivative_values[j] + terms[i].derivative_values[j]);
            }

            double cost = terms[i].cost;
            if (known_masks.count(next.mask)) cost += 2.5;
            out.push_back({"add " + terms[i].formula, next, cost});
        }
        return out;
    }

    std::string samples() const {
        std::ostringstream out;
        for (size_t i = 0; i < xs.size(); ++i) {
            if (i) out << ", ";
            out << "f(" << xs[i] << ")=" << integrand_values[i];
        }
        return out.str();
    }

    std::string hidden_formula() const {
        return "integral x^2*e^x dx = e^x*(x^2 - 2x + 2) + C";
    }

private:
    std::vector<double> xs{-2, -1, 0, 1, 2};
    std::vector<double> integrand_values;
    std::vector<IntegrationTerm> terms;
    std::set<uint32_t> known_masks{
        1u << 5,                 // e^x
        1u << 3,                 // e^x*x
        (1u << 3) | (1u << 6),   // e^x*x - e^x
        1u << 7,                 // x^3/3
        1u << 9,                 // sin(x)
    };

    template<typename Fn>
    void add_term(const std::string& formula, Fn derivative, double cost, bool exp_poly, bool correction) {
        std::vector<double> values;
        values.reserve(xs.size());
        for (double x : xs) {
            values.push_back(derivative(x));
        }
        terms.push_back({formula, values, cost, exp_poly, correction});
    }

    static double integrand(double x) {
        return x * x * std::exp(x);
    }

    bool matches_integrand(const std::vector<double>& values) const {
        if (values.size() != integrand_values.size()) return false;
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::abs(values[i] - integrand_values[i]) > 1e-8) return false;
        }
        return true;
    }

    double mean_abs_error(const std::vector<double>& values) const {
        double err = 0.0;
        for (size_t i = 0; i < values.size() && i < integrand_values.size(); ++i) {
            err += std::abs(values[i] - integrand_values[i]);
        }
        return err / static_cast<double>(integrand_values.size());
    }

    bool uses_exp_poly(uint32_t mask) const {
        for (size_t i = 0; i < terms.size(); ++i) {
            if ((mask & (1u << i)) && terms[i].exp_poly) return true;
        }
        return false;
    }

    bool uses_correction_chain(uint32_t mask) const {
        bool has_leading = mask & (1u << 0);
        bool has_linear_correction = mask & ((1u << 1) | (1u << 4));
        bool has_constant_correction = mask & ((1u << 2) | (1u << 6));
        return has_leading && has_linear_correction && has_constant_correction;
    }

    bool has_cancelling_pair(uint32_t mask) const {
        bool cancels_expx = (mask & (1u << 3)) && (mask & (1u << 4));
        bool cancels_exp = (mask & (1u << 5)) && (mask & (1u << 6));
        return cancels_expx || cancels_exp;
    }

    std::string append_formula(const std::string& formula, const std::string& term) const {
        if (formula == "0") return term;
        return "(" + formula + " + " + term + ")";
    }
};

void print_astar(const SearchResult<IntegralState>& result) {
    std::cout << "A* verified integration synthesis\n";
    std::cout << "  solved: " << (result.solved ? "yes" : "no") << "\n";
    std::cout << "  nodes expanded: " << result.nodes_expanded << "\n";
    std::cout << "  path cost: " << result.cost << "\n";
    for (size_t i = 0; i < result.path.size(); ++i) {
        std::cout << "  " << i + 1 << ". " << result.path[i].first
                  << " -> " << result.path[i].second.formula << "\n";
    }
    if (!result.path.empty()) {
        std::cout << "  antiderivative: " << result.path.back().second.formula << " + C\n";
    }
    std::cout << "\n";
}

void print_mcts(const MonteCarloResult<IntegralState>& result) {
    std::cout << "MCTS novelty integration synthesis\n";
    std::cout << "  solved: " << (result.solved ? "yes" : "no") << "\n";
    std::cout << "  simulations: " << result.simulations << "\n";
    std::cout << "  tree nodes: " << result.nodes_expanded << "\n";
    std::cout << "  reward: " << result.reward << "\n";
    for (size_t i = 0; i < result.path.size(); ++i) {
        std::cout << "  " << i + 1 << ". " << result.path[i].first
                  << " -> " << result.path[i].second.formula << "\n";
    }
    if (!result.path.empty()) {
        std::cout << "  antiderivative: " << result.path.back().second.formula << " + C\n";
    }
    std::cout << "\n";
}

int main() {
    IntegrationByPartsProblem problem;

    std::cout << "Graduate-level integration formula invention\n";
    std::cout << "Task: infer an antiderivative whose derivative matches x^2*e^x\n";
    std::cout << "Known memory: integral e^x, integral x*e^x, integral x^2, integral cos(x)\n";
    std::cout << "Derivative samples: " << problem.samples() << "\n";
    std::cout << "Hidden verifier formula: " << problem.hidden_formula() << "\n\n";

    auto astar = solve_astar(problem, 100000);
    print_astar(astar);

    MonteCarloConfig cfg;
    cfg.iterations = 3000;
    cfg.rollout_depth = 5;
    cfg.seed = 41;
    cfg.goal_reward = 45.0;
    cfg.novelty_weight = 2.2;
    cfg.stop_on_goal = false;
    auto mcts = solve_mcts(problem, cfg);
    print_mcts(mcts);

    return astar.solved && mcts.solved ? 0 : 1;
}
