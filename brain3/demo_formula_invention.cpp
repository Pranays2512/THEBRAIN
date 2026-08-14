#include "crisp/engines/reasoning/monte_carlo_tree.hpp"
#include "crisp/engines/reasoning/tree_reason.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace brain2::reasoning;

struct FormulaState {
    std::string expr;
    std::vector<double> outputs;
    int depth = 0;

    bool operator<(const FormulaState& other) const {
        return expr < other.expr;
    }
};

class FormulaInventionProblem : public SearchProblem<FormulaState> {
public:
    FormulaInventionProblem() {
        for (double x : xs) {
            target_outputs.push_back(target(x));
        }
    }

    FormulaState initial() const override {
        return {"x", xs, 1};
    }

    bool is_goal(const FormulaState& s) const override {
        return !seen.count(s.expr) && matches_target(s.outputs);
    }

    double heuristic(const FormulaState& s) const override {
        return output_error(s.outputs) + 0.05 * std::max(0, s.depth - 3);
    }

    double novelty(const FormulaState& s) const override {
        double score = seen.count(s.expr) ? -3.0 : 1.0;
        score += 0.6 * op_count(s.expr);
        score += 2.0 / (1.0 + output_error(s.outputs));
        if (has(s.expr, "*x") && has(s.expr, "+x")) score += 1.2;
        if (has(s.expr, "+1")) score += 0.5;
        if (s.depth > max_depth) score -= 4.0;
        return score;
    }

    std::vector<std::tuple<std::string, FormulaState, double>> moves(const FormulaState& s) const override {
        std::vector<std::tuple<std::string, FormulaState, double>> out;
        if (s.depth >= max_depth) return out;

        add_move(out, "add 1", "(" + s.expr + "+1)", s, [](double y, double) { return y + 1.0; });
        add_move(out, "add x", "(" + s.expr + "+x)", s, [](double y, double x) { return y + x; });
        add_move(out, "multiply by x", "(" + s.expr + "*x)", s, [](double y, double x) { return y * x; });
        add_move(out, "double", "(2*" + s.expr + ")", s, [](double y, double) { return 2.0 * y; });
        add_move(out, "subtract 1", "(" + s.expr + "-1)", s, [](double y, double) { return y - 1.0; });
        return out;
    }

    std::string target_description() const {
        return "f(x) = x*x + x + 1";
    }

    std::string examples() const {
        std::ostringstream out;
        for (size_t i = 0; i < xs.size(); ++i) {
            if (i) out << ", ";
            out << "f(" << xs[i] << ")=" << target_outputs[i];
        }
        return out.str();
    }

private:
    std::vector<double> xs{-2, -1, 0, 1, 2, 3};
    std::vector<double> target_outputs;
    int max_depth = 4;
    std::set<std::string> seen{"x", "(x+1)", "(x*x)", "(2*x)", "((x*x)+1)"};

    static double target(double x) {
        return x * x + x + 1.0;
    }

    bool matches_target(const std::vector<double>& outputs) const {
        if (outputs.size() != target_outputs.size()) return false;
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (std::abs(outputs[i] - target_outputs[i]) > 1e-9) return false;
        }
        return true;
    }

    double output_error(const std::vector<double>& outputs) const {
        double err = 0.0;
        for (size_t i = 0; i < outputs.size() && i < target_outputs.size(); ++i) {
            err += std::abs(outputs[i] - target_outputs[i]);
        }
        return err;
    }

    int op_count(const std::string& expr) const {
        return static_cast<int>(std::count(expr.begin(), expr.end(), '+')
                              + std::count(expr.begin(), expr.end(), '*')
                              + std::count(expr.begin(), expr.end(), '-'));
    }

    bool has(const std::string& expr, const std::string& needle) const {
        return expr.find(needle) != std::string::npos;
    }

    template<typename Fn>
    void add_move(std::vector<std::tuple<std::string, FormulaState, double>>& out,
                  const std::string& label,
                  const std::string& expr,
                  const FormulaState& s,
                  Fn fn) const {
        std::vector<double> next_outputs;
        next_outputs.reserve(s.outputs.size());
        for (size_t i = 0; i < s.outputs.size(); ++i) {
            double y = fn(s.outputs[i], xs[i]);
            if (!std::isfinite(y) || std::abs(y) > 10000.0) return;
            next_outputs.push_back(y);
        }
        double cost = seen.count(expr) ? 3.0 : 1.0;
        out.push_back({label, {expr, next_outputs, s.depth + 1}, cost});
    }
};

void print_astar_result(const SearchResult<FormulaState>& result) {
    std::cout << "A* disciplined synthesis\n";
    std::cout << "  solved: " << (result.solved ? "yes" : "no") << "\n";
    std::cout << "  nodes expanded: " << result.nodes_expanded << "\n";
    std::cout << "  path cost: " << result.cost << "\n";
    if (!result.path.empty()) {
        for (size_t i = 0; i < result.path.size(); ++i) {
            std::cout << "  " << i + 1 << ". " << result.path[i].first
                      << " -> " << result.path[i].second.expr << "\n";
        }
        std::cout << "  invented formula: " << result.path.back().second.expr << "\n";
    }
    std::cout << "\n";
}

void print_mcts_result(const MonteCarloResult<FormulaState>& result) {
    std::cout << "MCTS novelty synthesis\n";
    std::cout << "  solved: " << (result.solved ? "yes" : "no") << "\n";
    std::cout << "  simulations: " << result.simulations << "\n";
    std::cout << "  tree nodes: " << result.nodes_expanded << "\n";
    std::cout << "  reward: " << result.reward << "\n";
    if (!result.path.empty()) {
        for (size_t i = 0; i < result.path.size(); ++i) {
            std::cout << "  " << i + 1 << ". " << result.path[i].first
                      << " -> " << result.path[i].second.expr << "\n";
        }
        std::cout << "  invented formula: " << result.path.back().second.expr << "\n";
    }
    std::cout << "\n";
}

int main() {
    FormulaInventionProblem problem;
    std::cout << "Formula invention task\n";
    std::cout << "Known formulas: x, x+1, x*x, 2*x, x*x+1\n";
    std::cout << "Examples: " << problem.examples() << "\n";
    std::cout << "Hidden answer for verification: " << problem.target_description() << "\n\n";

    auto astar = solve_astar(problem, 50000);
    print_astar_result(astar);

    MonteCarloConfig cfg;
    cfg.iterations = 1500;
    cfg.rollout_depth = 4;
    cfg.seed = 23;
    cfg.goal_reward = 30.0;
    cfg.novelty_weight = 1.8;
    auto mcts = solve_mcts(problem, cfg);
    print_mcts_result(mcts);

    return astar.solved && mcts.solved ? 0 : 1;
}
