#pragma once
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <memory>
#include <queue>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace brain2 {
namespace synthesis {

struct ExprNode {
    std::string op; // "+", "-", "*", "/", "VAR", "CONST"
    std::string val; // "mass", "0.5" etc.
    std::shared_ptr<ExprNode> left;
    std::shared_ptr<ExprNode> right;
    
    std::string to_string() const {
        if (op == "VAR" || op == "CONST") return val;
        return "(" + left->to_string() + " " + op + " " + right->to_string() + ")";
    }
    
    // Evaluate for a single row
    double eval(const std::map<std::string, double>& env) const {
        if (op == "VAR") return env.at(val);
        if (op == "CONST") return std::stod(val);
        
        double l = left->eval(env);
        double r = right->eval(env);
        
        if (op == "+") return l + r;
        if (op == "-") return l - r;
        if (op == "*") return l * r;
        if (op == "/") {
            if (std::abs(r) < 1e-9) return INFINITY;
            return l / r;
        }
        return 0;
    }
};

class PolicyInduction {
public:
    double tolerance = 1e-5;

    // Evaluates expression over all rows, returning output vector
    std::vector<double> get_outputs(std::shared_ptr<ExprNode> expr, const std::vector<std::map<std::string, double>>& rows) {
        std::vector<double> outs;
        for (const auto& r : rows) {
            double v = expr->eval(r);
            if (std::isinf(v) || std::isnan(v)) return {};
            outs.push_back(v);
        }
        return outs;
    }

    // Proposer heuristic: absolute Pearson correlation with target.
    // 2.0 if perfect fit.
    double score_expr(std::shared_ptr<ExprNode> expr, const std::vector<std::map<std::string, double>>& rows, const std::string& target_col) {
        auto outs = get_outputs(expr, rows);
        if (outs.empty()) return -1.0;
        
        std::vector<double> tgts;
        for (const auto& r : rows) tgts.push_back(r.at(target_col));
        
        bool perfect = true;
        for (size_t i = 0; i < outs.size(); i++) {
            if (std::abs(outs[i] - tgts[i]) > tolerance) { perfect = false; break; }
        }
        if (perfect) return 2.0;

        // Calculate correlation
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
        int n = outs.size();
        for (int i = 0; i < n; i++) {
            sum_x += outs[i];
            sum_y += tgts[i];
            sum_xy += outs[i] * tgts[i];
            sum_x2 += outs[i] * outs[i];
            sum_y2 += tgts[i] * tgts[i];
        }
        
        double num = (n * sum_xy) - (sum_x * sum_y);
        double den = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
        if (den < 1e-9) return 0.0;
        return std::abs(num / den);
    }
    
    int get_size(std::shared_ptr<ExprNode> e) {
        if (!e) return 0;
        if (e->op == "VAR" || e->op == "CONST") return 1;
        return 1 + get_size(e->left) + get_size(e->right);
    }

    bool fits(std::shared_ptr<ExprNode> expr, const std::vector<std::map<std::string, double>>& rows, const std::string& target_col) {
        auto outs = get_outputs(expr, rows);
        if (outs.empty()) return false;
        for (size_t i = 0; i < outs.size(); i++) {
            if (std::abs(outs[i] - rows[i].at(target_col)) > tolerance) return false;
        }
        return true;
    }

    std::shared_ptr<ExprNode> induce(std::vector<std::map<std::string, double>> rows, const std::vector<std::string>& inputs, const std::string& target, int max_size = 7) {
        if (rows.size() < 4) return nullptr;
        
        // Pseudo-random split
        int cut = std::max(2, (int)(rows.size() * 0.6));
        std::vector<std::map<std::string, double>> train(rows.begin(), rows.begin() + cut);
        std::vector<std::map<std::string, double>> holdout(rows.begin() + cut, rows.end());
        
        std::vector<std::shared_ptr<ExprNode>> terminals;
        for (const auto& in : inputs) {
            terminals.push_back(std::make_shared<ExprNode>(ExprNode{"VAR", in, nullptr, nullptr}));
        }
        terminals.push_back(std::make_shared<ExprNode>(ExprNode{"CONST", "0.5", nullptr, nullptr}));
        terminals.push_back(std::make_shared<ExprNode>(ExprNode{"CONST", "2.0", nullptr, nullptr}));

        struct SearchNode {
            double priority;
            int tie;
            std::shared_ptr<ExprNode> expr;
            bool operator<(const SearchNode& o) const {
                if (std::abs(priority - o.priority) > 1e-9) return priority < o.priority;
                return tie > o.tie;
            }
        };

        std::priority_queue<SearchNode> pq;
        int tie = 0;
        
        for (auto t : terminals) {
            pq.push({score_expr(t, train, target), tie++, t});
        }
        
        std::vector<std::string> ops = {"+", "-", "*", "/"};
        int budget = 1500;
        
        while (!pq.empty() && budget-- > 0) {
            auto node = pq.top(); pq.pop();
            
            if (fits(node.expr, train, target) && fits(node.expr, holdout, target)) {
                return node.expr;
            }
            
            if (get_size(node.expr) >= max_size) continue;
            
            // Expand by combining with terminals only (bounded branching)
            for (const auto& op : ops) {
                for (auto t : terminals) {
                    auto e1 = std::make_shared<ExprNode>(ExprNode{op, "", node.expr, t});
                    auto e2 = std::make_shared<ExprNode>(ExprNode{op, "", t, node.expr});
                    
                    double p1 = score_expr(e1, train, target);
                    if (p1 > -0.5) pq.push({p1, tie++, e1});
                    
                    double p2 = score_expr(e2, train, target);
                    if (p2 > -0.5) pq.push({p2, tie++, e2});
                }
            }
        }
        
        return nullptr;
    }
};

} // namespace synthesis
} // namespace brain2
