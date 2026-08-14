#pragma once
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <optional>
#include <memory>
#include <iostream>

namespace brain2 {
namespace grounding {

class DimError : public std::runtime_error {
public:
    DimError(const std::string& msg) : std::runtime_error(msg) {}
};

// Assuming dimensions are represented as vectors of ints, e.g., {M, L, T}
using DimVec = std::vector<int>;

inline DimVec add_dims(const DimVec& a, const DimVec& b) {
    DimVec res(std::max(a.size(), b.size()), 0);
    for (size_t i = 0; i < a.size(); ++i) res[i] += a[i];
    for (size_t i = 0; i < b.size(); ++i) res[i] += b[i];
    return res;
}

inline DimVec sub_dims(const DimVec& a, const DimVec& b) {
    DimVec res(std::max(a.size(), b.size()), 0);
    for (size_t i = 0; i < a.size(); ++i) res[i] += a[i];
    for (size_t i = 0; i < b.size(); ++i) res[i] -= b[i];
    return res;
}

inline DimVec mul_dims(const DimVec& a, int k) {
    DimVec res = a;
    for (auto& x : res) x *= k;
    return res;
}

// AST structure for dimension evaluation (mimicking Python tuple expressions)
struct ExprNode {
    std::string op; // "+", "-", "*", "/", "^", "neg", "var", "const"
    std::string var_name;
    double const_val = 0.0;
    std::vector<std::shared_ptr<ExprNode>> children;
    
    static std::shared_ptr<ExprNode> make_var(const std::string& name) {
        auto n = std::make_shared<ExprNode>();
        n->op = "var";
        n->var_name = name;
        return n;
    }
    
    static std::shared_ptr<ExprNode> make_const(double val) {
        auto n = std::make_shared<ExprNode>();
        n->op = "const";
        n->const_val = val;
        return n;
    }
    
    static std::shared_ptr<ExprNode> make_op(const std::string& op, std::vector<std::shared_ptr<ExprNode>> children) {
        auto n = std::make_shared<ExprNode>();
        n->op = op;
        n->children = children;
        return n;
    }
};

inline std::optional<DimVec> dims_of(const std::shared_ptr<ExprNode>& expr, const std::map<std::string, DimVec>& units) {
    if (expr->op == "const") return std::nullopt; // Dimensionless
    
    if (expr->op == "var") {
        auto it = units.find(expr->var_name);
        if (it == units.end()) throw std::out_of_range("Unknown symbol " + expr->var_name);
        return it->second;
    }
    
    if (expr->op == "neg") return dims_of(expr->children[0], units);
    
    auto a = dims_of(expr->children[0], units);
    auto b = (expr->children.size() > 1) ? dims_of(expr->children[1], units) : std::nullopt;
    
    if (expr->op == "+" || expr->op == "-") {
        if (a.has_value() && b.has_value()) {
            if (a.value() != b.value()) throw DimError("Dimension mismatch in add/sub");
            return a;
        } else if (a.has_value()) {
            return a;
        } else if (b.has_value()) {
            return b;
        }
        return std::nullopt;
    }
    
    if (expr->op == "*") {
        if (!a.has_value()) return b;
        if (!b.has_value()) return a;
        return add_dims(*a, *b);
    }
    
    if (expr->op == "/") {
        if (!a.has_value() && !b.has_value()) return std::nullopt;
        if (!a.has_value()) {
            DimVec zero(b->size(), 0);
            return sub_dims(zero, *b);
        }
        if (!b.has_value()) return a;
        return sub_dims(*a, *b);
    }
    
    if (expr->op == "^") {
        if (expr->children[1]->op != "const") throw DimError("non-numeric exponent");
        if (!a.has_value()) return std::nullopt;
        return mul_dims(*a, (int)expr->children[1]->const_val);
    }
    
    throw DimError("unknown op " + expr->op);
}

// Struct representing a simplified Policy
struct PolicyDef {
    std::string target;
    std::shared_ptr<ExprNode> expr;
    std::vector<std::string> inputs;
};

inline std::optional<bool> dim_consistent(const PolicyDef& policy, const std::map<std::string, DimVec>& units) {
    std::optional<DimVec> d;
    try {
        d = dims_of(policy.expr, units);
    } catch (const DimError&) {
        return false;
    } catch (const std::out_of_range&) {
        return std::nullopt; // Unknown unit -> abstain
    }
    
    auto it = units.find(policy.target);
    if (it == units.end() || !d.has_value()) return std::nullopt;
    
    return d.value() == it->second;
}

inline double success_rate_feature(const PolicyDef& policy, const std::map<std::pair<std::string, std::string>, std::pair<int, int>>& history) {
    // Assuming history key is {target, combined_inputs_string} for simplicity
    std::string ins_str = "";
    for (const auto& i : policy.inputs) ins_str += i + ",";
    auto key = std::make_pair(policy.target, ins_str);
    
    auto it = history.find(key);
    int wins = 0, losses = 0;
    if (it != history.end()) {
        wins = it->second.first;
        losses = it->second.second;
    }
    return (double)(wins + 1) / (wins + losses + 2);
}

} // namespace grounding
} // namespace brain2
