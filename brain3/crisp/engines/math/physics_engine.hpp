#pragma once
/**
 * physics_engine.hpp — symbolic expression evaluation, variable isolation, and
 * physics law application.
 * Port of brain2/engines/math/physics_engine.py to C++.
 *
 * Provides:
 *   - ev(expr, env)       : evaluate an ExprNode with a variable environment
 *   - contains(expr, var) : test if a variable appears in an expression
 *   - isolate(expr, var, other) : rearrange expr=other to solve for var
 *   - PhysicsEngine       : store named laws and solve for any variable
 */

#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "crisp/engines/math/calculus_engine.hpp"

namespace brain2 {
namespace math {

class PhysicsError : public std::runtime_error {
public:
    explicit PhysicsError(const std::string& msg) : std::runtime_error(msg) {}
};

// ── Utilities ─────────────────────────────────────────────────────────────────

inline bool contains(const ExprPtr& expr, const std::string& target) {
    if (!expr) return false;
    if (is_var(expr) && expr->var == target) return true;
    for (const auto& c : expr->children)
        if (contains(c, target)) return true;
    return false;
}

inline double ev(const ExprPtr& expr, const std::map<std::string, double>& env) {
    if (!expr) throw PhysicsError("null expression");
    if (is_num(expr)) return expr->val;
    if (is_var(expr)) {
        auto it = env.find(expr->var);
        if (it == env.end()) throw PhysicsError("unknown value for '" + expr->var + "'");
        return it->second;
    }
    const auto& op = expr->op;
    if (op == "neg") return -ev(expr->children[0], env);
    if (op == "sin") return std::sin(ev(expr->children[0], env));
    if (op == "cos") return std::cos(ev(expr->children[0], env));
    if (op == "exp") return std::exp(ev(expr->children[0], env));
    if (op == "ln")  return std::log(ev(expr->children[0], env));
    double a = ev(expr->children[0], env);
    double b = ev(expr->children[1], env);
    if (op == "+") return a + b;
    if (op == "-") return a - b;
    if (op == "*") return a * b;
    if (op == "/") {
        if (b == 0.0) throw PhysicsError("division by zero");
        return a / b;
    }
    if (op == "^") return std::pow(a, b);
    throw PhysicsError("unknown operator: " + op);
}

/**
 * Solve expr=other for target, returning the rearranged expression tree.
 * Inverts operations outward: * -> /, + -> -, ^n -> ^(1/n).
 */
inline ExprPtr isolate(const ExprPtr& expr, const std::string& target, const ExprPtr& other) {
    if (is_var(expr) && expr->var == target) return other;
    if (!expr || expr->is_leaf()) throw PhysicsError("cannot isolate " + target + " in leaf");

    const auto& op = expr->op;
    const auto& a = expr->children[0];
    const auto& b = (expr->children.size() > 1) ? expr->children[1] : nullptr;

    if (op == "*") {
        if (contains(a, target))
            return isolate(a, target, ExprNode::make_op("/", {other, b}));
        else
            return isolate(b, target, ExprNode::make_op("/", {other, a}));
    }
    if (op == "/") {
        // a/b = other
        if (contains(a, target))
            return isolate(a, target, ExprNode::make_op("*", {other, b}));
        else
            // b = a/other
            return isolate(b, target, ExprNode::make_op("/", {a, other}));
    }
    if (op == "+") {
        if (contains(a, target))
            return isolate(a, target, ExprNode::make_op("-", {other, b}));
        else
            return isolate(b, target, ExprNode::make_op("-", {other, a}));
    }
    if (op == "-") {
        // a-b = other
        if (contains(a, target))
            return isolate(a, target, ExprNode::make_op("+", {other, b}));
        else
            // b = a-other
            return isolate(b, target, ExprNode::make_op("-", {a, other}));
    }
    if (op == "^") {
        // a^n = other -> a = other^(1/n)
        return isolate(a, target,
            ExprNode::make_op("^", {other, ExprNode::make_op("/", {ExprNode::make_num(1.0), b})}));
    }
    throw PhysicsError("cannot invert operator '" + op + "'");
}

// ── PhysicsEngine ─────────────────────────────────────────────────────────────

struct Law {
    std::string lhs_symbol;  // the isolated variable
    ExprPtr rhs_expr;        // the right-hand side expression tree
};

class PhysicsEngine {
    std::map<std::string, Law> laws_;

public:
    void add_law(const std::string& name, const std::string& lhs, const ExprPtr& rhs) {
        laws_[name] = {lhs, rhs};
    }

    std::vector<std::string> variable_names(const std::string& name) const {
        auto it = laws_.find(name);
        if (it == laws_.end()) throw PhysicsError("no law named '" + name + "'");
        std::vector<std::string> vars;
        vars.push_back(it->second.lhs_symbol);
        _collect_vars(it->second.rhs_expr, vars);
        return vars;
    }

    /**
     * Solve law `name` for `target` given knowns in env.
     * Returns (value, {formula_step, substituted_step}).
     */
    std::pair<double, std::vector<std::string>>
    solve(const std::string& name, const std::string& target,
          const std::map<std::string, double>& knowns) const {
        auto it = laws_.find(name);
        if (it == laws_.end()) throw PhysicsError("no law named '" + name + "'");
        const auto& law = it->second;

        ExprPtr tree;
        if (target == law.lhs_symbol) {
            tree = law.rhs_expr;
        } else {
            if (!contains(law.rhs_expr, target))
                throw PhysicsError("'" + target + "' is not in law '" + name + "'");
            tree = isolate(law.rhs_expr, target, ExprNode::make_var(law.lhs_symbol));
        }

        double value = ev(tree, knowns);
        std::string formula = target + " = " + render(tree);
        std::string substituted = target + " = " + render(_subst(tree, knowns)) + " = " + std::to_string(value);
        return {value, {formula, substituted}};
    }

    bool has_law(const std::string& name) const { return laws_.count(name) > 0; }

private:
    static void _collect_vars(const ExprPtr& e, std::vector<std::string>& out) {
        if (!e) return;
        if (is_var(e)) { out.push_back(e->var); return; }
        for (const auto& c : e->children) _collect_vars(c, out);
    }

    static ExprPtr _subst(const ExprPtr& expr, const std::map<std::string, double>& env) {
        if (!expr) return nullptr;
        if (is_var(expr)) {
            auto it = env.find(expr->var);
            if (it != env.end()) return ExprNode::make_num(it->second);
        }
        if (expr->is_leaf()) return expr;
        std::vector<ExprPtr> new_children;
        for (const auto& c : expr->children) new_children.push_back(_subst(c, env));
        return ExprNode::make_op(expr->op, new_children);
    }
};

} // namespace math
} // namespace brain2
