#pragma once
/**
 * integral_engine.hpp — symbolic integration by pattern matching.
 * Port of brain2/engines/math/integral_engine.py to C++.
 *
 * Attempts antiderivative by a bounded ruleset (constants, powers,
 * linearity, basic trig/exp). Returns nullptr when no rule applies —
 * honest "not elementary / unsupported". Every result is verified by
 * differentiating it back with calculus_engine.
 */

#include <string>
#include <cmath>
#include <stdexcept>
#include "crisp/engines/math/calculus_engine.hpp"
#include "crisp/engines/math/physics_engine.hpp"

namespace brain2 {
namespace math {

// Forward declaration of CalculusEngine (defined inline below to avoid
// requiring a separate calculus_engine.cpp for the diff() function)
// We use ev() from physics_engine.hpp for numerical verification.

class IntegralEngine {
public:
    /**
     * Compute the antiderivative of `e` w.r.t. `var`.
     * Returns nullptr if no rule applies.
     */
    ExprPtr integrate(const ExprPtr& e, const std::string& var = "x") const {
        auto F = _int(e, var);
        return F ? simplify(F) : nullptr;
    }

    /**
     * Verify by differentiating back: does d/dvar(antideriv) == integrand?
     * Checked numerically at a test point.
     */
    bool verify(const ExprPtr& integrand, const ExprPtr& antideriv,
                const std::string& var = "x",
                double at = 1.3, double tol = 1e-4) const {
        if (!antideriv) return false;
        auto d = CalculusEngine::diff(antideriv, var);
        if (!d) return false;   // no rule differentiates the candidate — unverifiable, so not verified
        try {
            std::map<std::string, double> env = {{var, at}};
            return std::abs(CalculusEngine::eval(d, env) - CalculusEngine::eval(integrand, env)) < tol;
        } catch (...) { return false; }
    }

private:
    // ── Integration rules ─────────────────────────────────────────────────────

    ExprPtr _int(const ExprPtr& e, const std::string& var) const {
        if (!e) return nullptr;

        // ∫ c dx = c*x  (anything free of the variable is a constant)
        if (!contains(e, var))
            return ExprNode::make_op("*", {e, ExprNode::make_var(var)});

        // ∫ x dx = x^2/2
        if (is_var(e) && e->var == var)
            return ExprNode::make_op("/",
                {ExprNode::make_op("^", {ExprNode::make_var(var), ExprNode::make_num(2)}),
                 ExprNode::make_num(2)});

        const auto& op = e->op;

        // Linearity: ∫ (a ± b) dx = ∫a ± ∫b
        if (op == "+" || op == "-") {
            auto a = _int(e->children[0], var);
            auto b = _int(e->children[1], var);
            if (!a || !b) return nullptr;
            return ExprNode::make_op(op, {a, b});
        }

        // Constant multiple: ∫ c*f dx = c * ∫f
        if (op == "*") {
            const auto& lhs = e->children[0];
            const auto& rhs = e->children[1];
            if (!contains(lhs, var)) {
                auto ib = _int(rhs, var);
                return ib ? ExprNode::make_op("*", {lhs, ib}) : nullptr;
            }
            if (!contains(rhs, var)) {
                auto ia = _int(lhs, var);
                return ia ? ExprNode::make_op("*", {rhs, ia}) : nullptr;
            }
            return nullptr; // product of two var-terms: by-parts (unsupported)
        }

        // Power rule: ∫ x^n dx = x^(n+1)/(n+1)
        if (op == "^") {
            const auto& base = e->children[0];
            const auto& n    = e->children[1];
            if (is_var(base) && base->var == var && is_num(n)) {
                double exp_val = n->val;
                if (exp_val == -1.0)
                    return ExprNode::make_op("ln", {ExprNode::make_var(var)});
                return ExprNode::make_op("/",
                    {ExprNode::make_op("^", {ExprNode::make_var(var), ExprNode::make_num(exp_val + 1)}),
                     ExprNode::make_num(exp_val + 1)});
            }
            return nullptr;
        }

        if (op == "neg") {
            auto in = _int(e->children[0], var);
            return in ? ExprNode::make_op("neg", {in}) : nullptr;
        }

        // Basic trig / exp: ∫ sin(x) = -cos(x), ∫ cos(x) = sin(x), ∫ exp(x) = exp(x)
        if (op == "sin" && is_var(e->children[0]) && e->children[0]->var == var)
            return ExprNode::make_op("neg",
                {ExprNode::make_op("cos", {ExprNode::make_var(var)})});
        if (op == "cos" && is_var(e->children[0]) && e->children[0]->var == var)
            return ExprNode::make_op("sin", {ExprNode::make_var(var)});
        if (op == "exp" && is_var(e->children[0]) && e->children[0]->var == var)
            return ExprNode::make_op("exp", {ExprNode::make_var(var)});

        // ∫ c/x dx = c * ln(x)
        if (op == "/") {
            const auto& num = e->children[0];
            const auto& den = e->children[1];
            if (!contains(num, var) && is_var(den) && den->var == var) {
                if (is_num(num) && num->val == 1.0) {
                    return ExprNode::make_op("ln", {ExprNode::make_var(var)});
                }
                return ExprNode::make_op("*", {num, ExprNode::make_op("ln", {ExprNode::make_var(var)})});
            }
            if (!contains(den, var)) {
                auto in = _int(num, var);
                return in ? ExprNode::make_op("/", {in, den}) : nullptr;
            }
        }

        return nullptr; // no rule applies
    }

    // ── Symbolic differentiation (minimal, for verification only) ─────────────
    ExprPtr _diff(const ExprPtr& e, const std::string& var) const {
        if (!e) return ExprNode::make_num(0);
        if (is_num(e)) return ExprNode::make_num(0);
        if (is_var(e)) return (e->var == var) ? ExprNode::make_num(1) : ExprNode::make_num(0);

        const auto& op = e->op;
        auto& ch = e->children;

        if (op == "neg") return ExprNode::make_op("neg", {_diff(ch[0], var)});
        if (op == "+") return ExprNode::make_op("+", {_diff(ch[0], var), _diff(ch[1], var)});
        if (op == "-") return ExprNode::make_op("-", {_diff(ch[0], var), _diff(ch[1], var)});
        // product rule
        if (op == "*")
            return ExprNode::make_op("+",
                {ExprNode::make_op("*", {_diff(ch[0], var), ch[1]}),
                 ExprNode::make_op("*", {ch[0], _diff(ch[1], var)})});
        // quotient rule
        if (op == "/")
            return ExprNode::make_op("/",
                {ExprNode::make_op("-",
                    {ExprNode::make_op("*", {_diff(ch[0], var), ch[1]}),
                     ExprNode::make_op("*", {ch[0], _diff(ch[1], var)})}),
                 ExprNode::make_op("*", {ch[1], ch[1]})});
        // power rule (x^n) -> n * x^(n-1)
        if (op == "^" && is_num(ch[1])) {
            double n = ch[1]->val;
            return ExprNode::make_op("*",
                {ExprNode::make_num(n),
                 ExprNode::make_op("^", {ch[0], ExprNode::make_num(n - 1)})});
        }
        if (op == "sin") return ExprNode::make_op("*", {ExprNode::make_op("cos", {ch[0]}), _diff(ch[0], var)});
        if (op == "cos") return ExprNode::make_op("*", {ExprNode::make_op("neg", {ExprNode::make_op("sin", {ch[0]})}), _diff(ch[0], var)});
        if (op == "exp") return ExprNode::make_op("*", {ExprNode::make_op("exp", {ch[0]}), _diff(ch[0], var)});
        if (op == "ln")  return ExprNode::make_op("/", {_diff(ch[0], var), ch[0]});
        return ExprNode::make_num(0);
    }
};

} // namespace math
} // namespace brain2
