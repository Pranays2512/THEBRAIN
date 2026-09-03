#pragma once
/**
 * algebra_engine.hpp — solve an equation for a variable, with verification.
 * Port of brain2/engines/math/algebra_engine.py to C++.
 *
 * Solves  left = right  for the unknown (appearing once) by:
 *   1. Identifying which side contains the variable
 *   2. Calling isolate() to rearrange
 *   3. Evaluating the result
 *   4. Back-substituting to verify  |lhs - rhs| < tol
 */

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include "crisp/engines/math/calculus_engine.hpp"
#include "crisp/engines/math/physics_engine.hpp"

namespace brain2 {
namespace math {

class AlgebraError : public std::runtime_error {
public:
    explicit AlgebraError(const std::string& msg) : std::runtime_error(msg) {}
};

class AlgebraEngine {
public:
    /**
     * Solve  equation (op="=", lhs, rhs)  for var.
     * Returns (value, steps) where steps[0]=original, steps[1]=rearranged, steps[2]=numeric.
     * Throws AlgebraError if var appears on both sides or not at all.
     */
    std::pair<double, std::vector<std::string>>
    solve(const ExprPtr& equation, const std::string& var = "x") const {
        if (!equation || equation->op != "=")
            throw AlgebraError("equation must be an ExprNode with op='='");
        if (equation->children.size() < 2)
            throw AlgebraError("equation must have exactly two children");

        const auto& left  = equation->children[0];
        const auto& right = equation->children[1];
        const bool lhs_has = contains(left,  var);
        const bool rhs_has = contains(right, var);

        if (!lhs_has && !rhs_has)
            throw AlgebraError("'" + var + "' is not in the equation");

        // ── Path 1: direct algebraic inversion (isolate) ─────────────────────
        // Tried FIRST whenever the variable occurs on exactly one side. This is
        // deliberate and load-bearing: isolate() returns the PRINCIPAL inverse,
        // so `x^2 = 49` yields 7 rather than -7. Root-finding is a strictly
        // wider net that would answer -7 (the smaller root), so making it the
        // fallback rather than the default means no equation that already
        // solved can change its answer.
        if (lhs_has != rhs_has) {
            const auto& expr_side = lhs_has ? left  : right;
            const auto& other     = lhs_has ? right : left;
            try {
                ExprPtr tree = isolate(expr_side, var, other);
                std::map<std::string, double> empty_env;
                double value = ev(tree, empty_env);
                if (std::isfinite(value)) {
                    value = std::round(value * 1e6) / 1e6;
                    if (_verify(equation, var, value)) {
                        return {value, {
                            render(left) + " = " + render(right),
                            var + " = " + render(simplify(tree)),
                            var + " = " + std::to_string(value)
                        }};
                    }
                }
            } catch (...) {
                // isolate() could not peel the variable out, or the isolated
                // tree still referenced it. Fall through — the polynomial path
                // handles exactly the cases isolate() structurally cannot.
            }
        }

        // ── Path 2: polynomial normal form ───────────────────────────────────
        // Collect both sides into  power -> coefficient  over `var`, subtract,
        // and solve by degree. This is what handles repeated occurrences of the
        // variable, which isolate() cannot see: its binary recursion descends
        // into ONE child per node, so a second occurrence in the sibling
        // subtree is silently folded into the "answer" instead of collected.
        Poly p, lp, rp;
        if (to_poly(left, var, lp) && to_poly(right, var, rp)) {
            for (const auto& kv : lp) p[kv.first] += kv.second;
            for (const auto& kv : rp) p[kv.first] -= kv.second;

            auto solved = _solve_poly(p, var, equation);
            if (!_verify(equation, var, solved.first))
                throw AlgebraError("internal: solution failed back-substitution");
            return solved;
        }

        if (lhs_has && rhs_has)
            throw AlgebraError("'" + var + "' on both sides and the equation is "
                               "not polynomial in '" + var + "'");
        throw AlgebraError("could not isolate '" + var + "' and the equation is "
                           "not polynomial in '" + var + "'");
    }

private:
    // power -> coefficient. Absent key means coefficient 0.
    using Poly = std::map<int, double>;

    static constexpr int  kMaxDegree = 8;
    static constexpr double kEps     = 1e-12;

    static std::string num(double v) {
        char b[64];
        std::snprintf(b, sizeof(b), "%.6g", v);
        return std::string(b);
    }

    // Highest power with a non-negligible coefficient. -1 == identically zero.
    static int poly_degree(const Poly& p) {
        int d = -1;
        for (const auto& kv : p)
            if (std::fabs(kv.second) > kEps && kv.first > d) d = kv.first;
        return d;
    }

    static double poly_at(const Poly& p, int k) {
        auto it = p.find(k);
        return it == p.end() ? 0.0 : it->second;
    }

    static bool poly_mul(const Poly& a, const Poly& b, Poly& out) {
        for (const auto& ka : a)
            for (const auto& kb : b) {
                const int power = ka.first + kb.first;
                if (power > kMaxDegree) return false;   // runaway guard
                out[power] += ka.second * kb.second;
            }
        return true;
    }

    /**
     * Flatten `e` into a polynomial in `var`. Returns false — never a partial
     * or approximate result — when the expression is not polynomial in `var`
     * (a second unknown, the variable in a denominator or exponent, or any
     * transcendental). Callers must treat false as "this method does not
     * apply" and fall back, NOT as "no solution".
     */
    static bool to_poly(const ExprPtr& e, const std::string& var, Poly& out) {
        if (!e) return false;

        if (e->op == "val") { out[0] += e->val; return true; }
        if (e->op == "var") {
            if (e->var != var) return false;   // a second unknown
            out[1] += 1.0;
            return true;
        }
        if (e->op == "neg") {
            if (e->children.size() != 1) return false;
            Poly a;
            if (!to_poly(e->children[0], var, a)) return false;
            for (const auto& kv : a) out[kv.first] -= kv.second;
            return true;
        }
        if (e->children.size() != 2) return false;

        Poly a, b;
        if (!to_poly(e->children[0], var, a)) return false;
        if (!to_poly(e->children[1], var, b)) return false;

        if (e->op == "+" || e->op == "-") {
            const double s = (e->op == "+") ? 1.0 : -1.0;
            for (const auto& kv : a) out[kv.first] += kv.second;
            for (const auto& kv : b) out[kv.first] += s * kv.second;
            return true;
        }
        if (e->op == "*") {
            Poly m;
            if (!poly_mul(a, b, m)) return false;
            for (const auto& kv : m) out[kv.first] += kv.second;
            return true;
        }
        if (e->op == "/") {
            // Only division by a constant keeps it polynomial. `12 / x` does
            // not — that case belongs to isolate().
            if (poly_degree(b) != 0) return false;
            const double d = poly_at(b, 0);
            if (std::fabs(d) < kEps) return false;
            for (const auto& kv : a) out[kv.first] += kv.second / d;
            return true;
        }
        if (e->op == "^") {
            if (poly_degree(b) != 0) return false;          // x in the exponent
            const double n = poly_at(b, 0);
            if (n < 0 || std::fabs(n - std::round(n)) > kEps) return false;
            const int k = static_cast<int>(std::lround(n));
            if (k > kMaxDegree) return false;
            Poly acc;
            acc[0] = 1.0;
            for (int i = 0; i < k; ++i) {
                Poly t;
                if (!poly_mul(acc, a, t)) return false;
                acc.swap(t);
            }
            for (const auto& kv : acc) out[kv.first] += kv.second;
            return true;
        }
        return false;   // sin / cos / exp / ln — not polynomial
    }

    static std::string render_poly(const Poly& p, const std::string& var) {
        std::string s;
        bool first = true;
        for (auto it = p.rbegin(); it != p.rend(); ++it) {
            const double c = it->second;
            const int    k = it->first;
            if (std::fabs(c) <= kEps) continue;
            if (!first)      s += (c < 0 ? " - " : " + ");
            else if (c < 0)  s += "-";
            const double a = std::fabs(c);
            if (k == 0) {
                s += num(a);
            } else {
                if (std::fabs(a - 1.0) > kEps) s += num(a) + "*";
                s += var;
                if (k >= 2) s += "^" + std::to_string(k);
            }
            first = false;
        }
        if (first) s = "0";
        return s + " = 0";
    }

    /**
     * Solve the collected polynomial. Every non-solution outcome throws with a
     * specific reason — a degenerate equation must never come back as a number.
     */
    static std::pair<double, std::vector<std::string>>
    _solve_poly(const Poly& p, const std::string& var, const ExprPtr& equation) {
        std::vector<std::string> steps;
        steps.push_back(render(equation->children[0]) + " = " +
                        render(equation->children[1]));
        steps.push_back("collect like terms:  " + render_poly(p, var));

        const int deg = poly_degree(p);

        if (deg < 0)
            throw AlgebraError("identity — every value of '" + var +
                               "' satisfies this equation");
        if (deg == 0)
            throw AlgebraError("no solution — constant terms do not balance (" +
                               num(poly_at(p, 0)) + " != 0)");

        if (deg == 1) {
            const double a = poly_at(p, 1), b = poly_at(p, 0);
            double x = std::round((-b / a) * 1e6) / 1e6;
            steps.push_back(var + " = " + num(-b) + " / " + num(a));
            steps.push_back(var + " = " + num(x));
            return {x, steps};
        }

        if (deg == 2) {
            const double a = poly_at(p, 2), b = poly_at(p, 1), c = poly_at(p, 0);
            const double disc = b * b - 4.0 * a * c;
            steps.push_back("discriminant = " + num(b * b) + " - " +
                            num(4.0 * a * c) + " = " + num(disc));
            if (disc < 0.0)
                throw AlgebraError("no real roots (discriminant " + num(disc) +
                                   " < 0)");
            const double s = std::sqrt(disc);
            double r1 = (-b - s) / (2.0 * a);
            double r2 = (-b + s) / (2.0 * a);
            if (r1 > r2) std::swap(r1, r2);
            r1 = std::round(r1 * 1e6) / 1e6;
            r2 = std::round(r2 * 1e6) / 1e6;
            if (std::fabs(r1 - r2) <= 1e-9) {
                steps.push_back("double root: " + var + " = " + num(r1));
            } else {
                steps.push_back("roots: " + var + " = " + num(r1) + ", " +
                                var + " = " + num(r2));
                steps.push_back("returning smallest real root: " + var + " = " +
                                num(r1));
            }
            return {r1, steps};
        }

        throw AlgebraError("degree " + std::to_string(deg) +
                           " polynomial — solver supports degree <= 2");
    }

    static bool _verify(const ExprPtr& equation, const std::string& var,
                        double value, double tol = 1e-6) {
        std::map<std::string, double> env = {{var, value}};
        try {
            double lval = ev(equation->children[0], env);
            double rval = ev(equation->children[1], env);
            return std::abs(lval - rval) < tol;
        } catch (...) {
            return false;
        }
    }
};

} // namespace math
} // namespace brain2
