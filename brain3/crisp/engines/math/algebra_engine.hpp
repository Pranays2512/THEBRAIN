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
#include <cmath>
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
        bool lhs_has = contains(left,  var);
        bool rhs_has = contains(right, var);

        if (lhs_has && rhs_has)
            throw AlgebraError("'" + var + "' on both sides — needs term collection");
        if (!lhs_has && !rhs_has)
            throw AlgebraError("'" + var + "' is not in the equation");

        const auto& expr_side = lhs_has ? left  : right;
        const auto& other     = lhs_has ? right : left;

        ExprPtr tree;
        try {
            tree = isolate(expr_side, var, other);
        } catch (const PhysicsError& e) {
            throw AlgebraError(std::string(e.what()));
        }

        std::map<std::string, double> empty_env;
        double value;
        try {
            value = ev(tree, empty_env);
        } catch (...) {
            // If tree still has variables, evaluate with empty (error if so)
            throw AlgebraError("isolated expression still contains unknowns");
        }

        // Round to 6 decimal places
        value = std::round(value * 1e6) / 1e6;

        std::vector<std::string> steps = {
            render(left) + " = " + render(right),
            var + " = " + render(simplify(tree)),
            var + " = " + std::to_string(value)
        };

        if (!_verify(equation, var, value))
            throw AlgebraError("internal: solution failed back-substitution");

        return {value, steps};
    }

private:
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
