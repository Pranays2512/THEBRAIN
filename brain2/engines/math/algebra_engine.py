#!/usr/bin/env python3
"""
algebra_engine.py — solve an equation for x, with the steps and a proof.

Generalizes the physics isolate() kernel from "rearrange a law" to "solve an
equation": given  left = right  with the unknown appearing once, isolate it,
evaluate, and VERIFY by substituting the answer back into the original equation
(so the solution is checked, not just produced).

    ae = AlgebraEngine()
    ae.solve(("=", ("+", ("*", 2, "x"), 3), 7))     # 2*x + 3 = 7  ->  x = 2

Honest scope: the unknown appears once (linear / single-power equations). Terms
on both sides, or x appearing twice, need term-collection — the next rung of the
search-based algebra (rewrite + solve), not this direct isolation.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engines.math.calculus_engine import render, simplify
from engines.math.physics_engine import contains, ev, isolate, PhysicsError


class AlgebraError(ValueError):
    pass


class AlgebraEngine:
    def solve(self, equation, var="x"):
        """Solve  left = right  for `var`. Returns (value, steps). Raises if the
        unknown appears on both sides (needs term collection) or not at all."""
        if not (isinstance(equation, tuple) and equation[0] == "="):
            raise AlgebraError("equation must be ('=', left, right)")
        left, right = equation[1], equation[2]
        lhs_has, rhs_has = contains(left, var), contains(right, var)
        if lhs_has and rhs_has:
            raise AlgebraError(f"'{var}' on both sides — needs term collection")
        if not (lhs_has or rhs_has):
            raise AlgebraError(f"'{var}' is not in the equation")

        expr_side, other = (left, right) if lhs_has else (right, left)
        try:
            tree = isolate(expr_side, var, other)
        except PhysicsError as e:
            raise AlgebraError(str(e))
        value = round(ev(tree, {}), 6)

        steps = [f"{render(equation[1])} = {render(equation[2])}",
                 f"{var} = {render(simplify(tree))}",
                 f"{var} = {value}"]
        if not self._verify(equation, var, value):
            raise AlgebraError("internal: solution failed back-substitution")
        return value, steps

    @staticmethod
    def _verify(equation, var, value, tol=1e-6):
        """Substitute the answer back into the ORIGINAL equation: both sides equal?"""
        env = {var: value}
        return abs(ev(equation[1], env) - ev(equation[2], env)) < tol


def _demo():
    ae = AlgebraEngine()
    eqs = [
        ("2*x + 3 = 7", ("=", ("+", ("*", 2, "x"), 3), 7)),
        ("5*x = 20", ("=", ("*", 5, "x"), 20)),
        ("x/3 + 1 = 4", ("=", ("+", ("/", "x", 3), 1), 4)),
        ("x^2 = 49", ("=", ("^", "x", 2), 49)),
        ("3*x - 5 = 10", ("=", ("-", ("*", 3, "x"), 5), 10)),
    ]
    print("=== algebra_engine — solve for x, verified ===\n")
    for label, eq in eqs:
        value, steps = ae.solve(eq)
        print(f"  {label}")
        print(f"      {steps[1]}  ->  {steps[2]}  (checked)\n")


if __name__ == "__main__":
    _demo()
