#!/usr/bin/env python3
"""
integral_engine.py — symbolic integration: rules SELECTED by form, may fail.

The contrast with differentiation is the point. Differentiation always succeeds:
one rule per node, composed deterministically. Integration must CHOOSE a rule by
the integrand's shape, and can fail outright — many functions (e.g. sin(x^2))
have no elementary antiderivative. That possibility of failure is what makes
integration the SEARCH case rather than the mechanical one.

Each result is verified by DIFFERENTIATING it back with the calculus engine —
integration checked by its own inverse:

    ie = IntegralEngine()
    ie.integrate(("^", "x", 2))      # -> x^3/3   (then d/dx(x^3/3) == x^2)

Honest scope: a bounded ruleset (constants, powers, sums, constant multiples,
basic trig/exp, 1/x). The hard cases — u-substitution, integration by parts —
are where real backtracking search lives (exponential); not built here. Returns
None when no rule in the set applies (honest "not elementary / unsupported").
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from calculus_engine import CalculusEngine, render, simplify
from physics_engine import contains, ev


class IntegralEngine:
    def __init__(self):
        self._ce = CalculusEngine()

    def integrate(self, e, var="x"):
        """Antiderivative of `e` w.r.t. `var` (constant of integration omitted),
        or None if no rule in the set applies."""
        F = self._int(e, var)
        return simplify(F) if F is not None else None

    def _int(self, e, var):
        # ∫ c dx = c*x   (anything free of the variable is constant)
        if not contains(e, var):
            return ("*", e, var)
        # ∫ x dx = x^2/2
        if e == var:
            return ("/", ("^", var, 2), 2)

        op = e[0]
        if op in ("+", "-"):                       # linearity
            a, b = self._int(e[1], var), self._int(e[2], var)
            return None if a is None or b is None else (op, a, b)

        if op == "*":                              # constant multiple only
            a, b = e[1], e[2]
            if not contains(a, var):
                ib = self._int(b, var)
                return ("*", a, ib) if ib is not None else None
            if not contains(b, var):
                ia = self._int(a, var)
                return ("*", b, ia) if ia is not None else None
            return None                            # product of two: by-parts (unsupported)

        if op == "^":                              # power rule
            base, n = e[1], e[2]
            if base == var and isinstance(n, (int, float)):
                if n == -1:
                    return ("ln", var)
                return ("/", ("^", var, n + 1), n + 1)
            return None

        if op == "sin" and e[1] == var:
            return ("neg", ("cos", var))
        if op == "cos" and e[1] == var:
            return ("sin", var)
        if op == "exp" and e[1] == var:
            return ("exp", var)
        if op == "/" and e[1] == 1 and e[2] == var:
            return ("ln", var)
        return None                                # no rule applies

    def verify(self, integrand, antideriv, var="x", at=1.3, tol=1e-4):
        """Differentiate the antiderivative back; does it recover the integrand?"""
        if antideriv is None:
            return False
        d = simplify(self._ce.diff(antideriv, var).expr)
        return abs(ev(d, {var: at}) - ev(integrand, {var: at})) < tol


def _demo():
    ie = IntegralEngine()
    cases = [
        ("x^2", ("^", "x", 2)),
        ("2*x", ("*", 2, "x")),
        ("x^2 + 3", ("+", ("^", "x", 2), 3)),
        ("cos(x)", ("cos", "x")),
        ("sin(x)", ("sin", "x")),
        ("1/x", ("/", 1, "x")),
        ("sin(x^2)  [no elementary form]", ("sin", ("^", "x", 2))),
    ]
    print("=== integral_engine — integrate, verified by differentiating back ===\n")
    for label, expr in cases:
        F = ie.integrate(expr)
        if F is None:
            print(f"  ∫ {label} dx  =  (no rule applies — honest fail)\n")
        else:
            ok = ie.verify(expr, F)
            print(f"  ∫ {label} dx  =  {render(F)} + C    [{'checked' if ok else 'WRONG'}]\n")


if __name__ == "__main__":
    _demo()
