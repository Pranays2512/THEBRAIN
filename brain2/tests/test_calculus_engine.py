#!/usr/bin/env python3
"""
test_calculus_engine.py — symbolic differentiation: rules compose, answers correct.

Pins both that the right policies fire (composition) and that the derivative is
actually correct — checked numerically against a finite difference, so the
symbolic result is verified, not just asserted by shape.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.math.calculus_engine import CalculusEngine, simplify

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def ev(e, xv):
    """Evaluate an expression tuple at x = xv."""
    if isinstance(e, (int, float)):
        return e
    if isinstance(e, str):
        return xv if e == "x" else 0.0
    op = e[0]
    if op == "neg":
        return -ev(e[1], xv)
    if op == "sin":
        return math.sin(ev(e[1], xv))
    if op == "cos":
        return math.cos(ev(e[1], xv))
    if op == "exp":
        return math.exp(ev(e[1], xv))
    if op == "ln":
        return math.log(ev(e[1], xv))
    a, b = ev(e[1], xv), ev(e[2], xv)
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b, "^": a ** b}[op]


def matches_numerically(expr, xv=1.3, h=1e-6):
    ce = CalculusEngine()
    d = simplify(ce.diff(expr).expr)
    numeric = (ev(expr, xv + h) - ev(expr, xv - h)) / (2 * h)
    return abs(ev(d, xv) - numeric) < 1e-4


def run():
    print("\nCalculusEngine — differentiation (policies compose, verified)")
    ce = CalculusEngine()

    # 1. single policies
    check("power rule: d/dx x^3 = 3*x^2", ce.diff(("^", "x", 3)).text == "3*x^2")
    check("trig rule: d/dx sin(x) = cos(x)", ce.diff(("sin", "x")).text == "cos(x)")

    # 2. composition: chain rule fires for sin(x^2)
    r = ce.diff(("sin", ("^", "x", 2)))
    check("chain rule composes with trig+power", "chain rule" in r.rules)
    check("chain result cos(x^2)*(2*x)", "cos(x^2)" in r.text and "2*x" in r.text)

    # 3. product rule pulls in several policies at once
    r2 = ce.diff(("*", ("^", "x", 2), ("sin", "x")))
    check("product composes power+trig",
          {"product rule", "power rule", "sin rule"} <= set(r2.rules))

    # 4. correctness, verified numerically (the real guarantee)
    exprs = [
        ("^", "x", 3),
        ("sin", "x"),
        ("*", ("^", "x", 2), ("sin", "x")),
        ("sin", ("^", "x", 2)),
        ("*", ("exp", "x"), ("ln", "x")),
        ("/", ("sin", "x"), "x"),
        ("cos", ("*", 2, "x")),
    ]
    allok = all(matches_numerically(e) for e in exprs)
    check("all derivatives match finite difference", allok)

    print(f"\nCalculus engine: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
