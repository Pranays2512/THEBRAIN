#!/usr/bin/env python3
"""
test_algebra_engine.py — solve equations for x by isolation, verified.

Pins correct solutions (checked by back-substitution inside solve), the steps,
and honest refusal when the unknown appears on both sides or not at all.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from algebra_engine import AlgebraEngine, AlgebraError

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nAlgebraEngine — solve for x, verified")
    ae = AlgebraEngine()

    check("2*x + 3 = 7 -> 2",
          ae.solve(("=", ("+", ("*", 2, "x"), 3), 7))[0] == 2)
    check("5*x = 20 -> 4",
          ae.solve(("=", ("*", 5, "x"), 20))[0] == 4)
    check("x/3 + 1 = 4 -> 9",
          ae.solve(("=", ("+", ("/", "x", 3), 1), 4))[0] == 9)
    check("x^2 = 49 -> 7",
          ae.solve(("=", ("^", "x", 2), 49))[0] == 7)
    check("3*x - 5 = 10 -> 5",
          ae.solve(("=", ("-", ("*", 3, "x"), 5), 10))[0] == 5)

    # unknown on the right side too
    check("10 = 2*x -> 5", ae.solve(("=", 10, ("*", 2, "x")))[0] == 5)

    # steps show the isolated form
    _, steps = ae.solve(("=", ("*", 5, "x"), 20))
    check("steps show x = 20/5", steps[1] == "x = 20/5")

    # honest refusals
    try:
        ae.solve(("=", ("+", "x", 1), ("*", 2, "x")))     # x both sides
        check("reject x on both sides", False)
    except AlgebraError:
        check("reject x on both sides", True)
    try:
        ae.solve(("=", ("+", 2, 3), 5))                   # no x
        check("reject equation without x", False)
    except AlgebraError:
        check("reject equation without x", True)

    print(f"\nAlgebra engine: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
