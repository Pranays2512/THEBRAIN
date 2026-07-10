#!/usr/bin/env python3
"""test_math_chat.py — plain-notation requests routed to the right engine."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from adapters.math_chat import MathChat

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nMathChat — ask in notation, routed to engines")
    mc = MathChat()

    d = mc.respond("differentiate sin(x^2)")
    check("differentiate routes to calculus", "cos(x^2)" in d and "chain rule" in d)

    d2 = mc.respond("what is the derivative of x^3?")
    check("'derivative of' phrasing works", "3*x^2" in d2)

    i = mc.respond("integrate cos(x)")
    check("integrate routes to integral engine", "sin(x)" in i and "checked" in i)

    i2 = mc.respond("integrate sin(x^2)")
    check("honest no-elementary-form", "no rule" in i2.lower())

    s = mc.respond("solve 2*x + 3 = 7 for x")
    check("solve routes to algebra, verified", "x = 2" in s and "verified" in s)

    s2 = mc.respond("solve x^2 = 49 for x")
    check("solve power equation", "x = 7" in s2)

    check("non-equation solve is honest",
          "not an equation" in mc.respond("solve x + 1").lower())
    check("parse error is reported",
          "couldn't parse" in mc.respond("differentiate sin x").lower())
    check("no intent -> guidance",
          "differentiate" in mc.respond("hello there").lower())

    print(f"\nMath chat: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
