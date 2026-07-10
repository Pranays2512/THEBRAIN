#!/usr/bin/env python3
"""
test_integral_engine.py — integration: rule selection, verified by its inverse.

Each antiderivative is checked by differentiating it back to the integrand (the
real guarantee). Also pins the honest difference from differentiation: when no
rule applies, integration returns None instead of a wrong answer.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.math.integral_engine import IntegralEngine

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nIntegralEngine — integrate, verified by differentiating back")
    ie = IntegralEngine()

    # every supported form must integrate AND verify against its derivative
    supported = [
        ("^", "x", 2),                        # power
        ("*", 2, "x"),                        # constant multiple
        ("+", ("^", "x", 2), 3),              # linearity + constant
        ("cos", "x"),
        ("sin", "x"),
        ("exp", "x"),
        ("/", 1, "x"),                        # -> ln(x)
        5,                                    # constant -> 5*x
    ]
    all_verified = True
    for e in supported:
        F = ie.integrate(e)
        if F is None or not ie.verify(e, F):
            all_verified = False
    check("all supported integrals verify (d/dx back == integrand)", all_verified)

    # specific shapes
    check("power: int x^2 = x^3/3 (derivative checks)",
          ie.verify(("^", "x", 2), ie.integrate(("^", "x", 2))))
    check("trig: int sin(x) = -cos(x)",
          ie.verify(("sin", "x"), ie.integrate(("sin", "x"))))

    # honest failure: no elementary antiderivative in the ruleset
    check("sin(x^2) -> None (honest, not faked)",
          ie.integrate(("sin", ("^", "x", 2))) is None)
    check("product of two x-functions -> None (by-parts unsupported)",
          ie.integrate(("*", "x", ("sin", "x"))) is None)

    print(f"\nIntegral engine: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
