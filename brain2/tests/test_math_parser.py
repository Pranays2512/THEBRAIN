#!/usr/bin/env python3
"""test_math_parser.py — exact parsing of math notation into expression tuples."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from math_parser import parse, ParseError

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nMathParser — notation -> tuples")
    check("power", parse("x^3") == ("^", "x", 3))
    check("function of power", parse("sin(x^2)") == ("sin", ("^", "x", 2)))
    check("precedence: * before +",
          parse("2*x + 3") == ("+", ("*", 2, "x"), 3))
    check("precedence: ^ before *",
          parse("2*x^2") == ("*", 2, ("^", "x", 2)))
    check("^ is right-associative",
          parse("2^3^2") == ("^", 2, ("^", 3, 2)))
    check("parentheses override",
          parse("(2 + 3)*x") == ("*", ("+", 2, 3), "x"))
    check("unary minus", parse("-cos(x)") == ("neg", ("cos", "x")))
    check("equation", parse("2*x + 3 = 7") == ("=", ("+", ("*", 2, "x"), 3), 7))
    check("float literal", parse("0.5*x") == ("*", 0.5, "x"))
    check("** accepted as ^", parse("x**2") == ("^", "x", 2))

    for bad in ["sin x", "2 +", "(x", "@"]:
        try:
            parse(bad)
            check(f"reject {bad!r}", False)
        except ParseError:
            check(f"reject {bad!r}", True)

    print(f"\nMath parser: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
