#!/usr/bin/env python3
"""
calculus_engine.py — symbolic differentiation: many rules COMPOSED in one pass.

This is the concrete answer to "can the brain switch between policies?" Each
differentiation rule (power, product, quotient, chain, trig, exp, log) is a
policy. A single expression forces several to compose — d/dx(sin(x^2)) needs the
chain rule, the trig rule, and the power rule together — and the engine applies
exactly the ones the expression demands, reporting which fired.

Differentiation is mechanical and complete, so the result is correct BY
CONSTRUCTION (no search, no guessing). Expressions are nested tuples:

    ("^", "x", 3)            x^3
    ("*", ("^","x",2), ("sin","x"))     x^2 * sin(x)
    ("sin", ("^","x",2))     sin(x^2)

    CalculusEngine().diff(expr) -> Result(expr, simplified_str, rules_used)

Honest scope: differentiation (a finite, mechanical ruleset). Integration is the
SEARCH case (rules can apply many ways, not mechanical) — a separate build.
"""

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

VAR = "x"


@dataclass
class Result:
    expr: tuple
    text: str
    rules: list = field(default_factory=list)


def _num(e):
    return isinstance(e, (int, float))


class CalculusEngine:
    def __init__(self):
        self._rules = None

    def diff(self, expr, var=VAR):
        """Differentiate `expr` w.r.t. `var`; return the simplified derivative
        plus the ordered list of distinct rules (policies) that fired."""
        self._rules = []
        d = self._d(expr, var)
        return Result(d, render(simplify(d)), list(dict.fromkeys(self._rules)))

    def _use(self, rule):
        self._rules.append(rule)

    def _d(self, e, var):
        # constant
        if _num(e):
            self._use("constant: d/dx c = 0")
            return 0
        # variable
        if isinstance(e, str):
            if e == var:
                self._use("variable: d/dx x = 1")
                return 1
            self._use("constant: d/dx (other symbol) = 0")
            return 0

        op = e[0]

        if op in ("+", "-"):
            self._use(f"{'sum' if op == '+' else 'difference'} rule")
            return (op, self._d(e[1], var), self._d(e[2], var))

        if op == "*":
            self._use("product rule")
            u, v = e[1], e[2]
            return ("+", ("*", self._d(u, var), v), ("*", u, self._d(v, var)))

        if op == "/":
            self._use("quotient rule")
            u, v = e[1], e[2]
            num = ("-", ("*", self._d(u, var), v), ("*", u, self._d(v, var)))
            return ("/", num, ("^", v, 2))

        if op == "^":                                # u^n, n constant
            self._use("power rule")
            u, n = e[1], e[2]
            inner = self._d(u, var)
            base = ("*", n, ("^", u, n - 1))
            if not _is_var(u, var):
                self._use("chain rule")
                return ("*", base, inner)
            return base

        if op in ("sin", "cos", "exp", "ln"):
            self._use(f"{op} rule")
            u = e[1]
            inner = self._d(u, var)
            if op == "sin":
                outer = ("cos", u)
            elif op == "cos":
                outer = ("neg", ("sin", u))
            elif op == "exp":
                outer = ("exp", u)
            else:                                    # ln
                outer = ("/", 1, u)
            if not _is_var(u, var):
                self._use("chain rule")
                return ("*", outer, inner)
            return outer

        raise ValueError(f"unknown operator: {op!r}")


def _is_var(e, var):
    return isinstance(e, str) and e == var


# ── simplification (so the answer is readable and minimal) ───────────────────
def simplify(e):
    if _num(e) or isinstance(e, str):
        return e
    op = e[0]
    if op == "neg":
        a = simplify(e[1])
        if _num(a):
            return -a
        return ("neg", a)
    args = [simplify(x) for x in e[1:]]
    a = args[0]
    b = args[1] if len(args) > 1 else None

    if op == "+":
        if a == 0:
            return b
        if b == 0:
            return a
        if _num(a) and _num(b):
            return a + b
    elif op == "-":
        if b == 0:
            return a
        if _num(a) and _num(b):
            return a - b
    elif op == "*":
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a
        if _num(a) and _num(b):
            return a * b
    elif op == "/":
        if a == 0:
            return 0
        if b == 1:
            return a
    elif op == "^":
        if b == 1:
            return a
        if b == 0:
            return 1
    return (op, *args)


# ── rendering ────────────────────────────────────────────────────────────────
def render(e):
    if _num(e):
        return str(e)
    if isinstance(e, str):
        return e
    op = e[0]
    if op == "neg":
        return f"-{render(e[1])}"
    if op in ("sin", "cos", "exp", "ln"):
        return f"{op}({render(e[1])})"
    a, b = render(e[1]), render(e[2])
    if op == "^":
        return f"{_paren(e[1])}^{b}"
    sym = {"+": " + ", "-": " - ", "*": "*", "/": "/"}[op]
    return f"{_paren(e[1])}{sym}{_paren(e[2])}"


def _paren(e):
    if _num(e) or isinstance(e, str) or e[0] in ("sin", "cos", "exp", "ln", "^", "neg"):
        return render(e)
    return f"({render(e)})"


def _demo():
    ce = CalculusEngine()
    cases = [
        ("x^3", ("^", "x", 3)),
        ("sin(x)", ("sin", "x")),
        ("x^2 * sin(x)", ("*", ("^", "x", 2), ("sin", "x"))),
        ("sin(x^2)", ("sin", ("^", "x", 2))),
        ("exp(x) * ln(x)", ("*", ("exp", "x"), ("ln", "x"))),
        ("sin(x) / x", ("/", ("sin", "x"), "x")),
    ]
    print("=== calculus_engine — differentiation, policies composed ===\n")
    for label, expr in cases:
        r = ce.diff(expr)
        print(f"  d/dx [ {label} ]  =  {r.text}")
        print(f"      policies: {', '.join(r.rules)}\n")


if __name__ == "__main__":
    _demo()
