#!/usr/bin/env python3
"""
math_chat.py — ask the math engines in plain notation.

Wires the exact math parser to the calculus / integral / algebra engines and
routes by intent word, so you can type the request instead of building tuples:

    > differentiate sin(x^2)      d/dx = cos(x^2)*(2*x)   [sin, power, chain]
    > integrate cos(x)            int = sin(x) + C        [checked]
    > solve 2*x + 3 = 7 for x     x = 2                   [(7-3)/2, verified]

Honest scope: math notation only (a formal grammar — parsing is exact here),
intents differentiate / integrate / solve. Not natural-language word problems.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.math.math_parser import parse, ParseError
from core.math.calculus_engine import CalculusEngine, render
from core.math.integral_engine import IntegralEngine
from core.math.algebra_engine import AlgebraEngine, AlgebraError


def _after(text, phrases):
    """Return the substring after the first matching phrase (longest first)."""
    low = text.lower()
    for p in sorted(phrases, key=len, reverse=True):
        k = low.find(p)
        if k != -1:
            return text[k + len(p):].strip()
    return text.strip()


class MathChat:
    def __init__(self):
        self.calc = CalculusEngine()
        self.intg = IntegralEngine()
        self.alg = AlgebraEngine()

    def respond(self, text):
        t = text.strip().rstrip("?.")
        low = t.lower()
        try:
            if "differentiate" in low or "derivative" in low:
                return self._diff(_after(t, ["differentiate", "derivative of", "derivative"]))
            if "integrate" in low or "integral" in low or "antiderivative" in low:
                return self._int(_after(t, ["integrate", "integral of", "integral",
                                            "antiderivative of", "antiderivative"]))
            if "solve" in low or "=" in t:
                expr = _after(t, ["solve for", "solve"])
                for tail in (" for x", " for y", " for "):
                    if expr.lower().endswith(tail.rstrip()):
                        expr = expr[: -len(tail)].strip()
                return self._solve(expr)
        except ParseError as e:
            return f"I couldn't parse that: {e}"
        return ("Ask me to differentiate, integrate, or solve "
                "(e.g. 'differentiate sin(x^2)').")

    def _diff(self, s):
        r = self.calc.diff(parse(s))
        return f"d/dx [ {s} ] = {r.text}    (rules: {', '.join(r.rules)})"

    def _int(self, s):
        F = self.intg.integrate(parse(s))
        if F is None:
            return f"∫ {s} dx — no rule in my set applies (likely no elementary form)."
        ok = self.intg.verify(parse(s), F)
        return f"∫ {s} dx = {render(F)} + C    [{'checked' if ok else 'unverified'}]"

    def _solve(self, s):
        node = parse(s)
        if not (isinstance(node, tuple) and node[0] == "="):
            return f"'{s}' is not an equation (needs '=')."
        try:
            value, steps = self.alg.solve(node)
        except AlgebraError as e:
            return f"Can't solve that: {e}"
        return f"{steps[1]}  ->  x = {value}   (from {steps[0]}, verified)"


def _demo():
    mc = MathChat()
    print("=== math_chat — ask in notation ===\n")
    for q in [
        "differentiate sin(x^2)",
        "what is the derivative of x^3?",
        "integrate cos(x)",
        "integrate sin(x^2)",
        "solve 2*x + 3 = 7 for x",
        "solve x^2 = 49 for x",
        "differentiate exp(x) * ln(x)",
    ]:
        print(f"  > {q}")
        print(f"    {mc.respond(q)}\n")


if __name__ == "__main__":
    _demo()
