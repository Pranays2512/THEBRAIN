#!/usr/bin/env python3
"""
physics_engine.py — apply taught physics laws, solving for ANY variable.

A law is one equation (lhs_symbol = rhs_expression). Given values for all but one
variable, the engine isolates the unknown symbolically — inverting operations
outward (x -> /, + -> -, ^n -> ^(1/n)) — then evaluates, showing the rearranged
formula and the numbers. It applies laws; it does not invent physics.

    pe = PhysicsEngine()
    pe.add_law("newton2", "F", ("*", "m", "a"))      # F = m*a
    pe.solve("newton2", "a", F=12, m=3)              # a = F/m = 12/3 = 4

Reuses the expression-tuple representation (and renderer) of calculus_engine. The
isolation routine is the kernel of the algebra solver too — generalized there.
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engines.math.calculus_engine import render


class PhysicsError(ValueError):
    pass


def contains(expr, target):
    if expr == target:
        return True
    if isinstance(expr, tuple):
        return any(contains(c, target) for c in expr[1:])
    return False


def ev(expr, env):
    """Evaluate an expression tuple with a {symbol: value} environment."""
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, str):
        if expr not in env:
            raise PhysicsError(f"unknown value for '{expr}'")
        return env[expr]
    op = expr[0]
    if op == "neg":
        return -ev(expr[1], env)
    if op in ("sin", "cos", "exp", "ln"):
        f = {"sin": math.sin, "cos": math.cos, "exp": math.exp, "ln": math.log}[op]
        return f(ev(expr[1], env))
    a, b = ev(expr[1], env), ev(expr[2], env)
    # Evaluate lazily — a dict literal would compute a**b even for a '+'/'*' op,
    # overflowing on large values.
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a / b
    if op == "^": return a ** b
    raise PhysicsError(f"unknown operator {op!r}")


def isolate(expr, target, other):
    """Solve  expr = other  for `target`, returning the expression tree that
    `target` equals. Inverts the operation around `target` step by step."""
    if expr == target:
        return other
    if not isinstance(expr, tuple):
        raise PhysicsError(f"cannot isolate {target} in {expr}")
    op, a, b = expr[0], expr[1], (expr[2] if len(expr) > 2 else None)

    if op == "*":
        return isolate(a, target, ("/", other, b)) if contains(a, target) \
            else isolate(b, target, ("/", other, a))
    if op == "/":                      # a/b = other
        return isolate(a, target, ("*", other, b)) if contains(a, target) \
            else isolate(b, target, ("/", a, other))   # b = a/other
    if op == "+":
        return isolate(a, target, ("-", other, b)) if contains(a, target) \
            else isolate(b, target, ("-", other, a))
    if op == "-":                      # a - b = other
        return isolate(a, target, ("+", other, b)) if contains(a, target) \
            else isolate(b, target, ("-", a, other))   # b = a - other
    if op == "^":                      # a^n = other -> a = other^(1/n)
        return isolate(a, target, ("^", other, ("/", 1, b)))
    raise PhysicsError(f"cannot invert operator '{op}'")


class PhysicsEngine:
    def __init__(self):
        self.laws = {}                 # name -> (lhs_symbol, rhs_expr)

    def add_law(self, name, lhs_symbol, rhs_expr):
        if not isinstance(lhs_symbol, str):
            raise PhysicsError("lhs must be a single variable symbol")
        self.laws[name] = (lhs_symbol, rhs_expr)

    def load_from_knowledge(self, facts):
        """Parse equation-like facts from the Knowledge Engine into executable laws.
        
        Scans all (subject, relation, object) triples for patterns like:
            second_law | can | written_as_f=ma  ->  F = m*a
            kinetic_energy | equals | 0.5*m*v^2 ->  KE = 0.5*m*v^2
        
        The parser converts simple infix equation strings into the tuple format
        the engine already understands. Returns the count of laws loaded.
        """
        loaded = 0
        for s, r, o in facts:
            # Pattern 1: "written_as_X=Y" or "equals_X=Y"
            eq_str = None
            if "written_as_" in o:
                eq_str = o.split("written_as_", 1)[1]
            elif "equals_" in o:
                eq_str = o.split("equals_", 1)[1]
            elif r in ("equals", "defined_as", "formula") and "=" in o:
                eq_str = o
            
            if eq_str and "=" in eq_str:
                try:
                    name = s.replace(" ", "_")
                    lhs_str, rhs_str = eq_str.split("=", 1)
                    lhs_str, rhs_str = lhs_str.strip(), rhs_str.strip()
                    if lhs_str and rhs_str:
                        rhs_tree = _parse_infix(rhs_str)
                        if rhs_tree is not None:
                            self.add_law(name, lhs_str.upper(), rhs_tree)
                            loaded += 1
                except Exception:
                    pass   # skip unparseable equations — honest fail
        return loaded

    def variables(self, name):
        lhs, rhs = self.laws[name]
        out = {lhs}

        def walk(e):
            if isinstance(e, str):
                out.add(e)
            elif isinstance(e, tuple):
                for c in e[1:]:
                    walk(c)
        walk(rhs)
        return out

    def solve(self, name, target, **knowns):
        """Solve law `name` for `target` given the other values. Returns
        (value, steps) where steps shows the rearranged formula and the numbers."""
        if name not in self.laws:
            raise PhysicsError(f"no law named '{name}'")
        lhs, rhs = self.laws[name]
        if target == lhs:
            tree = rhs                                   # already isolated
        else:
            if not contains(rhs, target):
                raise PhysicsError(f"'{target}' is not in law '{name}'")
            tree = isolate(rhs, target, lhs)             # rearrange for target
        value = ev(tree, knowns)
        formula = f"{target} = {render(tree)}"
        # substitute the known numbers into the rearranged formula for the trace
        subst = render(_subst(tree, knowns))
        return round(value, 6), [formula, f"{target} = {subst} = {round(value, 6)}"]


# ── infix equation parser: "m*a" -> ("*", "m", "a") ─────────────────────────
def _parse_infix(s):
    """Parse a simple infix equation string into the tuple format.
    Handles: *, /, +, -, ^, and 0.5 as a float literal.
    Operator precedence: ^ > * / > + -
    Returns None if unparseable."""
    s = s.strip()
    if not s:
        return None
    try:
        return _parse_add(s)
    except Exception:
        return None


def _parse_add(s):
    """Handle + and - (lowest precedence)."""
    depth = 0
    # scan right-to-left for + or - not inside parens
    for i in range(len(s) - 1, 0, -1):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        elif depth == 0 and s[i] in ('+', '-'):
            left = _parse_add(s[:i].strip())
            right = _parse_mul(s[i + 1:].strip())
            return (s[i], left, right)
    return _parse_mul(s)


def _parse_mul(s):
    """Handle * and / (medium precedence)."""
    depth = 0
    for i in range(len(s) - 1, 0, -1):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        elif depth == 0 and s[i] in ('*', '/'):
            left = _parse_mul(s[:i].strip())
            right = _parse_pow(s[i + 1:].strip())
            return (s[i], left, right)
    return _parse_pow(s)


def _parse_pow(s):
    """Handle ^ (highest precedence, right-associative)."""
    depth = 0
    for i in range(len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        elif depth == 0 and s[i] == '^':
            left = _parse_atom(s[:i].strip())
            right = _parse_pow(s[i + 1:].strip())
            return ("^", left, right)
    return _parse_atom(s)


def _parse_atom(s):
    """Parse a single token: number, variable, or parenthesized expression."""
    s = s.strip()
    if s.startswith('(') and s.endswith(')'):
        return _parse_add(s[1:-1])
    try:
        v = float(s)
        return int(v) if v == int(v) else v
    except ValueError:
        return s   # variable name


def _subst(expr, env):
    if isinstance(expr, str) and expr in env:
        return env[expr]
    if isinstance(expr, tuple):
        return (expr[0], *[_subst(c, env) for c in expr[1:]])
    return expr


def _demo():
    pe = PhysicsEngine()
    pe.add_law("newton2", "F", ("*", "m", "a"))                 # F = m*a
    pe.add_law("speed", "v", ("/", "d", "t"))                   # v = d/t
    pe.add_law("kinetic", "KE", ("*", 0.5, ("*", "m", ("^", "v", 2))))  # KE = ½mv²

    print("=== physics_engine — apply a law, solve for any variable ===\n")
    runs = [
        ("newton2", "F", dict(m=3, a=4)),
        ("newton2", "a", dict(F=12, m=3)),
        ("speed", "t", dict(d=100, v=20)),
        ("kinetic", "v", dict(KE=100, m=2)),
    ]
    for name, target, knowns in runs:
        val, steps = pe.solve(name, target, **knowns)
        kn = ", ".join(f"{k}={v}" for k, v in knowns.items())
        print(f"  [{name}] given {kn}, solve {target}:")
        for s in steps:
            print(f"      {s}")
        print()

    # Demo: loading a law from Knowledge Engine facts
    print("=== loading laws from Knowledge Engine facts ===\n")
    fake_facts = [
        ("second_law", "can", "written_as_F=m*a"),
        ("speed_law", "equals", "v=d/t"),
        ("kinetic_energy", "defined_as", "KE=0.5*m*v^2"),
    ]
    pe2 = PhysicsEngine()
    n = pe2.load_from_knowledge(fake_facts)
    print(f"  Loaded {n} laws from KB facts.")
    for name in pe2.laws:
        lhs, rhs = pe2.laws[name]
        print(f"    {name}: {lhs} = {render(rhs)}")
    if "second_law" in pe2.laws:
        val, steps = pe2.solve("second_law", "F", m=5, a=10)
        print(f"\n  Solve F from KB-learned second_law: F = {val}")
        for s in steps:
            print(f"      {s}")


if __name__ == "__main__":
    _demo()

