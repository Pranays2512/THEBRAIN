#!/usr/bin/env python3
"""
sympy_usub_proposer.py — the integration case where the proposer DOES win.

By-parts (sympy_integration_proposer.py) was non-discriminating: branch factor 2,
self-correcting, so blind == tree. U-SUBSTITUTION is the opposite: at each
integrand EVERY subexpression is a candidate u, most lead nowhere, and the wrong
choice recurses into the same shape (no self-correction). That high, dead-heavy
branching is exactly what a learned proposer prunes.

For ∫ (d/dx g)·h(g) dx the right u is g (the inner function). Candidates like u=x
produce a same-shaped subgoal that loops until the node cap. blind tries them in
SymPy's canonical order; the tree proposer learns "u = the argument of the outer
function" and goes straight there.

SymPy is the algebra substrate (diff, cancel, subs, elementary base integrals);
the search + proposer are ours. Every answer verified: diff(F) - integrand == 0.
"""

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from program_synth_tree import DecisionTree

x = sp.symbols("x")
U = sp.symbols("u_sub")
NODE_CAP = 4000
FUNCS = (sp.sin, sp.cos, sp.exp, sp.log)


# ── base oracle: elementary only (no function of a NONLINEAR argument) ────────
def has_nonlinear_arg(e):
    for f in e.atoms(sp.Function):
        a = f.args[0]
        if not (a.is_polynomial(x) and sp.degree(a, x) <= 1):
            return True
    return False


def base_integrate(e):
    if has_nonlinear_arg(e):
        return None
    try:
        F = sp.integrate(e, x)
    except Exception:
        return None
    return None if F.has(sp.Integral) else F


# ── u-substitution moves ─────────────────────────────────────────────────────
def subexprs(e):
    out = set()
    def rec(n):
        if n.has(x) and not n.is_Number:
            out.add(n)
        for a in n.args:
            rec(a)
    rec(e)
    out.discard(e)
    return sorted(out, key=sp.default_sort_key)   # deterministic move order


class Move:
    def __init__(self, g, subgoal):
        self.g, self.subgoal = g, subgoal


def usub_moves(e):
    moves = []
    for g in subexprs(e):
        gp = sp.diff(g, x)
        if gp == 0:
            continue
        q = sp.cancel(e / gp)
        qs = q.subs(g, U)
        if qs.has(x):
            continue                      # not a pure function of u -> invalid
        moves.append(Move(g, qs.subs(U, x)))   # rename u->x to recurse in one var
    return moves


def features(g, e):
    is_arg = any(f.args and f.args[0] == g for f in e.atoms(sp.Function))
    poly = g.is_polynomial(x)
    return np.array([
        1.0,
        1.0 if g.is_Pow else 0.0,
        float(sp.degree(g, x)) if poly and g.has(x) else 0.0,
        float(sp.count_ops(g)),
        1.0 if is_arg else 0.0,           # g is the arg of the outer function (the win)
        1.0 if g == x else 0.0,           # bare x (usually the trap)
    ], dtype=float)


# ── search ───────────────────────────────────────────────────────────────────
class Counter:
    def __init__(self): self.n = 0


def integrate_search(e, tree=None, counter=None, depth=14):
    if counter is None:
        counter = Counter()
    counter.n += 1
    if depth <= 0 or counter.n > NODE_CAP:
        return None
    F = base_integrate(e)
    if F is not None:
        return F
    moves = usub_moves(e)
    if tree is not None:
        moves.sort(key=lambda m: -float(tree.predict_dist(features(m.g, e))[1]))
    for m in moves:
        sub = integrate_search(m.subgoal, tree, counter, depth - 1)
        if sub is not None:
            return sub.subs(x, m.g)
    return None


def verify(e, F):
    return F is not None and sp.simplify(sp.diff(F, x) - e) == 0


# ── train the proposer ───────────────────────────────────────────────────────
def harvest(e, X, Y, depth=14):
    """Label EACH candidate move at this node by whether it actually leads to a
    solution (blind solve of its subgoal) — gives both classes, independent of
    try-order. Recurse into the winning move to harvest deeper nodes too."""
    F = base_integrate(e)
    if F is not None or depth <= 0:
        return F
    winner = None
    for m in usub_moves(e):
        sub = integrate_search(m.subgoal, None, Counter(), depth - 1)
        ok = sub is not None
        X.append(features(m.g, e)); Y.append(1 if ok else 0)
        if ok and winner is None:
            winner = m
    if winner is not None:
        harvest(winner.subgoal, X, Y, depth - 1)     # deeper nodes
        return integrate_search(winner.subgoal, None, Counter(), depth - 1).subs(x, winner.g)
    return None


def train_tree(corpus):
    X, Y = [], []
    for e in corpus:
        harvest(e, X, Y)
    print(f"  (training: {len(Y)} move-labels, {sum(Y)} positive)")
    if not X or len(set(Y)) < 2:
        return None
    return DecisionTree(n_ops=2, max_depth=6, min_samples=3).fit(np.array(X), np.array(Y))


# ── benchmark ────────────────────────────────────────────────────────────────
def make(n, f):
    inner = x**n
    return sp.diff(inner, x) * f(inner)          # (d/dx g)*h(g) -> u=g solves


def main():
    train = [make(n, f) for n in (2, 3) for f in (sp.sin, sp.cos, sp.exp)]
    test = [make(n, f) for n in (4, 5, 6) for f in (sp.sin, sp.cos, sp.exp)]
    tree = train_tree(train)

    print("=== sympy_usub_proposer — u-substitution, blind vs tree ===\n")
    print(f"  train {len(train)} (n=2,3), test {len(test)} (n=4,5,6)\n")
    print(f"  {'method':8s} {'avg nodes':>10s} {'solved':>9s}")
    rows = {}
    for name, t in (("blind", None), ("tree", tree)):
        nodes = solved = 0
        for e in test:
            c = Counter()
            F = integrate_search(e, t, c)
            nodes += c.n
            solved += 1 if verify(e, F) else 0
        rows[name] = (nodes / len(test), solved)
        print(f"  {name:8s} {rows[name][0]:10.1f} {solved:>6}/{len(test)}")
    bn, _ = rows["blind"]; tn, ts = rows["tree"]
    print(f"\n  tree explores ~{bn / max(tn,1e-9):.1f}x fewer nodes "
          f"(solve {ts}/{len(test)})")


if __name__ == "__main__":
    main()
