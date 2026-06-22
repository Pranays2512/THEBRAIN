#!/usr/bin/env python3
"""
sympy_integration_proposer.py — the integration proposer, done right.

proposer_experiment.py parked because hand-rolled symbolic integration blew up in
EXPRESSION SIZE (the wrong by-parts branch nested forever, each node slow). SymPy
fixes exactly that: it keeps expressions compact (simplify/diff/expand), so the
search terminates and NODE COUNT becomes the clean metric again.

SymPy is only the ALGEBRA substrate, not the solver — we do NOT call sp.integrate
on the whole problem (that would skip the search). The base oracle integrates only
ELEMENTARY terms (monomials, sin/cos/exp, 1/x, sums, constant multiples); a
product of two x-dependent factors must be cracked by a BY-PARTS SEARCH, and that
search is where the proposer earns its keep: pick u = the polynomial (degree
falls, terminates) vs u = the transcendental (degree climbs, wastes nodes).

blind = try by-parts splits in fixed order; tree = a decision-tree proposer orders
them by features. Every answer verified exactly: sp.diff(F) - integrand == 0.
"""

import os
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from program_synth_tree import DecisionTree

x = sp.symbols("x")
NODE_CAP = 3000


# ── base oracle: elementary antiderivatives only ─────────────────────────────
def needs_parts(e):
    """True if e contains a product of two x-dependent factors (needs by-parts)."""
    if isinstance(e, sp.Add):
        return any(needs_parts(t) for t in e.args)
    if isinstance(e, sp.Mul):
        xfac = [a for a in e.args if a.has(x)]
        return len(xfac) >= 2 or any(needs_parts(a) for a in xfac)
    return False


def base_integrate(e):
    """Antiderivative of an ELEMENTARY integrand, else None (forces search)."""
    if needs_parts(e):
        return None
    try:
        F = sp.integrate(e, x)
    except Exception:
        return None
    return None if F.has(sp.Integral) else F


# ── by-parts moves ───────────────────────────────────────────────────────────
class Move:
    def __init__(self, u, dv, V, subgoal):
        self.u, self.dv, self.V, self.subgoal = u, dv, V, subgoal


def candidate_moves(e):
    """By-parts splits of a product. Each picks one x-factor as u, the rest as dv
    (constants fold into dv); valid only if V=∫dv is elementary."""
    if not isinstance(e, sp.Mul):
        return []
    consts = [a for a in e.args if not a.has(x)]
    xfacs = [a for a in e.args if a.has(x)]
    c = sp.Mul(*consts) if consts else sp.Integer(1)
    moves = []
    for i in range(len(xfacs)):
        u = xfacs[i]
        dv = c * sp.Mul(*(xfacs[:i] + xfacs[i + 1:]))
        V = base_integrate(dv)
        if V is None:
            continue
        subgoal = sp.expand(V * sp.diff(u, x))
        moves.append(Move(u, dv, V, subgoal))
    return moves


def features(move):
    u, dv = move.u, move.dv
    poly = u.is_polynomial(x)
    deg = int(sp.degree(u, x)) if poly and u.has(x) else 0
    trans = lambda z: z.has(sp.sin) or z.has(sp.cos) or z.has(sp.exp) or z.has(sp.log)
    return np.array([1.0, float(deg), 1.0 if poly else 0.0,
                     1.0 if trans(dv) else 0.0, 1.0 if trans(u) else 0.0,
                     float(sp.count_ops(u))], dtype=float)


# ── the search (blind or tree-guided) ────────────────────────────────────────
class Counter:
    def __init__(self): self.n = 0


def integrate_search(e, tree=None, counter=None, depth=20):
    if counter is None:
        counter = Counter()
    counter.n += 1
    if depth <= 0 or counter.n > NODE_CAP:
        return None
    F = base_integrate(e)
    if F is not None:
        return F
    moves = candidate_moves(e)
    if tree is not None:                          # proposer: best split first
        moves.sort(key=lambda m: -float(tree.predict_dist(features(m))[1]))
    for m in moves:
        sub = integrate_search(m.subgoal, tree, counter, depth - 1)
        if sub is not None:
            return sp.expand(m.u * m.V - sub)
    return None


def verify(e, F):
    return F is not None and sp.simplify(sp.diff(F, x) - e) == 0


# ── train the tree proposer on solved-path moves ─────────────────────────────
def harvest(e, X, Y, counter=None, depth=20):
    if counter is None:
        counter = Counter()
    counter.n += 1
    if depth <= 0 or counter.n > NODE_CAP or base_integrate(e) is not None:
        return base_integrate(e)
    for m in candidate_moves(e):
        sub = harvest(m.subgoal, X, Y, counter, depth - 1)
        if sub is not None:
            X.append(features(m)); Y.append(1)        # on a winning path
            return sp.expand(m.u * m.V - sub)
        X.append(features(m)); Y.append(0)            # dead branch
    return None


def train_tree(corpus):
    X, Y = [], []
    for e in corpus:
        harvest(e, X, Y)
    if not X:
        return None
    return DecisionTree(n_ops=2, max_depth=6, min_samples=4).fit(np.array(X), np.array(Y))


# ── benchmark ────────────────────────────────────────────────────────────────
def make_corpus():
    polys = [x, x**2, x**3, x**4, x**5, x**6]
    trans = [sp.exp(x), sp.sin(x), sp.cos(x)]
    return [p * t for p in polys for t in trans] + [t * p for p in polys for t in trans]


def main():
    corpus = make_corpus()
    train = [e for e in corpus if any((e/g).is_polynomial(x) for g in
             (sp.exp(x), sp.sin(x), sp.cos(x)) if (e/g).is_polynomial(x))]
    # split by polynomial degree: train on low, test on high (generalization)
    def deg(e):
        p = [a for a in e.args if a.has(x) and a.is_polynomial(x)]
        return int(sp.degree(sp.Mul(*p), x)) if p else 0
    train = [e for e in corpus if deg(e) <= 2]
    test = [e for e in corpus if deg(e) >= 3]
    tree = train_tree(train)

    print("=== sympy_integration_proposer — blind vs tree (real search) ===\n")
    print(f"  train {len(train)} (deg<=2), test {len(test)} (deg>=3)\n")
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
    bn, bs = rows["blind"]; tn, ts = rows["tree"]
    print(f"\n  tree explores ~{bn / max(tn,1e-9):.1f}x fewer nodes "
          f"(solve {ts} vs {bs} of {len(test)})")


if __name__ == "__main__":
    main()
