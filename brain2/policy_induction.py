#!/usr/bin/env python3
"""
policy_induction.py — the brain DISCOVERS a policy from examples (induction).

Until now the executive only COMPOSED hand-given policies (conjecture->verify).
This is the next rung: given examples (input columns + a target column), INDUCE the
formula by bounded symbolic regression over {+,-,*,/}, then VERIFY it on held-out
examples before admitting it as a Policy. Induction is unsound (a fit on N rows is
a guess), so the held-out check is the gate — induction proposes, verification
disposes, exactly the discipline everywhere else.

    induce(rows, ["mass","accel"], "force")  ->  ("*","mass","accel")  (verified)

Honest scope: formulas up to a small tree depth over the given inputs (+ constants
0.5, 2). Deeper formulas (e.g. 1/2 m v^2) blow up blind enumeration — that's where
a proposer-GUIDED induction goes next (the same premise-selection idea), not built
here.
"""

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from means_ends import ev          # local lazy tuple-formula evaluator

OPS = ("+", "-", "*", "/")
CONSTS = (0.5, 2.0)


def _terminals(inputs):
    return list(inputs) + list(CONSTS)


def enumerate_exprs(inputs, max_depth):
    """All expression trees up to max_depth over inputs+constants (deduped by repr)."""
    layer = _terminals(inputs)                    # depth 0
    seen = {repr(e) for e in layer}
    all_exprs = list(layer)
    frontier = list(layer)
    for _ in range(max_depth):
        nxt = []
        for op in OPS:
            for a in all_exprs:
                for b in frontier:                # at least one new (depth-growing) child
                    e = (op, a, b)
                    if repr(e) not in seen:
                        seen.add(repr(e)); nxt.append(e)
                    e2 = (op, b, a)
                    if repr(e2) not in seen:
                        seen.add(repr(e2)); nxt.append(e2)
        all_exprs += nxt
        frontier = nxt
    return all_exprs


def _fits(expr, rows, target, tol):
    for r in rows:
        try:
            if abs(ev(expr, r) - r[target]) > tol:
                return False
        except (ZeroDivisionError, KeyError):
            return False
    return True


def induce(rows, inputs, target, tol=1e-6, max_depth=2):
    """Discover a formula target = f(inputs) that fits the TRAIN rows and is then
    VERIFIED on held-out rows. Returns the expr (tuple) or None. Shortest first."""
    random.Random(0).shuffle(rows)
    cut = max(2, int(len(rows) * 0.6))
    train, holdout = rows[:cut], rows[cut:]
    for expr in enumerate_exprs(inputs, max_depth):   # shortest-first (by layer)
        if _fits(expr, train, target, tol) and _fits(expr, holdout, target, tol):
            return expr                               # induced AND verified
    return None


def _render(e):
    if isinstance(e, tuple):
        return f"({_render(e[1])} {e[0]} {_render(e[2])})"
    return str(e)


# ── proposer-guided induction: best-first by fit, not blind enumeration ───────
import heapq

import numpy as np


def _outputs(expr, rows):
    try:
        return [ev(expr, r) for r in rows]
    except (ZeroDivisionError, KeyError, OverflowError):
        return None


def _score(expr, rows, target, tol):
    """The PROPOSER signal: how promising is this expression as a building block?
    2.0 if it already fits exactly; else |correlation| with the target (a piece
    that tracks the target is worth expanding). This is what guides the search."""
    outs = _outputs(expr, rows)
    if outs is None:
        return -1.0
    tgt = [r[target] for r in rows]
    if all(abs(o - t) <= tol for o, t in zip(outs, tgt)):
        return 2.0
    a, b = np.asarray(outs, float), np.asarray(tgt, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


MAX_SIZE = 9


def _search(rows, inputs, target, guided, budget=1500, tol=1e-6):
    """Best-first search. guided=True orders by fit (the proposer); guided=False
    orders by size (blind baseline). Expands by combining with TERMINALS only —
    bounded branching, builds depth via chains. Returns (expr, nodes)."""
    random.Random(0).shuffle(rows)
    cut = max(2, int(len(rows) * 0.6))
    train, holdout = rows[:cut], rows[cut:]
    terms = _terminals(inputs)
    seen = {repr(e) for e in terms}
    tie = 0

    def prio(e):
        return -_score(e, train, target, tol) if guided else _size(e)

    heap = []
    for e in terms:
        heapq.heappush(heap, (prio(e), tie, e)); tie += 1
    nodes = 0
    while heap and nodes < budget:
        _, _, e = heapq.heappop(heap)
        nodes += 1
        if _fits(e, train, target, tol) and _fits(e, holdout, target, tol):
            return e, nodes
        if _size(e) >= MAX_SIZE:
            continue
        for op in OPS:
            for t in terms:                       # combine with a terminal (bounded)
                for ne in ((op, e, t), (op, t, e)):
                    if repr(ne) not in seen:
                        seen.add(repr(ne))
                        heapq.heappush(heap, (prio(ne), tie, ne)); tie += 1
    return None, nodes


def guided_induce(rows, inputs, target, budget=1500, tol=1e-6):
    return _search(rows, inputs, target, True, budget, tol)


def blind_induce(rows, inputs, target, budget=1500, tol=1e-6):
    return _search(rows, inputs, target, False, budget, tol)


def _size(e):
    return 1 if not isinstance(e, tuple) else 1 + _size(e[1]) + _size(e[2])


# ── demo ──────────────────────────────────────────────────────────────────────
def _make(formula, inputs, n=12, seed=1):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r = {i: round(rng.uniform(1, 10), 2) for i in inputs}
        r["__t__"] = formula(r)
        rows.append(r)
    return rows


def _demo():
    cases = [
        ("force", ["mass", "accel"], lambda r: r["mass"] * r["accel"]),
        ("momentum", ["mass", "speed"], lambda r: r["mass"] * r["speed"]),
        ("density", ["mass", "volume"], lambda r: r["mass"] / r["volume"]),
        ("pe", ["mass", "gravity", "height"], lambda r: r["mass"] * r["gravity"] * r["height"]),
        ("ke", ["mass", "speed"], lambda r: 0.5 * r["mass"] * r["speed"] ** 2),  # 0.5*m*v*v, found at depth 2
    ]
    print("=== policy_induction — discover a formula from examples, verified ===\n")
    for name, inputs, f in cases:
        rows = _make(f, inputs)
        for r in rows:
            r[name] = r.pop("__t__")
        expr = induce(rows, inputs, name)
        if expr is None:
            print(f"  {name:9s} <- NOT FOUND at depth 2 "
                  f"(needs guided/deeper induction)")
        else:
            print(f"  {name:9s} = {_render(expr)}   (induced + held-out verified)")


def _demo_guided():
    # a deep formula: (mass*accel*height)/area  — depth 3, 4 inputs
    inputs = ["mass", "accel", "height", "area"]
    f = lambda r: r["mass"] * r["accel"] * r["height"] / r["area"]
    rows = _make(f, inputs, n=16)
    for r in rows:
        r["stress"] = r.pop("__t__")
    print("\n=== proposer-guided vs blind induction (deep formula, budget 1500) ===\n")
    print(f"  target: stress = (mass*accel*height)/area  (depth 3, 4 inputs)\n")
    be, bn = blind_induce([dict(r) for r in rows], inputs, "stress")
    ge, gn = guided_induce([dict(r) for r in rows], inputs, "stress")
    print(f"  blind   : {'FOUND ' + _render(be) if be else 'not found'}  "
          f"(nodes {bn})")
    print(f"  guided  : {'FOUND ' + _render(ge) if ge else 'not found'}  "
          f"(nodes {gn})")
    if ge and gn:
        print(f"\n  guided reaches the depth-3 formula the fit signal points at; "
              f"blind explores by size.")


if __name__ == "__main__":
    _demo()
    _demo_guided()
