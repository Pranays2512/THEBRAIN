#!/usr/bin/env python3
"""
composable_proposer.py — a learned PROPOSER guides compositional code synthesis.

composable_synth brute-enumerates the composition space. That explodes as the DSL
grows — exactly the warning from cross-domain policies. The fix is the same as
program_synth_tree: a learned proposer that, from FEATURES of the target's I/O,
predicts which primitive belongs in each slot, so the search evaluates the likely
programs first and finds the answer in far fewer tries.

  train: random programs -> (I/O features, slot choices)  -> a decision tree / slot
  solve: features(target) -> per-slot choice distributions -> order compositions by
         likelihood -> first exact fit. Count programs evaluated, blind vs guided.

The brain proposes WHICH pieces to compose (premise selection); the verifier still
confirms the survivor. This is what makes composable synthesis scale.

    python3 composable_proposer.py
"""

import math
import random

import numpy as np

from composable_synth import (INITS, RANGES, GUARDS, UPDATES, EARLIES, FINALS, run)
from program_synth_tree import DecisionTree

IK, RK = INITS, RANGES
GK, UK, EK, FK = list(GUARDS), list(UPDATES), list(EARLIES), list(FINALS)
SLOTS = [IK, RK, GK, UK, EK, FK]

ALL = [dict(init=i, lo=lo, hi=hi, guard=g, upd=u, early=e, final=f)
       for i in IK for (lo, hi) in RK for g in GK for u in UK for e in EK for f in FK]


def _idx(p):
    return (IK.index(p["init"]), RK.index((p["lo"], p["hi"])), GK.index(p["guard"]),
            UK.index(p["upd"]), EK.index(p["early"]), FK.index(p["final"]))


def io(p, ns=range(0, 13)):
    out = []
    for n in ns:
        try:
            out.append((n, run(p, n)))
        except Exception:
            return None
    return out


def feats(ex):
    ns = np.array([n for n, _ in ex], float)
    ys = np.array([y for _, y in ex], float)
    tri = ns * (ns + 1) / 2

    def corr(a, b):
        return 0.0 if a.std() < 1e-9 or b.std() < 1e-9 else abs(np.corrcoef(a, b)[0, 1])
    mono = float(all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1)))
    return np.array([
        corr(ys, ns), corr(ys, ns ** 2), corr(ys, tri),
        float((ys > ns).mean()), float((ys < ns).mean()), float((ys == ns).mean()),
        float(ys.mean() / (ns.max() + 1)), mono, float(ys.max() / (ys.mean() + 1)),
    ])


def train(n_tasks=4000, seed=1):
    rng = random.Random(seed)
    X, Ys = [], [[] for _ in SLOTS]
    for _ in range(n_tasks):
        p = rng.choice(ALL)
        ex = io(p)
        if ex is None or len({y for _, y in ex}) < 2:        # skip invalid/constant
            continue
        X.append(feats(ex))
        for s, idx in enumerate(_idx(p)):
            Ys[s].append(idx)
    X = np.array(X)
    return [DecisionTree(len(SLOTS[s]), max_depth=8, min_samples=5).fit(X, np.array(Ys[s]))
            for s in range(len(SLOTS))]


def order(trees, target_feats):
    dists = [t.predict_dist(target_feats) for t in trees]

    def score(p):
        return sum(math.log(max(dists[s][i], 1e-3)) for s, i in enumerate(_idx(p)))
    return sorted(ALL, key=score, reverse=True)


def search(examples, candidates):
    """Evaluate candidates in order; return (program, n_evaluated) of first exact fit."""
    cut = max(3, int(len(examples) * 0.6))
    tr, hd = examples[:cut], examples[cut:]
    for k, p in enumerate(candidates, 1):
        try:
            if all(run(p, n) == y for n, y in tr) and all(run(p, n) == y for n, y in hd):
                return p, k
        except Exception:
            pass
    return None, len(candidates)


def _demo():
    trees = train()
    tasks = {
        "fib":   [(0, 0), (1, 1), (2, 1), (3, 2), (7, 13), (10, 55)],
        "ndiv":  [(1, 1), (2, 2), (6, 4), (7, 2), (12, 6)],
        "cumstop": [(1, 1), (3, 2), (6, 3), (7, 4), (10, 4), (15, 5)],
    }
    print("=== composable_proposer — learned guidance over composition search ===\n")
    print(f"  space = {len(ALL)} compositions; proposer trained on random programs.\n")
    print(f"  {'task':9s} {'blind(rand)':>12s} {'guided':>8s}")
    tb = tg = 0
    for fn, ex in tasks.items():
        # FAIR blind: average evals over random orderings (fixed order accidentally
        # front-loads answers and isn't a fair baseline)
        nbs = []
        for s in range(7):
            sh = ALL[:]
            random.Random(s).shuffle(sh)
            nbs.append(search(ex, sh)[1])
        nb = sum(nbs) // len(nbs)
        _, ng = search(ex, order(trees, feats(ex)))      # proposer order
        tb += nb; tg += ng
        print(f"  {fn:9s} {nb:>12} {ng:>8}")
    print(f"\n  totals: blind(rand) {tb}, guided {tg}  ->  {tb / max(tg,1):.1f}x fewer programs")
    print("  evaluated. The proposer reads the target's I/O shape and tries the right")
    print("  compositions first — premise selection makes composable synthesis scale.")


if __name__ == "__main__":
    _demo()
