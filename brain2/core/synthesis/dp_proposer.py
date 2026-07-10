#!/usr/bin/env python3
"""
dp_proposer.py — learned proposer over the DP recurrence space.

dp_greedy_synth searched a tiny DP space (Kadane only). This expands it to a real
recurrence DSL (init x cur-update x best-update ~ 56 recurrences spanning max-subarray,
min-subarray, max-element, sum) and puts a learned PROPOSER on top: from features of
the task's I/O it predicts the recurrence pieces, ordering the search so the right
recurrence is found in far fewer tries. Gate = held-out examples (plus stress-vs-brute
where an oracle exists, e.g. Kadane).

Measured vs a FAIR random-order baseline (fixed enumeration order rigs it).

    python3 dp_proposer.py
"""

import math
import random

import numpy as np

from core.synthesis.program_synth_tree import DecisionTree

INITS = {"first": None, "zero": (0, 0)}      # 'first' uses arr[0] for both
CUR = {
    "max(x, cur + x)": lambda c, x: max(x, c + x), "min(x, cur + x)": lambda c, x: min(x, c + x),
    "cur + x": lambda c, x: c + x, "max(cur, x)": lambda c, x: max(c, x),
    "min(cur, x)": lambda c, x: min(c, x), "x": lambda c, x: x, "cur * x": lambda c, x: c * x,
}
BEST = {"max(best, cur)": lambda b, c: max(b, c), "min(best, cur)": lambda b, c: min(b, c),
        "best + cur": lambda b, c: b + c, "cur": lambda b, c: c}
IK, CK, BK = list(INITS), list(CUR), list(BEST)
ALL = [(i, c, b) for i in IK for c in CK for b in BK]


def run_dp(ik, ck, bk, arr):
    if not arr:
        return 0
    if ik == "first":
        cur = best = arr[0]
        seq = arr[1:]
    else:
        cur = best = 0
        seq = arr
    cf, bf = CUR[ck], BEST[bk]
    for x in seq:
        cur = cf(cur, x)
        best = bf(best, cur)
    return best


# ── reference algorithms (for generating tasks + Kadane stress oracle) ───────
def kadane(a): return max(sum(a[i:j + 1]) for i in range(len(a)) for j in range(i, len(a)))
def min_sub(a): return min(sum(a[i:j + 1]) for i in range(len(a)) for j in range(i, len(a)))
TASKS = {"kadane": kadane, "min_subarray": min_sub, "max_element": max, "total_sum": sum}


def feats(ex):
    sm = np.array([sum(a) for a, _ in ex], float)
    mx = np.array([max(a) for a, _ in ex], float)
    mn = np.array([min(a) for a, _ in ex], float)
    y = np.array([v for _, v in ex], float)

    def corr(u, v):
        return 0.0 if u.std() < 1e-9 or v.std() < 1e-9 else abs(np.corrcoef(u, v)[0, 1])
    return np.array([
        corr(y, sm), corr(y, mx), corr(y, mn),
        float((y == mx).mean()), float((y == sm).mean()), float((y == mn).mean()),
        float((y < 0).mean()), float((y >= mx).mean()), float((y <= mn).mean()),
    ])


def _rand_arr(rng):
    return [rng.randint(-9, 9) for _ in range(rng.randint(1, 7))]


def train(n=5000, seed=1):
    rng = random.Random(seed)
    X, Y = [], [[], [], []]
    for _ in range(n):
        rec = rng.choice(ALL)
        arrs = [_rand_arr(rng) for _ in range(6)]
        try:
            ex = [(a, run_dp(*rec, a)) for a in arrs]
        except Exception:
            continue
        if len({v for _, v in ex}) < 2:
            continue
        X.append(feats(ex))
        for s, k in enumerate(rec):
            Y[s].append([IK, CK, BK][s].index(k))
    X = np.array(X)
    return [DecisionTree(len([IK, CK, BK][s]), max_depth=8, min_samples=5).fit(X, np.array(Y[s]))
            for s in range(3)]


def order(trees, tf):
    d = [t.predict_dist(tf) for t in trees]
    def sc(rec):
        return sum(math.log(max(d[s][[IK, CK, BK][s].index(k)], 1e-3))
                   for s, k in enumerate(rec))
    return sorted(ALL, key=sc, reverse=True)


def search(ex, cands):
    cut = max(3, int(len(ex) * 0.6))
    tr, hd = ex[:cut], ex[cut:]
    for k, rec in enumerate(cands, 1):
        try:
            if all(run_dp(*rec, a) == y for a, y in tr) and \
               all(run_dp(*rec, a) == y for a, y in hd):
                return rec, k
        except Exception:
            pass
    return None, len(cands)


def _demo():
    trees = train()
    rng = random.Random(7)
    print("=== dp_proposer — learned guidance over the DP recurrence space ===\n")
    print(f"  space = {len(ALL)} recurrences.\n")
    print(f"  {'task':13s} {'blind(rand)':>12s} {'guided':>8s}  recurrence")
    tb = tg = 0
    for name, fn in TASKS.items():
        ex = [(a, fn(a)) for a in (_rand_arr(rng) for _ in range(8))]
        nbs = []
        for s in range(7):
            sh = ALL[:]; random.Random(s).shuffle(sh)
            nbs.append(search(ex, sh)[1])
        nb = sum(nbs) // len(nbs)
        rec, ng = search(ex, order(trees, feats(ex)))
        tb += nb; tg += ng
        r = f"cur={rec[1]}, best={rec[2]}" if rec else "—"
        print(f"  {name:13s} {nb:>12} {ng:>8}  {r}")
    print(f"\n  totals: blind(rand) {tb}, guided {tg}  ->  {tb / max(tg,1):.1f}x fewer")
    # Kadane: confirm the found recurrence vs brute oracle (the strong gate)
    ex = [(a, kadane(a)) for a in (_rand_arr(rng) for _ in range(8))]
    rec, _ = search(ex, order(trees, feats(ex)))
    stress = all(run_dp(*rec, a) == kadane(a) for a in (_rand_arr(rng) for _ in range(400)))
    print(f"  Kadane recurrence stress-vs-brute on 400 random arrays: "
          f"{'VERIFIED ✓' if stress else 'FAILED ✗'}")


if __name__ == "__main__":
    _demo()
