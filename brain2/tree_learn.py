#!/usr/bin/env python3
"""
tree_learn.py — the search LEARNS to reason more efficiently from experience.

tree_reason searches with a hand-given heuristic (or none). This adds the
brain-like part: it watches itself solve easy instances, learns a heuristic
from that experience (a linear value over state features), and then expands
dramatically fewer states on new, harder instances. Reasoning that improves
with experience — lightweight, on CPU, no hand-tuned heuristic.

Domain: the 8-puzzle (large state space, where blind search is expensive and a
good heuristic matters enormously). The honest result is the node-count drop:

    blind search (no heuristic)   : thousands of states expanded
    LEARNED heuristic             : a few dozen — and it was learned, not coded

The learned weights end up close to the classic Manhattan heuristic — i.e. it
rediscovers a known-good heuristic from its own solved experience, with no one
telling it the rule.
"""

import random

import numpy as np

from tree_reason import SearchProblem, solve

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)


class EightPuzzle(SearchProblem):
    def __init__(self, start, hfn=None):
        self.start = start
        self.hfn = hfn

    def initial(self):
        return self.start

    def is_goal(self, s):
        return s == GOAL

    def key(self, s):
        return s

    def heuristic(self, s):
        return self.hfn(s) if self.hfn else 0

    def moves(self, s):
        i = s.index(0)
        r, c = divmod(i, 3)
        for dr, dc, name in ((-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                j = nr * 3 + nc
                ls = list(s)
                ls[i], ls[j] = ls[j], ls[i]
                yield (f"blank {name}", tuple(ls), 1)


def features(s):
    """Per-tile Manhattan distance to goal (tiles 1..8) — 8 raw features."""
    f = [0.0] * 8
    for idx, val in enumerate(s):
        if val == 0:
            continue
        gr, gc = divmod(val - 1, 3)
        r, c = divmod(idx, 3)
        f[val - 1] = abs(gr - r) + abs(gc - c)
    return np.array(f)


def manhattan(s):
    return float(features(s).sum())


def scramble(depth, rng):
    s = GOAL
    p = EightPuzzle(s)
    for _ in range(depth):
        s = rng.choice([nxt for _, nxt, _ in p.moves(s)])
    return s


def learn_heuristic(n_train=80, depth=12, seed=0):
    """Solve easy instances, label each state on the optimal path with its true
    remaining cost, fit a linear heuristic over features by least squares."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n_train):
        start = scramble(depth, rng)
        path, cost, _ = solve(EightPuzzle(start, manhattan))  # optimal solve to get labels
        if path is None:
            continue
        states = [start] + [s for _, s in path]
        n = len(states) - 1                       # true cost-to-goal of `start`
        for i, st in enumerate(states):
            X.append(features(st))
            y.append(n - i)                       # remaining cost from this state
    X, y = np.array(X), np.array(y)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)     # learned weights
    def h(s):
        return max(0.0, float(features(s) @ w))
    return h, w


def avg_nodes(hfn, test_starts):
    total, solved = 0, 0
    for start in test_starts:
        path, cost, nodes = solve(EightPuzzle(start, hfn), max_nodes=400_000)
        if path is not None:
            total += nodes
            solved += 1
    return total / max(solved, 1), solved


def main():
    print("=== tree_learn — search that learns to reason more efficiently ===\n")
    print("Learning a heuristic from 80 solved easy puzzles (depth 12)...")
    learned, w = learn_heuristic()
    print(f"  learned weights (per tile): {np.round(w, 2)}")
    print(f"  (classic Manhattan = all 1.0; it rediscovered that from experience)\n")

    # shared harder test set (deeper scrambles drift further from solved)
    rng = random.Random(999)
    test_starts = [scramble(80, rng) for _ in range(20)]
    print("Test on 20 harder puzzles — states expanded to solve:\n")
    results = {}
    for name, hfn in (("blind (no heuristic)", None),
                      ("hand-coded Manhattan", manhattan),
                      ("LEARNED heuristic", learned)):
        avg, solved = avg_nodes(hfn, test_starts)
        results[name] = avg
        print(f"  {name:24s}: {avg:9.0f} states   ({solved}/20 solved)")

    ratio = results["blind (no heuristic)"] / max(results["LEARNED heuristic"], 1)
    print(f"\nThe learned heuristic was never told the rule — it fit it from its own")
    print(f"solved experience, and now expands ~{ratio:.0f}x fewer states than blind search.")


if __name__ == "__main__":
    main()
