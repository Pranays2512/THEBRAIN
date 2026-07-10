#!/usr/bin/env python3
"""
learned_guidance.py — hardened Learned search guidance (milestone #5).

The reasoning that improves with experience, made a clean reusable module. A
LearnedHeuristic fits a linear estimate of cost-to-goal over state features,
trained from solved instances, then guides the hardened search engine. The two
guarantees the tests pin: guided search stays CORRECT (still solves), and it
expands far fewer states than blind search.

    h = LearnedHeuristic(features)
    h.train(collect_examples(EightPuzzle, scramble, manhattan))
    solve(EightPuzzle(start, hfn=h))      # same engine, now guided

Domain-agnostic: give it a feature function and solved instances. Demonstrated
on the 8-puzzle (where it rediscovers the Manhattan heuristic from experience
and cuts search ~100x). Persistable; deterministic given a seed.
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engines.reasoning.tree_reason import solve
from engines.reasoning.tree_learn import EightPuzzle, features, manhattan, scramble


class GuidanceError(ValueError):
    pass


class LearnedHeuristic:
    """Linear cost-to-goal estimate over state features, fit by least squares."""

    def __init__(self, feature_fn, weights=None):
        if not callable(feature_fn):
            raise GuidanceError("feature_fn must be callable")
        self.feature_fn = feature_fn
        self.weights = None if weights is None else np.asarray(weights, dtype=float)

    def train(self, examples):
        """examples: iterable of (state, true_cost_to_goal)."""
        examples = list(examples)
        if not examples:
            raise GuidanceError("no training examples")
        X = np.array([self.feature_fn(s) for s, _ in examples], dtype=float)
        y = np.array([c for _, c in examples], dtype=float)
        self.weights, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def __call__(self, state):
        if self.weights is None:
            raise GuidanceError("heuristic used before train()")
        return max(0.0, float(self.feature_fn(state) @ self.weights))

    def save(self, path):
        if self.weights is None:
            raise GuidanceError("nothing to save; train() first")
        np.savetxt(path, self.weights)

    @classmethod
    def load(cls, path, feature_fn):
        return cls(feature_fn, weights=np.loadtxt(path))


def collect_examples(problem_factory, scramble_fn, base_heuristic,
                     n_tasks=80, depth=12, seed=0):
    """Solve easy instances with a base heuristic; label every state on each
    optimal path with its true remaining cost-to-goal."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_tasks):
        start = scramble_fn(depth, rng)
        path, _, _ = solve(problem_factory(start, base_heuristic))
        if path is None:
            continue
        states = [start] + [s for _, s in path]
        total = len(states) - 1
        for i, st in enumerate(states):
            out.append((st, total - i))
    return out


def _avg_nodes(hfn, starts, max_nodes=400_000):
    total = solved = 0
    for start in starts:
        path, _, nodes = solve(EightPuzzle(start, hfn), max_nodes=max_nodes)
        if path is not None:
            total += nodes
            solved += 1
    return total / max(solved, 1), solved


def _demo():
    print("LearnedHeuristic demo (8-puzzle):")
    h = LearnedHeuristic(features)
    h.train(collect_examples(EightPuzzle, scramble, manhattan))
    print(f"  learned weights ~ {np.round(h.weights, 2)} (Manhattan = all 1.0)")
    rng = random.Random(999)
    starts = [scramble(80, rng) for _ in range(15)]
    blind, _ = _avg_nodes(None, starts)
    learned, _ = _avg_nodes(h, starts)
    print(f"  blind {blind:.0f} states  ->  learned {learned:.0f} states  "
          f"(~{blind / max(learned, 1):.0f}x fewer)")


if __name__ == "__main__":
    _demo()
