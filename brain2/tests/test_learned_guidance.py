#!/usr/bin/env python3
"""
test_learned_guidance.py — hardening tests for Learned search guidance (#5).

Pins the two properties that matter: guided search stays CORRECT (its solutions
really solve), and it expands far fewer states than blind. Plus determinism,
persistence, and validation.
"""

import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.reasoning.tree_reason import solve
from core.reasoning.tree_learn import EightPuzzle, features, manhattan, scramble, GOAL
from core.reasoning.learned_guidance import LearnedHeuristic, collect_examples, GuidanceError

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def trained(seed=0):
    h = LearnedHeuristic(features)
    h.train(collect_examples(EightPuzzle, scramble, manhattan, n_tasks=60, seed=seed))
    return h


def applies(start, prog_moves):
    """Replay a solution path's moves and confirm it reaches the goal."""
    s = start
    for _, st in prog_moves:
        s = st
    return s == GOAL


def run():
    print("\nLearned guidance — hardening tests")
    h = trained()

    # 1. training fit (rediscovers Manhattan ~ all ones)
    check("trained weights present", h.weights is not None and len(h.weights) == 8)
    check("weights near Manhattan (~1)", np.allclose(h.weights, 1.0, atol=0.3))

    # 2. correctness: guided solutions actually solve, on many puzzles
    rng = random.Random(1)
    starts = [scramble(50, rng) for _ in range(12)]
    all_valid = True
    blind_nodes = guided_nodes = 0
    solved = 0
    for s in starts:
        pg, _, ng = solve(EightPuzzle(s, h), max_nodes=400_000)
        pb, _, nb = solve(EightPuzzle(s, None), max_nodes=400_000)
        if pg is not None:
            solved += 1
            all_valid = all_valid and applies(s, pg)
            guided_nodes += ng
            blind_nodes += nb
    check("guided solutions are valid (reach goal)", all_valid and solved == 12)

    # 3. speedup: guided expands far fewer states than blind
    ratio = blind_nodes / max(guided_nodes, 1)
    check(f"guided << blind (~{ratio:.0f}x fewer)", ratio > 8)

    # 4. determinism: same seed -> same weights
    check("deterministic training", np.allclose(trained(0).weights, trained(0).weights))

    # 5. persistence round-trip
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.txt")
        h.save(p)
        h2 = LearnedHeuristic.load(p, features)
        st = scramble(20, random.Random(7))
        check("save/load preserves heuristic", abs(h(st) - h2(st)) < 1e-6)

    # 6. validation
    try:
        LearnedHeuristic(features)(GOAL); check("reject use-before-train", False)
    except GuidanceError:
        check("reject use-before-train", True)
    try:
        LearnedHeuristic("not callable"); check("reject bad feature_fn", False)
    except GuidanceError:
        check("reject bad feature_fn", True)
    try:
        LearnedHeuristic(features).train([]); check("reject empty training", False)
    except GuidanceError:
        check("reject empty training", True)

    print(f"\nLearned guidance: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
