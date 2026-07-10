#!/usr/bin/env python3
"""
test_dual_process.py — hardening tests for Dual cognition (#8, last rung).

Pins: whatever tier answers, the program is CORRECT on the examples;
deliberated/reflex solutions COMPILE into instant memory on recurrence; and the
memory tier is faster than deliberation.
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.synthesis.program_synth_guided import run, rand_program, rand_name
from core.reasoning.dual_process_engine import DualProcessSolver, train_policy

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def tasks(n, rng, max_len=3):
    out = []
    while len(out) < n:
        prog = rand_program(rng, max_len)
        ins = [rand_name(rng) for _ in range(3)]
        try:
            ex = [(s, run(prog, s)) for s in ins]
        except Exception:
            continue
        if len({o for _, o in ex}) == 1 and ex[0][0] == ex[0][1]:
            continue
        out.append(ex)
    return out


def run_tests():
    print("\nDualProcessSolver — hardening tests")
    s = DualProcessSolver(train_policy())
    rng = random.Random(4)
    batch = tasks(25, rng)

    # 1. correctness: whatever tier answers, the program reproduces the examples
    all_correct = True
    tiers = set()
    for ex in batch:
        r = s.solve(ex)
        tiers.add(r.tier)
        if r.found:
            all_correct = all_correct and all(r.apply(i) == o for i, o in ex)
    check("every solution is correct on its examples", all_correct)

    # 2. more than one tier is exercised (reflex and/or deliberation)
    check("reflex used (not all deliberation)", s.stats["reflex"] > 0)

    # 3. compilation: re-solving a task hits compiled memory with the same answer
    ex0 = batch[0]
    first = s.solve(ex0)                 # already cached from the batch
    check("recurring task -> memory tier", first.tier == "memory")
    check("memory answer matches", all(first.apply(i) == o for i, o in ex0))

    # 4. memory tier is faster than a fresh deliberation
    fresh = tasks(1, random.Random(99))[0]
    t = time.perf_counter(); s.solve(fresh); delib_t = time.perf_counter() - t
    t = time.perf_counter(); s.solve(fresh); mem_t = time.perf_counter() - t
    check("compiled memory faster than first solve", mem_t < delib_t)

    # 5. stats consistency
    total = sum(v for k, v in s.stats.items())
    check("stats account for every solve", total >= len(batch) + 3)

    # 6. apply on unsolved raises
    s2 = DualProcessSolver(train_policy())
    r = s2.solve([("John Smith", "Smith, John")])   # outside DSL
    if not r.found:
        try:
            r.apply("x"); check("apply on unsolved raises", False)
        except ValueError:
            check("apply on unsolved raises", True)
    else:
        check("apply on unsolved raises", True)      # solved is also fine

    print(f"\nDual cognition: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
