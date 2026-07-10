#!/usr/bin/env python3
"""
check_library.py — a self-growing, PERSISTED library of verifiers (refuter -> invariant -> store).

Connects three pieces into one loop that makes the envelope grow across sessions:

  refuter finds a candidate BREAKS  ->  mine the invariants the CORRECT version satisfies
  ->  bank them in a persistent library (keyed by task)  ->  next session they pre-filter
      candidates immediately, no re-mining.

So every overfit the brain catches leaves behind a cheap, reusable check. The library is to
verifiers what BrainStore is to facts/functions: self-extension made permanent. Over sessions
the check set GROWS and prior checks are reused for free.

Honest scope: invariants are necessary-not-sufficient (from invariant_miner) — banked checks
reject cheaply, never falsely accept. Library is per-task-key; cross-task generalization of a
check is a further step.

    python3 check_library.py     # two sessions: session 1 learns from a break, session 2 reuses
"""

import json
import os

from core.synthesis import invariant_miner as IM
from core.synthesis import refuter as RF

LIB = os.path.join(os.path.dirname(__file__), "check_library")


class CheckLibrary:
    def __init__(self, path=LIB):
        self.path = path
        self.inv = {}                              # task_key -> [invariant names]
        self.load()

    def load(self):
        p = os.path.join(self.path, "invariants.json")
        if os.path.exists(p):
            self.inv = json.load(open(p))

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        json.dump(self.inv, open(os.path.join(self.path, "invariants.json"), "w"), indent=1)

    def learn_from_break(self, task, oracle, mine_inputs, holdout_inputs, kind, broken_fn):
        """A candidate broke -> diagnose it, then bank the invariants the oracle satisfies."""
        rep = RF.refute(broken_fn, oracle, kind)                  # confirm + diagnose the break
        train = [(x, oracle(*IM._norm(x))) for x in mine_inputs]
        hold = [(x, oracle(*IM._norm(x))) for x in holdout_inputs]
        admitted = IM.validate(IM.mine(train), hold)
        before = set(self.inv.get(task, []))
        self.inv[task] = sorted(before | admitted)
        self.save()
        return rep, admitted, sorted(set(self.inv[task]) - before)

    def prefilter(self, task, fn, probes):
        return IM.check(fn, probes, set(self.inv.get(task, [])))

    def known(self, task):
        return self.inv.get(task, [])


def _demo():
    import math
    import shutil
    shutil.rmtree(LIB, ignore_errors=True)        # fresh start for the demo
    overfit = lambda n: n * n                      # the wrong candidate synthesis produced

    print("=== check_library — refuter -> invariant -> persisted, growing across sessions ===\n")

    print("[session 1] a synthesized candidate n*n breaks; learn from it")
    lib = CheckLibrary()
    print("  library before:", lib.known("factorial") or "(empty)")
    rep, admitted, new = lib.learn_from_break(
        "factorial", math.factorial, [0, 1, 4, 5, 6], [7, 8, 9], "int1", overfit)
    print("  refuter: breaks_at %s  [%s]" % (rep["breaks_at"], rep["scope"]))
    print("  mined from the correct version + banked:", new)
    print("  library saved to disk.\n")

    print("[session 2] fresh process reloads the library — checks already there")
    lib2 = CheckLibrary()                          # reload from disk
    print("  library on load:", lib2.known("factorial"))
    ok, why = lib2.prefilter("factorial", overfit, [0, 1, 5])
    print("  same overfit n*n now pre-filtered instantly:", ok, "(%s)" % why)
    ok2, _ = lib2.prefilter("factorial", math.factorial, [0, 1, 5])
    print("  correct factorial still passes:", ok2)
    print("\n  Every break the brain catches leaves a reusable check behind. The verifier set")
    print("  grows across sessions — the self-growing envelope, made permanent.")


if __name__ == "__main__":
    _demo()
