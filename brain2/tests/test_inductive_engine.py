#!/usr/bin/env python3
"""
test_inductive_engine.py — learn rules from data, reject coincidences, reason.

Pins the honest core: a pattern that scores perfectly on training but fails the
hold-out set is rejected as spurious; only replicating patterns are promoted, and
promoted rules become usable in the reasoning engine.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.synthesis.inductive_engine import InductiveLearner
from core.reasoning.reasoning_engine import ReasoningEngine

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nInductiveLearner — mine, verify, promote")
    il = InductiveLearner()

    train = ([["rain", "wet_ground", "puddles"]] * 3 +
             [["study", "pass"]] * 3 +
             [["cat", "rainbow"]] * 2)            # coincidence in training only
    test = ([["rain", "wet_ground", "puddles"]] * 2 +
            [["study", "pass"]] * 2 +
            [["cat", "cloud"], ["cat", "wind"]])  # cat appears, rainbow does not

    promoted, rejected = il.mine(train, test)
    pset = {(r.a, r.b) for r in promoted}
    rset = {(a, b): why for a, b, why in rejected}

    # 1. real patterns survive
    check("real rule promoted: rain -> wet_ground", ("rain", "wet_ground") in pset)
    check("real rule promoted: study -> pass", ("study", "pass") in pset)

    # 2. coincidence rejected by the hold-out
    check("spurious rule NOT promoted", ("cat", "rainbow") not in pset)
    check("spurious rule rejected as spurious",
          "spurious" in rset.get(("cat", "rainbow"), ""))

    # 3. promoted rules score on BOTH splits
    check("promoted rules verified on hold-out",
          all(r.conf_test >= 0.7 for r in promoted))

    # 4. support threshold: a one-off pair is not even a candidate
    cand = il.candidates(train + [["fluke_a", "fluke_b"]], min_support=2, min_conf=0.8)
    check("below-support pair not mined", ("fluke_a", "fluke_b") not in cand)

    # 5. a rule strong in train but ABSENT in test is rejected
    t2_train = [["k", "m"]] * 3
    t2_test = [["k", "z"], ["k", "z"]]            # k present, m never follows
    p2, r2 = il.mine(t2_train, t2_test)
    check("train-only rule rejected on hold-out",
          ("k", "m") not in {(r.a, r.b) for r in p2})

    # 6. discovered rules are usable: reason transitively over them
    re = il.promote_into(ReasoningEngine(), promoted)
    ok, _ = re.reaches("rain", "leads_to", "puddles")
    check("reasoning uses the learned rules", ok)

    print(f"\nInductive engine: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
