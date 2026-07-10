#!/usr/bin/env python3
"""
test_curiosity_loop.py — idle learning driven by prediction error (idea #4).

Pins that error falls as the loop learns the predictable patterns, that learned
rules are the immediate successors, and that a genuinely random follower is never
learned — so curiosity correctly persists where no stable rule exists.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faculties.curiosity_loop import CuriosityLoop

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nCuriosityLoop — idle learning from prediction error")
    predictable = ([["rain", "wet_ground", "puddles"]] * 2 +
                   [["study", "pass"]] * 2 +
                   [["seed", "plant"]] * 2)
    noise = [["dice", "one"], ["dice", "two"], ["dice", "three"]]
    chunks = [predictable + noise[:1], predictable + noise[1:2],
              predictable + noise[2:3], predictable + noise[:1]]

    cl = CuriosityLoop()
    errors = []
    for ch in chunks:
        cl.observe(ch)
        _, err = cl.tick()
        errors.append(err)

    # 1. error falls as patterns are learned
    check("error decreases over ticks", errors[-1] < errors[0])
    check("error is monotonically non-increasing",
          all(errors[i + 1] <= errors[i] for i in range(len(errors) - 1)))

    # 2. predictable rules learned, as the IMMEDIATE successor
    check("learned rain -> wet_ground (immediate, not transitive puddles)",
          cl.predict.get("rain") == "wet_ground")
    check("learned study -> pass", cl.predict.get("study") == "pass")
    check("learned seed -> plant", cl.predict.get("seed") == "plant")

    # 3. the random follower is never learned -> curiosity persists
    check("random event 'dice' never learned", cl.predict.get("dice") is None)
    gaps = dict(cl.curiosity_gaps())
    check("dice remains a curiosity gap", "dice" in gaps and gaps["dice"] == 1.0)
    check("solved events drop out of curiosity",
          "rain" not in gaps and "study" not in gaps)

    print(f"\nCuriosity loop: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
