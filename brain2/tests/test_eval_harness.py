#!/usr/bin/env python3
"""
test_eval_harness.py — the trust metrics, and that the harness can detect error.

Pins that on the curated benchmark the brain is right whenever it answers, that
verified answers are 100% correct, that honest declines count as calibration (not
coverage), and — critically — that the harness actually FAILS a wrong answer
(otherwise the metrics are meaningless).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_harness import EvalHarness, Case, default_benchmark
from core.reasoning.neuro_bridge import Mind, Brain, RuleEyes, GrammarMouth

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def mind():
    m = Mind(RuleEyes(), Brain(), GrammarMouth())
    m.teach("apple", "isa", "fruit")
    m.teach("apple", "is", "red")
    return m


def run():
    print("\nEvalHarness — trust metrics")
    m, _ = EvalHarness(default_benchmark()).run(mind())

    # 1. the trust metric: every VERIFIED answer is correct
    check("verified-correct rate is 100%", m["verified_correct"] == m["verified"] > 0)

    # 2. right whenever it answered (on this curated set)
    check("accuracy on answered is 100%", m["correct_answered"] == m["answered"])

    # 3. honest declines exist and are scored as correct calibration, not coverage
    check("coverage below 100% (it declines)", m["answered"] < m["total"])
    check("calibration counts correct declines", m["correct"] > m["correct_answered"])

    # 4. the harness MUST catch a wrong answer (else metrics are worthless)
    wrong = [Case("differentiate x^3", "value", 999)]      # gold is wrong on purpose
    mw, _ = EvalHarness(wrong).run(mind())
    check("harness flags an incorrect answer", mw["correct"] == 0)
    check("a verified-but-wrong answer drops the trust metric",
          mw["verified"] == 1 and mw["verified_correct"] == 0)

    # 5. a deliberately unknown question is covered honestly
    unk = [Case("what is flibbertigibbet?", "unknown")]
    mu, _ = EvalHarness(unk).run(mind())
    check("honest unknown scored correct, not answered",
          mu["correct"] == 1 and mu["answered"] == 0)

    print(f"\nEval harness: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
