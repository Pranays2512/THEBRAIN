#!/usr/bin/env python3
"""
eval_harness.py — measure the brain the way a product pitch must: trust metrics.

Runs a benchmark set through the Mind (eyes -> brain -> mouth) and reports not
just accuracy but the numbers that distinguish this architecture:

  coverage             : fraction it answered at all (vs honestly declined)
  accuracy             : fraction correct over ALL items
  accuracy_on_answered : correct over the ones it chose to answer
  verified_correct_rate: of answers it marked VERIFIED, how many were right

The last one is the wedge — it should be 1.00. A system that is right whenever it
says "verified", and says "I don't know" otherwise, is trustworthy in a way a
bare LLM cannot be. Honest declines (no elementary integral, unknown entity) are
scored as CORRECT calibration, not failures — the brain is graded on knowing what
it knows, not on bluffing.

Honest scope: this is a curated benchmark over what the symbolic brain covers
(math, logic, taught knowledge) — a demonstration of the trust metric, NOT a
claim about MMLU. Broad MMLU needs the knowledge-ingestion work.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from neuro_bridge import Mind, Brain, RuleEyes, GrammarMouth


@dataclass
class Case:
    q: str
    mode: str           # "value" | "text" | "unknown"
    gold: object = None


def _match(case, ans):
    if case.mode == "unknown":
        return ans.known is False        # rewarded for honestly declining
    if not ans.known:
        return False
    if case.mode == "value":
        return ans.value == case.gold
    return str(case.gold).lower() in str(ans.value).lower()


class EvalHarness:
    def __init__(self, cases):
        self.cases = cases

    def run(self, mind):
        rows = []
        m = dict(total=0, answered=0, correct=0, correct_answered=0,
                 verified=0, verified_correct=0)
        for c in self.cases:
            ans = mind.brain.answer(mind.eyes.parse(c.q))
            correct = _match(c, ans)
            m["total"] += 1
            m["correct"] += int(correct)                 # incl. correct honest declines
            if ans.known:
                m["answered"] += 1
                m["correct_answered"] += int(correct)
            if ans.verified:
                m["verified"] += 1
                m["verified_correct"] += int(correct)
            rows.append((c.q, ans.known, ans.verified, correct, ans.value))
        return m, rows

    @staticmethod
    def report(m):
        def pct(a, b):
            return f"{(a / b * 100):.0f}%" if b else "n/a"
        print(f"  items                     {m['total']}")
        print(f"  coverage (answered)       {pct(m['answered'], m['total'])}")
        print(f"  calibration (incl declines){pct(m['correct'], m['total'])}")
        print(f"  accuracy on answered      {pct(m['correct_answered'], m['answered'])}")
        print(f"  VERIFIED-CORRECT RATE     {pct(m['verified_correct'], m['verified'])}"
              f"   ({m['verified_correct']}/{m['verified']})  <- the trust metric")


def default_benchmark():
    return [
        Case("differentiate x^3", "text", "3*x^2"),
        Case("differentiate sin(x^2)", "text", "cos(x^2)*(2*x)"),
        Case("what is the derivative of exp(x)*ln(x)?", "text", "exp(x)*ln(x)"),
        Case("integrate cos(x)", "text", "sin(x)"),
        Case("integrate x^2", "text", "x^3/3"),
        Case("integrate sin(x^2)", "unknown"),               # honest decline
        Case("solve 2*x + 3 = 7 for x", "value", 2),
        Case("solve x^2 = 49 for x", "value", 7),
        Case("solve 5*x = 20 for x", "value", 4),
        Case("I have 10 apples and give 3 away, how many do I have left?", "text", "7"),
        Case("what is apple?", "text", "fruit"),
        Case("what is unobtainium?", "unknown"),             # honest unknown
    ]


def _demo():
    mind = Mind(RuleEyes(), Brain(), GrammarMouth())
    mind.teach("apple", "isa", "fruit")
    mind.teach("apple", "is", "red")

    harness = EvalHarness(default_benchmark())
    m, rows = harness.run(mind)

    print("=== eval_harness — trust metrics over the brain's coverage ===\n")
    for q, known, verified, correct, val in rows:
        tag = "OK " if correct else "XX "
        flags = ("V" if verified else " ") + ("K" if known else "-")
        print(f"  [{tag}][{flags}] {q[:46]:46s} -> {val}")
    print()
    EvalHarness.report(m)


if __name__ == "__main__":
    _demo()
