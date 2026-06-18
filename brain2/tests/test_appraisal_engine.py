#!/usr/bin/env python3
"""
test_appraisal_engine.py — pragmatic appraisal of input (redesigned emotion).

Pins: utterance-type recognition from form, the multi-dimensional frame,
surprise-weighting (constant function words barely count), and robustness.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from appraisal_engine import AppraisalEngine

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nAppraisalEngine — pragmatic frame of input")
    ae = AppraisalEngine()

    # 1. the canonical example: greeting + question + tone, all at once
    a = ae.appraise("hey, how are you?")
    check("'hey how are you?' is a question", a.type == "question")
    check("  ...also greeting", a.frame["greeting"] >= 1)
    check("  ...also friendly + curious", a.frame["friendly"] >= 1 and a.frame["curious"] >= 1)
    check("  ...refers to the system (about_self)", a.frame["about_self"] >= 1)

    # 2. definition question
    a = ae.appraise("what is apple?")
    check("'what is apple?' is a definition question",
          a.type == "question" and a.frame["definition"] >= 1)

    # 3. command
    a = ae.appraise("tell me about you")
    check("'tell me about you' is a command about self",
          a.type == "command" and a.frame["about_self"] >= 1)

    # 4. statement (no strong markers)
    a = ae.appraise("the apple is red")
    check("plain statement classified as statement", a.type == "statement")

    # 5. wh-question without '?'
    check("wh-word alone signals question", ae.appraise("who is alice").type == "question")

    # 6. surprise weighting: a pile of common words stays low
    a = ae.appraise("it is in the of to and that")
    check("constant function words score low", a.frame["question"] < 0.8)

    # 7. inversion question (do you ...)
    check("inversion 'do you ...' -> question", ae.appraise("do you know him").type == "question")

    # 8. robustness: empty / None -> statement, no crash
    check("empty input -> statement", ae.appraise("").type == "statement")
    check("None input -> statement", ae.appraise(None).type == "statement")

    print(f"\nAppraisal engine: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
