#!/usr/bin/env python3
"""test_brain_chat.py — the unified router dispatches to the right faculty."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faculties.brain_chat import BrainChat

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def world():
    bc = BrainChat()
    for s, r, o in [("dog", "isa", "pet"), ("pet", "isa", "animal"),
                    ("apple", "isa", "fruit"), ("apple", "is", "red")]:
        bc.learn(s, r, o)
    for s, o in [("sunlight", "photosynthesis"), ("photosynthesis", "fruit")]:
        bc.learn(s, "leads_to", o)
    bc.set_transitive("isa")
    bc.set_transitive("leads_to")
    return bc


def run():
    print("\nBrainChat — one front door, dispatched")
    bc = world()

    # language faculties
    check("greeting -> conversation", bc.respond("hello").lower().startswith("hello"))
    check("describe -> conversation", "An apple is a fruit" in bc.respond("what is apple?"))
    check("category -> reasoning (chain)", "dog -> pet -> animal" in bc.respond("is a dog an animal?"))
    check("causal -> how/why chain",
          "leads to" in bc.respond("how does fruit grow?").lower())
    check("word problem -> arithmetic",
          bc.respond("I have 10 apples and give 3 away, how many do I have left?").startswith("7"))

    # math faculties
    check("differentiate -> calculus", "cos(x^2)" in bc.respond("differentiate sin(x^2)"))
    check("integrate -> integral", "sin(x)" in bc.respond("integrate cos(x)"))
    check("solve -> algebra (verified)",
          "x = 2" in bc.respond("solve 2*x + 3 = 7 for x"))

    # routing discipline: a plain fact question must NOT go to the math parser
    check("plain word stays in language",
          "don't know" in bc.respond("what is banana?").lower())
    check("math intent detected even with words",
          "3*x^2" in bc.respond("what is the derivative of x^3?"))

    print(f"\nBrain chat: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
