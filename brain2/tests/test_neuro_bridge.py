#!/usr/bin/env python3
"""
test_neuro_bridge.py — the IO contract: eyes -> brain -> mouth.

Pins that the brain operates on structured Query/Answer (not raw text), that math
answers are verified, unknowns are honest, and that Eyes/Mouth are swappable
without touching cognition (the LLM-as-eyes/mouth guarantee).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.reasoning.neuro_bridge import Mind, Brain, RuleEyes, GrammarMouth, Mouth, Query, Answer

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


class TerseMouth(Mouth):
    """A second mouth — proves rendering is swappable without touching the brain."""
    def render(self, a):
        if not a.known:
            return "?"
        return f"{a.value}" if a.kind != "language" else a.value


def run():
    print("\nNeuroBridge — eyes -> brain -> mouth")
    eyes, brain = RuleEyes(), Brain()
    brain.teach("apple", "isa", "fruit")
    brain.teach("apple", "is", "red")

    # 1. eyes turn text into a structured Query (the brain never sees raw text)
    check("eyes route differentiate", eyes.parse("differentiate x^3").kind == "differentiate")
    check("eyes route solve", eyes.parse("solve 2*x = 8 for x").kind == "solve")
    check("eyes route language", eyes.parse("what is apple?").kind == "language")
    check("eyes flag a parse error", eyes.parse("differentiate sin x").kind == "error")

    # 2. brain answers from structure, math is verified
    a_diff = brain.answer(Query("differentiate", {"expr": ("^", "x", 3)}))
    check("brain differentiates, verified", a_diff.verified and a_diff.value == "3*x^2")
    a_solve = brain.answer(eyes.parse("solve 2*x + 3 = 7 for x"))
    check("brain solves, verified, value=2", a_solve.verified and a_solve.value == 2)
    a_int = brain.answer(eyes.parse("integrate sin(x^2)"))
    check("brain honest on unintegrable", a_int.known is False)

    # 3. language path carries known-flag honestly
    a_lang = brain.answer(eyes.parse("what is banana?"))
    check("brain honest on unknown entity", a_lang.known is False)

    # 4. SWAPPABILITY: same brain Answer, two different mouths
    grammar, terse = GrammarMouth(), TerseMouth()
    a = brain.answer(eyes.parse("differentiate x^3"))
    check("grammar mouth renders a sentence", grammar.render(a) == "The derivative is 3*x^2.")
    check("terse mouth renders the same answer differently", terse.render(a) == "3*x^2")
    check("the brain answer was identical for both mouths", a.value == "3*x^2")

    # 4b. relational/inverse questions route through the query planner
    brain.teach("paris", "capital_of", "france")
    inv = brain.answer(eyes.parse("what is the capital of france?"))
    check("inverse query answered via planner",
          inv.known and "paris" in inv.value.lower())
    check("a plain describe still goes to conversation",
          "An apple is a fruit" in brain.answer(eyes.parse("what is apple?")).value)
    check("planner declines cleanly -> honest unknown",
          brain.answer(eyes.parse("what is the capital of narnia?")).known is False)

    # 5. full Mind pipeline
    mind = Mind(RuleEyes(), brain, GrammarMouth())
    check("end-to-end differentiate", "3*x^2" in mind.respond("differentiate x^3"))
    check("end-to-end solve verified", "x = 2" in mind.respond("solve 2*x + 3 = 7 for x"))
    check("end-to-end language", "An apple is a fruit" in mind.respond("what is apple?"))

    print(f"\nNeuro bridge: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
