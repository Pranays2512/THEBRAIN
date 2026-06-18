#!/usr/bin/env python3
"""
domain_demo.py — the payoff lap: a verifiable expert on a real domain.

Reads a family tree from plain text (no hand-typed triples), sets two
inheritance rules, and answers multi-hop questions it was NEVER directly told —
showing the derivation chain for every answer, and saying "no" / "I don't know"
honestly. Everything here is the hardened stack working end to end:

    text  -> fact_extractor  -> ReasoningEngine (facts + rules) -> answers + WHY
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from fact_extractor import FactExtractor
from reasoning_engine import ReasoningEngine

KNOWLEDGE = """
Tom is the parent of Sam.
Tom is the parent of Maya.
Sam is the parent of Kim.
Maya is the parent of Leo.
Kim is the parent of Ada.
"""


def main():
    print("=== domain_demo — a verifiable expert it LEARNED BY READING ===\n")

    # 1. read knowledge from text into the reasoning engine
    re = ReasoningEngine()
    fe = FactExtractor()
    facts = fe.extract(KNOWLEDGE)
    for s, r, o in facts:
        re.learn(s, r, o)
    print("Read this family tree from plain text:")
    for s, r, o in facts:
        print(f"    {s} is the {r} of {o}")

    # 2. give it two inheritance rules (the only thing hand-set)
    re.add_rule("parent", "parent", "grandparent")
    re.add_rule("parent", "grandparent", "great_grandparent")
    print("\nRules: parent∘parent → grandparent,  parent∘grandparent → great_grandparent\n")

    # 3. answer multi-hop questions it was NEVER told, with the derivation
    print("Questions (none of these were stored — all derived):\n")
    queries = [
        ("tom", "grandparent"),          # tom -> sam -> kim
        ("tom", "great_grandparent"),    # tom -> sam -> kim -> ada
        ("sam", "grandparent"),          # sam -> kim -> ada
        ("maya", "grandparent"),         # maya -> leo -> ?  (leo has no child)
    ]
    for subj, rel in queries:
        ans, why = re.ask(subj, rel)
        q = f"who is {subj.capitalize()}'s {rel.replace('_', '-')}?"
        if ans:
            print(f"  {q}")
            print(f"    -> {ans.capitalize()}")
            print(f"    because: {why}\n")
        else:
            print(f"  {q}")
            print(f"    -> none that I can derive.\n")

    # 4. honest negatives — it knows what it does NOT know
    print("Honest about its limits (distinguishes direct facts from derived ones):")
    print(f"  is Tom the parent of Kim?     -> "
          f"{'yes' if re.knows('tom', 'parent', 'kim') else 'no (Tom is Kim’s grandparent, not parent)'}")
    print(f"  is Tom the parent of Ada?     -> "
          f"{'yes' if re.knows('tom', 'parent', 'ada') else 'no (Tom is Ada’s great-grandparent)'}")
    print(f"  who is Zara's grandparent?    -> "
          f"{(re.ask('zara', 'grandparent')[0] or 'I don’t know anyone named Zara')}")


if __name__ == "__main__":
    main()
