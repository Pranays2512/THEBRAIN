#!/usr/bin/env python3
"""
world_chat.py — TALK to the brain about basic world knowledge.

world_demo answered with formatted print lines; this routes the same ConceptNet
common-sense knowledge through the full conversation loop, so the brain replies
in SENTENCES it generates — understand (intent) -> reason (facts + multi-parent
closure) -> produce (grammar). Category questions go through the engine's
transitive closure, so "is a dog an animal?" is answered with the chain shown.

    > what is a dog?      ->  A dog is a mammal. It can bark. ...
    > is a dog an animal? ->  Yes — dog -> pet -> animal.
    > is a dog a vehicle? ->  Not that I know of.

Same honest scope as conversation_engine: controlled question shapes, not
open-domain comprehension. Every word out is derived from a stored fact.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from conversation_engine import ConversationEngine
from world_knowledge import load_conceptnet


def build():
    # cap describe so dense concepts answer with a few facts, not a wall of text
    c = ConversationEngine(max_describe=3)
    for s, r, o in load_conceptnet():
        c.learn(s, r, o)
    c.set_transitive("isa")                  # IsA chains through the taxonomy
    return c


def main():
    print("=== world_chat — talking to the brain about the world (ConceptNet) ===\n")
    c = build()
    print(f"(loaded {len(c.r.kb.facts)} common-sense facts)\n")

    questions = [
        "hello",
        "what is a dog?",
        "what is a car?",
        "is a dog a mammal?",
        "is a dog an animal?",      # transitive — chain shown
        "is a dog a vehicle?",      # honest no
        "is a car a vehicle?",
        "what is a zorblax?",       # honest unknown
    ]
    for q in questions:
        print(f"  > {q}")
        print(f"    {c.respond(q)}\n")


if __name__ == "__main__":
    main()
