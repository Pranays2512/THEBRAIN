#!/usr/bin/env python3
"""
test_conversation_engine.py — the understand -> reason -> produce loop (capstone).

Pins the whole loop: appraisal routes utterance types, intent is recognized,
"it" is resolved to the topic via working memory, facts are retrieved, and the
answer is GENERATED with grammar (article, pronoun) — and unknowns are honest.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conversation_engine import ConversationEngine

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def apple_world():
    c = ConversationEngine()
    for s, r, o in [("apple", "isa", "fruit"), ("apple", "color", "red"),
                    ("apple", "grows_on", "tree"), ("apple", "has", "seeds")]:
        c.learn(s, r, o)
    return c


def run():
    print("\nConversationEngine — understand -> reason -> produce")
    c = apple_world()

    # 1. greeting routed by appraisal
    check("greeting handled", c.respond("hello").lower().startswith("hello"))

    # 2. describe: produced sentence with article + pronoun + multiple relations
    d = c.respond("what is apple?")
    check("describe starts 'An apple is a fruit'", d.startswith("An apple is a fruit"))
    check("describe uses pronoun 'It' for later facts", "It is red" in d)
    check("describe covers multiple relations", "tree" in d and "seeds" in d)

    # 3. confirm (yes)
    check("confirm true fact", c.respond("is apple a fruit?").lower().startswith("yes"))

    # 4. confirm (no / unknown value)
    check("confirm false value", "not" in c.respond("is apple blue?").lower())

    # 5. coreference: 'it' resolves to the current topic (apple)
    c.respond("what is apple?")                  # sets topic = apple
    check("'is it red?' -> yes via coreference", c.respond("is it red?").lower().startswith("yes"))
    check("'is it blue?' -> no via coreference", "not" in c.respond("is it blue?").lower())

    # 6. honest unknown
    check("unknown entity -> honest", "don't know" in c.respond("what is banana?").lower())

    # 7. topic switches with a new entity
    c.learn("dog", "isa", "animal")
    c.respond("what is dog?")
    check("topic switches to dog", c.respond("is it an animal?").lower().startswith("yes"))

    # 8. transitive reasoning surfaces in conversation
    c2 = ConversationEngine()
    for a, b in [("cat", "mammal"), ("mammal", "animal")]:
        c2.learn(a, "isa", b)
    c2.set_transitive("isa")
    # "is cat an animal?" should hold transitively
    check("transitive fact answered", c2.respond("is cat a animal?").lower().startswith("yes"))

    print(f"\nConversation loop: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
