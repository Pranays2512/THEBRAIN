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

    # 9. aggregation: many objects of ONE relation collapse into one clause
    agg = ConversationEngine()
    for o in ["canine", "pet", "animal"]:
        agg.learn("dog", "isa", o)
    out = agg.respond("what is dog?").lower()
    check("aggregates multi-valued relation into one clause",
          "a canine, a pet and an animal" in out)
    check("relation verb not repeated", out.count(" is ") == 1)

    # one object per relation is unchanged (no spurious coordination)
    check("single-object relations unaffected",
          c.respond("what is apple?").startswith("An apple is a fruit"))

    # 10. how/why -> narrate the causal chain through the topic
    howc = ConversationEngine()
    for s, o in [("sun", "plant_growth"), ("plant_growth", "fruit")]:
        howc.learn(s, "leads_to", o)
    howc.set_transitive("leads_to")
    ans = howc.respond("how does fruit grow?").lower()
    check("how-question narrates the chain",
          "sun leads to plant growth, which leads to fruit" in ans)
    check("unknown process is honest",
          "don't know how" in howc.respond("how does metal grow?").lower())

    # "in which ways" -> list every branch
    waysc = ConversationEngine()
    for s, o in [("vit", "immune"), ("immune", "fight"), ("vit", "energy")]:
        waysc.learn(s, "helps", o)
    waysc.set_transitive("helps")
    wout = waysc.respond("in which ways does vit help?").lower()
    check("ways lists multiple branches",
          "it also" in wout and "energy" in wout and "fight" in wout)

    # 11. arithmetic word problem routed through the loop
    mathc = ConversationEngine()
    check("subtraction word problem",
          mathc.respond("I have 10 apples and give 3 away, how many do I have left?").startswith("7"))
    check("addition word problem",
          mathc.respond("I have 5 marbles and find 4 more, how many do I have?").startswith("9"))
    check("how-many with no numbers isn't faked",
          "7" not in mathc.respond("how many legs does it have?"))

    print(f"\nConversation loop: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
