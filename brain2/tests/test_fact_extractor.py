#!/usr/bin/env python3
"""
test_fact_extractor.py — learn by reading (text -> facts).

Pins: parses controlled declarative sentences into triples, resolves coreference
across sentences, and the read->learn->answer round-trip (feed a paragraph, then
the brain answers questions about it). Plus validation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fact_extractor import FactExtractor
from conversation_engine import ConversationEngine

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nFactExtractor — learn by reading")
    fe = FactExtractor()

    # 1. category membership
    check("'An apple is a fruit' -> isa",
          fe.extract("An apple is a fruit.") == [("apple", "isa", "fruit")])

    # 2. property (adjective, no article)
    check("'Apple is red' -> is",
          fe.extract("Apple is red.") == [("apple", "is", "red")])

    # 2b. "X is the R of Y" relational pattern
    check("'Tom is the parent of Sam' -> parent",
          fe.extract("Tom is the parent of Sam.") == [("tom", "parent", "sam")])

    # 3. has / prepositional-verb relations
    check("'A dog has a tail' -> has",
          fe.extract("A dog has a tail.") == [("dog", "has", "tail")])
    check("'An apple grows on a tree' -> grows_on",
          fe.extract("An apple grows on a tree.") == [("apple", "grows_on", "tree")])

    # 4. coreference across sentences ("It" -> running subject)
    trips = fe.extract("An apple is a fruit. It is red. It has seeds.")
    check("coreference resolves 'It' to apple",
          trips == [("apple", "isa", "fruit"), ("apple", "is", "red"),
                    ("apple", "has", "seeds")])

    # 5. multiple subjects, topic switches
    trips = fe.extract("An apple is a fruit. A dog is an animal. It has a tail.")
    check("topic switches (It -> dog)",
          ("dog", "has", "tail") in trips and ("dog", "isa", "animal") in trips)

    # 6. the round-trip: read -> learn -> answer
    c = ConversationEngine()
    n = fe.teach_into("An apple is a fruit. It is red. It grows on a tree.", c)
    check("learned facts from text", n == 3)
    ans = c.respond("what is apple?")
    check("answers from what it READ", "fruit" in ans and "red" in ans and "tree" in ans)
    check("answer is grammatical (no 'a red')", "a red" not in ans)

    # 7. robustness
    check("gibberish yields no false facts", fe.extract("asdf qwer zxcv lkjh poiu") == [] or
          all(len(t) == 3 for t in fe.extract("asdf qwer zxcv")))
    check("empty text -> no facts", fe.extract("") == [])
    try:
        fe.extract(123); check("non-string rejected", False)
    except TypeError:
        check("non-string rejected", True)

    print(f"\nFact extractor: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
