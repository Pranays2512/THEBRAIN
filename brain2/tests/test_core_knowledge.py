#!/usr/bin/env python3
"""
test_core_knowledge.py — the vetted seed loads, chains, and answers.

Pins that the curated facts ingest cleanly, that the isa ladders support
transitive reasoning (dog -> mammal -> animal), and that the brain answers from
the seed while staying honest about what isn't in it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engines.knowledge.core_knowledge import CORE_FACTS, load_core
from engines.knowledge.knowledge_base import KnowledgeBase
from engines.reasoning.neuro_bridge import Brain, RuleEyes
from engines.reasoning.reasoning_engine import ReasoningEngine

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nCoreKnowledge — vetted seed")
    kb = KnowledgeBase()
    n = load_core(kb)
    check("seed ingests every vetted fact", n == len(CORE_FACTS))
    check("no duplicate facts in the seed", len(set(CORE_FACTS)) == len(CORE_FACTS))
    check("all triples well-formed", all(len(t) == 3 and all(t) for t in CORE_FACTS))

    # transitive ladders climb (dog -> mammal -> animal -> living_thing)
    re = ReasoningEngine()
    kb.into(re)
    ok, path = re.reaches("dog", "isa", "living_thing")
    check("isa ladder reaches the top transitively", ok)
    check("chain is the vetted ladder",
          path == ["dog", "mammal", "animal", "living_thing"])
    check("whale -> mammal too", re.reaches("whale", "isa", "animal")[0])

    # answers through the brain; honest about gaps
    brain = Brain()
    kb.into(brain)
    eyes = RuleEyes()
    check("answers an ingested entity",
          brain.answer(eyes.parse("what is a whale?")).known)
    check("honest about an un-ingested entity",
          brain.answer(eyes.parse("what is a unicorn?")).known is False)

    print(f"\nCore knowledge: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
