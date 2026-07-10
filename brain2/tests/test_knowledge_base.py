#!/usr/bin/env python3
"""
test_knowledge_base.py — ingest, normalize, dedupe, persist, raise coverage.

Pins the ingestion pipeline: multi-source triples are normalized and deduped,
stats are correct, the store round-trips, and loading it into a brain measurably
raises coverage while leaving un-ingested entities honestly unknown.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.knowledge.knowledge_base import KnowledgeBase, _coverage
from core.reasoning.neuro_bridge import Brain

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nKnowledgeBase — ingest, dedupe, persist, cover")
    kb = KnowledgeBase()

    # 1. normalize + dedupe
    kb.add("Dog", "IsA", "Animal")
    dup = kb.add("dog", "isa", "animal")               # same after normalize
    check("normalizes case/spaces", ("dog", "isa", "animal") in kb.facts)
    check("dedupes identical facts", dup is False and len(kb.facts) == 1)
    check("rejects self-loop", kb.add("x", "isa", "x") is False)
    kb.add("New York", "isa", "City")
    check("spaces -> underscores", ("new_york", "isa", "city") in kb.facts)

    # 2. multi-source ingest + stats
    n = kb.ingest_triples([("cat", "isa", "mammal"), ("cat", "can", "purr")], "demo")
    check("ingest_triples adds new facts", n == 2)
    kb.ingest_text("A whale is a mammal. It lives in the ocean.")
    s = kb.stats()
    check("stats count facts/entities/relations",
          s["facts"] == len(kb.facts) and s["entities"] > 0 and s["relations"] > 0)
    check("by-source tracked", s["by_source"].get("demo") == 2)

    # 3. persistence round-trip
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "kb.json")
        kb.save(p)
        kb2 = KnowledgeBase.load(p)
        check("save/load preserves facts", kb2.facts == kb.facts)

    # 4. ingestion raises brain coverage; un-ingested stays unknown
    qs = ["what is dog?", "what is cat?", "what is whale?", "what is zorblax?"]
    empty = Brain()
    a0, _ = _coverage(empty, qs)
    brain = Brain()
    kb.into(brain)
    a1, t = _coverage(brain, qs)
    check("empty brain covers nothing", a0 == 0)
    check("ingestion raises coverage", a1 > a0)
    check("un-ingested entity stays unknown", a1 < t)   # zorblax never ingested

    # 5. ConceptNet smoke (cached, large)
    big = KnowledgeBase()
    added = big.ingest_conceptnet()
    check("ConceptNet ingests many facts", added > 1000)

    print(f"\nKnowledge base: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
