#!/usr/bin/env python3
"""
test_llm_extractor.py — prose -> triples, with the grounding safety gate.

Pins that well-formed triples are kept and normalized, that BOTH JSON shapes
parse, and — critically — that a triple whose terms are NOT in the source text is
DROPPED (the model can't smuggle in facts the text never stated).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_extractor import LLMExtractor
from llm_adapter import StubClient
from knowledge_base import KnowledgeBase

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nLLMExtractor — prose -> grounded triples")
    text = "A whale is a mammal. It lives in the ocean."

    # array-of-arrays, with one hallucinated triple
    stub = StubClient({"whale": '[["whale","isa","mammal"],'
                                '["whale","lives_in","ocean"],'
                                '["dragon","can","breathe_fire"]]'})
    got = LLMExtractor(stub).extract(text)
    gset = set(got)
    check("keeps grounded triple (whale isa mammal)", ("whale", "isa", "mammal") in gset)
    check("keeps second grounded triple", ("whale", "lives_in", "ocean") in gset)
    check("DROPS hallucinated triple (dragon not in text)",
          ("dragon", "can", "breathe_fire") not in gset)

    # array-of-objects shape also parses + normalizes case/spaces
    stub2 = StubClient({"whale": '[{"subject":"Blue Whale","relation":"IsA","object":"Mammal"}]'})
    got2 = LLMExtractor(stub2).extract("A Blue Whale is a Mammal.")
    check("object-shape JSON parses + normalizes",
          ("blue_whale", "isa", "mammal") in set(got2))

    # garbage / empty model output -> no triples, no crash
    check("non-JSON output -> empty", LLMExtractor(StubClient({"x": "sorry!"})).extract("foo") == [])

    # drops into the ingestion pipeline like the grammar extractor
    kb = KnowledgeBase()
    kb.ingest_text(text, extractor=LLMExtractor(stub))
    check("extracted triples ingest via knowledge_base",
          ("whale", "isa", "mammal") in kb.facts and
          ("dragon", "can", "breathe_fire") not in kb.facts)

    print(f"\nLLM extractor: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
