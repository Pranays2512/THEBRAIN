#!/usr/bin/env python3
"""
knowledge_pack.py — fuse curated sources into one shippable, measured knowledge file.

Builds the brain's knowledge from the vetted core seed + ConceptNet (+ any TSV /
N-Triples dumps you pour in), dedupes, saves one JSON pack, and reports stats and
coverage on a sample question set. This is the "feed the brain" step turned into a
single artifact you can ship and reload.

    python3 knowledge_pack.py                 # build + report
    kb = build_pack("knowledge_pack.json", extra_tsv=["wikidata_subset.tsv"])

Honest scope: structured, curated sources. The full Wikidata dump (terabytes of
opaque Q/P codes) needs label resolution and is not a laptop job — but a labeled
subset exported as TSV/N-Triples drops straight in.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from knowledge_base import KnowledgeBase
from core_knowledge import load_core


def build_pack(out_path=None, extra_tsv=None, extra_nt=None, conceptnet=True):
    kb = KnowledgeBase()
    load_core(kb)
    if conceptnet:
        kb.ingest_conceptnet()
    for p in (extra_tsv or []):
        kb.ingest_tsv(p)
    for p in (extra_nt or []):
        kb.ingest_ntriples(p)
    if out_path:
        kb.save(out_path)
    return kb


SAMPLE_QS = [
    "what is a dog?", "what is a whale?", "is a dog an animal?",
    "what is the capital of france?", "what is water?", "what is a car?",
    "what is a unicorn?",                       # honest miss
]


def _coverage(kb, questions):
    from neuro_bridge import Brain, RuleEyes
    brain = Brain()
    kb.into(brain)
    eyes = RuleEyes()
    answered = sum(int(brain.answer(eyes.parse(q)).known) for q in questions)
    return answered, len(questions)


def _demo():
    print("=== knowledge_pack — build the brain's knowledge ===\n")
    kb = build_pack()
    s = kb.stats()
    print(f"pack: {s['facts']} facts, {s['entities']} entities, {s['relations']} relations")
    print(f"      by source: {s['by_source']}")
    a, t = _coverage(kb, SAMPLE_QS)
    print(f"\ncoverage on sample questions: {a}/{t}  (the unicorn is honestly unknown)")


if __name__ == "__main__":
    _demo()
