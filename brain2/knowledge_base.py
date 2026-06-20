#!/usr/bin/env python3
"""
knowledge_base.py — scalable knowledge ingestion (the coverage bottleneck).

Coverage = what the brain knows, so growing the brain means INGESTING knowledge.
This is the pipeline: pull triples from multiple curated sources (ConceptNet, or
plain text via the fact extractor), normalize, DEDUPE, persist, and report what's
in there — then load it all into the reasoning brain so coverage on real questions
goes up.

    kb = KnowledgeBase()
    kb.ingest_conceptnet()                 # curated common-sense triples
    kb.ingest_text("A whale is a mammal. It lives in the ocean.")
    kb.into(brain)                         # the brain now knows it
    kb.stats()                             # facts / entities / relations / by-source

Honest scope: ingestion of CURATED, trusted sources — not crawling the web. The
value is a clean, deduped, persistent store and a measurable coverage number, so
"feed the brain more" becomes a concrete, trackable step toward a product.
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from fact_extractor import FactExtractor
from world_knowledge import load_conceptnet


def _norm(token):
    return str(token).strip().lower().replace(" ", "_")


class KnowledgeBase:
    def __init__(self):
        self.facts = set()                 # unique (s, r, o)
        self.by_source = Counter()

    # ── ingestion ────────────────────────────────────────────────────────────
    def add(self, s, r, o, source="manual"):
        t = (_norm(s), _norm(r), _norm(o))
        if "" in t or t[0] == t[2]:
            return False
        if t in self.facts:
            return False
        self.facts.add(t)
        self.by_source[source] += 1
        return True

    def ingest_triples(self, triples, source):
        return sum(self.add(s, r, o, source) for s, r, o in triples)

    def ingest_conceptnet(self, min_weight=2.0):
        return self.ingest_triples(load_conceptnet(min_weight=min_weight), "conceptnet")

    def ingest_text(self, text, source="text", extractor=None):
        ex = extractor or FactExtractor()
        return self.ingest_triples(ex.extract(text), source)

    # ── stats / persistence ──────────────────────────────────────────────────
    def entities(self):
        return {s for s, _, _ in self.facts} | {o for _, _, o in self.facts}

    def relations(self):
        return {r for _, r, _ in self.facts}

    def stats(self):
        rels = Counter(r for _, r, _ in self.facts)
        return {
            "facts": len(self.facts),
            "entities": len(self.entities()),
            "relations": len(self.relations()),
            "by_source": dict(self.by_source),
            "top_relations": rels.most_common(6),
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"facts": sorted(self.facts), "by_source": dict(self.by_source)}, f)

    @classmethod
    def load(cls, path):
        kb = cls()
        with open(path) as f:
            d = json.load(f)
        kb.facts = {tuple(t) for t in d.get("facts", [])}
        kb.by_source = Counter(d.get("by_source", {}))
        return kb

    # ── loading into the brain ───────────────────────────────────────────────
    def into(self, engine, transitive=("isa",)):
        """Learn every fact into a brain (anything with .learn / .set_transitive)."""
        learn = engine.teach if hasattr(engine, "teach") else engine.learn
        n = sum(bool(learn(s, r, o)) for s, r, o in sorted(self.facts))
        present = self.relations()
        for rel in transitive:
            if rel in present:
                engine.set_transitive(rel)
        return n


def _coverage(brain, questions):
    from neuro_bridge import RuleEyes
    eyes = RuleEyes()
    answered = sum(int(brain.answer(eyes.parse(q)).known) for q in questions)
    return answered, len(questions)


def _demo():
    from neuro_bridge import Brain

    questions = ["what is dog?", "what is apple?", "what is car?",
                 "what is whale?", "what is zorblax?"]

    print("=== knowledge_base — ingest, then watch coverage rise ===\n")
    empty = Brain()
    a, t = _coverage(empty, questions)
    print(f"empty brain coverage: {a}/{t}\n")

    kb = KnowledgeBase()
    print(f"ingest ConceptNet:  +{kb.ingest_conceptnet()} facts")
    print(f"ingest text:        +{kb.ingest_text('A whale is a mammal. It lives in the ocean.')} facts")
    s = kb.stats()
    print(f"\nKB: {s['facts']} facts, {s['entities']} entities, {s['relations']} relations")
    print(f"    by source: {s['by_source']}")
    print(f"    top relations: {s['top_relations']}\n")

    brain = Brain()
    kb.into(brain)
    a2, t2 = _coverage(brain, questions)
    print(f"brain coverage after ingestion: {a2}/{t2}   (was {a}/{t})")
    print("  (zorblax still unknown — honest, not ingested)")


if __name__ == "__main__":
    _demo()
