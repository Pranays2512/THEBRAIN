#!/usr/bin/env python3
"""
core_knowledge.py — a small, hand-verified seed of high-quality world facts.

"Super quality data" honestly means CURATED and VETTED, not scraped. These are
clean (subject, relation, object) triples a person can check by eye, spanning a
few everyday domains, so the brain has a trustworthy base to reason over and to
chain (isa is transitive). Scale comes later from ConceptNet / Wikidata via
`knowledge_base`; this is the gold core.

    from core_knowledge import CORE_FACTS, load_core
    load_core(knowledge_base)        # ingest the vetted seed
"""

# every triple here is hand-checked true; isa chains are deliberate (dog -> mammal
# -> animal -> living_thing) so transitive reasoning has clean ladders to climb.
CORE_FACTS = [
    # ── taxonomy ladders (isa is transitive) ────────────────────────────────
    ("dog", "isa", "mammal"), ("cat", "isa", "mammal"), ("whale", "isa", "mammal"),
    ("human", "isa", "mammal"), ("sparrow", "isa", "bird"), ("eagle", "isa", "bird"),
    ("salmon", "isa", "fish"), ("frog", "isa", "amphibian"),
    ("mammal", "isa", "animal"), ("bird", "isa", "animal"), ("fish", "isa", "animal"),
    ("amphibian", "isa", "animal"), ("animal", "isa", "living_thing"),
    ("oak", "isa", "tree"), ("rose", "isa", "flower"), ("tree", "isa", "plant"),
    ("flower", "isa", "plant"), ("plant", "isa", "living_thing"),
    # ── properties / parts / abilities ──────────────────────────────────────
    ("dog", "can", "bark"), ("cat", "can", "meow"), ("bird", "can", "fly"),
    ("fish", "can", "swim"), ("whale", "lives_in", "ocean"), ("frog", "lives_in", "pond"),
    ("heart", "does", "pump_blood"), ("lung", "does", "exchange_gas"),
    ("mammal", "has", "lungs"), ("bird", "has", "feathers"), ("fish", "has", "gills"),
    ("tree", "has", "roots"), ("flower", "has", "petals"),
    # ── matter / chemistry basics ───────────────────────────────────────────
    ("water", "made_of", "hydrogen_and_oxygen"), ("water", "is", "liquid"),
    ("ice", "is", "solid"), ("steam", "is", "gas"),
    ("oxygen", "isa", "element"), ("hydrogen", "isa", "element"),
    ("gold", "isa", "metal"), ("iron", "isa", "metal"), ("metal", "isa", "material"),
    ("salt", "made_of", "sodium_and_chlorine"),
    # ── astronomy ───────────────────────────────────────────────────────────
    ("sun", "isa", "star"), ("earth", "isa", "planet"), ("mars", "isa", "planet"),
    ("moon", "orbits", "earth"), ("earth", "orbits", "sun"), ("planet", "isa", "body"),
    ("star", "emits", "light"),
    # ── geography (capitals / continents) ───────────────────────────────────
    ("paris", "capital_of", "france"), ("tokyo", "capital_of", "japan"),
    ("cairo", "capital_of", "egypt"), ("ottawa", "capital_of", "canada"),
    ("france", "isa", "country"), ("japan", "isa", "country"), ("egypt", "isa", "country"),
    ("france", "in", "europe"), ("japan", "in", "asia"), ("egypt", "in", "africa"),
    ("nile", "isa", "river"), ("everest", "isa", "mountain"),
    # ── everyday objects / use ──────────────────────────────────────────────
    ("knife", "used_for", "cutting"), ("pen", "used_for", "writing"),
    ("car", "isa", "vehicle"), ("bicycle", "isa", "vehicle"), ("vehicle", "used_for", "transport"),
    ("clock", "used_for", "telling_time"), ("apple", "isa", "fruit"),
    ("apple", "is", "edible"), ("fruit", "isa", "food"),
]


def load_core(kb):
    """Ingest the vetted seed into a KnowledgeBase. Returns facts added."""
    return kb.ingest_triples(CORE_FACTS, "core")


def _demo():
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from core.knowledge.knowledge_base import KnowledgeBase
    from neuro_bridge import Brain, RuleEyes

    kb = KnowledgeBase()
    print(f"=== core_knowledge — {load_core(kb)} vetted facts ===\n")
    s = kb.stats()
    print(f"{s['facts']} facts, {s['entities']} entities, {s['relations']} relations\n")

    brain = Brain()
    kb.into(brain)
    eyes = RuleEyes()
    for q in ["what is a whale?", "is a dog an animal?", "what is the capital of france?",
              "what is water?"]:
        ans = brain.answer(eyes.parse(q))
        print(f"  > {q}\n    known={ans.known}  {ans.value if ans.known else ''}")


if __name__ == "__main__":
    _demo()
