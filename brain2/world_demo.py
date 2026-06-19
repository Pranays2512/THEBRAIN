#!/usr/bin/env python3
"""
world_demo.py — the brain reasoning over basic human knowledge of the world.

Loads a curated common-sense subset of ConceptNet (real world knowledge — dogs
bark, birds fly, a car is a vehicle) into the brain's knowledge, then answers
everyday questions: lists what it knows, derives category membership through the
FULL IsA taxonomy (multi-parent transitive closure) with the chain shown, and is
honest about what it doesn't know.

Note: ConceptNet concepts have many parents (dog -> mammal, pet, canine, ...), so
membership is real transitive CLOSURE over the fact graph — not a single chain.
The binding memory holds the facts; reasoning traverses them.
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from reasoning_engine import ReasoningEngine
from world_knowledge import load_conceptnet


def pretty(w):
    return w.replace("_", " ")


def main():
    print("=== world_demo — reasoning over basic world knowledge (ConceptNet) ===\n")
    facts = load_conceptnet()
    re = ReasoningEngine()                       # the brain reasons over the facts
    for s, r, o in facts:
        re.learn(s, r, o)
    re.set_transitive("isa")                     # IsA chains through the taxonomy

    # direct multi-valued lookups for "what it knows" come from the fact graph
    rel_obj = defaultdict(set)
    for s, r, o in re.kb.facts:
        rel_obj[(s, r)].add(o)
    ents = {s for s, _, _ in re.kb.facts} | {o for _, _, o in re.kb.facts}
    print(f"Loaded {len(re.kb.facts)} common-sense facts about {len(ents)} concepts.\n")

    # category membership = the engine's native multi-parent closure (no longer
    # re-coded here — reaches() does BFS over the multi-valued fact graph).
    def is_a(c, target):
        return re.reaches(c, "isa", target)

    def holds(c, rel, obj):
        return obj in rel_obj.get((c, rel), set())

    # 1. what it knows (direct multi-valued lookup)
    print("What it knows:")
    for subj, rel, label in [("dog", "can", "a dog can"), ("apple", "has", "an apple has"),
                             ("car", "isa", "a car is"), ("knife", "used_for", "a knife is used for")]:
        vals = sorted(rel_obj.get((subj, rel), set()))[:4]
        if vals:
            print(f"  {label}: {', '.join(pretty(v) for v in vals)}")
    print()

    # 2. category membership through the full taxonomy, chain shown
    print("Category questions (derived through the IsA taxonomy):")
    for subj, target in [("dog", "mammal"), ("dog", "animal"), ("car", "vehicle"),
                         ("tree", "plant"), ("dog", "vehicle")]:
        ok, path = is_a(subj, target)
        if ok:
            print(f"  is a {pretty(subj)} a {pretty(target)}?  -> yes")
            print(f"      because: {' -> '.join(pretty(x) for x in path)}")
        else:
            print(f"  is a {pretty(subj)} a {pretty(target)}?  -> not that I can derive")
    print()

    # 3. honest capability checks
    print("Honest checks:")
    print(f"  can a bird fly?      -> {'yes' if holds('bird','can','fly') else 'not in what I know'}")
    print(f"  can a fish fly?      -> {'yes' if holds('fish','can','fly') else 'no (a fish can swim, not fly)'}")
    print(f"  is a rock alive?     -> {'yes' if is_a('rock','organism')[0] else 'not that I can derive'}")
    print(f"  what is a zorblax?   -> {'known' if 'zorblax' in ents else 'I have never heard of a zorblax'}")


if __name__ == "__main__":
    main()
