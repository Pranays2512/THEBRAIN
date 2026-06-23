#!/usr/bin/env python3
"""
ground_flow.py — perception -> grounded category -> real-world inheritance, end to end.

Unifies grounding with the knowledge base: a raw observation is recognized as a
known concept (grounding), that recognition AUTO-asserts (object, isa, concept) into
the reasoner, and the object inherits the concept's whole real-world web — taxonomy
chain + properties — with zero hand-entered facts about the object.

  see raw vector -> recognize 'dog' -> learn(obj, isa, dog)
                 -> inherit: obj isa mammal isa animal; obj can bark; obj has lungs

The brain looks at something unlabeled and knows what it is AND everything that
follows — perception flowing straight into real-world reasoning.

    venv2/bin/python3 ground_flow.py
"""

import numpy as np
import brain2
from reasoning_engine import ReasoningEngine
from core_knowledge import CORE_FACTS
from grounding import ground, recognize, ROWS, COLS, D

CONCEPTS = ["dog", "cat", "sparrow", "eagle", "salmon", "oak", "rose"]


def _demo():
    rng = np.random.default_rng(1)
    centers = {c: rng.uniform(-1, 1, D).astype("float32") for c in CONCEPTS}

    def obs(c):
        return (centers[c] + 0.18 * rng.standard_normal(D)).astype("float32")

    # ground the concepts on the SOM (self-organize + sparse labels)
    som = brain2.SOM(ROWS, COLS, D, init_lr=0.3)
    train = [(obs(c), i) for i, c in enumerate(CONCEPTS) for _ in range(50)]
    rng.shuffle(train)
    for _ in range(8):
        for v, _ in train:
            som.update(v, som.find_bmu(v), 1.0)
    centroids = ground(som, train[:len(CONCEPTS) * 5])

    # the real-world reasoner (curated facts) + transitive isa + inheritance
    re = ReasoningEngine()
    for s, r, o in CORE_FACTS:
        re.learn(s, r, o)
    re.set_transitive("isa")
    for prop in ("has", "can", "lives_in"):
        re.add_rule("isa", prop, prop)

    print("=== ground_flow — perceive -> recognize -> inherit real-world facts ===\n")
    for i in range(5):
        true = CONCEPTS[i % len(CONCEPTS)]
        sym = CONCEPTS[recognize(som, centroids, obs(true))]   # GROUND
        obj = f"thing{i}"
        re.learn(obj, "isa", sym)                              # AUTO-assert grounded fact
        ancestors = re.derive_all(obj, "isa")                  # inherited taxonomy
        cans = list(re.ask_all(obj, "can"))                    # inherited abilities
        has = list(re.ask_all(obj, "has"))                     # inherited parts
        print(f"  {obj}: perceived -> '{sym}'")
        print(f"      isa-chain: {ancestors}   can: {cans}   has: {has}")
    print("\n  each object: a raw vector in, recognized concept, then its full")
    print("  real-world web inherited — perception flows straight into reasoning.")


if __name__ == "__main__":
    _demo()
