#!/usr/bin/env python3
"""
ground_reason.py — close the loop: grounded PERCEPTION -> asserted FACT -> REASONING.

grounding.py recognizes a raw observation as a concept. This feeds that straight
into the reasoner: recognizing an object as "metal" asserts (object, isa, metal),
and the ReasoningEngine INHERITS the concept's properties to the object via a
composition rule (X isa Y AND Y property Z => X property Z). So a property the
brain was never told about the object is derived from what it PERCEIVED + what it
knows about the category.

  see raw vector -> recognize 'metal' -> learn (wire7, isa, metal)
                 -> reason: wire7 isa metal, metal property conductive
                 => wire7 property conductive        (grounded, then inferred)

Perception becomes a native reasoning input — the brain knows what it's looking at
AND what follows from that.

    venv2/bin/python3 ground_reason.py
"""

import brain2
from reasoning_engine import ReasoningEngine
from grounding import make_data, ground, recognize, SYMS, ROWS, COLS, D


def _demo():
    train, test = make_data()
    som = brain2.SOM(ROWS, COLS, D, init_lr=0.3)
    for _ in range(8):
        for v, _ in train:
            som.update(v, som.find_bmu(v), 1.0)
    centroids = ground(som, [(v, k) for v, k in train][:len(SYMS) * 5])

    # the reasoner: each concept has a known property + an inheritance rule
    re = ReasoningEngine()
    props = {"alpha": "conductive", "beta": "insulating",
             "gamma": "magnetic", "delta": "inert"}
    for concept, p in props.items():
        re.learn(concept, "property", p)            # category knowledge
    re.add_rule("isa", "property", "property")      # X isa Y & Y property Z => X property Z

    print("=== ground_reason — perceive -> recognize -> assert -> infer ===\n")
    # perceive several unlabeled objects; ground them; reason about each
    correct = 0
    # one object per category (test is grouped 30-per-class) -> spans concepts
    spread = [test[j] for j in (0, 30, 60, 90, 15, 75)]
    for i, (v, true_k) in enumerate(spread):
        obj = f"obj{i}"
        sym = SYMS[recognize(som, centroids, v)]     # GROUND: recognize the category
        re.learn(obj, "isa", sym)                    # ASSERT the grounded fact
        inferred, why = re.ask(obj, "property")      # INFER its property
        truth = props[SYMS[true_k]]
        correct += (inferred == truth)
        print(f"  {obj}: perceived -> '{sym}'  =>  {obj} property = {inferred}"
              f"   [{'ok' if inferred == truth else 'MISS, true ' + truth}]")
    print(f"\n  {correct}/6 properties correctly inferred from PERCEPTION alone.")
    print("  e.g. " + (why or "") )
    print("\n  the brain saw raw vectors, recognized categories, asserted them as")
    print("  facts, and INFERRED properties it was never told about the objects.")


if __name__ == "__main__":
    _demo()
