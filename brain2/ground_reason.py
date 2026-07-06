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

PROPS = {"alpha": "conductive", "beta": "insulating",
         "gamma": "magnetic", "delta": "inert"}


def ground_and_reason(reasoner=None, epochs=8):
    """Perceive raw vectors -> recognize categories (SOM) -> ASSERT grounded facts ->
    INFER properties never told. Returns {results, correct, total}. The reusable core
    (callable from the front); `reasoner` lets a caller reuse a live ReasoningEngine."""
    train, test = make_data()
    som = brain2.SOM(ROWS, COLS, D, init_lr=0.3)
    for _ in range(epochs):
        for v, _ in train:
            som.update(v, som.find_bmu(v), 1.0)
    centroids = ground(som, [(v, k) for v, k in train][:len(SYMS) * 5])

    re = reasoner or ReasoningEngine()
    for concept, p in PROPS.items():
        re.learn(concept, "property", p)            # category knowledge
    re.add_rule("isa", "property", "property")      # X isa Y & Y property Z => X property Z

    results, correct = [], 0
    for i, (v, true_k) in enumerate([test[j] for j in (0, 30, 60, 90, 15, 75)]):
        obj = f"obj{i}"
        sym = SYMS[recognize(som, centroids, v)]     # GROUND: recognize the category
        re.learn(obj, "isa", sym)                    # ASSERT the grounded fact
        inferred, _why = re.ask(obj, "property")     # INFER its property
        truth = PROPS[SYMS[true_k]]
        correct += (inferred == truth)
        results.append({"obj": obj, "perceived": sym, "inferred": inferred, "truth": truth})
    return {"results": results, "correct": correct, "total": len(results)}


def _demo():
    print("=== ground_reason — perceive -> recognize -> assert -> infer ===\n")
    r = ground_and_reason()
    for x in r["results"]:
        ok = x["inferred"] == x["truth"]
        print(f"  {x['obj']}: perceived -> '{x['perceived']}'  =>  property = {x['inferred']}"
              f"   [{'ok' if ok else 'MISS, true ' + x['truth']}]")
    print(f"\n  {r['correct']}/{r['total']} properties correctly inferred from PERCEPTION alone.")
    print("  the brain saw raw vectors, recognized categories, asserted them as facts,")
    print("  and INFERRED properties it was never told about the objects.")


if __name__ == "__main__":
    _demo()
