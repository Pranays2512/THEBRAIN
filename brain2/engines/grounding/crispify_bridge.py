#!/usr/bin/env python3
"""
crispify_bridge.py — feed the C++ PolicyEngine from the C++ BindingMemory.

The membrane, made concrete at the vector seam. The C++ BindingMemory is FUZZY
(stores triples as vectors, recalls by similarity with a confidence). The C++
PolicyEngine is CRISP (exact arithmetic over verified facts). The bridge:

  query BindingMemory  ->  (object_vector, confidence)  ->  GATE on confidence
     accept (conf >= floor): decode the scalar -> a crisp fact for the engine
     reject (low conf):      return None -> honest "unknown"

So fuzzy associative recall PROPOSES, the confidence gate DISPOSES, and only crisp,
high-confidence facts reach the reasoner. Verified empirically: exact bound facts
recall at conf 1.0; an unstored fact recalls at ~0.6 and is rejected. Both halves
are C++; this is the only Python — the encode/decode + the gate.

Run:  venv2/bin/python3 crispify_bridge.py
"""

import brain2
import numpy as np

D = 64
CONF = 0.9                      # crispify threshold: known facts ~1.0, unknown ~0.6


def vec(sym):
    """Deterministic symbol -> unit vector (a tiny fixed codebook)."""
    r = np.random.default_rng(abs(hash(sym)) % (2 ** 32))
    v = r.standard_normal(D).astype("float32")
    return v / np.linalg.norm(v)


AXIS = vec("__magnitude__")     # scalar rides along this fixed axis


class BindingFactStore:
    """Crisp fact source backed by the fuzzy C++ BindingMemory."""
    def __init__(self):
        self.bm = brain2.BindingMemory(D, 1000)

    def bind(self, entity, rel, value):
        o = (float(value) * AXIS).astype("float32")
        self.bm.bind(vec(entity), vec(rel), o)

    def fact(self, entity, rel):
        o, conf = self.bm.query(vec(entity), vec(rel))
        if conf < CONF:                          # crispify gate: reject fuzzy recall
            return None
        return float(np.dot(np.asarray(o), AXIS))


def _demo():
    store = BindingFactStore()
    for r, v in [("mass", 1000.0), ("accel", 20.0), ("speed", 300.0)]:
        store.bind("rocket", r, v)

    mem = brain2.PolicyMemory()
    mem.add("force", ["mass", "accel"], ["*", "mass", "accel"])
    mem.add("power", ["force", "speed"], ["*", "force", "speed"])

    eng = brain2.PolicyEngine(mem, store.fact, True)
    print("=== crispify_bridge — C++ BindingMemory -> gate -> C++ PolicyEngine ===\n")
    print("  facts live as VECTORS in BindingMemory; the engine pulls them through")
    print(f"  a confidence gate (>= {CONF}):\n")
    for rel in ("mass", "force", "power", "wisdom"):
        v = eng.solve("rocket", rel)
        print(f"  rocket.{rel:7s} = {'(unknown — recall below gate)' if v is None else f'{v:.4g}'}")
    print("\n  force/power are COMPUTED by the crisp reasoner from facts recalled")
    print("  out of the fuzzy vector memory; 'wisdom' recalls weakly and is rejected.")


if __name__ == "__main__":
    _demo()
