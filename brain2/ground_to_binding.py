#!/usr/bin/env python3
"""
ground_to_binding.py — grounded perceptions auto-flow into the brain's binding memory.

Closes perception -> memory -> reasoning entirely inside the C++ brain. A perceived
quantity is decoded, then WRITTEN as a vector triple into the brain's own
BindingMemory (the hippocampal store) — not a Python dict. Reasoning then pulls the
fact back out of that memory through a confidence gate and the PolicyEngine
computes with it. The brain stores what it perceives, and reasons from what it
stored.

  perceive -> decode value -> brain.binding.bind(entity, rel, value)   [C++ memory]
  reason   -> brain.binding.query(entity, rel) -> gate -> PolicyEngine.solve

    venv2/bin/python3 ground_to_binding.py
"""

import numpy as np
import brain2

D = 32                                  # must match the brain's n_dims
CONF = 0.9


def vec(sym):
    r = np.random.default_rng(abs(hash(sym)) % (2 ** 32))
    v = r.standard_normal(D).astype("float32")
    return v / np.linalg.norm(v)


AXIS = vec("__magnitude__")


def _demo():
    rng = np.random.default_rng(0)
    brain = brain2.Brain(som_rows=4, som_cols=4, n_dims=D)
    bm = brain.binding                  # the brain's OWN hippocampal store

    # a learned decoder (perception -> number), like ground_numeric
    enc_axes = {a: rng.standard_normal(D) for a in ("mass", "accel")}
    def encode(vals):
        v = sum(vals[a] * enc_axes[a] for a in vals) + 0.04 * rng.standard_normal(D)
        return v.astype("float32")
    labeled = [{a: rng.uniform(2, 9) for a in enc_axes} for _ in range(40)]
    Vm = np.array([encode(x) for x in labeled])
    Wd = {a: np.linalg.lstsq(Vm, np.array([x[a] for x in labeled]), rcond=None)[0]
          for a in enc_axes}

    # PolicyEngine pulls facts FROM the brain's binding memory (gated)
    def fact(e, r):
        o, conf = bm.query(vec(e), vec(r))
        return float(np.dot(np.asarray(o), AXIS)) if conf >= CONF else None
    mem = brain2.PolicyMemory()
    mem.add("force", ["mass", "accel"], ["*", "mass", "accel"])
    eng = brain2.PolicyEngine(mem, fact, True)

    print("=== ground_to_binding — perceive -> brain.binding -> reason ===\n")
    hits = 0
    for i in range(6):
        vals = {a: round(rng.uniform(2, 9), 2) for a in enc_axes}
        obs = encode(vals)
        dec = {a: float(obs @ Wd[a]) for a in enc_axes}        # GROUND the numbers
        ent = f"obj{i}"
        for a, val in dec.items():                             # WRITE into brain memory
            bm.bind(vec(ent), vec(a), (val * AXIS).astype("float32"))
        got = eng.solve(ent, "force")                          # READ from brain memory + compute
        true = vals["mass"] * vals["accel"]
        ok = got is not None and abs(got - true) / true < 0.1
        hits += ok
        print(f"  obj{i}: bound mass~{dec['mass']:.2f}, accel~{dec['accel']:.2f} "
              f"-> force={got:.1f} (true {true:.1f})  [{'ok' if ok else 'off'}]")
    print(f"\n  brain.binding now holds {bm.size} grounded triples; {hits}/6 forces")
    print("  correct — facts perceived, stored in the C++ hippocampal memory, and")
    print("  reasoned from there. Perception -> memory -> reasoning, all in the brain.")


if __name__ == "__main__":
    _demo()
