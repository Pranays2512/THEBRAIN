#!/usr/bin/env python3
"""
ground_numeric.py — ground CONTINUOUS quantities, feed them to the policy engine.

grounding.py grounded categories (symbols). This grounds NUMBERS: a perception
encodes measurable quantities (mass, accel); a learned decoder reads them back;
the recovered values are asserted as crisp facts; the C++ PolicyEngine computes a
derived quantity (force = mass*accel) from quantities it PERCEIVED, not was told.

  observation vector -> decode (mass, accel) -> brain.teach_fact -> policy_solve(force)

The decoder is calibrated from a few labeled observations (a grounded "sensor"),
then verified on fresh ones. So the numeric reasoning chain is fed by perception
end to end, and checked against truth.

    venv2/bin/python3 ground_numeric.py
"""

import numpy as np
import brain2

D = 16
ATTRS = ["mass", "accel"]


def _demo():
    rng = np.random.default_rng(0)
    axes = {a: rng.standard_normal(D) for a in ATTRS}     # encoding directions

    def encode(vals):
        v = sum(vals[a] * axes[a] for a in ATTRS) + 0.05 * rng.standard_normal(D)
        return v.astype("float32")

    # calibrate the decoder from labeled observations (grounded sensor): learn
    # w_a such that observation . w_a ~= value of attribute a.
    labeled = [{a: rng.uniform(2, 9) for a in ATTRS} for _ in range(40)]
    V = np.array([encode(x) for x in labeled])
    W = {a: np.linalg.lstsq(V, np.array([x[a] for x in labeled]), rcond=None)[0]
         for a in ATTRS}

    def decode(v):
        return {a: float(v @ W[a]) for a in ATTRS}

    brain = brain2.Brain(som_rows=4, som_cols=4, n_dims=D)
    brain.policy_add("force", ["mass", "accel"], ["*", "mass", "accel"])

    print("=== ground_numeric — perceive quantities -> numeric facts -> compute ===\n")
    print("  decoder calibrated from 40 labeled observations; tested on fresh ones:\n")
    hits = 0
    for i in range(8):
        vals = {a: round(rng.uniform(2, 9), 2) for a in ATTRS}
        dec = decode(encode(vals))                        # GROUND the numbers
        for a in ATTRS:
            brain.teach_fact(f"obj{i}", a, dec[a])        # assert as crisp facts
        got = brain.policy_solve(f"obj{i}", "force")      # PolicyEngine computes
        true = vals["mass"] * vals["accel"]
        ok = abs(got - true) / true < 0.1
        hits += ok
        print(f"  obj{i}: perceived mass~{dec['mass']:.2f} accel~{dec['accel']:.2f} "
              f"-> force={got:.1f}  (true {true:.1f})  [{'ok' if ok else 'off'}]")
    print(f"\n  {hits}/8 forces within 10% — computed by the reasoner from quantities")
    print("  it PERCEIVED (decoded from raw vectors), not values it was told.")


if __name__ == "__main__":
    _demo()
