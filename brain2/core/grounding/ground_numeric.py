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


def ground_and_compute(n=8, tol=0.1):
    """Perceive continuous quantities (decode from raw vectors) -> assert as crisp facts ->
    the C++ PolicyEngine COMPUTES from perceived values. Returns {hits,total,results}. The
    reusable core (callable from the front)."""
    rng = np.random.default_rng(0)
    axes = {a: rng.standard_normal(D) for a in ATTRS}

    def encode(vals):
        v = sum(vals[a] * axes[a] for a in ATTRS) + 0.05 * rng.standard_normal(D)
        return v.astype("float32")

    labeled = [{a: rng.uniform(2, 9) for a in ATTRS} for _ in range(40)]
    V = np.array([encode(x) for x in labeled])
    W = {a: np.linalg.lstsq(V, np.array([x[a] for x in labeled]), rcond=None)[0]
         for a in ATTRS}

    def decode(v):
        return {a: float(v @ W[a]) for a in ATTRS}

    brain = brain2.Brain(som_rows=4, som_cols=4, n_dims=D)
    brain.policy_add("force", ["mass", "accel"], ["*", "mass", "accel"])

    hits, results = 0, []
    for i in range(n):
        vals = {a: round(rng.uniform(2, 9), 2) for a in ATTRS}
        dec = decode(encode(vals))
        for a in ATTRS:
            brain.teach_fact(f"obj{i}", a, dec[a])
        got = brain.policy_solve(f"obj{i}", "force")
        true = vals["mass"] * vals["accel"]
        ok = abs(got - true) / true < tol
        hits += ok
        results.append({"mass": round(dec["mass"], 2), "accel": round(dec["accel"], 2),
                        "force": round(got, 1), "true": round(true, 1), "ok": bool(ok)})
    return {"hits": hits, "total": n, "results": results}


def _demo():
    print("=== ground_numeric — perceive quantities -> numeric facts -> compute ===\n")
    r = ground_and_compute()
    for x in r["results"]:
        print(f"  perceived mass~{x['mass']} accel~{x['accel']} -> force={x['force']} "
              f"(true {x['true']})  [{'ok' if x['ok'] else 'off'}]")
    print(f"\n  {r['hits']}/{r['total']} forces within 10% — computed from PERCEIVED quantities.")


if __name__ == "__main__":
    _demo()
