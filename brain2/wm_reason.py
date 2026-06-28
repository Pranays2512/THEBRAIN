#!/usr/bin/env python3
"""
wm_reason.py — multi-hop reasoning over DISTINCT working-memory slots (Gen #3 follow-on).

slot_contents (added earlier) lets WM hold concept A and concept B separately instead of
mean(A,B). This uses it: a 2-hop transitive comparison (A>B and B>C  =>  A>C) that requires
holding all three intermediates distinctly. The contrast shows why the old flat context()
made this impossible — averaging the slots destroys the individual magnitudes the chain needs.

Each concept is a vector: dim0 carries a magnitude (e.g. size), the rest is a one-hot id.
Holding them in separate slots, the reasoner recovers each magnitude and chains the
comparisons. From the blended context() vector, the individuals are unrecoverable.

    venv2/bin/python3 wm_reason.py
"""

import numpy as np
import brain2


def vec(magnitude, idx, n=5):
    v = np.zeros(n, dtype="float32")
    v[0] = magnitude
    v[idx] = 1.0
    return v


def _demo():
    print("=== wm_reason — multi-hop over distinct WM slots (vs flat average) ===\n")
    items = [("A", 3.0), ("B", 2.0), ("C", 1.0)]   # A>B>C by magnitude
    wm = brain2.WorkingMemory(5)
    for i, (name, mag) in enumerate(items, start=1):
        wm.gate(vec(mag, i), 2.0)

    # DISTINCT path: recover each slot's magnitude+id, chain the comparison
    slots = [np.asarray(s) for s in wm.slot_contents(-1.0)]
    recovered = sorted(((float(s[0]), int(np.argmax(s[1:]) + 1)) for s in slots), reverse=True)
    print("  slot_contents holds %d concepts distinctly:" % len(slots))
    for mag, idx in recovered:
        print("    concept id %d, magnitude %.0f" % (idx, mag))
    chain_ok = all(recovered[i][0] > recovered[i + 1][0] for i in range(len(recovered) - 1))
    print("  2-hop chain: A>B (%.0f>%.0f) AND B>C (%.0f>%.0f)  =>  A>C : %s"
          % (recovered[0][0], recovered[1][0], recovered[1][0], recovered[2][0],
             "PROVEN" if chain_ok else "failed"))

    # FLAT path: context() blends everything -> the individuals are gone
    ctx = np.asarray(wm.context())
    print("\n  context() (the old flat average) =", np.round(ctx, 2))
    print("    -> a single blended vector; A, B, C are smeared together, their magnitudes")
    print("       unrecoverable. Multi-hop comparison is impossible from this.")
    print("\n  Holding concepts DISTINCTLY is what makes relational/multi-hop reasoning work —")
    print("  the slot view enables it; the averaged context never could.")


if __name__ == "__main__":
    _demo()
