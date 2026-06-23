#!/usr/bin/env python3
"""
stream_grounding.py — grounding through the FULL brain.perceive pipeline.

grounding.py drove the SOM directly. This streams observations through
brain.perceive — the whole cognitive loop (SOM + predictive coding + episodic
memory + working memory + emotion + attention) — so grounding happens inside the
actual brain, not a bare map. Surprising observations get stored as episodes;
prediction error falls as the brain learns the world; the SOM still organizes, so
symbols ground and new observations are recognized.

  stream raw observations -> brain.perceive (full loop) -> SOM organizes +
  episodes stored on surprise -> ground symbols on brain.som -> recognize held-out.

    venv2/bin/python3 stream_grounding.py
"""

import brain2
from grounding import make_data, ground, recognize, SYMS, ROWS, COLS, D, K


def _demo():
    train, test = make_data()
    brain = brain2.Brain(som_rows=ROWS, som_cols=COLS, n_dims=D)

    early, late, episodes, attended = [], [], 0, 0
    for ep in range(4):
        for v, _ in train:
            r = brain.perceive(v)                 # the FULL cognitive loop
            (early if ep == 0 else late).append(r.prediction_error)
            episodes += int(r.episodic_stored)
            attended += int(r.attention_passed)

    # ground + recognize on the brain's OWN SOM (now trained via perceive)
    centroids = ground(brain.som, [(v, k) for v, k in train][:K * 5])
    ok = sum(recognize(brain.som, centroids, v) == k for v, k in test)

    print("=== stream_grounding — grounding through brain.perceive (full loop) ===\n")
    print(f"  streamed {4 * len(train)} observations through the cognitive loop")
    print(f"  prediction error: {sum(early)/len(early):.3f} (epoch 0) -> "
          f"{sum(late)/len(late):.3f} (later)   — the brain learned the world")
    print(f"  episodes stored on surprise: {episodes}   attention-passed: {attended}")
    print(f"  held-out recognition (grounded on the brain's SOM): {ok}/{len(test)} "
          f"= {ok/len(test):.0%}")
    print("\n  grounding now runs inside the real brain — perception, memory,")
    print("  emotion and attention all engaged, symbols still anchored in data.")


if __name__ == "__main__":
    _demo()
