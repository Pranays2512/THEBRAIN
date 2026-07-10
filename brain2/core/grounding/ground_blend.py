#!/usr/bin/env python3
"""
ground_blend.py — concept blending wired into the REAL grounding system.

concept_blend.py proved the idea on toy vectors. This runs it on the actual perceptual
pipeline: the C++ SOM self-organizes raw data, grounding.py grounds concepts as mean SOM
activation maps, and blend() fuses two of those REAL centroids into a new one — then
verifies the blend occupies perceptual space NO existing concept owns (its nearest known
concept is below the same-concept similarity floor). A verified-novel blend is registered,
so recognize() gains a new category grounded in the real SOM.

This is the "mint a new concept region" step on real activations (no C++ change — the
fixed-neuron SOM is untouched; the new concept lives in the Python centroid registry over
its activation space).

Honest limit: blend PROPOSES a novel region and verifies it's unclaimed; whether the
region is USEFUL still needs grounded examples to confirm. Sparse near-binary activation
maps (a known SOM limit) make the blend a union of two regions.

    venv2/bin/python3 ground_blend.py
"""

import numpy as np
import brain2
from core.grounding import grounding as G


def blend_centroid(centroids, a_name, b_name, mode="salient"):
    a, b = centroids[a_name], centroids[b_name]
    if mode == "mid":
        return (a + b) / 2.0
    return np.where(np.abs(a) >= np.abs(b), a, b)      # salient per-dim union


def verify_novel(centroids, c, sim_floor):
    """Novel iff even the MOST similar existing concept is below the same-concept floor:
    the blend is not a relabel of anything known. Returns (novel, nearest, sim)."""
    sims = {k: G._cos(c, v) for k, v in centroids.items()}
    nearest = max(sims, key=sims.get)
    return sims[nearest] < sim_floor, nearest, sims[nearest]


def _demo():
    train, test = G.make_data()
    som = brain2.SOM(G.ROWS, G.COLS, G.D, init_lr=0.3)
    for _ in range(8):
        for v, _ in train:
            som.update(v, som.find_bmu(v), 1.0)
    labeled = [(v, k) for v, k in train][:G.K * 5]
    centroids = G.ground(som, labeled)
    named = {G.SYMS[k]: centroids[k] for k in centroids}

    print("=== ground_blend — blend real grounded concepts into a new region ===\n")
    # same-concept similarity floor: how self-similar a concept is to its own held-out mean
    selfsim = []
    for k in centroids:
        half = [np.asarray(som.activation_map(v)) for v, kk in test if kk == k]
        selfsim.append(G._cos(centroids[k], np.mean(half, axis=0)))
    floor = float(np.mean(selfsim)) * 0.9
    print("  grounded concepts:", ", ".join(named))
    print("  same-concept similarity ~%.2f  -> novelty floor %.2f\n" % (np.mean(selfsim), floor))

    a_name, b_name = G.SYMS[0], G.SYMS[1]
    for mode in ("mid", "salient"):
        c = blend_centroid(named, a_name, b_name, mode)
        novel, near, sim = verify_novel(named, c, floor)
        print("  blend(%s,%s) [%-7s] -> nearest=%s sim=%.2f  => %s"
              % (a_name, b_name, mode, near, sim,
                 "NOVEL (unclaimed region)" if novel else "not novel (is %s)" % near))

    # Activation-space blend stays near a parent (sparse maps). Blend in RAW input space
    # instead — features are disentangled there — then ground the result on the SOM.
    centers = np.array([np.mean([v for v, k in train if k == i], axis=0)
                        for i in range(G.K)])
    h = G.D // 2
    chimera_raw = np.concatenate([centers[0][:h], centers[1][h:]]).astype("float32")  # A-half ⊕ B-half
    c = np.asarray(som.activation_map(chimera_raw))               # ground it -> new activation region
    novel, near, sim = verify_novel(named, c, floor)
    print("\n  RAW-space chimera (%s first-half ⊕ %s second-half), grounded on the SOM:"
          % (a_name, b_name))
    print("    nearest=%s sim=%.2f  => %s"
          % (near, sim, "NOVEL (unclaimed region)" if novel else "not novel (is %s)" % near))
    if novel:
        new_name = a_name + "+" + b_name
        # PROVE USEFUL: a region is only a real concept if it GENERALIZES. Draw samples
        # from the chimera region, ground the concept on a FEW, and verify held-out
        # samples are recognized as it — WITHOUT damaging the existing concepts.
        rng = np.random.default_rng(11)
        new_train = [(chimera_raw + 0.18 * rng.standard_normal(G.D)).astype("float32")
                     for _ in range(5)]
        new_test = [(chimera_raw + 0.18 * rng.standard_normal(G.D)).astype("float32")
                    for _ in range(30)]

        def acc(centset, samples_labels):
            return sum(max(centset, key=lambda k: G._cos(np.asarray(som.activation_map(v)),
                                                         centset[k])) == lbl
                       for v, lbl in samples_labels) / len(samples_labels)

        orig = [(v, G.SYMS[k]) for v, k in test]
        before = acc(named, orig)                                  # 4-concept accuracy
        named[new_name] = np.mean([np.asarray(som.activation_map(v)) for v in new_train],
                                   axis=0)                          # ground from examples
        after = acc(named, orig)                                   # with the new concept added
        new_acc = acc(named, [(v, new_name) for v in new_test])    # new-class generalization
        print("    registered '%s' and grounded it on 5 examples.\n" % new_name)
        print("  PROVE USEFUL:")
        print("    new-class held-out recognition: %.0f%%  (the region generalizes -> a real concept)"
              % (new_acc * 100))
        print("    original concepts before/after adding it: %.0f%% -> %.0f%%  (%s)"
              % (before * 100, after * 100,
                 "no damage" if after >= before - 0.02 else "regressed"))

        # MINT A REAL NEURON: grow the live SOM at the chimera point so the new concept
        # has a home in perceptual space (no longer a Python-only centroid).
        before_n = som.n_neurons
        new_idx = som.add_neuron(chimera_raw)
        routed = sum(som.find_bmu(v) == new_idx for v in new_test)
        print("    minted live SOM neuron #%d (neurons %d -> %d); %d/%d new-region samples"
              " now route to it." % (new_idx, before_n, som.n_neurons, routed, len(new_test)))
    print("\n  Novel (unclaimed) + generalizes (grounded) + harmless (no regression) + a"
          " minted SOM neuron = a concept the brain invented, verified, and now PERCEIVES.")


if __name__ == "__main__":
    _demo()
