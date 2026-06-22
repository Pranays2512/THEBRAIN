#!/usr/bin/env python3
"""
grounding.py — the brain forms its OWN concepts from raw observation, then grounds
symbols on them (a step toward generality: meaning anchored in data, not just told).

Until now facts/symbols were TOLD ("rocket mass 1000"). Grounding is the brain
seeing raw observation vectors, self-organizing them with the SOM into concept
regions, and attaching a symbol to each region — so the symbol "alpha" MEANS "this
part of perceptual space," recognizable from new raw input it was never told about.

  1. observe unlabeled vectors  -> SOM self-organizes (unsupervised structure)
  2. a FEW labeled examples     -> ground a symbol onto each SOM region (sparse, like
                                   a human pointing: "this is X")
  3. new raw observation        -> SOM -> region -> recognized symbol  (grounded!)
  VERIFY: recognition accuracy on held-out observations (does the grounding
  generalize? — the same verifier discipline, now for perception).

Then it connects to the reasoner: perceive raw data -> recognize the concept ->
recall a property of it. The brain knows WHAT it's looking at, then reasons.

    venv2/bin/python3 grounding.py
"""

import numpy as np
import brain2

D = 8
ROWS = COLS = 16
K = 4                                  # number of concepts to form
SYMS = ["alpha", "beta", "gamma", "delta"]


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1, 1, size=(K, D)).astype("float32")
    def sample(k):
        return (centers[k] + 0.18 * rng.standard_normal(D)).astype("float32")
    train = [(sample(k), k) for k in range(K) for _ in range(60)]
    rng.shuffle(train)
    test = [(sample(k), k) for k in range(K) for _ in range(30)]
    return train, test


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def ground(som, labeled):
    """Ground each symbol on the SOM region its labeled examples light up: the
    symbol's prototype = mean ACTIVATION MAP (the full distributed SOM response,
    far richer than a single BMU's grid position)."""
    acts = {k: [] for k in range(K)}
    for v, k in labeled:
        acts[k].append(np.asarray(som.activation_map(v)))
    return {k: np.mean(acts[k], axis=0) for k in acts if acts[k]}


def recognize(som, centroids, v):
    a = np.asarray(som.activation_map(v))
    return max(centroids, key=lambda k: _cos(a, centroids[k]))


def _demo():
    train, test = make_data()
    som = brain2.SOM(ROWS, COLS, D, init_lr=0.3)

    # 1. self-organize on UNLABELED observations (several epochs)
    for _ in range(8):
        for v, _ in train:
            som.update(v, som.find_bmu(v), 1.0)

    # 2. ground symbols from a FEW labeled examples per concept (sparse)
    labeled = [(v, k) for v, k in train][:K * 5]      # only 5 labels each
    centroids = ground(som, labeled)

    # 3. + VERIFY: recognize held-out raw observations
    ok = sum(recognize(som, centroids, v) == k for v, k in test)
    print("=== grounding — brain forms concepts from raw data, grounds symbols ===\n")
    print(f"  {K} concepts, SOM {ROWS}x{COLS}, grounded from {K*5} labels "
          f"({K*5}/{len(train)} of training labeled)")
    print(f"  held-out recognition: {ok}/{len(test)} = {ok/len(test):.0%}   "
          f"(symbols generalize to unseen observations)\n")

    # 4. connect to the reasoner: each grounded concept carries a property;
    #    perceive raw data -> recognize -> recall the property (knows what it sees)
    props = {0: "conductive", 1: "insulating", 2: "magnetic", 3: "inert"}
    print("  perceive raw observation -> recognize concept -> recall property:")
    v_test, true_k = test[0]
    k = recognize(som, centroids, v_test)
    print(f"    saw a vector -> recognized '{SYMS[k]}' (true {SYMS[true_k]}) "
          f"-> property: {props[k]}")
    print("\n  the symbol 'alpha' now MEANS a region of perceptual space — grounded,")
    print("  recognized from raw input, not a string it was told.")


if __name__ == "__main__":
    _demo()
