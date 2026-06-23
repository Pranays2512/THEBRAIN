#!/usr/bin/env python3
"""
unsupervised_grounding.py — concept DISCOVERY with no labels at all.

grounding.py used a few labels to ground symbols. This uses NONE: the brain
self-organizes raw observations (SOM), then clusters the activation patterns into
groups — discovering the category structure on its own. Labels are only consulted
AFTER, to measure whether what it discovered matches reality (purity). Naming a
cluster ("this group is metal") is a separate, later, sparse act.

  raw observations -> SOM -> cluster activation maps (k-means, unsupervised)
  -> measure purity against true categories (did it find the real structure?)

High purity = the brain carved the world at its real joints without being told.

    venv2/bin/python3 unsupervised_grounding.py
"""

import numpy as np
import brain2
from grounding import make_data, ROWS, COLS, D, K


def _one_kmeans(X, k, rng, iters=50):
    cent = X[rng.choice(len(X), k, replace=False)]
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any() else cent[j]
                        for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    inertia = ((X - cent[lab]) ** 2).sum()
    return lab, inertia


def kmeans(X, k, restarts=10, seed=0):
    """Best of several random restarts (k-means is init-sensitive)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        lab, inertia = _one_kmeans(X, k, rng)
        if best is None or inertia < best[1]:
            best = (lab, inertia)
    return best[0]


def purity(clusters, truth, k):
    """Fraction correct if each cluster is named by its majority true label."""
    total = 0
    for j in range(k):
        members = truth[clusters == j]
        if len(members):
            total += np.bincount(members, minlength=k).max()
    return total / len(truth)


def _demo():
    train, _ = make_data()
    som = brain2.SOM(ROWS, COLS, D, init_lr=0.3)
    for _ in range(8):
        for v, _ in train:
            som.update(v, som.find_bmu(v), 1.0)

    X = np.array([som.activation_map(v) for v, _ in train])   # NO labels used
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)  # cosine clustering
    truth = np.array([k for _, k in train])
    clusters = kmeans(X, K)
    p = purity(clusters, truth, K)

    print("=== unsupervised_grounding — discover concepts with NO labels ===\n")
    print(f"  {len(train)} raw observations, SOM {ROWS}x{COLS}, k-means k={K} on")
    print(f"  activation patterns — zero labels used to cluster.\n")
    print(f"  cluster purity vs true categories: {p:.0%}")
    print("  (the brain carved the data into the real categories on its own;")
    print("  naming each cluster is a separate, later, sparse act.)")


if __name__ == "__main__":
    _demo()
