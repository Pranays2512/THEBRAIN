#!/usr/bin/env python3
"""
concept_blend.py — the novelty primitive: make a concept that exists in NEITHER parent.

Analogy/transfer finds the nearest EXISTING concept (generalization). Blending does the
opposite: deliberately fuse two DISTANT concepts into a point that lands in EMPTY feature
space — outside every known category. That empty point is a candidate new solution space
(the thing the architecture otherwise structurally lacks). A blend is only interesting if
it is verifiably NOVEL: its nearest known concept is farther than the cluster radius, so
it is classified as neither parent nor anything else.

Pure-python (feature vectors as lists) so it runs anywhere. In the real brain these are
SOM centroids; a verified-novel blend would mint a new SOM neuron at that point and train
it on the combined features.

Honest limit: blending PROPOSES novelty (an empty, reachable region). It does not prove
the new concept is USEFUL — that still needs grounding/verification downstream. It widens
the space; the verifier still decides what's real.
"""

import math


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest(v, concepts, exclude=()):
    best, bd = None, float("inf")
    for name, c in concepts.items():
        if name in exclude:
            continue
        d = dist(v, c)
        if d < bd:
            best, bd = name, d
    return best, bd


def blend(a_name, b_name, concepts, mode="salient"):
    """Fuse two concepts. 'salient' keeps, per dimension, the more pronounced feature of
    the two (|value| larger) -> a chimera carrying BOTH parents' strong traits. 'mid' is
    the midpoint (less novel, sits on the boundary)."""
    a, b = concepts[a_name], concepts[b_name]
    if mode == "mid":
        return [(x + y) / 2 for x, y in zip(a, b)]
    return [x if abs(x) >= abs(y) else y for x, y in zip(a, b)]   # salient union


def verify_novel(v, concepts, radius):
    """A blend is NOVEL iff its nearest known concept is farther than the cluster radius
    (it occupies empty space -> a genuinely new region, not a relabel of something known).
    Returns (is_novel, nearest_known, distance)."""
    near, d = nearest(v, concepts)
    return d > radius, near, d


def propose(a_name, b_name, concepts, radius, mode="salient"):
    v = blend(a_name, b_name, concepts, mode)
    novel, near, d = verify_novel(v, concepts, radius)
    return {"vector": v, "novel": novel, "nearest": near, "distance": d}


def _demo():
    # features: [wings, lays_eggs, gills, fins]
    concepts = {
        "bird":    [1, 1, 0, 0],
        "fish":    [0, 0, 1, 1],
        "reptile": [0, 1, 0, 0],
        "insect":  [1, 0, 0, 0],
    }
    radius = 1.0   # within ~1.0 of a centroid -> "that's just <known concept>"
    print("=== concept_blend — invent a point outside every known category ===\n")
    print("known concepts (features = [wings, lays_eggs, gills, fins]):")
    for n, c in concepts.items():
        print("   %-8s %s" % (n, c))

    print("\nblend(bird, fish):")
    for mode in ("mid", "salient"):
        r = propose("bird", "fish", concepts, radius, mode)
        tag = "NOVEL (new region)" if r["novel"] else "not novel (near %s)" % r["nearest"]
        print("   %-8s -> %s   nearest=%s d=%.2f  => %s"
              % (mode, r["vector"], r["nearest"], r["distance"], tag))

    r = propose("bird", "fish", concepts, radius, "salient")
    if r["novel"]:
        concepts["chimera"] = r["vector"]
        print("\n  minted new concept 'chimera' =", r["vector"],
              "(wings+eggs+gills+fins) — exists in neither parent, occupies empty space.")
    print("\n  Blend PROPOSES the new region; grounding/verification still decides if it's useful.")


if __name__ == "__main__":
    _demo()
