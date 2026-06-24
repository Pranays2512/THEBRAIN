#!/usr/bin/env python3
"""
scale_test.py — where does the brain break? ConceptNet at scale + SOM growth.

Honest scaling measurement (the #3 weak-spot probe):
  * ConceptNet: stream up to 100k real English assertions into the ReasoningEngine;
    time loading and graph reasoning (transitive closure / inheritance) at 10k/50k/
    100k. Graph reasoning uses the fact adjacency (should scale ~linearly); the
    fuzzy binding-memory recall (ask) is the part that degrades — measured separately.
  * SOM: find_bmu is brute-force O(N_neurons). Time it across map sizes to show the
    wall (correctness-first; an approximate index would be needed past some size).

    venv2/bin/python3 scale_test.py
"""

import gzip
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from reasoning_engine import ReasoningEngine

CN = os.path.join(os.path.dirname(__file__), "train", "conceptnet-assertions-5.7.0.csv.gz")


def stream_en(limit):
    out = []
    with gzip.open(CN, "rt", encoding="utf-8") as f:
        for line in f:
            c = line.split("\t")
            if len(c) < 4 or not (c[2].startswith("/c/en/") and c[3].startswith("/c/en/")):
                continue
            sp, op = c[2].split("/"), c[3].split("/")
            if len(sp) < 4 or len(op) < 4:
                continue
            out.append((sp[3], c[1].split("/")[-1].lower(), op[3]))
            if len(out) >= limit:
                break
    return out


def conceptnet_scale():
    print("=== ConceptNet at scale (load + graph reasoning) ===\n")
    print(f"  {'facts':>8s} {'load s':>8s} {'entities':>9s} {'closure ms/q':>13s} "
          f"{'avg ancestors':>14s}")
    for N in (10_000, 50_000, 100_000):
        triples = stream_en(N)
        t0 = time.time()
        re = ReasoningEngine()
        for s, r, o in triples:
            re.learn(s, r, o)
        re.set_transitive("isa")
        load = time.time() - t0
        ents = {s for s, _, _ in triples} | {o for _, _, o in triples}
        subs = [s for s, r, _ in triples if r == "isa"]
        sample = random.Random(0).sample(subs, min(200, len(subs))) if subs else []
        t0 = time.time()
        total_anc = sum(len(re.closure(s, "isa")) for s in sample)
        qms = (time.time() - t0) * 1000 / max(len(sample), 1)
        print(f"  {len(triples):>8} {load:>8.2f} {len(ents):>9} {qms:>13.2f} "
              f"{total_anc / max(len(sample),1):>14.1f}")


def som_scale():
    import brain2
    import numpy as np
    print("\n=== SOM find_bmu (brute-force O(N_neurons)) ===\n")
    print(f"  {'map':>9s} {'neurons':>8s} {'ms/find_bmu':>12s}")
    rng = np.random.default_rng(0)
    for side in (16, 32, 64, 128, 256):
        som = brain2.SOM(side, side, 32)
        vs = [rng.standard_normal(32).astype("float32") for _ in range(200)]
        t0 = time.time()
        for v in vs:
            som.find_bmu(v)
        ms = (time.time() - t0) * 1000 / len(vs)
        print(f"  {side}x{side:<5} {side*side:>8} {ms:>12.3f}")
    print("\n  O(N): ~16x neurons -> ~16x time. Past ~1M neurons brute force is the wall;")
    print("  an approximate index (LSH/graph) would be needed — at the cost of exact BMU.")


if __name__ == "__main__":
    conceptnet_scale()
    try:
        som_scale()
    except Exception as e:
        print(f"\n(SOM scale skipped: {e})")
