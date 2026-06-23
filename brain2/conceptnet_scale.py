#!/usr/bin/env python3
"""
conceptnet_scale.py — reason over a BIGGER real-world corpus (ConceptNet 5.7).

Loads ~1.5k real English ConceptNet triples into the ReasoningEngine and reasons
the same way as the curated seed: transitive taxonomy, property inheritance,
multi-hop chaining — now at 20x the scale. Reports the corpus shape, query latency,
and how many facts are DERIVABLE beyond the stored ones (coverage).

    python3 conceptnet_scale.py
"""

import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from reasoning_engine import ReasoningEngine

DATA = os.path.join(os.path.dirname(__file__), "train", "conceptnet_en_subset.json")


def build():
    facts = json.load(open(DATA))
    re = ReasoningEngine()
    for s, r, o in facts:
        re.learn(s, r, o)
    rels = Counter(r for _, r, _ in facts)
    # taxonomy relations are transitive; properties inherit down them
    tax = [r for r in ("isa", "is_a", "IsA", "type_of", "kind_of") if r in rels]
    for t in tax:
        re.set_transitive(t)
        for prop in ("can", "has", "used_for", "part_of", "at_location", "desires"):
            if prop in rels:
                re.add_rule(t, prop, prop)
    return re, facts, rels, tax


def _demo():
    re, facts, rels, tax = build()
    ents = {s for s, _, _ in facts} | {o for _, _, o in facts}
    print("=== conceptnet_scale — reasoning over ~1.5k real-world facts ===\n")
    print(f"  {len(facts)} triples, {len(ents)} concepts, {len(rels)} relations")
    print(f"  top relations: {dict(rels.most_common(6))}")
    print(f"  taxonomy (transitive): {tax or '(none named isa)'}\n")

    # sample queries: inherited properties + multi-hop, timed
    tax0 = tax[0] if tax else None
    samples = [s for s, r, _ in facts if r == tax0][:4] if tax0 else []
    t0 = time.time()
    for subj in samples:
        anc = re.derive_all(subj, tax0)
        cans = re.ask_all(subj, "can") if "can" in rels else {}
        line = f"  {subj}: {tax0}-ancestors={anc[:4]}"
        if cans:
            line += f"  inherited can={list(cans)[:3]}"
        print(line)
    dt = (time.time() - t0) * 1000

    # coverage: derivable taxonomy ancestors beyond stored
    if tax0:
        stored = sum(1 for _, r, _ in facts if r == tax0)
        subjects = {s for s, r, _ in facts if r == tax0}
        derived = sum(len(re.closure(s, tax0)) for s in subjects)
        print(f"\n  {tax0}: {stored} stored -> {derived} derivable ancestor relations "
              f"({derived/max(stored,1):.1f}x)")
    print(f"  {len(samples)} multi-hop inheritance queries in {dt:.1f} ms "
          f"(~{dt/max(len(samples),1):.1f} ms each) — scales fine.")


if __name__ == "__main__":
    _demo()
