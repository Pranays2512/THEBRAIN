#!/usr/bin/env python3
"""
scale_knowledge.py — reason over REAL-WORLD knowledge, not toy physics.

Loads the curated world facts (core_knowledge: taxonomy, chemistry, astronomy,
geography) into the ReasoningEngine, sets transitivity, and adds inheritance +
multi-hop rules. Then it answers questions that require CHAINING real facts the way
it chains physics policies — and measures how many facts it can DERIVE beyond the
ones stored (coverage is combinatorial, the same lever as compositional.py, now on
real knowledge).

    python3 scale_knowledge.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core_knowledge import CORE_FACTS
from reasoning_engine import ReasoningEngine


def build():
    re = ReasoningEngine()
    for s, r, o in CORE_FACTS:
        re.learn(s, r, o)
    re.set_transitive("isa")                          # taxonomy ladders climb
    # property inheritance: X isa Y AND Y <prop> Z  =>  X <prop> Z
    for prop in ("has", "can", "does", "is", "lives_in"):
        re.add_rule("isa", prop, prop)
    re.add_rule("capital_of", "in", "continent_of")   # multi-hop geography
    return re


def _demo():
    re = build()
    print("=== scale_knowledge — chaining REAL-WORLD facts ===\n")
    print(f"  loaded {len(CORE_FACTS)} curated facts (taxonomy, chemistry, "
          f"astronomy, geography)\n")

    qs = [
        ("whale", "isa", "animal", "taxonomy chain"),
        ("dog", "has", None, "inherited property"),
        ("sparrow", "can", None, "inherited ability"),
        ("paris", "continent_of", None, "multi-hop: capital -> country -> continent"),
        ("gold", "isa", "material", "metal -> material"),
    ]
    for subj, rel, obj, note in qs:
        if obj is not None:
            hit, path = re.reaches(subj, rel, obj)
            chain = " -> ".join(path) if path else "(stored/derived)"
            print(f"  {subj} {rel} {obj}? {hit}   [{note}]   {chain}")
        else:
            # ask_all fans out over ALL isa-ancestors (single-best ask would jump
            # to the top of the transitive chain and miss where the property lives)
            allv = re.ask_all(subj, rel)
            vals = ", ".join(sorted(allv)) or "(none)"
            print(f"  {subj} {rel}? -> {vals}   [{note}]")
            if allv:
                print(f"      {list(allv.values())[0]}")
    print()

    # coverage: how many isa-ancestors derivable beyond direct facts (combinatorial)
    entities = sorted({s for s, r, _ in CORE_FACTS if r == "isa"})
    derived = sum(len(re.closure(e, "isa")) for e in entities)
    direct = sum(1 for _, r, _ in CORE_FACTS if r == "isa")
    print(f"  isa coverage: {direct} stored 'isa' facts -> {derived} derivable "
          f"ancestor relations ({derived/direct:.1f}x) via transitive closure.")
    print("  Real knowledge, reasoned the same way as physics: chain, inherit, derive.")


if __name__ == "__main__":
    _demo()
