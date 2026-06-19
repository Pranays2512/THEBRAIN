#!/usr/bin/env python3
"""
discovery.py — the discovery cycle: two proposers, one verifier, one memory.

Ties the quartet together. Induction proposes rules from observed data; analogy
proposes rules by transferring structure from a known domain. BOTH hypotheses
pass the same verify-before-promote gate (held-out / experimental evidence), and
the survivors install into ONE reasoning engine — so discovered and borrowed
knowledge become equally trustworthy, because both earned it.

    ds = DiscoverySystem()
    ds.discover(episodes_train, episodes_test, source_facts, target_facts, experiments)
    ds.engine.reaches("switch", "leads_to", "current")   # uses what it discovered

Honest scope: proposes genuinely (mine patterns, transfer across domains) but
trusts nothing unverified. No causation claims, no open-ended invention — every
promoted rule replicated on held-out data or was confirmed by an experiment.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from reasoning_engine import ReasoningEngine
from inductive_engine import InductiveLearner, _before, _has
from analogy_engine import AnalogyEngine


class DiscoverySystem:
    def __init__(self):
        self.engine = ReasoningEngine()
        self.ind = InductiveLearner()
        self.ana = AnalogyEngine()

    def _confirmed(self, a, b, episodes, min_support=2, min_conf=0.7):
        """Does evidence support a -> b? (the verify gate, reused for analogy)."""
        support = sum(_before(ep, a, b) for ep in episodes)
        seen = sum(_has(ep, a) for ep in episodes)
        return seen > 0 and support >= min_support and support / seen >= min_conf

    def discover(self, train, test, source_facts=None, target_facts=None, experiments=None):
        """Run both proposers through the verifier; install survivors. Returns
        (mined, transferred_ok, transferred_unverified)."""
        # proposer 1: induction from observed episodes
        mined, _ = self.ind.mine(train, test)
        for r in mined:
            self.engine.learn(r.a, "leads_to", r.b)

        # proposer 2: analogy transfer, each transferred fact verified by experiment
        transferred_ok, transferred_no = [], []
        if source_facts and target_facts:
            _, transfers = self.ana.map_domains(source_facts, target_facts)
            evidence = experiments or []        # targeted experiments, not unrelated data
            for ts, rel, to, src in transfers:
                if self._confirmed(ts, to, evidence):
                    self.engine.learn(ts, rel, to)
                    transferred_ok.append((ts, rel, to, src))
                else:
                    transferred_no.append((ts, rel, to, src))

        self.engine.set_transitive("leads_to")
        return mined, transferred_ok, transferred_no


def _demo():
    # observed temporal data in the electricity domain (for induction)
    train = ([["switch", "battery", "current"]] * 3 + [["short", "spark"]] * 3)
    test = ([["switch", "battery", "current"]] * 2 + [["short", "spark"]] * 2)
    # structural facts of two domains (for analogy)
    water = [("pump", "increases", "flow"), ("pipe", "resists", "flow"),
             ("flow", "depends_on", "pressure"), ("pump", "raises", "pressure")]
    elec = [("battery", "increases", "current"), ("resistor", "resists", "current"),
            ("current", "depends_on", "voltage")]
    # experiments confirming the analogical prediction battery -> voltage
    experiments = [["battery", "voltage"]] * 3

    ds = DiscoverySystem()
    mined, ok, no = ds.discover(train, test, water, elec, experiments)

    print("=== discovery — induction + analogy, one verifier ===\n")
    print("Mined from data (verified on hold-out):")
    for r in mined:
        print(f"  {r.a} -> {r.b}")
    print("\nTransferred by analogy + confirmed by experiment:")
    for ts, rel, to, src in ok:
        print(f"  {ts} {rel} {to}   (analog of {src[0]} {src[1]} {src[2]})")
    if no:
        print("\nTransferred but UNconfirmed (kept as hypothesis, not promoted):")
        for ts, rel, to, src in no:
            print(f"  {ts} {rel} {to}")

    print("\nReasoning with everything it discovered:")
    ok2, path = ds.engine.reaches("switch", "leads_to", "current")
    print(f"  switch leads to current? -> {'yes' if ok2 else 'no'}"
          + (f"  ({' -> '.join(path)})" if ok2 else ""))
    ans, _ = ds.engine.ask("battery", "raises")
    print(f"  what does a battery raise? -> {ans}   (discovered by analogy)")


if __name__ == "__main__":
    _demo()
