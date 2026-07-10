#!/usr/bin/env python3
"""
inductive_engine.py — learn rules from data, keep only the verified ones.

Ideas #1 (hypothesis generation) + #3 (simulation/testing) as one honest loop.
Instead of being TOLD every rule, the brain scans observed episodes for patterns
("B tends to follow A"), proposes provisional rules, then VERIFIES each against a
held-out split — promoting the ones that hold and rejecting coincidences.

    il = InductiveLearner()
    promoted, rejected = il.mine(train_episodes, test_episodes)
    il.promote_into(reasoning_engine, promoted)   # discovered rules become usable

The point is the gate, not the guess: generation is cheap (you can always propose
A -> B); the value is rejecting spurious correlations. A pattern that scores 100%
on training but fails on the hold-out set is superstition, not a rule. Honest
scope: co-occurrence rules over symbolic episodes, with support/confidence
thresholds and a hold-out check — it does not infer causation, only association
that REPLICATES. Real causal discovery needs intervention, not just observation.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.reasoning.reasoning_engine import ReasoningEngine


@dataclass
class Rule:
    a: str
    b: str
    conf_train: float
    conf_test: float
    support: int


def _has(episode, a):
    return a in episode


def _before(episode, a, b):
    return a in episode and b in episode and episode.index(a) < episode.index(b)


class InductiveLearner:
    def candidates(self, train, min_support, min_conf):
        """Provisional rules from training data: A -> B with enough support and
        confidence (= fraction of episodes containing A in which B follows A)."""
        pairs = set()
        for ep in train:
            for i, a in enumerate(ep):
                for b in ep[i + 1:]:
                    if a != b:
                        pairs.add((a, b))
        out = {}
        for a, b in pairs:
            support = sum(_before(ep, a, b) for ep in train)
            seen_a = sum(_has(ep, a) for ep in train)
            conf = support / seen_a if seen_a else 0.0
            if support >= min_support and conf >= min_conf:
                out[(a, b)] = (support, conf)
        return out

    def mine(self, train, test, min_support=2, min_conf=0.8,
             verify_conf=0.7, min_test=2):
        """Propose rules from `train`, verify on held-out `test`. Returns
        (promoted [Rule], rejected [(a, b, reason)])."""
        cand = self.candidates(train, min_support, min_conf)
        promoted, rejected = [], []
        for (a, b), (support, conf) in sorted(cand.items()):
            seen_a = sum(_has(ep, a) for ep in test)
            holds = sum(_before(ep, a, b) for ep in test)
            if seen_a < min_test:
                rejected.append((a, b, f"untested (only {seen_a} hold-out cases)"))
                continue
            conf_test = holds / seen_a
            if conf_test >= verify_conf:
                promoted.append(Rule(a, b, round(conf, 3), round(conf_test, 3), support))
            else:
                rejected.append(
                    (a, b, f"spurious — train {conf:.0%} but hold-out {conf_test:.0%}"))
        return promoted, rejected

    def promote_into(self, engine, promoted, relation="leads_to"):
        """Install verified rules as facts the reasoning engine can chain."""
        for r in promoted:
            engine.learn(r.a, relation, r.b)
        engine.set_transitive(relation)
        return engine


def _demo():
    train = (
        [["rain", "wet_ground", "puddles"]] * 3 +
        [["study", "pass"]] * 3 +
        [["cat", "rainbow"]] * 2        # coincidence in training only
    )
    test = (
        [["rain", "wet_ground", "puddles"]] * 2 +
        [["study", "pass"]] * 2 +
        [["cat", "cloud"], ["cat", "wind"]]   # cat appears, rainbow does NOT
    )

    il = InductiveLearner()
    promoted, rejected = il.mine(train, test)

    print("=== inductive_engine — learn rules, verify, promote ===\n")
    print("Promoted (held up on the hold-out set):")
    for r in promoted:
        print(f"  {r.a} -> {r.b}   (train {r.conf_train:.0%}, hold-out {r.conf_test:.0%})")
    print("\nRejected:")
    for a, b, why in rejected:
        print(f"  {a} -> {b}   ({why})")

    # discovered rules become usable knowledge
    re = il.promote_into(ReasoningEngine(), promoted)
    ok, path = re.reaches("rain", "leads_to", "puddles")
    print("\nReasoning with the DISCOVERED rules:")
    print(f"  does rain lead to puddles?  -> {'yes' if ok else 'no'}", end="")
    if ok:
        print(f"   ({' -> '.join(path)})")
    else:
        print()


if __name__ == "__main__":
    _demo()
