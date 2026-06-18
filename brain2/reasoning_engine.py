#!/usr/bin/env python3
"""
reasoning_engine.py — hardened Reasoning layer (milestone #2, productionized).

The Knowledge layer retrieves facts and chains a SINGLE relation (transitive
closure). Reasoning adds what it can't: COMPOSITION RULES across DIFFERENT
relations — "X parent Y and Y parent Z => X grandparent Z" — derived by
backward chaining and explained with the rule that fired.

    re = ReasoningEngine()
    re.learn("tom", "parent", "sam"); re.learn("sam", "parent", "kid")
    re.add_rule("parent", "parent", "grandparent")
    re.ask("tom", "grandparent")    -> ("kid", "tom parent sam AND sam parent kid => ...")

Builds on the hardened KnowledgeEngine (facts grounded in the binding memory);
the rule layer composes relations on top. Same hardening discipline: input
validation, cycle safety, persistence, explanations.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from knowledge_engine import KnowledgeEngine, KnowledgeError


class ReasoningEngine:
    def __init__(self, kb=None):
        self.kb = kb or KnowledgeEngine()
        self.rules = []          # (a, b, c): X a Y & Y b Z  =>  X c Z
        self.transitive = set()  # relations r where r chains: X r Y r Z => X r Z

    # ── facts + rules ────────────────────────────────────────────────────────
    def learn(self, subj, rel, obj):
        return self.kb.learn(subj, rel, obj)

    def add_rule(self, prem1, prem2, concl):
        """X prem1 Y AND Y prem2 Z  =>  X concl Z."""
        for r in (prem1, prem2, concl):
            KnowledgeEngine._norm(r)
        rule = (prem1.strip(), prem2.strip(), concl.strip())
        if rule not in self.rules:
            self.rules.append(rule)

    def set_transitive(self, rel):
        self.transitive.add(KnowledgeEngine._norm(rel))

    # ── inference (backward chaining) ────────────────────────────────────────
    def ask(self, subj, rel, max_depth=8):
        """Return (object, explanation) for `subj rel ?`, deriving it through
        transitive closure and composition rules, or (None, None)."""
        subj, rel = KnowledgeEngine._norm(subj), KnowledgeEngine._norm(rel)
        return self._ask(subj, rel, max_depth, set())

    def _ask(self, subj, rel, depth, seen):
        if depth <= 0 or (subj, rel) in seen:
            return None, None
        seen = seen | {(subj, rel)}

        # 1. transitive same-relation chaining (binding memory's strength)
        if rel in self.transitive:
            chain = self.kb.derive(subj, rel)
            if len(chain) >= 2:
                return chain[-1], f"{(' ' + rel + ' ').join(chain)}  (transitive)"

        # 2. a direct stored fact
        obj, _ = self.kb.ask(subj, rel, hops=1)
        if obj is not None:
            return obj, f"{subj} {rel} {obj}  (direct)"

        # 3. composition rules: X a Y AND Y b Z  =>  X rel Z
        for prem1, prem2, concl in self.rules:
            if concl != rel:
                continue
            mid, _ = self._ask(subj, prem1, depth - 1, seen)
            if mid is None:
                continue
            z, _ = self._ask(mid, prem2, depth - 1, seen)
            if z is not None:
                return z, (f"{subj} {prem1} {mid} AND {mid} {prem2} {z}  =>  "
                           f"{subj} {rel} {z}   [rule {prem1}∘{prem2}→{rel}]")
        return None, None

    def explain(self, subj, rel):
        _, why = self.ask(subj, rel)
        return why

    def knows(self, subj, rel, obj):
        ans, _ = self.ask(subj, rel)
        return ans == KnowledgeEngine._norm(obj)

    # ── persistence (facts + rules) ──────────────────────────────────────────
    def save(self, path):
        import json
        self.kb.save(path + ".kb.json")
        with open(path, "w") as f:
            json.dump({"rules": self.rules, "transitive": sorted(self.transitive)}, f, indent=2)

    @classmethod
    def load(cls, path):
        import json
        re = cls(kb=KnowledgeEngine.load(path + ".kb.json"))
        with open(path) as f:
            d = json.load(f)
        re.rules = [tuple(r) for r in d.get("rules", [])]
        re.transitive = set(d.get("transitive", []))
        return re


def _demo():
    re = ReasoningEngine()
    for s, r, o in [("tom", "parent", "sam"), ("sam", "parent", "kid"),
                    ("kid", "parent", "baby")]:
        re.learn(s, r, o)
    re.add_rule("parent", "parent", "grandparent")
    re.add_rule("parent", "grandparent", "great_grandparent")
    print("ReasoningEngine demo:")
    for rel in ("grandparent", "great_grandparent"):
        ans, why = re.ask("tom", rel)
        print(f'  tom {rel}? -> {ans}')
        print(f'    because: {why}')


if __name__ == "__main__":
    _demo()
