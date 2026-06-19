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
from collections import defaultdict, deque

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

    # ── multi-valued inference (ALL bindings, not first) ─────────────────────
    # ask() returns one answer (first binding it finds). But a subject can have
    # many: two parents -> two grandparents; a rule premise that matches several
    # mids fans out to several conclusions. ask_all enumerates EVERY derivable
    # object, exploring all premise bindings instead of committing to the first.
    def ask_all(self, subj, rel, max_depth=8):
        """All objects derivable for (subj, rel) via direct facts, transitive
        closure, and composition rules — the multi-valued ask(). Returns
        {object: explanation}, one shortest explanation per object."""
        subj, rel = KnowledgeEngine._norm(subj), KnowledgeEngine._norm(rel)
        return self._ask_all(subj, rel, max_depth, frozenset())

    def _ask_all(self, subj, rel, depth, seen):
        if depth <= 0 or (subj, rel) in seen:
            return {}
        seen = seen | {(subj, rel)}
        out = {}                                     # setdefault keeps first (shortest) reason

        # 1. transitive closure (already multi-valued)
        if rel in self.transitive:
            for o, path in self.closure(subj, rel).items():
                out.setdefault(o, f"{(' ' + rel + ' ').join(path)}  (transitive)")

        # 2. ALL direct facts for (subj, rel), not just the best-recalled one
        for o in sorted(self._adjacency(rel).get(subj, ())):
            out.setdefault(o, f"{subj} {rel} {o}  (direct)")

        # 3. composition rules, fanning out over EVERY premise binding
        for prem1, prem2, concl in self.rules:
            if concl != rel:
                continue
            for mid in self._ask_all(subj, prem1, depth - 1, seen):
                for z in self._ask_all(mid, prem2, depth - 1, seen):
                    out.setdefault(z, (f"{subj} {prem1} {mid} AND {mid} {prem2} {z}  =>  "
                                       f"{subj} {rel} {z}   [rule {prem1}∘{prem2}→{rel}]"))
        return out

    # ── multi-parent transitive closure (DAG, not single chain) ──────────────
    # KnowledgeEngine.derive and binding-memory recall return ONE best object
    # per (subject, relation) — fine for a chain (a -> b -> c), wrong for real
    # taxonomies where a concept has MANY parents (dog isa pet, mammal, canine).
    # Closure traverses the multi-valued FACT graph: every ancestor, each with
    # a shortest derivation path. This is the honest fix the ConceptNet demo
    # surfaced, now native in the engine instead of re-coded per demo.
    def _adjacency(self, rel):
        """(subject) -> set(objects) for one relation, from the fact graph.
        Multi-valued: the ground truth the single-best recall can't represent."""
        adj = defaultdict(set)
        for s, r, o in self.kb.facts:
            if r == rel:
                adj[s].add(o)
        return adj

    def closure(self, subj, rel, max_nodes=100000):
        """Every object reachable from `subj` by following `rel` transitively,
        each mapped to a SHORTEST derivation path [subj, ..., obj].

        BFS over the multi-valued fact graph. The graph may contain cycles, so
        a visited set is required for termination. Excludes `subj` itself.
        Returns {obj: [subj, ..., obj]}.
        """
        subj, rel = KnowledgeEngine._norm(subj), KnowledgeEngine._norm(rel)
        adj = self._adjacency(rel)
        # paths[node] holds the shortest path discovered to `node`.
        paths = {subj: [subj]}

        # BFS: the first time a node is reached is via a shortest path, so the
        # explanation shows the most direct ancestry. `paths` is also the
        # visited set, which makes cyclic graphs terminate.
        frontier = deque([subj])
        while frontier and len(paths) <= max_nodes:
            node = frontier.popleft()
            for nbr in sorted(adj.get(node, ())):     # sorted -> deterministic
                if nbr not in paths:
                    paths[nbr] = paths[node] + [nbr]
                    frontier.append(nbr)

        paths.pop(subj, None)          # report ancestors, not the start node
        return paths

    def reaches(self, subj, rel, target):
        """(bool, path) — is `target` a transitive `rel`-ancestor of `subj`,
        and the shortest chain that shows it (e.g. dog -> pet -> animal)."""
        target = KnowledgeEngine._norm(target)
        paths = self.closure(subj, rel)
        return (target in paths), paths.get(target)

    def derive_all(self, subj, rel):
        """All transitive `rel`-ancestors of `subj`, sorted — the multi-parent
        generalization of KnowledgeEngine.derive (which follows one chain)."""
        return sorted(self.closure(subj, rel))

    # ── causal / process chains (how / why) ──────────────────────────────────
    # A process is a chain on a single causal relation (e.g. leads_to): each
    # step causes the next. "How does X happen?" walks BACKWARD (what produces
    # X); "how does X help?" walks FORWARD (what X causes). Same traversal, the
    # adjacency just reversed. Returns the steps in story order (cause first).
    def _bfs_paths(self, start, adj):
        paths = {start: [start]}
        frontier = deque([start])
        while frontier:
            node = frontier.popleft()
            for nbr in sorted(adj.get(node, ())):
                if nbr not in paths:
                    paths[nbr] = paths[node] + [nbr]
                    frontier.append(nbr)
        paths.pop(start, None)
        return paths

    def process_chain(self, topic, rel, direction="forward"):
        """The longest causal chain through `topic` on relation `rel`, in story
        order (cause -> ... -> effect). direction='forward' follows rel out of
        `topic` (its effects); 'backward' follows rel into `topic` (its causes).
        Returns [] if no chain. Linear processes give the obvious full chain."""
        topic, rel = KnowledgeEngine._norm(topic), KnowledgeEngine._norm(rel)
        fwd, rev = defaultdict(set), defaultdict(set)
        for s, r, o in self.kb.facts:
            if r == rel:
                fwd[s].add(o)
                rev[o].add(s)
        if direction == "forward":
            paths = self._bfs_paths(topic, fwd)
            if not paths:
                return []
            end = max(paths, key=lambda k: len(paths[k]))
            return paths[end]                           # [topic, ..., effect]
        paths = self._bfs_paths(topic, rev)
        if not paths:
            return []
        root = max(paths, key=lambda k: len(paths[k]))
        return list(reversed(paths[root]))              # [root, ..., topic]

    def process_branches(self, topic, rel):
        """Every forward causal chain from `topic` to a leaf (a node with no
        further `rel`-edge) — the branches of "in which ways does X ...?".
        Each chain is in story order [topic, ..., leaf]."""
        topic, rel = KnowledgeEngine._norm(topic), KnowledgeEngine._norm(rel)
        fwd = defaultdict(set)
        for s, r, o in self.kb.facts:
            if r == rel:
                fwd[s].add(o)
        paths = self._bfs_paths(topic, fwd)
        leaves = [n for n in paths if not fwd.get(n)]   # ends of each branch
        return [paths[leaf] for leaf in sorted(leaves, key=lambda n: paths[n])]

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
