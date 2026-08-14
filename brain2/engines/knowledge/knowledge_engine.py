#!/usr/bin/env python3
"""
knowledge_engine.py — hardened Knowledge layer (milestone #1, productionized).

The brain_repl demo proved the mechanism: learn facts online, derive the
unstated, explain the chain. This is the hardened version — a clean, validated,
persistent API around the binding memory, with input validation, idempotency,
graceful handling of unknowns and cycles, and save/load.

    kb = KnowledgeEngine()
    kb.learn("alice", "manages", "bob")
    kb.learn("bob", "manages", "carol")
    kb.ask("alice", "manages", hops=2)        -> ("carol", confidence)
    kb.explain("alice", "manages")            -> "alice -> bob -> carol"
    kb.save("org.json"); KnowledgeEngine.load("org.json")

Facts are the source of truth (persisted as JSON); the binding memory is a
derived index, rebuilt deterministically by replay — so save/load is exact and
does not depend on C++ serialization.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

DEFAULT_DIMS = 64


class KnowledgeError(ValueError):
    """Invalid input to the knowledge engine."""


class KnowledgeEngine:
    def __init__(self, n_dims=DEFAULT_DIMS, som=32, seed=1):
        self.n_dims = int(n_dims)
        self._som = int(som)
        self._seed = int(seed)
        self._b = brain2.Brain(som_rows=som, som_cols=som,
                               n_dims=self.n_dims, hidden_dim=128, seed=seed)
        self._b.auto_replay = False
        self._vecs = {}
        self._entities = set()       # tokens seen as subject or object (decode set)
        self.facts = []              # ordered list of (subj, rel, obj), unique
        self._factset = set()        # same facts as a set — O(1) dedup (was O(N)/call)
        self._fact_dict = {}         # fast exact lookup (subj, rel) -> obj

    # ── token -> deterministic concept vector ────────────────────────────────
    def _vec(self, token):
        v = self._vecs.get(token)
        if v is None:
            h = abs(hash(token)) % (2 ** 32)
            v = np.random.default_rng(h).standard_normal(self.n_dims).astype(np.float32)
            self._vecs[token] = v
        return v

    @staticmethod
    def _norm(token):
        if not isinstance(token, str):
            raise KnowledgeError(f"token must be a string, got {type(token).__name__}")
        t = token.strip().lower()
        if not t:
            raise KnowledgeError("token must be non-empty")
        return t

    # ── learning ─────────────────────────────────────────────────────────────
    def learn(self, subj, rel, obj):
        """Store a fact. Idempotent: re-learning the same fact is a no-op."""
        subj, rel, obj = self._norm(subj), self._norm(rel), self._norm(obj)
        if (subj, rel, obj) in self._factset:        # O(1), was O(N) rebuild/call
            return False
        self._b.bind_triple(self._vec(subj), self._vec(rel), self._vec(obj))
        self._entities.update((subj, obj))
        self.facts.append((subj, rel, obj))
        self._factset.add((subj, rel, obj))
        self._fact_dict[(subj, rel)] = obj
        return True

    def _fact_set(self):
        return self._factset

    # ── retrieval ────────────────────────────────────────────────────────────
    def _decode(self, vec):
        v = np.asarray(vec, dtype=np.float32)
        if v.size == 0 or np.linalg.norm(v) < 1e-8 or not self._entities:
            return None, 0.0
        names = sorted(self._entities)
        M = np.stack([self._vec(n) for n in names])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        sims = M @ (v / np.linalg.norm(v))
        i = int(np.argmax(sims))
        return names[i], float(sims[i])

    # A true (subject, relation) match scores ~1.0; a query whose subject is
    # absent but whose relation matches scores ~0.5 (the binding memory averages
    # subject and relation similarity). This threshold rejects those dead-end
    # false positives while keeping exact transitive derivations (~1.0).
    MATCH_THRESHOLD = 0.7

    def ask(self, subj, rel, hops=1, threshold=MATCH_THRESHOLD):
        """Return (object_token, confidence) reachable from subj via rel in
        `hops` steps, or (None, 0.0) if unknown."""
        subj, rel = self._norm(subj), self._norm(rel)
        if (subj, rel) in self._fact_dict:
            return self._fact_dict[(subj, rel)], 1.0
            
        if subj not in self._entities:
            return None, 0.0
        hops = max(1, int(hops))
        vec, conf = self._b.binding_query(self._vec(subj), self._vec(rel),
                                          True, float(threshold), hops)
        tok, sim = self._decode(vec)
        if tok is None or tok == subj:
            return None, 0.0
        return tok, round(float(conf), 4)

    def derive(self, subj, rel, max_hops=8):
        """Reconstruct the single-hop chain from subj following rel. Returns the
        list of tokens [subj, ..., endpoint]; stops on dead end or cycle."""
        subj, rel = self._norm(subj), self._norm(rel)
        chain, cur, seen = [subj], subj, {subj}
        for _ in range(max_hops):
            vec, _ = self._b.binding_query(self._vec(cur), self._vec(rel),
                                           True, self.MATCH_THRESHOLD, 1)
            nxt, _ = self._decode(vec)
            if nxt is None or nxt in seen:
                break
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        return chain

    def explain(self, subj, rel):
        """Human-readable derivation, or None if nothing follows."""
        chain = self.derive(subj, rel)
        if len(chain) < 2:
            return None
        return f"  {(' ' + rel + ' ').join(chain)}"

    def knows(self, subj, rel, obj, max_hops=8):
        """Does subj relate to obj via rel (directly or transitively)?"""
        obj = self._norm(obj)
        chain = self.derive(subj, rel, max_hops=max_hops)
        return obj in chain[1:]

    def all_facts(self):
        return list(self.facts)

    # ── persistence (facts are the source of truth) ──────────────────────────
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"n_dims": self.n_dims, "som": self._som, "seed": self._seed,
                       "facts": self.facts}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        kb = cls(n_dims=d.get("n_dims", DEFAULT_DIMS),
                 som=d.get("som", 32), seed=d.get("seed", 1))
        for s, r, o in d.get("facts", []):
            kb.learn(s, r, o)
        return kb


def _demo():
    kb = KnowledgeEngine()
    for s, r, o in [("alice", "manages", "bob"), ("bob", "manages", "carol"),
                    ("carol", "manages", "dave")]:
        kb.learn(s, r, o)
    print("KnowledgeEngine demo:")
    obj, conf = kb.ask("alice", "manages", hops=3)
    print(f'  ask("alice","manages",hops=3) -> {obj!r} (conf {conf})')
    print(f'  explain: {kb.explain("alice", "manages")}')
    print(f'  knows(alice manages dave)? {kb.knows("alice","manages","dave")}')


if __name__ == "__main__":
    _demo()
