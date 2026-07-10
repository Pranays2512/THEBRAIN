#!/usr/bin/env python3
"""
semantic_memory.py — the memory doing what a dict cannot.

The binding memory's strength is similarity-based retrieval, but with random
token vectors that strength is dead (everything is exact-match, so a dict wins).
This gives it REAL embeddings (GloVe): now semantically-near tokens are close in
vector space, so it GENERALIZES — answer a query about "car" from a fact stored
about "automobile", because they mean nearly the same thing. A dict returns
nothing; the vector memory returns the answer.

    sm = SemanticMemory()
    sm.learn("automobile", "has", "engine")
    sm.ask("car", "has")          -> ("engine", confidence)   # never stored "car"
    sm.similar("car")             -> ["automobile", "vehicle", ...]

Honest limit: it is only as good as the embeddings. GloVe-50 captures strong
synonymy (car~automobile, dog~puppy) but is noisy on subtler pairs and (a known
quirk) rates antonyms as similar. Better embeddings -> better generalization.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

GLOVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glove.6B.50d.txt")
N_DIMS = 50
_GLOVE = None       # module-level cache (load once)


def _load_glove():
    global _GLOVE
    if _GLOVE is None:
        g = {}
        with open(GLOVE_PATH, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                g[p[0]] = np.array(p[1:], dtype=np.float32)
        _GLOVE = g
    return _GLOVE


class SemanticError(ValueError):
    pass


class SemanticMemory:
    # score = (subject_sim + relation_sim) / 2. With the relation matched
    # exactly (sim 1.0), a 0.8 threshold requires subject_sim >= 0.6 — i.e. only
    # genuinely-similar subjects generalize; unrelated ones (sim < 0.6) are
    # rejected. Tuned against GloVe-50.
    MATCH_THRESHOLD = 0.8

    def __init__(self, threshold=MATCH_THRESHOLD):
        self.threshold = threshold
        self.glove = _load_glove()
        self.b = brain2.Brain(som_rows=32, som_cols=32, n_dims=N_DIMS, hidden_dim=128, seed=1)
        self.b.auto_replay = False
        self._tokens = set()
        self.facts = []

    def _vec(self, token):
        v = self.glove.get(token.lower())
        if v is None:                          # OOV -> deterministic random
            h = abs(hash(token)) % (2 ** 32)
            v = np.random.default_rng(h).standard_normal(N_DIMS).astype(np.float32)
        return v

    @staticmethod
    def _norm(t):
        if not isinstance(t, str) or not t.strip():
            raise SemanticError("token must be a non-empty string")
        return t.strip()

    def learn(self, subj, rel, obj):
        subj, rel, obj = self._norm(subj), self._norm(rel), self._norm(obj)
        if (subj, rel, obj) in set(self.facts):
            return False
        self.b.bind_triple(self._vec(subj), self._vec(rel), self._vec(obj))
        self._tokens.update((subj, obj))
        self.facts.append((subj, rel, obj))
        return True

    def _decode(self, vec):
        v = np.asarray(vec, dtype=np.float32)
        if v.size == 0 or np.linalg.norm(v) < 1e-8 or not self._tokens:
            return None
        names = sorted(self._tokens)
        M = np.stack([self._vec(n) for n in names])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        return names[int(np.argmax(M @ (v / np.linalg.norm(v))))]

    def ask(self, subj, rel):
        """Return (object, confidence) for subj rel ?, matching SEMANTICALLY
        near subjects/relations; (None, 0.0) if nothing is similar enough."""
        subj, rel = self._norm(subj), self._norm(rel)
        vec, conf = self.b.binding_query(self._vec(subj), self._vec(rel),
                                         True, self.threshold, 1)
        if conf is None or conf < self.threshold:
            return None, 0.0
        tok = self._decode(vec)
        return (tok, round(float(conf), 3)) if tok else (None, 0.0)

    def save(self, path):
        """Persist the learned triples; vector bindings rebuild on load by replaying them,
        so a session's associative memory survives restart."""
        import json
        json.dump(self.facts, open(path, "w"))

    def replay(self, triples):
        """Re-learn a list of (subj, rel, obj) triples (used on load)."""
        for s, r, o in triples:
            self.learn(s, r, o)
        return self

    def similar(self, token, k=5):
        """Top-k known tokens semantically closest to `token` (what a dict
        can't do)."""
        token = self._norm(token)
        q = self._vec(token); qn = np.linalg.norm(q)
        if qn < 1e-8 or not self._tokens:
            return []
        cands = [t for t in self._tokens if t != token]
        sims = [(t, float(self._vec(t) @ q / (np.linalg.norm(self._vec(t)) * qn)))
                for t in cands]
        sims.sort(key=lambda x: -x[1])
        return [t for t, _ in sims[:k]]


def _demo():
    sm = SemanticMemory()
    for s, r, o in [("automobile", "has", "engine"), ("dog", "has", "tail"),
                    ("doctor", "treats", "patients")]:
        sm.learn(s, r, o)
    print("SemanticMemory demo (generalizes via meaning, never stored the query word):")
    for s, r in [("car", "has"), ("puppy", "has"), ("physician", "treats"), ("apple", "has")]:
        obj, conf = sm.ask(s, r)
        print(f"  ask({s!r},{r!r}) -> {obj!r}  (conf {conf})")
    print(f"  similar('car') -> {sm.similar('car', 3)}")


if __name__ == "__main__":
    _demo()
