#!/usr/bin/env python3
"""type_oracle.py — wires event_verify's injected `type_of` to a real taxonomy.

The membrane's selectional check needs a token's TYPE. The crisp source is the `isa` ladder
in `core_knowledge` (hand-checked, transitive): a token's type set is its full isa-closure
(dog -> mammal -> animal -> living_thing). Crisp beats fuzzy clustering here — exact and
explainable — and stays standalone (no C++/GloVe needed to run).

Honesty preserved: an UNKNOWN token returns None, so the membrane ABSTAINS (never guesses) —
exactly the three-valued contract. The `__call__` disposal path is CRISP-ONLY: fuzzy never
decides admit/reject (that would trade honest abstention for a guess). Instead an optional
`similar` hook (nearest-token from semantic_memory/context_embed) feeds `grow()`: fuzzy
CONJECTURES an isa edge -> a verify callback DISPOSES -> the edge is admitted into the crisp
closure. From then on the token disposes exactly, like any hand-checked isa fact. Fuzzy
narrows the teacher's work; it never crosses the membrane.

Usage:  oracle = TypeOracle();  admit(ev, store, oracle, constraints)
        oracle.grow("puppy", verify)   # conjecture->verify->admit into the taxonomy
(Open-language track — closes the 'type_of is a plug point' gap; fuzzy grows, crisp disposes.)"""

from collections import defaultdict


def _isa_closure(triples):
    """token -> frozenset of all isa-ancestors (including itself). Transitive over `isa`."""
    parents = defaultdict(set)
    for s, r, o in triples:
        if r == "isa":
            parents[s].add(o)
    closure = {}

    def anc(tok, seen):
        if tok in closure:
            return closure[tok]
        acc = {tok}
        for p in parents.get(tok, ()):
            if p not in seen:
                acc |= anc(p, seen | {p})
        return acc

    for tok in list(parents):
        closure[tok] = frozenset(anc(tok, {tok}))
    # objects that are only ever ancestors (e.g. "animal") still map to themselves + up
    for o in {o for _, r, o in triples if r == "isa"}:
        if o not in closure:
            closure[o] = frozenset(anc(o, {o}))
    return closure


class TypeOracle:
    def __init__(self, triples=None, similar=None, sim_threshold=0.6):
        if triples is None:
            from core.knowledge.core_knowledge import CORE_FACTS
            triples = CORE_FACTS
        self.closure = _isa_closure(triples)
        self.similar = similar                  # optional token -> [(other, score)] fuzzy hook
        self.sim_threshold = sim_threshold

    def __call__(self, token):
        """CRISP disposal path: frozenset of the token's isa-closure, or None if unknown
        (-> membrane abstains). Fuzzy never answers here; it only feeds grow()."""
        return self.closure.get(token)

    def types(self, token):
        return self.__call__(token)

    # ── fuzzy-proposes / crisp-disposes: taxonomy growth (off the disposal path) ──
    def suggest_parent(self, token):
        """CONJECTURE only: nearest in-taxonomy neighbor above threshold -> (neighbor, score),
        or None. Never used to admit/reject an event — only to propose an isa edge to verify."""
        if self.similar is None:
            return None
        for other, score in self.similar(token):
            if score >= self.sim_threshold and other in self.closure:
                return other, score
        return None

    def admit_isa(self, token, parent):
        """Grow the crisp store after verification: token inherits parent's closure. Returns
        the new type set. (The one place the closure gains a token from outside the taxonomy.)"""
        base = self.closure.get(parent, frozenset({parent}))
        self.closure[token] = frozenset({token} | set(base))
        return self.closure[token]

    def grow(self, token, verify):
        """conjecture (fuzzy neighbor) -> verify (crisp/teacher) -> admit into the taxonomy.
        verify: (token, neighbor) -> bool. Returns the new crisp type set, or None if there was
        no confident neighbor or verification refused (token stays unknown -> abstain)."""
        s = self.suggest_parent(token)
        if s is None:
            return None
        neighbor, _ = s
        if not verify(token, neighbor):
            return None
        return self.admit_isa(token, neighbor)


import math


def build_similar_from_vectors(vectors, vocab, k=5):
    """Generic adapter: `similar(token) -> [(other, cosine)]` ranking `vocab` by cosine to the
    query, using an embedding table `vectors` (token -> sequence of floats). Pure-python, so it
    is standalone-testable; `vectors` can be GloVe, context_embed, anything. Only candidates in
    `vocab` (the taxonomy tokens) are ranked — suggest_parent needs an in-taxonomy neighbor."""
    vocab = [v for v in vocab if v in vectors]

    def _norm(v):
        return math.sqrt(sum(x * x for x in v))

    def similar(token):
        q = vectors.get(token)
        if q is None:
            return []
        qn = _norm(q)
        if qn == 0:
            return []
        out = []
        for t in vocab:
            if t == token:
                continue
            v = vectors[t]
            vn = _norm(v)
            if vn == 0:
                continue
            out.append((t, sum(a * b for a, b in zip(q, v)) / (qn * vn)))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    return similar


def build_similar_from_semantic(sm, vocab, k=5):
    """GloVe-backed adapter over a semantic_memory.SemanticMemory (has `.glove`). Needs venv2
    (brain2 + GloVe); import guarded at the call site. TypeOracle never hard-depends on the
    heavy stack — it only ever receives the resulting `similar` callable."""
    return build_similar_from_vectors(sm.glove, vocab, k=k)
