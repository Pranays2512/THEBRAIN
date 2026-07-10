#!/usr/bin/env python3
"""concept_memory.py — from shared structure to NAMED, REUSABLE concept.

factorizer/curiosity_cross discover that two domains share a shape; this store gives the
shape a name and a life-cycle: candidate -> (used PROMOTE_AT times in verified solutions) ->
promoted. Promotion is the statistical admit gate — a concept earns first-class status by
proving reusable, not by being found once. A promoted concept is a hypothesis with a good
track record, NOT a truth: it still goes through verification every time it's used; promotion
changes what the proposer PROPOSES, never what the verifier ACCEPTS. Shape variables are
UPPERCASE strings; recognize() pattern-matches a concrete expr and returns the binding.
(Plan Phase A, Task 11 — the naming/promotion lifecycle factorizer was missing.)"""

import json

PROMOTE_AT = 3


def _match_shape(shape, expr, bind):
    if isinstance(shape, str) and shape.isupper():       # shape variable
        if shape in bind and bind[shape] != expr:
            return None
        bind[shape] = expr
        return bind
    if isinstance(shape, tuple) and isinstance(expr, tuple) and len(shape) == len(expr):
        for s, e in zip(shape, expr):
            if _match_shape(s, e, bind) is None:
                return None
        return bind
    return bind if shape == expr else None


class ConceptMemory:
    def __init__(self, promote_at=PROMOTE_AT):
        self.concepts = {}          # name -> {shape, sources, uses, status}
        self.promote_at = promote_at
        self._n = 0

    def register(self, shape, sources):
        for name, c in self.concepts.items():            # dedupe by exact shape
            if c["shape"] == shape:
                return name
        name = "concept_%d" % self._n
        self._n += 1
        self.concepts[name] = {"shape": shape, "sources": list(sources),
                               "uses": 0, "status": "candidate"}
        return name

    def recognize(self, expr):
        for name, c in self.concepts.items():
            bind = _match_shape(c["shape"], expr, {})
            if bind is not None and bind:
                return name, bind
        return None

    def record_use(self, name):
        c = self.concepts[name]
        c["uses"] += 1
        if c["uses"] >= self.promote_at and c["status"] == "candidate":
            c["status"] = "promoted"

    def status(self, name):
        return self.concepts[name]["status"]

    def save(self, path):
        def enc(e):
            return [enc(x) for x in e] if isinstance(e, tuple) else e
        data = {n: {**c, "shape": enc(c["shape"])} for n, c in self.concepts.items()}
        with open(path, "w") as f:
            json.dump({"n": self._n, "concepts": data}, f, indent=2)

    @classmethod
    def load(cls, path):
        def dec(e):
            return tuple(dec(x) for x in e) if isinstance(e, list) else e
        m = cls()
        with open(path) as f:
            d = json.load(f)
        m._n = d["n"]
        for n, c in d["concepts"].items():
            m.concepts[n] = {**c, "shape": dec(c["shape"])}
        return m
