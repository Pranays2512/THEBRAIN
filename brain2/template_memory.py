#!/usr/bin/env python3
"""template_memory.py — durable grammar store with the conjecture->verify->admit gate.

Mirrors PolicyMemory/PolicyLearner: induction PROPOSES a template (slotting / anti-unify),
verification against held-out labeled examples DISPOSES. Only verified templates are stored;
parsing is exact and explainable. Statement templates carry entity+value slots; question
templates carry an entity slot and answer with the rel. (Plan Phase A, Tasks 3/6.)"""

import json

from parse_template import Template, tokenize, match, slot_example, anti_unify


class TemplateMemory:
    def __init__(self, entities=None):
        self.templates = []                     # list[Template], most-literal first
        self.entities = set(entities or ())

    # ── use ──────────────────────────────────────────────────────────────────
    def parse(self, sentence):
        toks = tokenize(sentence, normalized=True)
        for t in self.templates:
            b = match(t, toks, self.entities)
            if b is not None and "entity" in b and "value" in b:
                return b["entity"], t.rel, b["value"]
        return None

    def parse_question(self, sentence):
        toks = tokenize(sentence, normalized=True)
        for t in self.templates:
            b = match(t, toks, self.entities)
            if b is not None and "entity" in b and "value" not in b:
                return b["entity"], t.rel
        return None

    # ── learn: conjecture -> verify -> admit ─────────────────────────────────
    def _verified(self, t, examples):
        for sent, lab in examples:
            b = match(t, tokenize(sent, normalized=True), self.entities)
            if b is None or b.get("entity") != tokenize(lab["entity"], normalized=True)[0] \
                    or float(b.get("value", float("nan"))) != float(lab["value"]) \
                    or t.rel != lab["rel"]:
                return False
        return True

    def _q_verified(self, t, examples):
        for sent, lab in examples:
            b = match(t, tokenize(sent, normalized=True), self.entities)
            if b is None or b.get("entity") != tokenize(lab["entity"], normalized=True)[0] \
                    or t.rel != lab["rel"]:
                return False
        return True

    def _admit(self, conjectures, examples, holdout, verify):
        for i in range(len(conjectures)):
            for j in range(i + 1, len(conjectures)):
                g = anti_unify(conjectures[i], conjectures[j])
                if g is not None:
                    conjectures.append(g)
        admitted = 0
        for t in conjectures:
            if t not in self.templates and verify(t, examples) and \
               any(verify(t, [h]) for h in holdout):
                self.templates.append(t)
                admitted += 1
        # most-literal first: fewer wildcards = higher precision (shadowing guard)
        self.templates.sort(key=lambda t: sum(1 for it in t.items if it[0] == "any"))
        return admitted

    def learn(self, examples, holdout):
        return self._admit([slot_example(s, l) for s, l in examples],
                           examples, holdout, self._verified)

    def _q_slot(self, sentence, label):
        toks = tokenize(sentence, normalized=True)
        ent = tokenize(label["entity"], normalized=True)[0]
        items = tuple(("slot", "entity", "entity") if tok == ent else ("w", tok) for tok in toks)
        return Template(rel=label["rel"], items=items)

    def learn_question(self, examples, holdout):
        return self._admit([self._q_slot(s, l) for s, l in examples],
                           examples, holdout, self._q_verified)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path):
        data = [{"rel": t.rel, "items": [list(it) for it in t.items]} for t in self.templates]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path, entities=None):
        m = cls(entities=entities)
        with open(path) as f:
            data = json.load(f)
        for d in data:
            m.templates.append(Template(d["rel"], tuple(tuple(it) for it in d["items"])))
        return m
