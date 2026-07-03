#!/usr/bin/env python3
"""reading_loop.py — grow language from raw text, not from teacher-made pairs.

The pipeline: template-parse a sentence -> build an Event -> event_verify disposes -> on
ADMIT the parse re-enters template induction. Three disciplines make this safe to run
unattended:

  * Anti-collapse gate — ONLY verified parses (ADMIT) and trusted teacher labels feed
    induction. A rejected/abstained parse never trains the grammar, so the loop can't
    hallucinate itself into a corner.
  * Fragment-level active learning — a sentence the grammar can't parse escalates to the
    teacher; teacher labels buffer per-relation and induce a new template once >=2 exist
    (so anti-unify + a real held-out check run). Teacher touched per-miss, not per-sentence.
  * Measurable decay — escalation rate is tracked; as templates accumulate it must fall.
    That number is the honest 'is the teacher still needed' signal.

(Open-language track, Gap 2 — uses Gap 1's Event + membrane and Gap 3's context stack.)"""

import re
from collections import Counter

from event_form import fact_as_event, event_as_fact, Event
from event_verify import EventStore, admit, ADMIT
from discourse import ContextStack, _PRONOUNS, link_events, _CONNECTIVES
from event_parse import parse_event


class ReadingLoop:
    def __init__(self, tm, store=None, type_of=None, constraints=None, teacher=None):
        self.tm = tm                                   # TemplateMemory (owns entities + grammar)
        self.store = store or EventStore()
        if type_of is None:                            # wire the real isa-taxonomy by default
            try:
                from type_oracle import TypeOracle
                type_of = TypeOracle()
            except Exception:
                type_of = lambda _t: None              # degrade gracefully if taxonomy absent
        self.type_of = type_of
        self.constraints = constraints or {}
        self.teacher = teacher                          # sentence -> label dict, or None
        self.ctx = ContextStack(type_of=self.type_of)
        self.verified = []                              # induction pool (anti-collapse)
        self._buf = {}                                  # rel -> [(sentence, label)] from teacher
        self._eid = 0
        self.stats = Counter()

    def _next_id(self):
        self._eid += 1
        return self._eid

    def _admit_parse(self, entity, rel, value, sentence):
        ev = fact_as_event(entity, rel, value, eid=self._next_id())
        d = admit(ev, self.store, self.type_of, self.constraints)
        self.stats[d] += 1
        if d == ADMIT:
            self.ctx.push_entity(entity)
            self.ctx.push_event(ev.id)
            self.verified.append((sentence, {"entity": entity, "rel": rel, "value": value}))
        return d, ev

    def _escalate(self, sentence):
        """Fragment the grammar couldn't parse -> ask the trusted teacher, buffer, induce."""
        self.stats["escalated"] += 1
        if self.teacher is None:
            return None
        label = self.teacher(sentence)
        self.stats["teacher_calls"] += 1
        if not label:
            return None
        self.tm.entities.add(label["entity"])
        buf = self._buf.setdefault(label["rel"], [])
        buf.append((sentence, label))
        if len(buf) >= 2:                               # enough to anti-unify + hold out
            self.tm.learn(buf[:-1], holdout=[buf[-1]])
        return label

    def read(self, sentence):
        """Read one sentence. Returns (disposition, event_or_None). Dispositions:
        'admit'/'reject'/'abstain' (parsed) or 'escalated' (grammar miss)."""
        self.stats["seen"] += 1
        p = self.tm.parse(sentence)
        if p is not None:
            self.stats["parsed"] += 1
            entity, rel, value = p
            d, ev = self._admit_parse(entity, rel, value, sentence)
            return d, ev
        label = self._escalate(sentence)
        if label is not None:                           # retry once with the new template
            p = self.tm.parse(sentence)
            if p is not None:
                self.stats["parsed"] += 1
                return self._admit_parse(*p, sentence)
        return "escalated", None

    def read_corpus(self, sentences, batch=10):
        """Read many; return per-batch escalation rate (the decay curve) + final stats."""
        curve = []
        for i in range(0, len(sentences), batch):
            before = self.stats["escalated"]
            for s in sentences[i:i + batch]:
                self.read(s)
            chunk = min(batch, len(sentences) - i)
            curve.append((self.stats["escalated"] - before) / max(chunk, 1))
        return {"escalation_curve": curve, "stats": dict(self.stats),
                "facts": [event_as_fact(e) for e in self.store.events]}


class EventReader:
    """Read prose as EVENTS, not just facts: parse -> resolve coref -> membrane -> discourse
    links. A sentence is split on a connective into clauses (each a clause = one event), so
    'A because B' yields two events joined by CAUSE. Pronoun agents/patients resolve against
    the ContextStack before the membrane sees them, so type checks run on the referent."""

    def __init__(self, entities, verbs, store=None, type_of=None, constraints=None):
        self.entities = set(entities)
        self.verbs = set(verbs)
        self.store = store or EventStore()
        if type_of is None:
            try:
                from type_oracle import TypeOracle
                type_of = TypeOracle()
            except Exception:
                type_of = lambda _t: None
        self.type_of = type_of
        self.constraints = constraints or {}
        self.ctx = ContextStack(type_of=self.type_of)
        self.events = []                                # admitted Events
        self.relations = []                            # admitted Relations
        self.stats = Counter()
        self._eid = 0

    def _next_id(self):
        self._eid += 1
        return self._eid

    def _resolve(self, ev):
        """Swap pronoun roles for their referent on the context stack (coref)."""
        a, p = ev.agent, ev.patient
        if a in _PRONOUNS:
            a = self.ctx.resolve(a) or a
        if p in _PRONOUNS:
            p = self.ctx.resolve(p) or p
        return Event(ev.verb, a, p, ev.time, ev.polarity, self._next_id())

    def _read_clause(self, clause):
        ev = parse_event(clause, self.entities, self.verbs, self.type_of)
        if ev is None:
            self.stats["nomatch"] += 1
            return None, None
        ev = self._resolve(ev)
        d = admit(ev, self.store, self.type_of, self.constraints)
        self.stats[d] += 1
        if d == ADMIT:
            for tok in (ev.agent, ev.patient):
                if tok and tok not in _PRONOUNS:
                    self.ctx.push_entity(tok)
            self.ctx.push_event(ev.id)
            self.events.append(ev)
        return d, ev

    def read(self, sentence):
        """Read one sentence. Returns (events, relation_or_None). Splits on the first
        connective into two clauses and links their events with the typed Relation."""
        words = re.findall(r"[a-z']+", sentence.lower())
        ci = next((i for i, w in enumerate(words) if w in _CONNECTIVES), None)
        if ci is None:
            _, ev = self._read_clause(sentence)
            return [e for e in (ev,) if e], None
        conn = words[ci]
        d1, e1 = self._read_clause(" ".join(words[:ci]))
        d2, e2 = self._read_clause(" ".join(words[ci + 1:]))
        rel = None
        if e1 is not None and e2 is not None and d1 == ADMIT and d2 == ADMIT:
            rel = link_events([conn], e1.id, e2.id)
            if rel is not None:
                self.relations.append(rel)
                self.store.add_relation(rel)
        return [e for e in (e1, e2) if e], rel

