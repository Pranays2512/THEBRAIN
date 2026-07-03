#!/usr/bin/env python3
"""event_form.py — the richer logical form that holds open language.

FACT: obj|prop|num and LAW: target=expr are too shallow for prose — no slot for negation,
causality, agent/patient, tense. This is the extension (not replacement): an Event carries
verb + roles + time + polarity, and a Relation typed-links two events (CAUSE/CONTRAST/
SEQUENCE). FACT/LAW remain degenerate cases (a stative Event with patient=value), so
everything downstream re-targets to Events without throwing away the numeric core.

Membrane note: an Event is a CONJECTURE until event_verify admits it. This module only
builds/serializes the shape — it owns no truth. (Open-language track, Gap 1.)"""

from dataclasses import dataclass, field, asdict

POS, NEG = 1, -1                        # polarity
CAUSE, CONTRAST, SEQUENCE = "CAUSE", "CONTRAST", "SEQUENCE"
_REL_KINDS = frozenset((CAUSE, CONTRAST, SEQUENCE))


@dataclass(frozen=True)
class Event:
    verb: str
    agent: str = None
    patient: str = None
    time: str = None                   # coarse tense/marker: past|present|future|None
    polarity: int = POS
    id: int = 0

    def key(self):
        """Identity for contradiction checks: same claim modulo polarity/id."""
        return (self.verb, self.agent, self.patient, self.time)

    def negated(self):
        return Event(self.verb, self.agent, self.patient, self.time, -self.polarity, self.id)


@dataclass(frozen=True)
class Relation:
    kind: str                          # CAUSE|CONTRAST|SEQUENCE
    e1: int                            # Event.id
    e2: int

    def __post_init__(self):
        if self.kind not in _REL_KINDS:
            raise ValueError("unknown relation kind %r" % self.kind)


def fact_as_event(obj, prop, value, eid=0):
    """A FACT is a stative Event: verb=prop, agent=obj, patient=the value (as str)."""
    return Event(verb=prop, agent=obj, patient=str(value), time="present", polarity=POS, id=eid)


def event_as_fact(ev):
    """Inverse for the numeric core: (obj, prop, patient) — only for positive stative events."""
    if ev.polarity != POS:
        return None
    return (ev.agent, ev.verb, ev.patient)


# ── serialization (JSON-friendly) ───────────────────────────────────────────
def dump_event(ev):
    return asdict(ev)


def load_event(d):
    return Event(**d)


def dump_relation(r):
    return asdict(r)


def load_relation(d):
    return Relation(**d)
