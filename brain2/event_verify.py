#!/usr/bin/env python3
"""event_verify.py — the membrane for events. The numeric gate does not apply to prose, so
events get a WEAKER but still crisp contract:

  1. Polarity non-contradiction — the store may not hold EVENT(v,a,p,t,+) and (v,a,p,t,-).
  2. Selectional type constraints — a verb restricts the type of its agent/patient
     (type_of maps a token to a type via SOM clusters / semantic memory, injected).

Three-valued, exactly like domain_features' dimensional filter: admit / reject / abstain.
ABSTAIN is the load-bearing case — a CONSTRAINED verb whose role type we don't yet know is
held, never guessed. (An UNCONSTRAINED verb has no selectional claim to check, so a
numeric-core fact flows straight through.) Abstained events do not enter the truth store and
do not contradict; they are the escalation queue for the reading loop.

Fuzzy proposes an Event; this disposes. (Open-language track, Gap 1 — contract before scale.)"""

ADMIT, REJECT, ABSTAIN = "admit", "reject", "abstain"


class EventStore:
    """Crisp store of admitted events + typed relations, keyed for contradiction checks."""

    def __init__(self):
        self.events = []                       # admitted Event objects
        self.relations = []                    # admitted Relation objects
        self._by_key = {}                      # Event.key() -> polarity already present

    def contradicts(self, ev):
        seen = self._by_key.get(ev.key())
        return seen is not None and seen != ev.polarity

    def has(self, ev):
        return self._by_key.get(ev.key()) == ev.polarity

    def _commit(self, ev):
        self.events.append(ev)
        self._by_key[ev.key()] = ev.polarity

    def add_relation(self, r):
        self.relations.append(r)


def check_types(ev, type_of, constraints):
    """True (nothing to object to: verb unconstrained, or every checkable role satisfies its
    restriction), False (a known role type is disallowed -> reject), None (a CONSTRAINED verb
    has a role whose type is unknown -> abstain and escalate, never guess)."""
    spec = constraints.get(ev.verb)
    if spec is None:
        return True                            # no selectional claim to check; numeric core vouches
    unknown = False
    for role in ("agent", "patient"):
        tok = getattr(ev, role)
        allowed = spec.get(role)
        if tok is None or allowed is None:
            continue
        typ = type_of(tok)
        # type_of may return a single type (str) or a set/closure of types (isa chain).
        types = ({typ} if isinstance(typ, str) else set(typ)) if typ else set()
        if not types:
            unknown = True                     # cannot judge this role yet
            continue
        if not (set(allowed) & types):
            return False
    return None if unknown else True


def classify(ev, store, type_of, constraints):
    """The disposition of a single conjectured event, before any commit."""
    if store.contradicts(ev):
        return REJECT
    t = check_types(ev, type_of, constraints)
    if t is False:
        return REJECT
    if t is None:
        return ABSTAIN                         # unjudgeable -> hold, escalate; never write
    return ADMIT


def admit(ev, store, type_of, constraints):
    """Classify and, on ADMIT, commit to the store. Returns the disposition string."""
    d = classify(ev, store, type_of, constraints)
    if d == ADMIT and not store.has(ev):
        store._commit(ev)
    return d
