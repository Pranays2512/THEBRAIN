#!/usr/bin/env python3
"""verb_learn.py — the capstone: LEARN a verb's selectional constraint from reading.

Positional detection recovers structure on an unknown verb, but the event ABSTAINS (held) —
we can't verify a verb we don't understand. This closes that: watch how a verb is used and
INDUCE its selectional restriction, so its events start disposing crisply (admit/reject).

The same conjecture->verify->admit membrane as type_oracle.grow, applied to verbs:
  * observe   — record the type-closures of each use's agent/patient (types from the noun
                oracle, which itself may have been grown from fuzzy neighbors).
  * conjecture — the constraint is the INTERSECTION of the observed role type-closures (the
                shared supertype), minus universal roots so it doesn't collapse to 'anything'.
  * verify    — a held-out use must satisfy the conjecture (no counterexample), AND the verb
                must have been seen >= promote_at times (a track record, not one sighting —
                like concept promotion; a learned constraint is a good hypothesis, still
                re-checked by the membrane on every future use, never a proof).

Admitting a verb makes it TRUSTED: its events move held -> admit/reject. (Open-language track
— read-only comprehension becomes learning-from-prose.)"""

# categories too general to be a useful restriction (would admit anything)
_ROOTS = frozenset({"living_thing", "thing", "entity", "object", "concept"})


def _generalize(closures, frac=1.0):
    """The shared selectional type across uses. A type is kept if it appears in at least
    `frac` of the TYPED observations (untyped fillers are ignored, not counted against).
    frac=1.0 is strict intersection (every use must share it — brittle: one shallow-typed or
    outlier filler collapses the constraint). frac<1.0 is frequency-thresholded generalization:
    robust to sparse taxonomy gaps and one-off contexts, which is what real corpora have.
    Universal roots are removed either way (they'd admit anything)."""
    import math
    from collections import Counter
    typed = [set(c) for c in closures if c]
    n = len(typed)
    if n == 0:
        return frozenset()
    cnt = Counter()
    for s in typed:
        cnt.update(s)
    need = n if frac >= 1.0 else max(1, math.ceil(frac * n))
    return frozenset(t for t, k in cnt.items() if k >= need) - _ROOTS


def _intersect(closures):
    """Back-compat alias: strict intersection (frac=1.0)."""
    return _generalize(closures, 1.0)


class VerbLearner:
    def __init__(self, type_of, promote_at=2, frac=1.0):
        self.type_of = type_of
        self.promote_at = promote_at
        self.frac = frac                       # generalization threshold (1.0 = strict intersect)
        self.obs = {}                          # verb -> [(agent_types, patient_types)]
        self.constraints = {}                  # verb -> {agent:set, patient:set}  (admitted)

    def observe(self, ev):
        """Record one use: the type-closures of this event's agent and patient."""
        a = self.type_of(ev.agent) if ev.agent else None
        p = self.type_of(ev.patient) if ev.patient else None
        self.obs.setdefault(ev.verb, []).append((a, p))

    def _conjecture(self, obs):
        spec = {}
        agent_c = _generalize([a for a, _ in obs], self.frac)
        patient_c = _generalize([p for _, p in obs], self.frac)
        if agent_c:
            spec["agent"] = set(agent_c)
        if patient_c:
            spec["patient"] = set(patient_c)
        return spec

    def _satisfies(self, spec, obs):
        for a, p in obs:
            if spec.get("agent") and a and not (spec["agent"] & set(a)):
                return False
            if spec.get("patient") and p and not (spec["patient"] & set(p)):
                return False
        return True

    def learn(self, verb, holdout=()):
        """conjecture -> verify -> admit. Returns the admitted constraint, or None (held:
        too few sightings, empty conjecture, or a held-out counterexample)."""
        obs = self.obs.get(verb, [])
        if len(obs) < self.promote_at:
            return None
        spec = self._conjecture(obs)
        if not spec:                           # no typed evidence -> nothing to constrain on
            return None
        if not self._satisfies(spec, holdout):  # counterexample -> usage not consistent enough
            return None
        self.constraints[verb] = spec
        return spec

    def acquire(self, holdouts=None):
        """Try to learn every observed-but-unlearned verb. Returns the set newly admitted."""
        holdouts = holdouts or {}
        learned = set()
        for verb in list(self.obs):
            if verb not in self.constraints and self.learn(verb, holdouts.get(verb, ())):
                learned.add(verb)
        return learned
