#!/usr/bin/env python3
"""event_predict.py — predictive processing for the event stream (the default-state upgrade).

Brains don't idle waiting for input; the cortex CONSTANTLY predicts the next input and learns
from the error (Friston/Clark predictive processing). The reactive front only answers when
asked. This bolts prediction onto the event intake:

    predict next event  ->  parse the real one  ->  SURPRISE = prediction error  ->  learn

Surprise is the real, SEMANTIC novelty signal (an *unexpected event*), not the shallow lexical
novelty (an unseen token) the front used before. It drives what a brain does with error:
  * high surprise -> worth storing (episodic), worth attention
  * where you're wrong -> where to learn (update the priors)
  * high-error regions -> curiosity targets (the gaps to read toward)

Membrane: prediction is a fuzzy expectation — it never asserts truth. The parse + crisp
membrane still own what actually happened. Learning is Hebbian in spirit: consecutive events
that co-occur strengthen the transition, so the brain comes to expect what usually follows."""

import math
from collections import Counter, defaultdict


class EventPredictor:
    """Learns verb-transition + base-rate statistics over the event stream and scores how
    surprising each new event is given the previous one. Cheap: two count tables."""

    def __init__(self, blend=0.7, smoothing=0.5):
        self.trans = defaultdict(Counter)     # prev_verb -> Counter(next_verb)
        self.base = Counter()                  # overall verb frequency
        self.blend = blend                     # weight on the transition vs base rate
        self.k = smoothing                     # add-k Laplace

    # ── predict ──────────────────────────────────────────────────────────────
    def prob_verb(self, prev_event, verb):
        """P(next verb = `verb` | previous event), blending transition + base rate."""
        bt = sum(self.base.values())
        p_base = (self.base[verb] + self.k) / (bt + self.k * (len(self.base) + 1)) if bt else None
        if prev_event is not None and self.trans[prev_event.verb]:
            tr = self.trans[prev_event.verb]
            tt = sum(tr.values())
            p_tr = (tr[verb] + self.k) / (tt + self.k * (len(tr) + 1))
            return self.blend * p_tr + (1 - self.blend) * (p_base or p_tr)
        return p_base                          # None if the predictor has seen nothing yet

    def expect(self, prev_event, k=3):
        """The top-k verbs the brain expects next (for introspection / curiosity)."""
        src = self.trans[prev_event.verb] if (prev_event and self.trans[prev_event.verb]) else self.base
        return [v for v, _ in src.most_common(k)]

    # ── prediction error ───────────────────────────────────────────────────────
    def surprise(self, prev_event, event):
        """Surprisal of the observed verb, -log2 p, normalized to [0,1]. No history -> 1.0
        (everything is maximally surprising to a brain that has predicted nothing yet)."""
        p = self.prob_verb(prev_event, event.verb)
        if p is None:
            return 1.0
        bits = -math.log2(max(p, 1e-6))
        return min(bits / 10.0, 1.0)           # ~10 bits = "never saw this coming"

    # ── learn (Hebbian: co-occurring events strengthen the transition) ──────────
    def learn(self, prev_event, event):
        self.base[event.verb] += 1
        if prev_event is not None:
            self.trans[prev_event.verb][event.verb] += 1
