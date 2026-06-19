#!/usr/bin/env python3
"""
curiosity_loop.py — idle-time learning driven by prediction error (idea #4).

When not being asked anything, the brain explores its own knowledge gaps. It
predicts what follows each observed event; where it is wrong (or has no rule), the
prediction error is high — that is curiosity. Each idle tick it accumulates more
observations, mines + verifies new rules (the inductive engine) to cover the
high-error areas, and its error drops. Attention then moves to whatever is still
unpredictable.

    cl = CuriosityLoop()
    cl.run([chunk1, chunk2, ...])      # idle ticks, each folds in more data

Honest scope: curiosity = prediction error; exploration = the verify-before-
promote inductive loop. It learns patterns that REPLICATE and stays curious where
none exist (a random follower never yields a verified rule, so the gap persists —
the loop cannot invent a pattern that isn't there). It does not discover new math;
it fills gaps with rules the data actually supports.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from reasoning_engine import ReasoningEngine
from inductive_engine import InductiveLearner


class CuriosityLoop:
    def __init__(self, min_support=2, min_conf=0.8, verify_conf=0.7, min_test=2):
        self.engine = ReasoningEngine()
        self.learner = InductiveLearner()
        self.observed = []
        self.predict = {}             # event -> predicted next event (verified rules only)
        self._thr = dict(min_support=min_support, min_conf=min_conf,
                         verify_conf=verify_conf, min_test=min_test)

    def observe(self, episodes):
        self.observed.extend(episodes)

    def _split(self):
        n = len(self.observed)
        cut = max(1, int(0.7 * n))
        return self.observed[:cut], self.observed[cut:]

    def tick(self):
        """One idle step: mine + verify over all observations so far, install any
        newly verified rules, and report what was learned and the new error."""
        train, test = self._split()
        if not test:
            return [], self.error()
        promoted, _ = self.learner.mine(train, test, **self._thr)

        # all verified rules go to the reasoning engine (incl. transitive ones);
        # the prediction table keeps only the IMMEDIATE successor per event, so a
        # rule like rain->puddles (non-adjacent) doesn't fight rain->wet_ground.
        imm = {}
        for a, b in self._transitions():
            imm[(a, b)] = imm.get((a, b), 0) + 1
        by_source = {}
        for r in promoted:
            self.engine.learn(r.a, "leads_to", r.b)
            by_source.setdefault(r.a, []).append(r.b)
        self.engine.set_transitive("leads_to")

        learned = []
        for a, bs in by_source.items():
            best = max(bs, key=lambda b: imm.get((a, b), 0))
            if imm.get((a, best), 0) and self.predict.get(a) != best:
                self.predict[a] = best
                learned.append((a, best))
        return learned, self.error()

    # ── prediction error = curiosity signal ──────────────────────────────────
    def _transitions(self):
        for ep in self.observed:
            for i in range(len(ep) - 1):
                yield ep[i], ep[i + 1]

    def error(self):
        total = wrong = 0
        for a, b in self._transitions():
            total += 1
            if self.predict.get(a) != b:
                wrong += 1
        return round(wrong / total, 3) if total else 0.0

    def curiosity_gaps(self, top=3):
        """Events ranked by how badly the brain predicts them (its curiosity)."""
        tot, bad = {}, {}
        for a, b in self._transitions():
            tot[a] = tot.get(a, 0) + 1
            if self.predict.get(a) != b:
                bad[a] = bad.get(a, 0) + 1
        scored = [(a, round(bad.get(a, 0) / tot[a], 2)) for a in tot if bad.get(a, 0)]
        return sorted(scored, key=lambda x: -x[1])[:top]

    def run(self, chunks):
        print("start: no rules, no observations yet\n")
        for k, chunk in enumerate(chunks, 1):
            self.observe(chunk)
            learned, err = self.tick()
            desc = ", ".join(f"{a}->{b}" for a, b in learned) or "(nothing new verified)"
            print(f"idle tick {k}: learned {desc}")
            print(f"            error now {err:.0%}; still curious about {self.curiosity_gaps()}\n")
        return self.error()


def _demo():
    # predictable patterns + one genuinely random follower (dice)
    predictable = ([["rain", "wet_ground", "puddles"]] * 2 +
                   [["study", "pass"]] * 2 +
                   [["seed", "plant"]] * 2)
    noise = [["dice", "one"], ["dice", "two"], ["dice", "three"]]
    chunks = [predictable + noise[:1],
              predictable + noise[1:2],
              predictable + noise[2:3],
              predictable + noise[:1]]

    print("=== curiosity_loop — idle learning driven by prediction error ===\n")
    cl = CuriosityLoop()
    cl.run(chunks)
    print(f"converged: error {cl.error():.0%}. "
          f"Remaining curiosity (unlearnable): {cl.curiosity_gaps()}")


if __name__ == "__main__":
    _demo()
