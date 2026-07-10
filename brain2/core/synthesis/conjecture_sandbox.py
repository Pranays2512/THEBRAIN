#!/usr/bin/env python3
"""
conjecture_sandbox.py — the brain TESTS its own guesses (active experimentation).

So far verification was passive: check a candidate against data it was given. This lets the
brain ACT on an unverified guess it feels confident enough to test — design an experiment,
run it in a sandbox, and judge the conjecture against the knowledge it ALREADY TRUSTS. This
is how a scientist promotes a hunch: not by being told, but by testing it.

  conjecture (an unverified formula/rule)
     -> DESIGN a test: generate diverse scenarios spanning the input space
     -> RUN it in the sandbox and derive what a TRUSTED principle says the answer must be
     -> JUDGE: survives every self-generated test -> PROVISIONAL ADMIT (confidence up);
        contradicted -> REJECT with the counterexample.

The oracle is the brain's own VERIFIED knowledge (here: energy conservation). No external
answer key, no human — the brain bootstraps new beliefs from trusted ones by experiment.

Honest limit: it can only test a guess against principles it ALREADY trusts (or internal
consistency). A guess with NO trusted anchor and no consistency handle stays unverifiable —
the sandbox extends reach where a trusted principle can adjudicate, not everywhere.
"""

import math
import random

G = 9.8


def trusted_KE(m, h):
    """TRUSTED principle: energy conservation -> a mass dropped from height h has KE = m*g*h
    at the bottom, with speed v = sqrt(2 g h). This is the brain's verified anchor."""
    v = math.sqrt(2 * G * h)
    return v, m * G * h


def design_and_test(conjecture, n=40, tol=0.01, seed=0):
    """Design experiments (random drops), run the conjecture vs the trusted KE, judge it."""
    rng = random.Random(seed)
    worst = 0.0
    counter = None
    for _ in range(n):
        m = rng.uniform(0.5, 10)
        h = rng.uniform(1, 100)
        v, ke_true = trusted_KE(m, h)
        try:
            ke_guess = conjecture(m, v)
        except Exception:
            return False, "raised", (m, v)
        rel = abs(ke_guess - ke_true) / (abs(ke_true) + 1e-9)
        if rel > worst:
            worst, counter = rel, (round(m, 2), round(v, 2), round(ke_true, 1), round(ke_guess, 1))
    survived = worst <= tol
    return survived, worst, counter


def _demo():
    print("=== conjecture_sandbox — the brain tests its own guesses against trusted knowledge ===\n")
    print("  trusted anchor: energy conservation (a drop gives KE = m*g*h, v = sqrt(2gh)).")
    print("  the brain designs random drop experiments and tests each conjecture:\n")

    conjectures = [
        ("KE = m*v",       lambda m, v: m * v),               # wrong (units, scaling)
        ("KE = m*v^2",     lambda m, v: m * v * v),           # right shape, wrong factor
        ("KE = 0.5*m*v^2", lambda m, v: 0.5 * m * v * v),     # correct
        ("KE = 0.5*m*v^3", lambda m, v: 0.5 * m * v ** 3),    # wrong power
    ]
    for name, fn in conjectures:
        survived, worst, counter = design_and_test(fn)
        if survived:
            print("  %-15s -> PROVISIONAL ADMIT (survived 40 self-designed tests, worst err %.2f%%)"
                  % (name, worst * 100))
        else:
            print("  %-15s -> REJECT (worst err %.0f%%, e.g. m=%s v=%s: true %s vs guess %s)"
                  % (name, worst * 100, counter[0], counter[1], counter[2], counter[3]))
    print("\n  The brain promoted the ONE conjecture that survives experiments it designed itself,")
    print("  judged by a principle it already trusts — bootstrapping a new verified law from a")
    print("  guess, with no external answer key. Active experimentation, not passive fitting.")


if __name__ == "__main__":
    _demo()
