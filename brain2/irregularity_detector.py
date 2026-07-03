#!/usr/bin/env python3
"""
irregularity_detector.py — know the boundary: detect domains that have NO checkable regularity.

Self-verifiers reach exactly as far as the open world is REGULAR. The other part — chaotic,
one-off, subjective, genuinely novel — has no invariant to mint, no law to fit. The brain
can't conquer it, but it must KNOW it's there: detect the absence of regularity and abstain
honestly, instead of hallucinating a law that isn't real.

Given (input -> output) data, it checks three things and declares REGULAR vs IRREGULAR:
  1. functional?   same input -> same output (else not even a function -> irregular)
  2. law fits?     a simple law (linear / power) predicts HELD-OUT within tolerance
  3. invariant?    a mined invariant survives held-out

REGULAR  -> the brain can verify here; mine/induce/reason.
IRREGULAR-> no regularity survives -> the brain ABSTAINS (this is the unverifiable part).

This is epistemic humility as a mechanism: the detector is the brain's MAP of its own reach.
"""

import math


def _functional(pairs, tol=1e-6):
    seen = {}
    for x, y in pairs:
        if x in seen and abs(seen[x] - y) > tol + 1e-6 * abs(y):
            return False
        seen[x] = y
    return True


def _law_error(train, holdout):
    """Best held-out relative error fitting y=m*x+b and y=k*x^p (whichever is better).
    Prefers the verified C++ port (brain2.law_error) when available, else the Python
    reference below. harden_regress locks brain2.law_error == _law_error_py."""
    from cpp_accel import cpp
    b = cpp()
    if b is not None:
        return b.law_error(train, holdout)
    return _law_error_py(train, holdout)


def _law_error_py(train, holdout):
    """Pure-Python reference (the equality target for the C++ port)."""
    n = len(train)
    xs = [x for x, _ in train]; ys = [y for _, y in train]
    best = 1e9
    # linear least squares
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    b = my - m * mx
    err = max(abs((m * x + b) - y) / (abs(y) + 1e-9) for x, y in holdout)
    best = min(best, err)
    # power law via log-log (needs positives)
    if all(x > 0 and y > 0 for x, y in train + holdout):
        lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
        lmx, lmy = sum(lx) / n, sum(ly) / n
        lden = sum((v - lmx) ** 2 for v in lx) or 1e-9
        p = sum((v - lmx) * (w - lmy) for v, w in zip(lx, ly)) / lden
        k = math.exp(lmy - p * lmx)
        err2 = max(abs(k * x ** p - y) / (abs(y) + 1e-9) for x, y in holdout)
        best = min(best, err2)
    return best


def assess(train, holdout, law_tol=0.05):
    if not train or not holdout:
        return "IRREGULAR", "insufficient data (need train + held-out points)"
    if not _functional(train + holdout):
        return "IRREGULAR", "not even a function (same input -> different outputs)"
    # Regularity must mean PREDICTABILITY: a law that generalizes to held-out. Surviving
    # invariants alone do NOT establish it — they are necessary-not-sufficient and can hold
    # on noise by chance (e.g. y>=x on random positives). So the verdict is the law fit.
    err = _law_error(train, holdout)
    if err <= law_tol:
        return "REGULAR", "a law predicts held-out within %.0f%% (err %.1f%%)" % (law_tol * 100, err * 100)
    return "IRREGULAR", "no law predicts held-out (best err %.0f%%) -> unverifiable" % (err * 100)


def _demo():
    import random
    print("=== irregularity_detector — the brain's map of where it CAN'T verify ===\n")

    kepler_tr = [(0.387, 0.241), (0.723, 0.615), (1.0, 1.0), (1.524, 1.881), (5.203, 11.862)]
    kepler_ho = [(9.537, 29.457), (19.191, 84.02), (30.07, 164.8)]

    rng = random.Random(0)
    noise_tr = [(i, rng.uniform(0, 100)) for i in range(1, 8)]      # no relation
    noise_ho = [(i, rng.uniform(0, 100)) for i in range(8, 12)]

    linear_tr = [(x, 2 * x + 3 + rng.uniform(-0.2, 0.2)) for x in range(1, 8)]  # noisy but real
    linear_ho = [(x, 2 * x + 3) for x in range(8, 12)]

    incons = [(1, 5), (1, 9), (2, 3), (2, 8)]                       # same x, different y

    for name, (tr, ho) in [("planetary (Kepler)", (kepler_tr, kepler_ho)),
                            ("noisy linear law", (linear_tr, linear_ho)),
                            ("random noise", (noise_tr, noise_ho)),
                            ("inconsistent/subjective", (incons, incons))]:
        verdict, why = assess(tr, ho)
        act = "mine + reason" if verdict == "REGULAR" else "ABSTAIN (out of verifiable reach)"
        print("  %-24s -> %-9s  %s\n%29s%s" % (name, verdict, why, "", "=> " + act))
    print("\n  The brain detects regularity vs its absence. Where no regularity survives, it")
    print("  ABSTAINS honestly — it knows the edge of what verification can touch.")


if __name__ == "__main__":
    _demo()
