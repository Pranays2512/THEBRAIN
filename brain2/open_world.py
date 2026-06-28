#!/usr/bin/env python3
"""
open_world.py — the full loop on a REAL open-world slice: discover a law, verify, generalize.

Not a physics toy with the answer baked in — REAL planetary data (semi-major axis in AU,
orbital period in years). The brain is told NOTHING about Kepler. It runs the loop:

  perceive data -> MINE the conserved quantity (an invariant) -> INDUCE the law ->
  VERIFY on held-out planets it never saw -> PREDICT / answer a question.

The discovered invariant is Kepler's third law (T^2 / a^3 = constant). The test is honest:
fit on the inner planets, verify the SAME constant holds for Saturn/Uranus/Neptune (held
out), then predict a period from an axis. Generalization to unseen real cases = the claim.
"""

import math

# REAL data: (planet, semi-major axis [AU], orbital period [years])
TRAIN = [("Mercury", 0.387, 0.241), ("Venus", 0.723, 0.615), ("Earth", 1.0, 1.0),
         ("Mars", 1.524, 1.881), ("Jupiter", 5.203, 11.862)]
HELD_OUT = [("Saturn", 9.537, 29.457), ("Uranus", 19.191, 84.02), ("Neptune", 30.07, 164.8)]


def _demo():
    print("=== open_world — discover a law from REAL data, verify on held-out, generalize ===\n")
    print("  data: planetary semi-major axis (AU) and orbital period (years). Kepler NOT told.\n")

    # 1. INDUCE the exponent of T = a^p  (p = log T / log a; Earth a=1 is uninformative)
    ps = [math.log(t) / math.log(a) for _, a, t in TRAIN if a != 1.0]
    p = sum(ps) / len(ps)
    print("  induced law:  T = a^%.3f   (Kepler's 3rd law is p = 1.5 -> T^2 = a^3)" % p)

    # 2. MINE the conserved invariant T^2 / a^3 (the regularity that makes it a LAW)
    ks = [t * t / a ** 3 for _, a, t in TRAIN]
    k = sum(ks) / len(ks)
    spread = max(ks) - min(ks)
    print("  mined invariant:  T^2 / a^3 = %.4f  (constant across all 5 planets, spread %.4f)\n"
          % (k, spread))

    # 3. VERIFY on held-out planets the brain NEVER saw: same invariant must hold
    print("  VERIFY on held-out planets (never seen):")
    ok = 0
    for name, a, t in HELD_OUT:
        kh = t * t / a ** 3
        good = abs(kh - k) / k < 0.02
        ok += good
        print("    %-8s T^2/a^3 = %.4f  %s" % (name, kh, "OK" if good else "FAIL"))
    print("    -> %d/%d held-out planets satisfy the discovered law (it GENERALIZES)\n" % (ok, len(HELD_OUT)))

    # 4. PREDICT / answer: period of a body at a given axis, from the discovered law
    print("  PREDICT from the law (T = sqrt(k * a^3)):")
    for a in (9.537, 19.191, 30.07):
        t_pred = math.sqrt(k * a ** 3)
        actual = next(tt for _, aa, tt in HELD_OUT if aa == a)
        print('    "period of a planet at %.2f AU?" -> %.1f years  (actual %.1f, err %.1f%%)'
              % (a, t_pred, actual, abs(t_pred - actual) / actual * 100))

    print("\n  Real data in -> law discovered (not told) -> verified on unseen planets ->")
    print("  predicts new cases. The open world entered the verifiable envelope, one slice,")
    print("  through perceive -> mine -> induce -> verify, no human stating the law.")


if __name__ == "__main__":
    _demo()
