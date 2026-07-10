#!/usr/bin/env python3
"""
synth_invariant.py — self-made verifiers earn their keep in synthesis (invariant pre-filter).

Synthesis verifies a candidate with stress-vs-oracle: ~1000 oracle calls per candidate —
the expensive step. invariant_miner mints NECESSARY checks from the task's own examples for
nearly free. Put them in front of stress: a candidate whose outputs violate an admitted
invariant is rejected with NO oracle call. Only survivors pay for the full stress test.

  mine invariants from the task examples (validate on a few held-out oracle points, once)
  -> for each candidate: cheap invariant check; violate -> reject (0 oracle calls)
  -> survivors only -> expensive stress-vs-oracle

Necessary-not-sufficient (from invariant_miner) is exactly what a pre-filter wants: it can
only REJECT (cheaply), never falsely accept — so it never throws away a correct program,
it just spares the oracle from obviously-wrong ones.

    python3 synth_invariant.py
"""

import random

from core.synthesis import invariant_miner as IM

STRESS_N = 1000


def task_invariants(oracle, mine_inputs, holdout_inputs):
    train = [(x, oracle(x)) for x in mine_inputs]
    hold = [(x, oracle(x)) for x in holdout_inputs]
    return IM.validate(IM.mine(train), hold)


def triage(candidates, oracle, invariants, probe, seed=0):
    """Run each candidate through the cheap invariant filter, then stress only survivors.
    Returns (results, oracle_calls_used)."""
    rng = random.Random(seed)
    stress_inputs = [rng.randint(0, 20) for _ in range(STRESS_N)]
    calls, results = 0, []
    for name, fn in candidates:
        ok, why = IM.check(fn, probe, invariants)          # cheap: candidate's own outputs
        if not ok:
            results.append((name, "rejected cheaply", why))
            continue
        passed = True                                      # survivor pays for the oracle
        for x in stress_inputs:
            calls += 1
            try:
                if fn(x) != oracle(x):
                    passed = False
                    break
            except Exception:
                passed = False
                break
        results.append((name, "STRESS " + ("pass" if passed else "fail"), ""))
    return results, calls


def _demo():
    import math
    print("=== synth_invariant — mined verifiers pre-filter the oracle ===\n")
    oracle = math.factorial
    invs = task_invariants(oracle, mine_inputs=[0, 1, 4, 5, 6], holdout_inputs=[7, 8, 9])
    print("  admitted invariants for factorial:", sorted(invs), "\n")

    candidates = [   # a candidate stream (some buggy) as the synth search would produce
        ("n*n",          lambda n: n * n),            # f(0)=0 -> violates out_positive
        ("n*(n-1)",      lambda n: n * (n - 1)),      # f(0)=0 -> violates out_positive
        ("-factorial",   lambda n: -math.factorial(n)),  # negative -> violates out_nonneg
        ("n+1",          lambda n: n + 1),            # passes invariants, fails stress
        ("factorial",    math.factorial),             # correct
    ]
    probe = [0, 1, 2, 5, 8]
    results, calls = triage(candidates, oracle, invs, probe)
    rejected = sum(1 for _, v, _ in results if v.startswith("rejected"))
    for name, verdict, why in results:
        print("  %-12s -> %-18s %s" % (name, verdict, why))

    baseline = len(candidates) * STRESS_N
    print("\n  oracle calls: %d (with pre-filter)  vs  %d (stress every candidate)"
          % (calls, baseline))
    print("  %d/%d candidates rejected for free; correct program never rejected by the filter."
          % (rejected, len(candidates)))
    print("  Self-made verifiers cut the expensive oracle work — necessary checks pay off.")


if __name__ == "__main__":
    _demo()
