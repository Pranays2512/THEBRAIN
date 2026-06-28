#!/usr/bin/env python3
"""
invariant_miner.py — the brain makes its OWN verifiers (self-growing the envelope).

Verifiers are the envelope: what the brain can check is what it can trust. Until now every
verifier was hand-coded. This mints them from experience, the way a person turns a reliable
regularity into a rule they check against ("if it violates conservation, I'm wrong"):

  1. MINE     candidate invariants that hold across solved examples (sign, monotonicity,
              out>=input, bounds, parity, ...).
  2. VALIDATE each on HELD-OUT cases — an invariant earns trust only by surviving cases it
              could have broken (this kills coincidences that held in the sample).
  3. ADMIT    survivors as cheap checks.
  4. FILTER   a new candidate WITHOUT an oracle: violate an admitted invariant -> reject.
  5. DEMOTE   any invariant that ever rejects a known-correct output (it was spurious).

Honest ceiling: self-minted invariants are NECESSARY, not sufficient. Violating one means
DEFINITELY wrong; satisfying all of them does NOT prove correct. So this is a fast
error-detector / pre-filter, not a completeness proof — exactly how humans use such checks.
"""

# Each invariant is a predicate over (input_value, output_value).
PREDICATES = {
    "out_nonneg":   lambda x, y: y >= 0,
    "out_positive": lambda x, y: y > 0,
    "out_ge_input": lambda x, y: y >= x,
    "out_le_input": lambda x, y: y <= x,
    "out_even":     lambda x, y: y % 2 == 0,
    "out_le_1000":  lambda x, y: y <= 1000,      # plausible on small samples, usually spurious
}


def _holds(name, examples):
    p = PREDICATES[name]
    try:
        return all(p(x, y) for x, y in examples)
    except Exception:
        return False


def mine(train):
    """Candidate invariants: predicates true on ALL training examples, plus monotonicity."""
    cands = {n for n in PREDICATES if _holds(n, train)}
    s = sorted(train)
    if all(s[i][1] <= s[i + 1][1] for i in range(len(s) - 1)):
        cands.add("monotonic_increasing")
    return cands


def validate(cands, holdout):
    """Keep only invariants that ALSO hold on held-out cases (earned, not coincidental)."""
    admitted = set()
    for n in cands:
        if n == "monotonic_increasing":
            s = sorted(holdout)
            if all(s[i][1] <= s[i + 1][1] for i in range(len(s) - 1)):
                admitted.add(n)
        elif _holds(n, holdout):
            admitted.add(n)
    return admitted


def check(fn, probe_inputs, admitted):
    """Filter a candidate fn with NO oracle: reject if any admitted invariant is violated."""
    pairs = []
    for x in probe_inputs:
        try:
            pairs.append((x, fn(x)))
        except Exception:
            return False, "raised on %s" % str(x)
    for n in admitted:
        if n == "monotonic_increasing":
            s = sorted(pairs)
            if not all(s[i][1] <= s[i + 1][1] for i in range(len(s) - 1)):
                return False, "violates monotonic_increasing"
        elif not _holds(n, pairs):
            return False, "violates %s" % n
    return True, "passes all %d admitted invariants" % len(admitted)


def demote(admitted, correct_examples):
    """Drop any invariant a KNOWN-correct output violates (it was spurious / over-trusted)."""
    keep, dropped = set(), []
    for n in admitted:
        if n == "monotonic_increasing":
            keep.add(n); continue
        if _holds(n, correct_examples):
            keep.add(n)
        else:
            dropped.append(n)
    return keep, dropped


def _demo():
    import math
    print("=== invariant_miner — the brain mints its own verifiers ===\n")
    train   = [(0, 1), (1, 1), (4, 24), (5, 120), (6, 720)]      # factorial, solved
    holdout = [(7, 5040), (8, 40320), (9, 362880)]               # more correct cases

    cands = mine(train)
    print("  mined from training (hold on all 5):", sorted(cands))
    admitted = validate(cands, holdout)
    print("  ADMITTED after held-out validation   :", sorted(admitted))
    print("    -> 'out_le_1000' held on training by luck but FAILS held-out (5040) — dropped.\n")

    # Use the self-made verifiers to reject a buggy candidate WITH NO ORACLE
    buggy = lambda n: n * n           # not factorial; f(0)=0
    ok, why = check(buggy, [0, 1, 2, 5, 8], admitted)
    print("  candidate n*n vs the minted checks (no oracle): %s  (%s)" % (ok, why))
    good = math.factorial
    ok2, why2 = check(good, [0, 1, 2, 5, 8], admitted)
    print("  candidate factorial vs the minted checks       : %s  (%s)\n" % (ok2, why2))

    # DEMOTE: if an admitted invariant ever rejects a known-correct output, drop it.
    # factorial(1)=1 is odd, so 'out_even' would have been wrong to trust — demotion catches it.
    forced = admitted | {"out_even"}
    kept, dropped = demote(forced, train + holdout)
    print("  demotion (an invariant that rejects a correct output is spurious):")
    print("    forced-in:", sorted(forced), "\n    kept:", sorted(kept), " dropped:", dropped)
    print("\n  Self-made verifiers: validated before trust, revised on misfire. They CATCH")
    print("  errors without an answer key — necessary checks, not proofs of correctness.")


if __name__ == "__main__":
    _demo()
