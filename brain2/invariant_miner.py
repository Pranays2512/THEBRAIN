#!/usr/bin/env python3
"""
invariant_miner.py — the brain makes its OWN verifiers (self-growing the envelope).

Verifiers are the envelope: what the brain can check is what it can trust. Until now every
verifier was hand-coded. This mints them from experience, the way a person turns a reliable
regularity into a rule they check against ("if it violates conservation, I'm wrong"):

  1. MINE     candidate invariants that hold across solved examples.
  2. VALIDATE each on HELD-OUT cases — an invariant earns trust only by surviving cases it
              could have broken (kills coincidences that held in the sample).
  3. ADMIT    survivors as cheap checks.
  4. FILTER   a new candidate WITHOUT an oracle: violate an admitted invariant -> reject.
  5. DEMOTE   any invariant that ever rejects a known-correct output (it was spurious).

Arity-general: an example is (args, y) where args is a tuple (scalars auto-wrapped), so the
same machinery mines 1-arg tasks (factorial) and 2-arg tasks (gcd: out divides both inputs,
out <= min). Richer vocabulary = a wider, cheaper pre-filter.

Honest ceiling: invariants are NECESSARY, not sufficient. Violating one => DEFINITELY wrong;
satisfying all != proven correct. A fast error-detector / pre-filter, not a completeness proof.
"""

from math import prod


def _norm(x):
    return x if isinstance(x, tuple) else (x,)


# Each invariant is a predicate over (args_tuple, output). All arity-general.
PREDICATES = {
    "out_nonneg":       lambda a, y: y >= 0,
    "out_positive":     lambda a, y: y > 0,
    "out_ge_max_arg":   lambda a, y: y >= max(a),
    "out_le_min_arg":   lambda a, y: y <= min(a),
    "out_ge_first":     lambda a, y: y >= a[0],
    "out_le_first":     lambda a, y: y <= a[0],
    "out_le_prod":      lambda a, y: (y <= abs(prod(a))) if all(v != 0 for v in a) else True,
    "out_divides_args": lambda a, y: y > 0 and all(v % y == 0 for v in a),
    "out_even":         lambda a, y: y % 2 == 0,
    "out_le_1000":      lambda a, y: y <= 1000,        # usually spurious; validation should kill it
}


def _holds(name, examples):
    p = PREDICATES[name]
    try:
        return all(p(_norm(x), y) for x, y in examples)
    except Exception:
        return False


def _monotone(examples):
    """1-arg only: sorted by input, output non-decreasing."""
    if any(len(_norm(x)) != 1 for x, _ in examples):
        return False
    s = sorted((( _norm(x)[0], y) for x, y in examples))
    return all(s[i][1] <= s[i + 1][1] for i in range(len(s) - 1))


def mine(train):
    cands = {n for n in PREDICATES if _holds(n, train)}
    if _monotone(train):
        cands.add("monotonic_increasing")
    return cands


def validate(cands, holdout):
    admitted = set()
    for n in cands:
        if n == "monotonic_increasing":
            if _monotone(holdout):
                admitted.add(n)
        elif _holds(n, holdout):
            admitted.add(n)
    return admitted


def check(fn, probe_inputs, admitted):
    """Filter a candidate fn with NO oracle: reject if any admitted invariant is violated.
    probe_inputs are scalars or arg-tuples; fn is called with *args."""
    pairs = []
    for x in probe_inputs:
        args = _norm(x)
        try:
            pairs.append((args, fn(*args)))
        except Exception:
            return False, "raised on %s" % str(x)
    for n in admitted:
        if n == "monotonic_increasing":
            if not _monotone([(a, y) for a, y in pairs]):
                return False, "violates monotonic_increasing"
        elif not all(PREDICATES[n](a, y) for a, y in pairs):
            return False, "violates %s" % n
    return True, "passes all %d admitted invariants" % len(admitted)


def demote(admitted, correct_examples):
    """Drop any invariant a KNOWN-correct output violates (it was spurious / over-trusted)."""
    keep, dropped = set(), []
    for n in admitted:
        if n == "monotonic_increasing":
            keep.add(n) if _monotone(correct_examples) else dropped.append(n)
        elif _holds(n, correct_examples):
            keep.add(n)
        else:
            dropped.append(n)
    return keep, dropped


# ── functional invariants (probe-based) ───────────────────────────────────────
# Value invariants above constrain the OUTPUT. Functional invariants are properties of the
# FUNCTION itself — checked by probing it, not read off the I/O examples: commutativity,
# idempotence, involution. They catch errors value checks miss (e.g. a-b vs gcd: a-b can
# satisfy the value bounds but is NOT commutative).
FUNCTIONAL = {
    "commutative": (2, lambda f, a, b: f(a, b) == f(b, a)),
    "idempotent":  (1, lambda f, x: f(f(x)) == f(x)),
    "involutive":  (1, lambda f, x: f(f(x)) == x),
}


def _probes(arity, seed=0, n=12):
    import random
    rng = random.Random(seed)
    if arity == 1:
        return [(rng.randint(0, 20),) for _ in range(n)]
    return [(rng.randint(1, 20), rng.randint(1, 20)) for _ in range(n)]


def mine_functional(f, arity, seed=0):
    """Probe f; return the functional invariants it satisfies."""
    probes = _probes(arity, seed)
    out = set()
    for name, (ar, pred) in FUNCTIONAL.items():
        if ar != arity:
            continue
        try:
            if all(pred(f, *p) for p in probes):
                out.add(name)
        except Exception:
            pass
    return out


def check_functional(g, arity, admitted, seed=1):
    """A candidate must satisfy the admitted functional invariants (probed, no oracle)."""
    probes = _probes(arity, seed)
    for name in admitted:
        ar, pred = FUNCTIONAL[name]
        try:
            if not all(pred(g, *p) for p in probes):
                return False, "violates %s" % name
        except Exception:
            return False, "raised checking %s" % name
    return True, "passes functional invariants"


def _demo():
    import math
    print("=== invariant_miner — the brain mints its own verifiers (arity-general) ===\n")

    # 1-arg: factorial
    train   = [(0, 1), (1, 1), (4, 24), (5, 120), (6, 720)]
    holdout = [(7, 5040), (8, 40320), (9, 362880)]
    admitted = validate(mine(train), holdout)
    print("  factorial -> admitted:", sorted(admitted))
    print("    (out_le_1000 mined but DROPPED on held-out 5040)")
    buggy = lambda n: n * n
    print("    n*n rejected:", check(buggy, [0, 1, 5], admitted)[1])
    print("    factorial passes:", check(math.factorial, [0, 1, 5], admitted)[0])

    # 2-arg: gcd — the richer/relational invariants (divides both inputs, <= min)
    print("\n  gcd (2-arg) -> the multi-arg invariants kick in:")
    g_train = [((12, 8), 4), ((48, 36), 12), ((7, 5), 1), ((100, 80), 20)]
    g_hold  = [((9, 6), 3), ((14, 21), 7), ((17, 13), 1)]
    g_adm = validate(mine(g_train), g_hold)
    print("    admitted:", sorted(g_adm))
    bad_gcd = lambda a, b: a + b               # not gcd; doesn't divide the inputs
    print("    a+b rejected:", check(bad_gcd, [(12, 8), (7, 5)], g_adm)[1])
    print("    real gcd passes:", check(math.gcd, [(12, 8), (7, 5), (100, 80)], g_adm)[0])

    # demotion still works
    kept, dropped = demote(admitted | {"out_even"}, train + holdout)
    print("\n  demotion: forced out_even ->", "dropped" if "out_even" in dropped else "kept")

    # functional invariants (probe-based): properties of the function, caught by probing
    print("\n  functional invariants (probe-based, what value checks miss):")
    fg = mine_functional(math.gcd, 2)
    print("    gcd  -> %s" % sorted(fg))
    print("      a-b rejected:", check_functional(lambda a, b: a - b, 2, fg)[1],
          "(not commutative)")
    print("      gcd passes:", check_functional(math.gcd, 2, fg)[0])
    fa = mine_functional(abs, 1)
    print("    abs  -> %s  (abs(abs(x))==abs(x))" % sorted(fa))
    fn = mine_functional(lambda x: -x, 1)
    print("    neg  -> %s  (-(-x)==x)" % sorted(fn))
    print("\n  Mined, validated, revisable checks — value AND functional, 1-arg AND 2-arg.")


if __name__ == "__main__":
    _demo()
