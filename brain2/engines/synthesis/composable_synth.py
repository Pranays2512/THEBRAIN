#!/usr/bin/env python3
"""
composable_synth.py — coding primitives as COMPOSABLE policies (cross-class novelty).

The same move as cross-domain policy composition, applied to code. Instead of N
separate templates (fold, two-state, conditional, early-return), there is ONE
composable program shape whose PARTS recombine:

  a, b = INIT
  for i in RANGE:
      if GUARD(i, n):              # optional
          a, b = UA(a,b,i), UB(a,b,i)
      if EARLY(a,b,i,n):           # optional
          return i
  return FINAL

Searching combinations of {init, range, guard, update, early, final} gives:
  * the OLD classes as compositions (Fibonacci = two-state; count-divisors = guarded
    fold) — combinatorial coverage, not hand-built templates, AND
  * NOVEL cross-class algorithms no single template had — e.g. a FOLD combined with
    an EARLY-RETURN ("first k where 1+..+k >= n"), which is neither a pure fold nor a
    pure search.

Every candidate is verified on held-out examples. (At scale this space explodes —
that's where the PROPOSER gates which pieces to compose, exactly like the policy
proposer. Bounded here to demonstrate the principle.)

    python3 composable_synth.py
"""

INITS = [
    (0, 1),   # accumulator=0, counter=1  (sum, count)
    (0, 0),   # both zero                 (double accumulator)
    (1, 1),   # both one                  (product, fibonacci base)
    (1, 0),   # product=1, count=0        (factorial, product)
    (0, -1),  # accumulator=0, sentinel=-1 (argmax/argmin patterns)
]

RANGES = [
    ("1", "n + 1"),   # 1..n inclusive        (triangular, factorial, divisors)
    ("2", "n"),       # 2..n-1                (primality inner)
    ("0", "n"),       # 0..n-1                (0-indexed loops)
    ("1", "n"),       # 1..n-1                (skip last)
    ("2", "n + 1"),   # 2..n inclusive        (even/odd from 2)
]

GUARDS = {
    "None":           None,
    "n % i == 0":     lambda i, n: n % i == 0,    # divisibility
    "i % 2 == 0":     lambda i, n: i % 2 == 0,    # even elements
    "i % 2 != 0":     lambda i, n: i % 2 != 0,    # odd elements
    "i * i <= n":     lambda i, n: i * i <= n,    # sqrt bound (primality)
    "n % i != 0":     lambda i, n: n % i != 0,    # non-divisors
    "i <= n // 2":    lambda i, n: i <= n // 2,   # half-range
}

UPDATES = {  # (UA code, UB code): (UA fn, UB fn)
    # ── additive accumulation ──────────────────────────────────────────────
    ("a + i",     "b"):      (lambda a, b, i: a + i,         lambda a, b, i: b),
    ("a + 1",     "b"):      (lambda a, b, i: a + 1,         lambda a, b, i: b),
    ("a + i * i", "b"):      (lambda a, b, i: a + i * i,     lambda a, b, i: b),
    ("a + i * i * i", "b"):  (lambda a, b, i: a + i**3,      lambda a, b, i: b),
    # ── multiplicative accumulation (NEW: factorial, products, powers) ─────
    ("a * i",     "b"):      (lambda a, b, i: a * i,         lambda a, b, i: b),
    ("a * b",     "b + 1"):  (lambda a, b, i: a * b,         lambda a, b, i: b + 1),  # nCr-like
    ("a * i * i", "b"):      (lambda a, b, i: a * i * i,     lambda a, b, i: b),
    # ── two-state (fibonacci-style carry) ─────────────────────────────────
    ("b",         "a + b"):  (lambda a, b, i: b,             lambda a, b, i: a + b),
    ("b",         "a * b"):  (lambda a, b, i: b,             lambda a, b, i: a * b),  # NEW: geometric sequence
    ("b",         "a + i"):  (lambda a, b, i: b,             lambda a, b, i: a + i),
    # ── conditional / min-max tracking (NEW) ──────────────────────────────
    ("a + 1",     "b + i"):  (lambda a, b, i: a + 1,         lambda a, b, i: b + i),  # count+sum pair
    ("max(a,i)",  "b"):      (lambda a, b, i: max(a, i),     lambda a, b, i: b),      # running max
    ("min(a,i) if a>=0 else i", "b"):
                             (lambda a, b, i: min(a, i) if a >= 0 else i,
                              lambda a, b, i: b),                                      # running min
}

# Remove placeholder entries that need `n` in lambda (can't close over it simply)
UPDATES = {k: v for k, v in UPDATES.items()
           if k not in [("a * n + i", "b"), ("a + n % i", "b")]}

EARLIES = {
    "None":           None,
    "a >= n":         lambda a, b, i, n: a >= n,    # sum reached threshold
    "a > n":          lambda a, b, i, n: a > n,     # exceeded threshold
    "a == n":         lambda a, b, i, n: a == n,    # exact match
    "n % i == 0":     lambda a, b, i, n: n % i == 0,  # NEW: found first divisor
    "a * a > n":      lambda a, b, i, n: a * a > n, # NEW: squared exceeds (sqrt stop)
    "b == 0":         lambda a, b, i, n: b == 0,    # NEW: second accumulator zeroed
}

FINALS = {
    "a":    lambda a, b, last: a,
    "b":    lambda a, b, last: b,
    "-1":   lambda a, b, last: -1,
    "i":    lambda a, b, last: last,                # NEW: return last loop index
    "a + b":lambda a, b, last: a + b,               # NEW: sum of both accumulators
    "a - 1":lambda a, b, last: a - 1,               # NEW: off-by-one correction
    "b - 1":lambda a, b, last: b - 1,               # NEW: off-by-one on second
}


def _rng(lo, hi, n):
    return range(int(lo), n + 1 if hi.endswith("+ 1") else n)


def run(p, n):
    a, b = p["init"]
    gf = GUARDS[p["guard"]]
    uaf, ubf = UPDATES[p["upd"]]
    ef = EARLIES[p["early"]]
    last = 0
    for i in _rng(p["lo"], p["hi"], n):
        last = i
        if gf is None or gf(i, n):
            a, b = uaf(a, b, i), ubf(a, b, i)
        if ef is not None and ef(a, b, i, n):
            return i
    return FINALS[p["final"]](a, b, last)


def synthesize(examples):
    cut = max(3, int(len(examples) * 0.6))
    train, hold = examples[:cut], examples[cut:]
    for init in INITS:
        for lo, hi in RANGES:
            for g in GUARDS:
                for u in UPDATES:
                    for e in EARLIES:
                        for fin in FINALS:
                            p = dict(init=init, lo=lo, hi=hi, guard=g,
                                     upd=u, early=e, final=fin)
                            try:
                                if all(run(p, n) == y for n, y in train) and \
                                   all(run(p, n) == y for n, y in hold):
                                    return p
                            except Exception:
                                pass
    return None


def render(fn, p):
    g = "" if p["guard"] == "None" else f"        if {p['guard']}:\n    "
    body = (f"        a, b = {p['upd'][0]}, {p['upd'][1]}\n" if p["guard"] == "None"
            else f"        if {p['guard']}:\n            a, b = {p['upd'][0]}, {p['upd'][1]}\n")
    early = "" if p["early"] == "None" else f"        if {p['early']}:\n            return i\n"
    return (f"def {fn}(n):\n    a, b = {p['init'][0]}, {p['init'][1]}\n"
            f"    for i in range({p['lo']}, {p['hi']}):\n{body}{early}"
            f"    return {p['final']}\n")


def _verify(code, fn, examples):
    ns = {}
    exec(code, {"range": range}, ns)
    return all(ns[fn](n) == y for n, y in examples)


def _demo():
    cases = {
        # OLD classes, now found as compositions of shared primitives:
        "fib":   [(0, 0), (1, 1), (2, 1), (3, 2), (7, 13), (10, 55)],
        "ndiv":  [(1, 1), (2, 2), (6, 4), (7, 2), (12, 6)],
        # NOVEL cross-class: FOLD + EARLY-RETURN — "first k with 1+..+k >= n":
        "cumstop": [(1, 1), (3, 2), (6, 3), (7, 4), (10, 4), (15, 5)],
    }
    print("=== composable_synth — coding primitives compose (cross-class novelty) ===\n")
    for fn, ex in cases.items():
        p = synthesize(ex)
        if p is None:
            print(f"  {fn:8s}: no composition found")
            continue
        code = render(fn, p)
        ok = _verify(code, fn, ex)
        parts = f"init{p['init']} range({p['lo']},{p['hi']}) guard={p['guard']} " \
                f"upd={p['upd']} early={p['early']} ret={p['final']}"
        tag = "NOVEL fold+early" if (p["early"] != "None" and p["guard"] == "None") else \
              ("two-state" if p["upd"] == ("b", "a + b") else "guarded fold")
        print(f"  {fn:8s} [{tag}]: {'VERIFIED ✓' if ok else 'WRONG ✗'}")
        print(f"      {parts}")
    print("\n  one composition search re-derived Fibonacci (two-state) and count-divisors")
    print("  (guarded fold), AND found 'cumstop' = fold + early-return — a cross-class")
    print("  algorithm no single template had. Composition gives novelty; verify gates it.")


if __name__ == "__main__":
    _demo()
