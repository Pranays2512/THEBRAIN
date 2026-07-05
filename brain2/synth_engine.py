#!/usr/bin/env python3
"""
synth_engine.py — one proposer-guided synthesis engine over all the spaces.

Unifies the scattered synthesizers (formula / accumulator-loop / two-state+conditional
/ while / early-return / list+nested / DP) behind ONE entry point. A task is a list of
(args, output) examples + an input kind; the engine ROUTES to the spaces that apply,
searches each (proposer-guided where built), and returns the first program that
VERIFIES on every example. Every backend emits a Python function; the engine verifies
uniformly by running it.

Then a BENCHMARK: a suite of classic algorithm tasks run through the engine, reporting
coverage (how many the brain writes, no LLM) and which space solved each.

    python3 synth_engine.py
"""

import composable_synth as C
import loop_synth3 as L3
import loop_synth4 as L4
import dp_proposer as DP


# ── backends: each takes examples in (args, out) form, returns Python code or None
def _int1(examples):
    return [(a[0], y) for a, y in examples]


def b_composable(ex):
    p = C.synthesize(_int1(ex))
    return C.render("f", p) if p else None


# Learned accelerator over the SAME composable space: a proposer reads the target's I/O
# shape and orders the compositions so the likely program is tried first (premise
# selection). Trained once per process, cached. Guided runs before brute b_composable;
# on a miss the engine falls through to brute, and solve() re-verifies either way — so
# the accelerator only ever changes SPEED, never correctness.
_CP = {"trees": None, "failed": False}


def _cp_trees():
    import composable_proposer as CP
    if _CP["trees"] is None and not _CP["failed"]:
        try:
            _CP["trees"] = CP.train()
        except Exception:
            _CP["failed"] = True
    return _CP["trees"]


def b_composable_guided(ex):
    import composable_proposer as CP
    trees = _cp_trees()
    if trees is None:
        return None
    data = _int1(ex)
    try:
        cand = CP.order(trees, CP.feats(data))
        p, _ = CP.search(data, cand)
    except Exception:
        p = None
    return C.render("f", p) if p else None


def b_early(ex):
    r = L3.synth_search(_int1(ex))
    return L3.render("f", *r) if r else None


def b_while(ex):
    r = L3.synth_while([(a, y) for a, y in ex])      # args already (a,b)
    return L3.render("f", *r) if r else None


def b_list(ex):
    data = [(a[0], y) for a, y in ex]
    for synth in (L4.synth_fold, L4.synth_nested, L4.synth_sort):
        r = synth(data)
        if r:
            return L4.render("f", *r)
    return None


def b_member(ex):
    r = L4.synth_member([(a, y) for a, y in ex])     # args (lst, t)
    return L4.render("f", *r) if r else None


def b_dp(ex):
    data = [(a[0], y) for a, y in ex]
    cut = max(3, int(len(data) * 0.6))
    for rec in DP.ALL:
        try:
            if all(DP.run_dp(*rec, a) == y for a, y in data[:cut]) and \
               all(DP.run_dp(*rec, a) == y for a, y in data[cut:]):
                ik, ck, bk = rec
                init = "cur = best = lst[0]\n    seq = lst[1:]" if ik == "first" \
                    else "cur = best = 0\n    seq = lst"
                return (f"def f(lst):\n    if not lst: return 0\n    {init}\n"
                        f"    for x in seq:\n        cur = {ck}\n        best = {bk}\n"
                        f"    return best\n")
        except Exception:
            pass
    return None


ROUTES = {              # input kind -> ordered backends to try
    "int1":  [("composable_guided", b_composable_guided), ("composable", b_composable), ("early", b_early)],
    "int2":  [("while", b_while)],
    "list":  [("list", b_list), ("dp", b_dp)],
    "listt": [("member", b_member)],
}


def solve(examples, kind):
    for name, backend in ROUTES.get(kind, []):
        try:
            code = backend(examples)
        except Exception:
            code = None
        if code and _verify(code, examples):
            return name, code
    return None, None


_SAFE = {"range": range, "len": len, "list": list, "max": max, "min": min, "set": set}


def _verify(code, examples):
    ns = {}
    exec(code, _SAFE, ns)
    f = ns["f"]
    try:
        return all(f(*a) == y for a, y in examples)
    except Exception:
        return False


import random as _rnd

GEN = {
    "int1":  lambda r: (r.randint(0, 25),),
    "int2":  lambda r: (r.randint(1, 40), r.randint(1, 40)),
    "list":  lambda r: ([r.randint(-9, 9) for _ in range(r.randint(1, 8))],),
    "listt": lambda r: (lambda L: (L, r.choice(L) if L and r.random() < .5
                                   else r.randint(0, 9)))(
        [r.randint(0, 9) for _ in range(r.randint(0, 6))]),
}


def stress(code, oracle, kind, n=1000, seed=0):
    """Run the synthesized program vs the real reference on n random inputs.
    Returns (survived, counterexample). 'Fits examples' -> 'survives 1000 cases'."""
    ns = {}
    exec(code, _SAFE, ns)
    f = ns["f"]
    rng = _rnd.Random(seed)
    for _ in range(n):
        args = GEN[kind](rng)
        try:
            exp = oracle(*args)
        except Exception:
            continue                         # oracle undefined here -> skip
        try:
            got = f(*args)
        except Exception:
            return False, args
        if got != exp:
            return False, args
    return True, None


# ── benchmark suite ───────────────────────────────────────────────────────────
def _ex(kind, fn, inputs):
    return [((x,), fn(x)) if kind in ("int1", "list") else (x, fn(*x)) for x in inputs]


BENCH = [
    ("triangular", "int1", lambda n: n * (n + 1) // 2, [1, 2, 3, 5, 8, 10]),
    ("factorial", "int1", lambda n: __import__("math").factorial(n), [0, 1, 4, 5, 6]),
    ("count_div", "int1", lambda n: sum(1 for i in range(1, n + 1) if n % i == 0),
        [1, 2, 6, 7, 12, 16]),
    ("smallest_factor", "int1", lambda n: next((i for i in range(2, n) if n % i == 0), n),
        [15, 7, 9, 11, 25, 8]),
    ("is_prime", "int1", lambda n: n >= 2 and all(n % i for i in range(2, n)),
        [1, 2, 3, 4, 5, 9, 11, 15]),
    ("sum_sq", "int1", lambda n: sum(i * i for i in range(1, n + 1)), [1, 2, 3, 4, 5]),
    ("gcd", "int2", __import__("math").gcd, [(12, 8), (48, 36), (7, 5), (100, 80), (17, 13)]),
    ("sum_list", "list", sum, [[1, 2, 3], [5, 5], [], [4], [2, 3, 4]]),
    ("max_list", "list", max, [[3, 1, 4, 1, 5], [2, 2], [7], [9, 1],
                               [-3, -5, -1], [-2, -9]]),   # negatives disambiguate init
    ("product", "list", lambda a: __import__("math").prod(a), [[1, 2, 3, 4], [5], [2, 3]]),
    ("max_subarray", "list", lambda a: max(sum(a[i:j + 1]) for i in range(len(a))
        for j in range(i, len(a))), [[1, -2, 3, 4], [-1, -2], [2, 3], [5, -1, 5]]),
    ("contains", "listt", lambda lst, t: t in lst,
        [([1, 2, 3], 2), ([1, 2, 3], 9), ([], 1), ([4, 5], 5)]),
    ("sort_asc", "list", sorted, [[3, 1, 2], [5, 4], [1], [2, 3, 1]]),
    ("has_dup", "list", lambda a: len(set(a)) != len(a),
        [[1, 2, 3], [1, 2, 2], [], [5, 5]]),
]


def _fib_task():
    def fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
    return ("fibonacci", "int1", fib, [0, 1, 2, 3, 7, 10])


def _demo():
    bench = BENCH + [_fib_task()]
    print("=== synth_engine — match real Python fns, gated by stress-vs-oracle ===\n")
    print(f"  {'task':16s} {'examples':>9s} {'stress(1000)':>13s}  [space]")
    weak = strong = 0
    for name, kind, fn, inputs in bench:
        ex = _ex(kind, fn, inputs)
        space, code = solve(ex, kind)
        if code is None:
            print(f"  {name:16s} {'—':>9s} {'—':>13s}  [—]")
            continue
        weak += 1
        surv, ce = stress(code, fn, kind)
        strong += surv
        tag = "SURVIVED ✓" if surv else f"OVERFIT ✗ {ce}"
        print(f"  {name:16s} {'fit ✓':>9s} {tag:>13s}  [{space}]")
    print(f"\n  fits-examples: {weak}/{len(bench)}    survives-1000-stress: {strong}/{len(bench)}")
    print("  the gap = overfits that example-fitting would have shipped. stress catches them.")


if __name__ == "__main__":
    _demo()
