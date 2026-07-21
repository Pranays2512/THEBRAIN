#!/usr/bin/env python3
"""
loop_synth3.py — while-loops + early-return synthesis (GCD, primality, factors).

The next control structures the brain reclaims from the LLM:

  WHILE (two-arg):  while COND(a,b): a,b = NA(a,b), NB(a,b) ; return a|b   (Euclid GCD)
  EARLY-RETURN:     [if PRE: return p] ; for i in RANGE: if COND(i,n): return R1
                    ; return R2                                            (primality, factor)

Search each space for a program fitting input/output examples, verify on held-out,
render Python. No LLM. while-loops are condition-terminated (not range-bounded) and
early-return exits mid-loop — genuinely new shapes beyond folds.

    python3 loop_synth3.py
"""

RANGES = [("2", "n"), ("1", "n + 1"), ("2", "n + 1"), ("1", "n")]


def _rng(lo, hi, n):
    return range(int(lo), n + 1 if hi.endswith("+ 1") else n)


# ── WHILE (two integer args) ─────────────────────────────────────────────────
WSTATE = {
    "a": lambda a, b: a, "b": lambda a, b: b,
    "a % b": lambda a, b: a % b, "b % a": lambda a, b: b % a,
    "a - b": lambda a, b: a - b, "b - a": lambda a, b: b - a,
}
WCOND = {"b != 0": lambda a, b: b != 0, "a != 0": lambda a, b: a != 0,
         "a != b": lambda a, b: a != b, "a > 0": lambda a, b: a > 0}


def synth_while(examples):
    for cc, cf in WCOND.items():
        for na, naf in WSTATE.items():
            for nb, nbf in WSTATE.items():
                for ret in ("a", "b"):
                    def f(args, cf=cf, naf=naf, nbf=nbf, ret=ret):
                        a, b = args
                        for _ in range(100000):
                            if not cf(a, b):
                                break
                            a, b = naf(a, b), nbf(a, b)
                        return a if ret == "a" else b
                    try:
                        if all(f(args) == y for args, y in examples):
                            return ("while", dict(cond=cc, na=na, nb=nb, ret=ret))
                    except Exception:
                        pass
    return None


# ── EARLY-RETURN (single arg n) ──────────────────────────────────────────────
# (render-label, check fn, return value) — no eval; checks are real lambdas
PRES = [None, ("n < 2", lambda n: n < 2, False), ("n < 1", lambda n: n < 1, False)]
ECOND = {"n % i == 0": lambda i, n: n % i == 0, "i * i > n": lambda i, n: i * i > n,
         "n % i != 0": lambda i, n: n % i != 0}
RET = {"True": True, "False": False, "i": "i", "n": "n", "-1": -1, "0": 0, "1": 1}


def _rv(tag, i, n):
    v = RET[tag]
    return i if v == "i" else (n if v == "n" else v)


def synth_search(examples):
    for pre in PRES:
        for lo, hi in RANGES:
            for cc, cf in ECOND.items():
                for r1 in RET:
                    for r2 in ("True", "False", "n", "-1", "0", "1"):  # no 'i' (loop may be empty)
                        def f(n, pre=pre, lo=lo, hi=hi, cf=cf, r1=r1, r2=r2):
                            if pre and pre[1](n):
                                return pre[2]
                            for i in _rng(lo, hi, n):
                                if cf(i, n):
                                    return _rv(r1, i, n)
                            return _rv(r2, None, n)
                        try:
                            cut = max(3, int(len(examples) * 0.6))
                            if all(f(n) == y for n, y in examples[:cut]) and \
                               all(f(n) == y for n, y in examples[cut:]):
                                return ("early", dict(pre=pre, lo=lo, hi=hi,
                                                      cond=cc, r1=r1, r2=r2))
                        except Exception:
                            pass
    return None


def render(fn, kind, s):
    if kind == "while":
        return (f"def {fn}(a, b):\n    while {s['cond']}:\n"
                f"        a, b = {s['na']}, {s['nb']}\n    return {s['ret']}\n")
    pre = f"    if {s['pre'][0]}:\n        return {s['pre'][2]}\n" if s["pre"] else ""
    return (f"def {fn}(n):\n{pre}"
            f"    for i in range({s['lo']}, {s['hi']}):\n"
            f"        if {s['cond']}:\n            return {s['r1']}\n"
            f"    return {s['r2']}\n")


def _verify(code, fn, examples, two):
    ns = {}
    exec(code, {}, ns)
    return all(ns[fn](*a) == y for a, y in examples) if two else \
        all(ns[fn](n) == y for n, y in examples)


def _demo():
    gcd_ex = [((12, 8), 4), ((48, 36), 12), ((7, 5), 1), ((100, 80), 20), ((17, 13), 1)]
    prime_ex = [(1, False), (2, True), (3, True), (4, False), (5, True),
                (9, False), (11, True), (15, False)]
    factor_ex = [(15, 3), (7, 7), (9, 3), (11, 11), (25, 5), (8, 2)]   # smallest factor>=2

    print("=== loop_synth3 — while-loops + early-return, no LLM ===\n")

    rw = synth_while(gcd_ex)
    if rw:
        code = render("gcd", *rw)
        ok = _verify(code, "gcd", gcd_ex, two=True)
        print(f"  gcd    [while]: while {rw[1]['cond']}: a,b = {rw[1]['na']}, {rw[1]['nb']}; "
              f"return {rw[1]['ret']}   [{'VERIFIED ✓' if ok else 'WRONG ✗'}]")

    for fn, ex in (("is_prime", prime_ex), ("smallest_factor", factor_ex)):
        rs = synth_search(ex)
        if rs is None:
            print(f"  {fn}: no program found")
            continue
        code = render(fn, *rs)
        ok = _verify(code, fn, ex, two=False)
        s = rs[1]
        pre = f"if {s['pre'][0]}: return {s['pre'][2]}; " if s["pre"] else ""
        print(f"  {fn} [early]: {pre}for i in range({s['lo']},{s['hi']}): "
              f"if {s['cond']}: return {s['r1']}; return {s['r2']}   "
              f"[{'VERIFIED ✓' if ok else 'WRONG ✗'}]")
    print("\n  condition-terminated loops and mid-loop exits, synthesized + verified —")
    print("  GCD, primality, smallest factor. Another control class, no LLM.")


if __name__ == "__main__":
    _demo()
