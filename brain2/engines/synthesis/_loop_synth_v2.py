#!/usr/bin/env python3
"""
loop_synth2.py — richer algorithm synthesis: two accumulators + conditionals.

loop_synth did single-accumulator folds. This adds two control structures, taking
more algorithm classes away from the LLM and giving them to the verified synthesizer:

  TWO-STATE:   a,b = ia,ib ; for i in RANGE: a,b = NA(a,b,i), NB(a,b,i) ; return a|b
  COND-FOLD:   acc = INIT ; for i in RANGE: if COND(i,n): acc = UPDATE(acc,i) ; return acc

Search each space for a program fitting input/output examples, verify on held-out,
render Python. No LLM. Covers Fibonacci (two-state), count-of-divisors / sum-of-evens
(cond-fold) — branching / multi-state algorithms, not just folds.

    python3 loop_synth2.py
"""

RANGES = [("0", "n"), ("1", "n + 1"), ("1", "n"), ("2", "n + 1")]


def _rng(lo, hi, n):
    return range(int(lo), n + 1 if hi.endswith("+ 1") else n)


# two-state expressions over (a, b, i)
STATE = {
    "a": lambda a, b, i: a, "b": lambda a, b, i: b, "a + b": lambda a, b, i: a + b,
    "a + i": lambda a, b, i: a + i, "b + i": lambda a, b, i: b + i,
    "a * i": lambda a, b, i: a * i, "a + 1": lambda a, b, i: a + 1,
}
INITS2 = [(0, 1), (1, 1), (1, 0), (0, 0)]

# conditional-fold pieces
CONDS = {
    "n % i == 0": lambda i, n: n % i == 0, "i % 2 == 0": lambda i, n: i % 2 == 0,
    "i % 2 == 1": lambda i, n: i % 2 == 1, "True": lambda i, n: True,
    "i % 2 != 0": lambda i, n: i % 2 != 0,
}
UPD = {"acc + 1": lambda acc, i: acc + 1, "acc + i": lambda acc, i: acc + i,
       "acc * i": lambda acc, i: acc * i}


def _fit(fn, examples):
    cut = max(3, int(len(examples) * 0.6))
    return (all(fn(n) == y for n, y in examples[:cut]) and
            all(fn(n) == y for n, y in examples[cut:]))


def synth_two(examples):
    for ia, ib in INITS2:
        for lo, hi in RANGES:
            for na, naf in STATE.items():
                for nb, nbf in STATE.items():
                    for ret in ("a", "b"):
                        def f(n, ia=ia, ib=ib, lo=lo, hi=hi, naf=naf, nbf=nbf, ret=ret):
                            a, b = ia, ib
                            for i in _rng(lo, hi, n):
                                a, b = naf(a, b, i), nbf(a, b, i)
                            return a if ret == "a" else b
                        try:
                            if _fit(f, examples):
                                return ("two", dict(ia=ia, ib=ib, lo=lo, hi=hi,
                                                    na=na, nb=nb, ret=ret))
                        except Exception:
                            pass
    return None


def synth_cond(examples):
    for init in (0, 1):
        for lo, hi in RANGES:
            for cc, cf in CONDS.items():
                for uc, uf in UPD.items():
                    def f(n, init=init, lo=lo, hi=hi, cf=cf, uf=uf):
                        acc = init
                        for i in _rng(lo, hi, n):
                            if cf(i, n):
                                acc = uf(acc, i)
                        return acc
                    try:
                        if _fit(f, examples):
                            return ("cond", dict(init=init, lo=lo, hi=hi, cond=cc, upd=uc))
                    except Exception:
                        pass
    return None


def synthesize(examples):
    if len(examples) < 4:
        return None
    return synth_cond(examples) or synth_two(examples)


def render(fn, kind, s):
    if kind == "two":
        return (f"def {fn}(n):\n    a, b = {s['ia']}, {s['ib']}\n"
                f"    for i in range({s['lo']}, {s['hi']}):\n"
                f"        a, b = {s['na']}, {s['nb']}\n    return {s['ret']}\n")
    return (f"def {fn}(n):\n    acc = {s['init']}\n"
            f"    for i in range({s['lo']}, {s['hi']}):\n"
            f"        if {s['cond']}:\n            acc = {s['upd']}\n    return acc\n")


def _verify(code, fn, examples):
    ns = {}
    exec(code, {}, ns)
    return all(ns[fn](n) == y for n, y in examples)


def _demo():
    cases = {
        "fib":  [(0, 0), (1, 1), (2, 1), (3, 2), (7, 13), (10, 55)],       # two-state
        "ndiv": [(1, 1), (2, 2), (6, 4), (7, 2), (12, 6)],                 # count divisors
        "sumev": [(1, 0), (2, 2), (4, 6), (6, 12), (8, 20)],               # sum of evens<=n
        "sumodd": [(1, 1), (3, 4), (5, 9), (7, 16)],                       # sum of odds<=n
    }
    print("=== loop_synth2 — two accumulators + conditionals, no LLM ===\n")
    for fn, ex in cases.items():
        res = synthesize(ex)
        if res is None:
            print(f"  {fn:7s}: no program found")
            continue
        kind, s = res
        code = render(fn, kind, s)
        ok = _verify(code, fn, ex)
        last = " / ".join(l.strip() for l in code.splitlines()[1:-1])
        print(f"  {fn:7s} [{kind}]: {last}   [{'VERIFIED ✓' if ok else 'WRONG ✗'}]")
    print("\n  branching + multi-state algorithms, synthesized from examples and")
    print("  verified — the brain reclaims another class from the LLM.")


if __name__ == "__main__":
    _demo()
