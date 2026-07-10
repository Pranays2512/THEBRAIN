#!/usr/bin/env python3
"""
refuter.py — the make/break loop's BREAK half: decompose a built rule by attacking it.

The brain composes (synth/induction) and the verifier confirms ("does it pass?").
The refuter is the verifier turned aggressive: "WHERE does it FAIL, and where does it
still HOLD?" Breaking a rule is how you find the gap a NEW rule must fill — so this is
both a diagnostic (pinpoint an overfit) and the generator of the next conjecture.

Given a program (code defining f) or any callable, plus an oracle and a task kind:
  - hunt a counterexample (random + structured edge probes),
  - characterise the VALID SCOPE (for int1, the integer ranges where it holds; for
    lists, which structural property breaks it — empty / singleton / negatives /
    duplicates / sorted).

Honest limit: like everything here, refutation needs an oracle. No oracle -> can only
report "unbroken in N samples", not "correct". It finds where two things DISAGREE.
"""

import math
import random

from core.synthesis import synth_engine as SE


def _load(code_or_fn):
    if callable(code_or_fn):
        return code_or_fn
    ns = {}
    exec(code_or_fn, SE._SAFE, ns)
    return ns["f"]


def _eq(f, oracle, args):
    try:
        return f(*args) == oracle(*args)
    except Exception:
        try:
            oracle(*args)          # oracle defined but f threw -> a real break
            return False
        except Exception:
            return None            # oracle undefined here -> not a fair test


def _int1_scope(f, oracle, lo=0, hi=60):
    """Scan single-int inputs; return the integer intervals where f == oracle."""
    holds = []
    for n in range(lo, hi + 1):
        r = _eq(f, oracle, (n,))
        if r is True:
            holds.append(n)
    if not holds:
        return "holds nowhere in [%d,%d]" % (lo, hi)
    # compress contiguous runs into intervals
    runs, start, prev = [], holds[0], holds[0]
    for n in holds[1:]:
        if n == prev + 1:
            prev = n
        else:
            runs.append((start, prev)); start = prev = n
    runs.append((start, prev))
    txt = " ∪ ".join("[%d,%d]" % (a, b) if a != b else "{%d}" % a for a, b in runs)
    full = runs == [(lo, hi)]
    return ("holds for ALL n in [%d,%d]" % (lo, hi)) if full else ("holds for n in " + txt)


def _list_probes(f, oracle):
    """Which structural property of a list breaks it? Probe named edge classes."""
    cases = {
        "empty":      [[]],
        "singleton":  [[7], [-3]],
        "negatives":  [[-3, -5, -1], [-2, -9, -4]],
        "duplicates": [[5, 5, 5], [2, 2, 3, 3]],
        "sorted":     [[1, 2, 3, 4], [0, 5, 9]],
        "reverse":    [[9, 5, 1], [4, 3, 2, 1]],
        "mixed":      [[3, -1, 4, -1, 5], [-2, 8, -6, 7]],
    }
    broken = []
    for name, inputs in cases.items():
        for lst in inputs:
            if _eq(f, oracle, (lst,)) is False:
                broken.append(name); break
    if not broken:
        return "holds on all probed list classes"
    return "BREAKS on: " + ", ".join(broken)


def refute(code_or_fn, oracle, kind, n=2000, seed=0):
    """Attack a rule. Returns a dict:
        robust      — bool, no counterexample found
        breaks_at   — first counterexample input (or None)
        fail_rate   — fraction of fair random samples that disagreed
        scope       — human description of where it HOLDS / which property breaks it
    """
    f = _load(code_or_fn)
    rng = random.Random(seed)
    breaks_at, fails, fair = None, 0, 0
    for _ in range(n):
        args = SE.GEN[kind](rng)
        r = _eq(f, oracle, args)
        if r is None:
            continue
        fair += 1
        if r is False:
            fails += 1
            if breaks_at is None:
                breaks_at = args
    if kind == "int1":
        scope = _int1_scope(f, oracle)
    elif kind in ("list", "listt"):
        scope = _list_probes(f, oracle) if kind == "list" else "list+target task"
    else:
        scope = "scope-scan not defined for kind=%s" % kind
    return {
        "robust": breaks_at is None,
        "breaks_at": breaks_at,
        "fail_rate": (fails / fair) if fair else 0.0,
        "scope": scope,
    }


def _demo():
    print("=== refuter — break a rule, find where it holds ===\n")
    # 1. an OVERFIT program: max_list with init=0 (fits positives, breaks on all-negatives)
    overfit = ("def f(a):\n"
               "    m = 0\n"
               "    for x in a:\n"
               "        if x > m: m = x\n"
               "    return m\n")
    r = refute(overfit, max, "list")
    print("max_list with init=0 (classic overfit):")
    print("   robust:", r["robust"], "| breaks_at:", r["breaks_at"],
          "| fail_rate: %.2f" % r["fail_rate"])
    print("   scope:", r["scope"], "\n")

    # 2. a formula valid only in a RANGE: triangular-number approx n^2/2 (off by n/2)
    approx = lambda n: (n * n) // 2
    true_tri = lambda n: n * (n + 1) // 2
    r = refute(approx, true_tri, "int1")
    print("n^2//2 vs true triangular n(n+1)/2:")
    print("   robust:", r["robust"], "| breaks_at:", r["breaks_at"])
    print("   scope:", r["scope"], "\n")

    # 3. a CORRECT program: nothing to break
    good = ("def f(a):\n"
            "    m = a[0]\n"
            "    for x in a:\n"
            "        if x > m: m = x\n"
            "    return m\n")
    r = refute(good, max, "list")
    print("correct max_list:")
    print("   robust:", r["robust"], "| scope:", r["scope"])
    print("\n  The refuter pinpoints the overfit's failing region — the gap a new rule must fill.")


if __name__ == "__main__":
    _demo()
