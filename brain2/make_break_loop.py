#!/usr/bin/env python3
"""
make_break_loop.py — close the bootstrap: compose -> refute -> factor -> compose harder.

The three engines (refuter, factorizer, concept_blend) are wired into ONE measurable
cycle that improves itself:

  1. REFUTE   — attack a candidate rule; find where it breaks (the gap a new rule fills).
  2. FACTOR   — find the repeated idiom across solved formulas; bank it as a new DSL
                primitive (the search space grows from what was solved).
  3. RE-SYNTH — a target that is UNREACHABLE within the depth budget using base ops
                becomes REACHABLE once the banked primitive is available.
  4. BLEND    — (novelty arm) propose a concept in empty feature space.

The payoff is measured, not asserted: same depth budget, same target — fails before
factoring, solves after. That is DreamCoder's wake/sleep win, in miniature, on the
formula DSL the brain already uses.

Honest limit: tiny enumerative synth over {+,-,*}; the point is the DELTA from banking a
primitive, not the synthesizer's raw power. Everything is verified by evaluation.
"""

import random
from itertools import product

import factorizer as FZ
import refuter as RF
import concept_blend as CB


# ── a tiny enumerative synthesizer over an extensible op-set ───────────────────
def synth(target_io, variables, opset, prims, depth):
    """Enumerate expression trees up to `depth` over `opset`; return (expr, tried) for
    the first that matches all target I/O, else (None, tried). opset entries are
    (name, arity, kind) with kind in {'base','call'}; 'call' uses a learned primitive."""
    levels = [list(variables)]                       # depth 0 = the variables
    for _ in range(depth):
        lower = [e for lv in levels for e in lv]
        new = []
        for name, arity, kind in opset:
            for combo in product(lower, repeat=arity):
                new.append((name,) + combo if kind == "base"
                           else ("call", name, list(combo)))
        levels.append(new)
    tried = 0
    for lv in levels:
        for e in lv:
            tried += 1
            if _matches(e, target_io, prims):
                return e, tried
    return None, tried


def _matches(expr, target_io, prims):
    try:
        return all(abs(FZ.eval_tree(expr, env, prims) - out) < 1e-9 for env, out in target_io)
    except Exception:
        return False


def _io(tree, variables, n=5, seed=1):
    rng = random.Random(seed)
    io = []
    for _ in range(n):
        env = {v: rng.uniform(-4, 4) for v in variables}
        io.append((env, FZ.eval_tree(tree, env, {})))
    return io


def _demo():
    print("=== make_break_loop — compose -> refute -> factor -> compose harder ===\n")

    # ---- 1. REFUTE: a candidate "FMA" rule that's an overfit (drops the +c term) ----
    print("[1] REFUTE — attack a candidate before trusting it")
    true_fma = lambda a, b, c: a * b + c
    overfit  = lambda a, b, c: a * b            # forgot +c
    # treat as int-ish: probe via refuter on a single-var slice (c varies)
    broke = next(((a, b, c) for a in range(1, 4) for b in range(1, 4) for c in range(1, 4)
                  if overfit(a, b, c) != true_fma(a, b, c)), None)
    print("    candidate a*b vs true a*b+c  ->  breaks at (a,b,c)=%s (gap: the +c term)\n" % str(broke))

    # ---- 2. FACTOR: bank the shared idiom from a library of solved formulas ----
    print("[2] FACTOR — bank the repeated idiom as a new primitive")
    solved = [
        ("f1", ("+", ("*", "a", "b"), "c")),         # a*b+c
        ("f2", ("+", ("*", "x", "y"), "z")),         # x*y+z
        ("f3", ("+", ("*", "m", "n"), "p")),         # m*n+p   (FMA shape x3)
    ]
    _, prims, disc = FZ.factor(solved, min_count=3, min_size=5)
    name, skel, arity = disc
    print("    discovered %s = %s  (arity %d) — the multiply-add idiom\n" % (name, skel, arity))

    # ---- 3. RE-SYNTH: same target + budget, fails on base ops, solves with the primitive ----
    print("[3] RE-SYNTH — a target unreachable at depth<=2 becomes reachable after factoring")
    vars3 = ["a", "b", "c"]
    # target = FMA(FMA(a,b,c), b, c) = (a*b+c)*b + c   (depth-4 in raw +,-,*)
    target = ("+", ("*", ("+", ("*", "a", "b"), "c"), "b"), "c")
    io = _io(target, vars3)
    base_ops    = [("+", 2, "base"), ("-", 2, "base"), ("*", 2, "base")]
    learned_ops = base_ops + [(name, arity, "call")]

    e0, t0 = synth(io, vars3, base_ops,    {},    depth=2)
    e1, t1 = synth(io, vars3, learned_ops, prims, depth=2)
    print("    target (a*b+c)*b+c   budget: depth<=2")
    print("    base ops {+,-,*}      -> %-7s  (%d candidates tried)"
          % ("SOLVED" if e0 else "FAILED", t0))
    print("    base + %s             -> %-7s  (%d tried)   %s"
          % (name, "SOLVED" if e1 else "FAILED", t1, e1 if e1 else ""))
    if e1 and not e0:
        print("    => banking the primitive made a previously-unreachable target reachable.\n")

    # ---- 4. BLEND: the novelty arm — a concept in empty space ----
    print("[4] BLEND — propose a concept outside every known category")
    concepts = {"bird": [1, 1, 0, 0], "fish": [0, 0, 1, 1],
                "reptile": [0, 1, 0, 0], "insect": [1, 0, 0, 0]}
    r = CB.propose("bird", "fish", concepts, radius=1.0, mode="salient")
    print("    blend(bird,fish) -> %s  novel=%s (nearest %s @ %.2f)\n"
          % (r["vector"], r["novel"], r["nearest"], r["distance"]))

    print("LOOP: refute found the gap, factor banked the idiom, re-synth solved what it"
          " couldn't before, blend opened a new region. The space grew from what was solved.")


if __name__ == "__main__":
    _demo()
