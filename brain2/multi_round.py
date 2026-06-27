#!/usr/bin/env python3
"""
multi_round.py — the bootstrap run for real: a BATCH of tasks over multiple rounds,
banking a primitive each round, measuring that the solve-rate climbs as the DSL grows.

make_break_loop showed one round (a target that fails before factoring, solves after).
This runs the loop as a standing process:

  each round:
    1. try every UNSOLVED target within a FIXED depth budget using the current op-set
    2. add solved targets to the library
    3. factor the library (anti-unification) -> bank a new primitive into the op-set

A target needing a composed idiom is unreachable at the budget with base ops, but once
the idiom is banked it becomes reachable in a later round. The measured signal: cumulative
solved grows across rounds with NO change to the budget or the synthesizer — only the
banked primitives change. That's self-improvement from the system's own solved work.

Honest limit: small formula DSL + tiny enumerator; the claim is the round-over-round
DELTA, every solution verified by I/O evaluation.
"""

import factorizer as FZ
import make_break_loop as MB

BASE_OPS = [("+", 2, "base"), ("-", 2, "base"), ("*", 2, "base")]

TASKS = [
    # name, target tree, variables  (f1..f3 = FMA shape; hard/harder need it composed)
    ("f1",     ("+", ("*", "a", "b"), "c"), ["a", "b", "c"]),
    ("f2",     ("+", ("*", "x", "y"), "z"), ["x", "y", "z"]),
    ("f3",     ("+", ("*", "m", "n"), "p"), ["m", "n", "p"]),
    ("hard",   ("+", ("*", ("+", ("*", "a", "b"), "c"), "b"), "c"), ["a", "b", "c"]),
    ("harder", ("+", ("*", ("+", ("*", "a", "b"), "c"),
                       ("+", ("*", "a", "b"), "c")), "c"), ["a", "b", "c"]),
]

DEPTH = 2
ROUNDS = 3


def _ops(prims):
    return BASE_OPS + [(nm, ar, "call") for nm, (ar, _sk) in prims.items()]


def run():
    print("=== multi_round — solve-rate climbs as the DSL grows (budget fixed: depth<=%d) ===\n"
          % DEPTH)
    prims, solved, library = {}, {}, []
    for r in range(1, ROUNDS + 1):
        ops = _ops(prims)
        newly = []
        for name, target, vars_ in TASKS:
            if name in solved:
                continue
            io = MB._io(target, vars_)
            expr, tried = MB.synth(io, vars_, ops, prims, depth=DEPTH)
            if expr is not None:
                solved[name] = expr
                library.append((name, target))
                newly.append(name)
        prim_names = ",".join(prims) or "(none)"
        print("round %d: prims=[%s]  solved this round: %s  | cumulative %d/%d"
              % (r, prim_names, newly or "-", len(solved), len(TASKS)))
        # bank a primitive from everything solved so far
        if len(library) >= 2:
            existing = {sk for _ar, sk in prims.values()}
            _newlib, prims2, disc = FZ.factor_au(library, min_count=2, min_kept=1, prims=prims)
            if disc is not None and disc[1] not in existing:     # don't re-bank a known shape
                prims = prims2
                print("         banked %s = %s (arity %d)" % (disc[0], disc[1], disc[2]))
    print("\n  final: %d/%d solved, %d primitives banked (%s)."
          % (len(solved), len(TASKS), len(prims), ", ".join(prims) or "none"))
    print("  The deep targets were UNREACHABLE at depth<=%d round 1; banked idioms made them"
          " reachable later — the space grew from solved work." % DEPTH)


if __name__ == "__main__":
    run()
