#!/usr/bin/env python3
"""
policy_pack.py — stress the executive + proposer with REAL multi-domain policies.

Loads a pack of physics + chemistry composition rules into one shared policy
memory (the "unified cross-subject policy memory" idea), with MULTIPLE policies
per target and several deliberately dead routes (their leaf inputs aren't known
for the sample). Then runs a battery of queries, blind vs proposer, and reports
the aggregate work saved. As the pack grows and dead routes multiply, the
proposer's edge climbs — the same premise-selection win that hits 5.4x in
program_synth_tree, here over cross-domain policies.

Membrane intact: facts stay in the ReasoningEngine, policies in the pack; they
meet only on the blackboard. Every answer is still produced by the verified
executive — the proposer only orders the search.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from reasoning_engine import ReasoningEngine
from means_ends import Policy
from policy_proposer import MultiPolicyMemory, Solver

# ── the pack: physics + chemistry composition rules ──────────────────────────
# (target, inputs, formula).  Several targets have >1 route on purpose.
# Multi-route targets list the DEAD route first (needs a missing fact), so blind
# trips into it before backtracking; the proposer scores it low and skips it.
PACK = [
    # physics
    ("force",     ("mass", "accel"),          ("*", "mass", "accel")),
    ("weight",    ("mass", "gravity"),        ("*", "mass", "gravity")),       # dead (no gravity)
    ("momentum",  ("force", "time"),          ("*", "force", "time")),         # route1 DEAD (no time)
    ("momentum",  ("mass", "speed"),          ("*", "mass", "speed")),         # route2 live
    ("ke",        ("mass", "speed"),          ("*", 0.5, ("*", "mass", ("^", "speed", 2)))),
    ("pe",        ("mass", "gravity", "height"), ("*", ("*", "mass", "gravity"), "height")),
    ("work",      ("power", "time"),          ("*", "power", "time")),         # route1 DEAD (cycle+time)
    ("work",      ("force", "distance"),      ("*", "force", "distance")),     # route2 live
    ("power",     ("work", "time"),           ("/", "work", "time")),          # route1 DEAD (needs time)
    ("power",     ("force", "speed"),         ("*", "force", "speed")),        # route2 live
    ("pressure",  ("weight", "area"),         ("/", "weight", "area")),        # route1 DEAD (weight dead)
    ("pressure",  ("force", "area"),          ("/", "force", "area")),         # route2 live
    ("impulse",   ("force", "time"),          ("*", "force", "time")),         # dead if no time
    # chemistry
    ("moles",     ("mass", "molar_mass"),     ("/", "mass", "molar_mass")),
    ("molarity",  ("moles", "volume"),        ("/", "moles", "volume")),
    ("density",   ("mass", "volume"),         ("/", "mass", "volume")),
    ("conc_mass", ("molarity", "molar_mass"), ("*", "molarity", "molar_mass")),
]


def build_memory():
    mem = MultiPolicyMemory()
    for t, ins, expr in PACK:
        mem.add(Policy(t, ins, expr))
    return mem


def build_facts():
    kb = ReasoningEngine()
    # 'time' and 'gravity'/'height' deliberately absent -> some routes dead-end.
    facts = {"mass": 2.0, "accel": 9.8, "speed": 30.0, "distance": 5.0,
             "area": 0.25, "molar_mass": 18.0, "volume": 0.5}
    for r, v in facts.items():
        kb.learn("sample", r, str(v))
    return kb


def _demo():
    kb, mem = build_facts(), build_memory()
    queries = ["force", "work", "power", "ke", "momentum", "pressure",
               "moles", "molarity", "density", "conc_mass", "impulse"]

    print("=== policy_pack — proposer over unified physics+chem policy memory ===")
    print(f"  {len(PACK)} policies, facts known: mass,accel,speed,distance,area,"
          f"molar_mass,volume  (time/gravity/height MISSING)\n")
    print(f"  {'query':10s} {'answer':>14s} {'blind':>7s} {'prop':>6s}")
    tot_b = tot_p = 0
    for q in queries:
        sb = Solver(kb, mem, use_proposer=False); ab = sb.solve("sample", q)
        sp = Solver(kb, mem, use_proposer=True);  ap = sp.solve("sample", q)
        tot_b += sb.work
        tot_p += sp.work
        ans = "—" if ap is None else f"{ap:.3g}"
        print(f"  {q:10s} {ans:>14s} {sb.work:>7} {sp.work:>6}")
    print(f"\n  TOTAL work — blind {tot_b}, proposer {tot_p}  "
          f"=>  {tot_b / max(tot_p,1):.2f}x less search")
    print("  proposer skips routes whose leaf inputs aren't known (e.g. power via")
    print("  work/time when 'time' is missing). Same answers, less search; the")
    print("  verifier-backed executive still produces every value.")


if __name__ == "__main__":
    _demo()
