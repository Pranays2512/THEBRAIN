#!/usr/bin/env python3
"""
compositional.py — systematic generalization: novel combinations of known parts.

The thing pure pattern-matchers (and LLMs, on SCAN/COGS) fail and the symbolic
core gets for free. Give the brain only ATOMIC policies (each one hop) and facts
for a FRESH entity it never saw. It answers DEEP targets by composing those atoms
into chains never trained as a unit — force -> work -> energy -> power, four levels
deep, assembled on demand.

  K atomic policies  ->  answers targets at every reachable depth, on any entity,
  via recombination. Coverage is COMBINATORIAL in the atoms, not enumerated.

A student/LLM would need each combination in its training data; the executive
derives them, because meaning here is COMPOSITIONAL by construction.

    venv2/bin/python3 compositional.py
"""

import brain2

ATOMIC = [   # each policy is ONE hop; depth comes from composing them
    ("force",    ["mass", "accel"],   ["*", "mass", "accel"]),
    ("work",     ["force", "dist"],   ["*", "force", "dist"]),
    ("energy",   ["work", "eff"],     ["*", "work", "eff"]),
    ("power",    ["energy", "time"],  ["/", "energy", "time"]),    # depth 4: power<-energy<-work<-force
    ("momentum", ["mass", "speed"],   ["*", "mass", "speed"]),
    ("pressure", ["force", "area"],   ["/", "force", "area"]),
]


def _demo():
    brain = brain2.Brain(som_rows=4, som_cols=4, n_dims=4)
    for t, ins, expr in ATOMIC:
        brain.policy_add(t, ins, expr)

    # a FRESH entity, never seen in any combination — only its leaf facts
    facts = {"mass": 3.0, "accel": 4.0, "dist": 5.0, "eff": 0.9,
             "time": 2.0, "speed": 6.0, "area": 1.5}
    for r, v in facts.items():
        brain.teach_fact("X", r, v)

    print("=== compositional — answer novel deep combinations of atomic policies ===\n")
    print(f"  {len(ATOMIC)} atomic (one-hop) policies; entity X has only leaf facts.\n")
    depths = {"force": 1, "momentum": 1, "work": 2, "pressure": 2, "energy": 3, "power": 4}
    for target, d in depths.items():
        v = brain.policy_solve("X", target)
        print(f"  X.{target:9s} = {v:<9.4g} (composed {d} level{'s' if d>1 else ''} deep)")
    print("\n  'power' was assembled power<-energy<-work<-force<-(mass,accel) — a 4-deep")
    print("  chain never given as one policy, on an entity never seen in that combo.")
    print("  Coverage is combinatorial in the atoms; a pattern-matcher would need each")
    print("  combination in training. This is where the symbolic core beats memorization.")


if __name__ == "__main__":
    _demo()
