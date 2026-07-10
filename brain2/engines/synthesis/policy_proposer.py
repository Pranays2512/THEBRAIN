#!/usr/bin/env python3
"""
policy_proposer.py — the Proposer wired onto the executive's policy memory.

means_ends.py gave each target ONE policy, so there was nothing to choose. Real
reasoning has MANY ways to reach a goal, most of them dead ends for the facts you
actually have. The Proposer is what makes that tractable: instead of trying
policies blindly (and wasting a whole sub-derivation on one whose leaf inputs
aren't available), it scores each candidate by how GROUNDABLE its inputs are and
tries the likely winner first. This is premise selection — the same idea proven
at 5.4x in program_synth_tree, now over the policy store.

The score here is a structural feature (fraction of a policy's inputs that bottom
out in known facts). A trained proposer would weight several such features; this
one feature is already the signal, and the membrane is intact — the proposer only
ORDERS the search, the verifier-backed executive still produces the answer.

  power = force*speed   OR   power = energy/time     (two policies, one target)
  facts: mass, accel, speed   (time is MISSING)
  blind tries energy/time first -> wastes the whole energy subtree -> backtracks.
  proposer scores force*speed higher (all inputs groundable) -> no waste.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engines.reasoning.reasoning_engine import ReasoningEngine
from engines.reasoning.means_ends import Need, Policy, FactSource, ev


# ── policy memory with MANY policies per target ──────────────────────────────
class MultiPolicyMemory:
    def __init__(self):
        self.by_target = {}

    def add(self, p: Policy):
        self.by_target.setdefault(p.target, []).append(p)

    def candidates(self, target):
        return self.by_target.get(target, [])

    def __contains__(self, target):
        return target in self.by_target


# ── the proposer: score a policy by how groundable its inputs are ────────────
def groundable(rel, mem, kb, seen=frozenset()):
    """Optimistic estimate in [0,1]: can `rel` be resolved for the entity?
    1 if it's a fact, else the best candidate policy's mean input groundability,
    0 if it's a leaf with no fact and no policy (a dead end)."""
    if rel in seen:
        return 0.0
    if kb(rel):                                  # fact present
        return 1.0
    cands = mem.candidates(rel)
    if not cands:
        return 0.0                               # leaf, not a fact -> ungroundable
    return max(
        sum(groundable(i, mem, kb, seen | {rel}) for i in p.inputs) / len(p.inputs)
        for p in cands)


def proposer_score(policy, mem, kb):
    """Mean groundability of a policy's inputs — higher = likelier to succeed."""
    return sum(groundable(i, mem, kb) for i in policy.inputs) / len(policy.inputs)


# ── solver that orders candidate policies (blind or proposer) + counts work ──
class Solver:
    def __init__(self, kb, mem, use_proposer):
        self.kb = kb
        self.mem = mem
        self.use_proposer = use_proposer
        self.work = 0                            # sub-goal attempts = search cost
        self.solved = {}

    def _fact(self, rel):                        # entity is fixed per solve()
        ans, _ = self.kb.ask(self.entity, rel)
        return ans is not None

    def solve(self, entity, rel):
        self.entity = entity
        return self._solve(Need(entity, rel), frozenset())

    def _solve(self, need, seen):
        self.work += 1
        if need in seen:                         # cycle guard (force->pressure->force)
            return None
        if need in self.solved:
            return self.solved[need]
        ans, _ = self.kb.ask(need.subject, need.rel)
        if ans is not None:
            self.solved[need] = float(ans)
            return float(ans)
        cands = self.mem.candidates(need.rel)
        if self.use_proposer:                    # premise selection: best first
            cands = sorted(cands, key=lambda p: -proposer_score(p, self.mem, self._fact))
        for p in cands:
            env, ok = {}, True
            for r in p.inputs:
                v = self._solve(Need(need.subject, r), seen | {need})
                if v is None:
                    ok = False
                    break
                env[r] = v
            if ok:
                val = ev(p.expr, env)
                self.solved[need] = val
                return val
        return None


# ── demo ─────────────────────────────────────────────────────────────────────
def _demo():
    kb = ReasoningEngine()
    for s, r, o in [("rocket", "mass", "1000"), ("rocket", "accel", "20"),
                    ("rocket", "speed", "300")]:               # NOTE: no 'time'
        kb.learn(s, r, o)

    mem = MultiPolicyMemory()
    mem.add(Policy("force",  ("mass", "accel"), ("*", "mass", "accel")))
    mem.add(Policy("energy", ("mass", "speed"), ("*", 0.5, ("*", "mass", ("^", "speed", 2)))))
    # two ways to get power; the energy/time one is a dead end (no 'time' fact).
    # add the DEAD one first so blind wastes the energy subtree before backtracking.
    mem.add(Policy("power", ("energy", "time"), ("/", "energy", "time")))
    mem.add(Policy("power", ("force", "speed"), ("*", "force", "speed")))

    print("=== policy_proposer — premise selection over policy memory ===\n")
    print("  two policies for 'power'; 'time' is missing so energy/time dead-ends.\n")
    for use in (False, True):
        s = Solver(kb, mem, use_proposer=use)
        ans = s.solve("rocket", "power")
        tag = "proposer" if use else "blind   "
        print(f"  {tag}:  power = {ans}   work (sub-goal attempts) = {s.work}")

    sb = Solver(kb, mem, False); sb.solve("rocket", "power")
    sp = Solver(kb, mem, True);  sp.solve("rocket", "power")
    print(f"\n  proposer does {sb.work / max(sp.work,1):.1f}x less work — it scored "
          f"force*speed (all inputs groundable) over energy/time (time missing),")
    print("  so it never explored the dead energy subtree. Same answer, less search.")


if __name__ == "__main__":
    _demo()
