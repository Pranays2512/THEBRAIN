#!/usr/bin/env python3
"""
dual_process_engine.py — hardened Dual cognition (milestone #8, the last rung).

Reflex (System 1) + deliberation (System 2) + compilation, as a clean solver.
Three tiers, fastest first:
  1. compiled memory — a cache of already-solved tasks (instant)
  2. policy reflex   — greedy rollout of the learned tree policy, NO search
  3. deliberation    — full search for the genuinely novel
and every deliberated (or reflex) solution is COMPILED into the cache, so a
recurring task is answered instantly next time.

    s = DualProcessSolver(train_policy())
    r = s.solve(examples)        # r.tier in {memory, reflex, deliberation}
    r.found, r.apply(new_input)

Composes hardened pieces: the tree policy (reflex) and the search engine
(deliberation). Whatever tier answers, the program is correct on the examples.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engines.reasoning.tree_reason import solve
from engines.synthesis.program_synth_guided import OPS, run, Synthesize
from engines.synthesis.program_synth_tree import DecisionTree, collect, tree_scores


def train_policy(seed=1):
    X, y = collect(seed=seed)
    return DecisionTree(len(OPS)).fit(X, y)


class DualResult:
    def __init__(self, found, program, tier):
        self.found = found
        self.program = program
        self.tier = tier            # "memory" | "reflex" | "deliberation"

    def apply(self, s):
        if not self.found:
            raise ValueError("no program to apply")
        return run(self.program, s)

    def __repr__(self):
        return f"DualResult(found={self.found}, tier={self.tier!r}, program={self.program})"


class DualProcessSolver:
    def __init__(self, policy, max_len=6):
        self.policy = policy
        self.max_len = max_len
        self.cache = {}
        self.stats = {"memory": 0, "reflex": 0, "deliberation": 0, "unsolved": 0}

    def _reflex(self, examples):
        """System 1: greedy rollout of the policy, no search."""
        prog = ()
        for _ in range(self.max_len):
            sc = tree_scores(self.policy, prog, examples)
            if sc is None:
                return None
            prog = prog + (max(OPS, key=lambda o: sc[o]),)
            try:
                if all(run(prog, i) == o for i, o in examples):
                    return prog
            except Exception:
                return None
        return None

    def solve(self, examples, max_nodes=200_000):
        sig = tuple(examples)
        if sig in self.cache:                       # 1. compiled memory
            self.stats["memory"] += 1
            return DualResult(True, self.cache[sig], "memory")

        prog = self._reflex(examples)               # 2. policy reflex
        if prog is not None:
            self.cache[sig] = prog
            self.stats["reflex"] += 1
            return DualResult(True, prog, "reflex")

        path, _, _ = solve(Synthesize(examples, max_len=self.max_len, prior=None),
                           max_nodes)               # 3. deliberation
        if path is None:
            self.stats["unsolved"] += 1
            return DualResult(False, None, "deliberation")
        prog = path[-1][1] if path else ()
        self.cache[sig] = prog                      # compile into reflex memory
        self.stats["deliberation"] += 1
        return DualResult(True, prog, "deliberation")


def _demo():
    s = DualProcessSolver(train_policy())
    ex = [("John Smith", "JS"), ("Mary Jane", "MJ")]
    print("DualProcessSolver demo:")
    r1 = s.solve(ex)
    print(f"  first time:  tier={r1.tier}, program={r1.program}")
    r2 = s.solve(ex)
    print(f"  second time: tier={r2.tier} (compiled to instant memory)")
    print(f"  stats: {s.stats}")


if __name__ == "__main__":
    _demo()
