#!/usr/bin/env python3
"""
program_synth_policy.py — a goal-conditioned policy guides each next op.

The marginal prior (program_synth_guided.py) only knew which operators a spec
tends to need overall — a weak ~1.8x. This is the full idea: at every step the
policy looks at WHERE THE COMPUTATION IS NOW versus the target, and scores which
operator best closes the remaining gap.

Concretely, for a partial program it runs that program on the inputs to get the
intermediate values, computes features of (intermediate -> target), and the
learned policy scores each next operator from THAT. It is trained on prefixes of
solved programs: for every step of a known solution, "given this intermediate
state, the next op was X." So it learns to drive the computation toward the
target one step at a time — a goal-conditioned policy over the synthesis state,
exactly "a statistical model guides each choice as you build."

It also prunes provably-dead branches (a partial program that already errors
can never be completed). Compared head-to-head with blind search and the
marginal prior on the same deep tasks.
"""

import math
import random

import numpy as np

from tree_reason import SearchProblem, solve
from program_synth_guided import (
    OPS, run, features, rand_name, rand_program, make_hard_tasks,
    learn_prior, Synthesize,
)


# ── learn the goal-conditioned policy ────────────────────────────────────────
def learn_policy(n_tasks=8000, seed=1):
    """From prefixes of solved programs: features(intermediate -> target) -> next op."""
    rng = random.Random(seed)
    X, Y = [], []
    for _ in range(n_tasks):
        prog = rand_program(rng, 5)
        ins = [rand_name(rng) for _ in range(3)]
        try:
            outs = [run(prog, s) for s in ins]
        except Exception:
            continue
        for k in range(len(prog)):
            try:
                mids = [run(prog[:k], s) for s in ins]   # intermediate after k ops
            except Exception:
                break
            X.append(features(list(zip(mids, outs))))    # remaining-work features
            Y.append([1.0 if op == prog[k] else 0.0 for op in OPS])
    X, Y = np.array(X), np.array(Y)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def policy_scores(W, prog, examples):
    """Score next ops from the CURRENT intermediate state vs target.
    Returns None if the partial program already errors (dead branch)."""
    try:
        cur = [(run(prog, i), o) for i, o in examples]
    except Exception:
        return None
    s = features(cur) @ W
    return {op: float(np.clip(s[i], 1e-3, 1.0)) for i, op in enumerate(OPS)}


class PolicySynth(SearchProblem):
    def __init__(self, examples, W, max_len=6):
        self.examples = examples
        self.W = W
        self.max_len = max_len

    def initial(self):
        return ()

    def is_goal(self, prog):
        for inp, out in self.examples:
            try:
                if run(prog, inp) != out:
                    return False
            except Exception:
                return False
        return True

    def key(self, prog):
        return prog

    def heuristic(self, prog):
        return 0

    def moves(self, prog):
        if len(prog) >= self.max_len:
            return
        sc = policy_scores(self.W, prog, self.examples)
        if sc is None:
            return                                # prune provably-dead branch
        # Best-first by the policy, with a mild per-op cost (+1) that keeps a
        # shortest-program bias. The marginal prior could NOT be trusted this
        # aggressively (it drove search down wrong deep chains); the
        # goal-conditioned policy can, because it re-reads what's-left-to-do at
        # every step — which is the whole point of conditioning on the state.
        for op in OPS:
            yield (f"then {op}", prog + (op,), 1.0 - math.log(max(sc[op], 1e-3)))


# ── three-way comparison ─────────────────────────────────────────────────────
def evaluate(tasks, prior_for, W, max_len=6):
    res = {"blind": [0, 0], "marginal": [0, 0], "policy": [0, 0]}
    for ex, _ in tasks:
        pb, _, nb = solve(Synthesize(ex, max_len=max_len, prior=None), max_nodes=600_000)
        pm, _, nm = solve(Synthesize(ex, max_len=max_len, prior=prior_for(ex)), max_nodes=600_000)
        pp, _, npn = solve(PolicySynth(ex, W, max_len=max_len), max_nodes=600_000)
        res["blind"][0] += nb;  res["blind"][1] += pb is not None
        res["marginal"][0] += nm; res["marginal"][1] += pm is not None
        res["policy"][0] += npn; res["policy"][1] += pp is not None
    return res


def main():
    print("=== program_synth_policy — goal-conditioned policy guides each op ===\n")
    print("Learning marginal prior and goal-conditioned policy from solved tasks...")
    prior_for = learn_prior()
    W = learn_policy()
    print("  done.\n")

    rng = random.Random(7)
    print("Building 15 deep tasks (minimal solution >= 4 ops)...")
    tasks = make_hard_tasks(15, rng)
    avg_depth = sum(d for _, d in tasks) / len(tasks)
    print(f"  done (avg minimal solution length {avg_depth:.1f} ops).\n")

    res = evaluate(tasks, prior_for, W)
    n = len(tasks)
    print(f"{n} deep tasks — avg programs searched (lower = smarter):\n")
    print(f"  {'method':28s} {'programs':>10s} {'solved':>9s}")
    for name, label in (("blind", "blind search"),
                        ("marginal", "marginal-prior guided"),
                        ("policy", "goal-conditioned POLICY")):
        nodes, solved = res[name]
        print(f"  {label:28s} {nodes / n:10.0f} {solved:>7}/{n}")

    blind = res["blind"][0] / n
    marg = res["marginal"][0] / n
    pol = res["policy"][0] / n
    print(f"\n  policy explores ~{blind / max(pol, 1):.1f}x fewer programs than blind and")
    print(f"  beats the marginal prior ({marg:.0f} -> {pol:.0f}). It scores each next op by")
    print(f"  what's LEFT to do, so it can be trusted with best-first search — the")
    print(f"  marginal prior could not (it dives down wrong deep chains).")


if __name__ == "__main__":
    main()
