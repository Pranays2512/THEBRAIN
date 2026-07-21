#!/usr/bin/env python3
"""
program_synth_tree.py — the policy as a decision TREE.

The reasoning engine is a search tree; its guide should be a tree too. The
linear policy capped at ~3.8x because it can't model feature INTERACTIONS —
"if the intermediate still has spaces AND the target is short -> initials" is
not a weighted sum. A decision tree splits on exactly those interactions.

It is also the right model for this project: lightweight, fast (traverse, no
forward pass), trains with no backprop (greedy splits), dependency-free, and
INTERPRETABLE — the policy is readable if-then rules.

Same goal-conditioned setup as program_synth_policy: at each step, features of
(intermediate -> target), the tree predicts a distribution over the next op,
and best-first search follows it (dead branches pruned). Compared head-to-head
with blind search and the linear policy on the same deep tasks and features.
"""

import math
import random

import numpy as np

from engines.reasoning.tree_reason import SearchProblem, solve
from engines.synthesis.program_synth_guided import OPS, run, features, rand_name, rand_program, make_hard_tasks
from engines.synthesis.program_synth_policy import learn_policy, policy_scores as linear_scores, PolicySynth


# ── a small, dependency-free multiclass decision tree ────────────────────────
class DecisionTree:
    def __init__(self, n_ops, max_depth=10, min_samples=15):
        self.n_ops = n_ops
        self.max_depth = max_depth
        self.min_samples = min_samples

    def _dist(self, y):
        c = np.bincount(y, minlength=self.n_ops).astype(float) + 0.1   # smoothing
        return c / c.sum()

    def _gini(self, y):
        d = self._dist(y)
        return 1.0 - float((d * d).sum())

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples or len(set(y)) == 1:
            return ("leaf", self._dist(y))
        base = self._gini(y) * len(y)
        best = None
        for f in range(X.shape[1]):
            mask = X[:, f] > 0.5
            nl, nr = int(mask.sum()), int((~mask).sum())
            if nl == 0 or nr == 0:
                continue
            cost = self._gini(y[mask]) * nl + self._gini(y[~mask]) * nr
            gain = base - cost
            if best is None or gain > best[0]:
                best = (gain, f, mask)
        if best is None or best[0] <= 1e-9:
            return ("leaf", self._dist(y))
        _, f, mask = best
        return ("node", f,
                self._build(X[mask], y[mask], depth + 1),
                self._build(X[~mask], y[~mask], depth + 1))

    def fit(self, X, y):
        self.root = self._build(np.asarray(X), np.asarray(y), 0)
        return self

    def predict_dist(self, x):
        node = self.root
        while node[0] == "node":
            node = node[2] if x[node[1]] > 0.5 else node[3]
        return node[1]


# ── training data: prefixes of solved programs (goal-conditioned) ────────────
def collect(n_tasks=8000, seed=1):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n_tasks):
        prog = rand_program(rng, 5)
        ins = [rand_name(rng) for _ in range(3)]
        try:
            outs = [run(prog, s) for s in ins]
        except Exception:
            continue
        for k in range(len(prog)):
            try:
                mids = [run(prog[:k], s) for s in ins]
            except Exception:
                break
            X.append(features(list(zip(mids, outs))))
            y.append(OPS.index(prog[k]))
    return np.array(X), np.array(y)


def tree_scores(tree, prog, examples):
    try:
        cur = [(run(prog, i), o) for i, o in examples]
    except Exception:
        return None
    d = tree.predict_dist(features(cur))
    return {op: float(max(d[i], 1e-3)) for i, op in enumerate(OPS)}


class TreeSynth(SearchProblem):
    def __init__(self, examples, tree, max_len=6):
        self.examples = examples
        self.tree = tree
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
        sc = tree_scores(self.tree, prog, self.examples)
        if sc is None:
            return
        for op in OPS:
            yield (f"then {op}", prog + (op,), 1.0 - math.log(max(sc[op], 1e-3)))


def main():
    print("=== program_synth_tree — the policy as a decision tree ===\n")
    print("Collecting prefix data and fitting linear policy + decision tree...")
    X, y = collect()
    W = learn_policy()
    tree = DecisionTree(len(OPS)).fit(X, y)
    print(f"  done ({len(y)} training steps).\n")

    rng = random.Random(7)
    print("Building 15 deep tasks (minimal solution >= 4 ops)...")
    tasks = make_hard_tasks(15, rng)
    print(f"  done (avg solution {sum(d for _, d in tasks) / len(tasks):.1f} ops).\n")

    rows = {"blind": [0, 0], "linear": [0, 0], "tree": [0, 0]}
    for ex, _ in tasks:
        from engines.synthesis.program_synth_guided import Synthesize
        pb, _, nb = solve(Synthesize(ex, max_len=6, prior=None), max_nodes=600_000)
        pl, _, nl = solve(PolicySynth(ex, W, max_len=6), max_nodes=600_000)
        pt, _, nt = solve(TreeSynth(ex, tree, max_len=6), max_nodes=600_000)
        rows["blind"][0] += nb; rows["blind"][1] += pb is not None
        rows["linear"][0] += nl; rows["linear"][1] += pl is not None
        rows["tree"][0] += nt; rows["tree"][1] += pt is not None

    n = len(tasks)
    print(f"{n} deep tasks — avg programs searched:\n")
    print(f"  {'method':28s} {'programs':>10s} {'solved':>9s}")
    for k, label in (("blind", "blind search"),
                     ("linear", "linear policy"),
                     ("tree", "DECISION-TREE policy")):
        nodes, solved = rows[k]
        print(f"  {label:28s} {nodes / n:10.0f} {solved:>7}/{n}")
    blind, tree_n = rows["blind"][0] / n, rows["tree"][0] / n
    print(f"\n  tree policy explores ~{blind / max(tree_n, 1):.1f}x fewer programs than blind")
    print(f"  — it splits on feature INTERACTIONS the linear model can't, stays")
    print(f"  lightweight, trains with no backprop, and is readable if-then rules.")


if __name__ == "__main__":
    main()
