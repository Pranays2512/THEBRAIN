#!/usr/bin/env python3
"""
online_proposer2.py — richer features (transfer) + loop signals (sandbox/refuter) for the proposer.

online_proposer learned space-order keyed only by `kind`. That can't separate task FAMILIES
inside one kind: e.g. 'list' holds both sort tasks (solved by the 'list' space) and
subarray tasks (solved by 'dp'). Keyed by kind alone, learning one helps and HURTS the
other. This keys priors on a richer feature SIGNATURE of the task (list-vs-scalar output,
negatives, output-grows, output-exceeds-max-element), so each family learns its own best
space — and a NEW task transfers to the right space by matching signature, not by repetition.

Loop signals feed the same priors:
  sandbox/stress survived -> reward the space (this shape holds)
  refuter: candidate broke under stress -> penalize the space (this shape breaks here)

Measured vs kind-only and vs the static order on a MIXED workload (sort + subarray
interleaved): the feature proposer solves both families near-optimally; kind-only can't.

Honest limit: a hand-picked feature set; learning the features themselves is the next rung.
"""

import random

from engines.synthesis import synth_engine as SE
from engines.synthesis.online_proposer import OnlineProposer, fixed_solve


def _flat(a):
    out = []
    for x in a:
        out += list(x) if isinstance(x, (list, tuple)) else [x]
    return [v for v in out if isinstance(v, (int, float))]


def features(examples, kind):
    """A task signature: cheap structural flags that separate families needing different spaces."""
    f = {"scalar" if kind in ("int1", "int2") else "listish"}
    args = [a for a, _ in examples]
    ys = [y for _, y in examples]
    if all(isinstance(y, list) for y in ys):
        f.add("out_list")                            # sort-shaped
    else:
        f.add("out_scalar")
    in_vals = [v for a in args for v in _flat(a)]
    if any(v < 0 for v in in_vals):
        f.add("neg_in")
    # output can EXCEED the largest single input element (subarray/DP signature)
    exceeds = False
    for a, y in examples:
        fv = _flat(a)
        if fv and isinstance(y, (int, float)) and y > max(fv):
            exceeds = True
    if exceeds:
        f.add("out_exceeds_max")
    return frozenset(f)


class FeatureProposer:
    def __init__(self):
        self.prior = {}                              # feature-signature -> {space: weight}

    def _order(self, kind, sig):
        backs = SE.ROUTES.get(kind, [])
        w = self.prior.setdefault(sig, {n: 1.0 for n, _ in backs})
        for n, _ in backs:
            w.setdefault(n, 1.0)
        return sorted(backs, key=lambda nb: w[nb[0]], reverse=True)

    def solve(self, examples, kind, oracle=None):
        sig = features(examples, kind)
        attempts = 0
        for name, backend in self._order(kind, sig):
            attempts += 1
            try:
                code = backend(examples)
            except Exception:
                code = None
            if not code or not SE._verify(code, examples):
                continue
            # sandbox/refuter signal: does it SURVIVE stress, or BREAK?
            survived = SE.stress(code, oracle, kind)[0] if oracle else True
            if survived:
                self.prior[sig][name] += 2.0         # this shape holds -> prefer it
                return name, code, attempts
            self.prior[sig][name] -= 1.0             # refuter: broke under stress -> avoid
        return None, None, attempts


def _demo():
    rng = random.Random(1)
    msub = lambda a: max(sum(a[i:j + 1]) for i in range(len(a)) for j in range(i, len(a)))

    def rlist(n=6):
        return [rng.randint(-9, 9) for _ in range(n)]

    # MIXED 'list' workload: sort tasks (family A) + subarray tasks (family B), interleaved.
    workload = []
    for _ in range(5):
        workload.append(("subarray", "list", msub, [rlist() for _ in range(5)]))
    for _ in range(5):
        workload.append(("sort", "list", sorted, [rlist() for _ in range(5)]))
    rng.shuffle(workload)

    print("=== online_proposer2 — richer features (transfer) + sandbox/refuter signal ===\n")
    kindp = OnlineProposer()
    featp = FeatureProposer()
    ft = kt = ft2 = 0
    print("  task       static  kind-only  feature   (attempts)")
    for name, kind, oracle, ins in workload:
        ex = SE._ex(kind, oracle, ins)
        _, _, fa = fixed_solve(ex, kind)
        _, _, ka = kindp.solve(ex, kind)
        _, _, fa2 = featp.solve(ex, kind, oracle)
        ft += fa; kt += ka; ft2 += fa2
        print("  %-9s %5d %9d %8d" % (name, fa, ka, fa2))
    print("\n  total attempts:  static %d   kind-only %d   feature %d" % (ft, kt, ft2))
    print("  feature signatures learned:")
    for sig, w in featp.prior.items():
        fam = "sort-family" if "out_list" in sig else "subarray-family"
        print("    %-15s -> %s" % (fam, {k: round(v, 1) for k, v in w.items()}))
    print("\n  Keyed by kind alone, one family's learning hurts the other. Keyed by feature")
    print("  signature, each family learns its own best space and NEW tasks transfer by match.")


if __name__ == "__main__":
    _demo()
