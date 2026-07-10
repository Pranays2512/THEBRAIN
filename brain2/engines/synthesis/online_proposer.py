#!/usr/bin/env python3
"""
online_proposer.py — fix the weakest part: a proposer that LEARNS from every verified outcome.

The synth engine's proposer is STATIC: ROUTES tries the synthesis spaces in a fixed order
(e.g. list-tasks always try 'list' then 'dp'), so a workload whose answers live in the
second space wastes an attempt on every single task. The proposer never learns.

This wraps the SAME backends + the SAME verifier, but orders the spaces by LEARNED priors
and rewards the space that actually solved each task. Over a session the proposer adapts to
the workload: spaces that keep working get tried first, so attempts-per-solve drop. The
fix for the propose/verify asymmetry — generation that improves from experience.

Measured against the static order on the same tasks: fewer total backend attempts, with no
loss of solutions (same verifier gates every candidate). Honest limit: learns space-ORDER
from outcomes; richer per-task features (beyond kind) are the next rung.
"""

from engines.synthesis import synth_engine as SE


class OnlineProposer:
    def __init__(self):
        self.prior = {}                              # kind -> {space_name: weight}

    def _order(self, kind):
        backs = SE.ROUTES.get(kind, [])
        w = self.prior.setdefault(kind, {n: 1.0 for n, _ in backs})
        return sorted(backs, key=lambda nb: w[nb[0]], reverse=True)

    def solve(self, examples, kind):
        attempts = 0
        for name, backend in self._order(kind):
            attempts += 1
            try:
                code = backend(examples)
            except Exception:
                code = None
            if code and SE._verify(code, examples):
                self.prior[kind][name] += 2.0       # LEARN: this space worked -> try it first
                return name, code, attempts
        return None, None, attempts


def fixed_solve(examples, kind):
    """The static baseline: SE.ROUTES order, no learning."""
    attempts = 0
    for name, backend in SE.ROUTES.get(kind, []):
        attempts += 1
        try:
            code = backend(examples)
        except Exception:
            code = None
        if code and SE._verify(code, examples):
            return name, code, attempts
    return None, None, attempts


def _demo():
    import random
    msub = lambda a: max(sum(a[i:j + 1]) for i in range(len(a)) for j in range(i, len(a)))
    rng = random.Random(0)

    def rand_list(n=6):
        return [rng.randint(-9, 9) for _ in range(n)]

    # A DP-heavy 'list' workload: max-subarray tasks (answer lives in the 'dp' space, which
    # ROUTES tries SECOND) interleaved with a couple of fold tasks (the 'list' space).
    workload = []
    for _ in range(6):
        ins = [rand_list() for _ in range(5)]
        workload.append(("max_subarray", "list", msub, ins))
    for _ in range(2):
        ins = [rand_list() for _ in range(5)]
        workload.append(("sum_list", "list", sum, ins))
    rng.shuffle(workload)

    print("=== online_proposer — a proposer that learns the right space from outcomes ===\n")
    prop = OnlineProposer()
    fixed_total = online_total = 0
    print("  task          static  online   (attempts to solve)")
    for name, kind, oracle, ins in workload:
        ex = SE._ex(kind, oracle, ins)
        _, _, fa = fixed_solve(ex, kind)
        _, _, oa = prop.solve(ex, kind)
        fixed_total += fa; online_total += oa
        print("  %-13s %4d %7d" % (name, fa, oa))
    print("\n  total backend attempts:  static %d   vs   online %d  (%.0f%% fewer)"
          % (fixed_total, online_total, (1 - online_total / fixed_total) * 100))
    print("  learned space priors for 'list':", {k: round(v, 1) for k, v in prop.prior["list"].items()})
    print("  The proposer learned 'dp' solves this workload and tries it first — attempts drop,")
    print("  same verifier, no lost solutions. Generation that improves from experience.")


if __name__ == "__main__":
    _demo()
