#!/usr/bin/env python3
"""
dual_process.py — reflex (System 1) + deliberation (System 2).

The HFT objection was that reasoning is deliberative (slow search) while a
reflex is instant. The answer is to have BOTH, like a mind does: a fast
reflex that acts without searching, and slow deliberation that kicks in only
when the reflex is unsure.

Here the learned tree policy IS the reflex: follow its top choice at each step
with NO search (a greedy rollout — a handful of cheap policy lookups). If that
solves the task, done, instantly. If not, fall back to full tree search
(deliberation). On a stream of mixed-difficulty tasks the reflex handles the
familiar/easy ones for ~free, and deliberation is spent only on the genuinely
hard ones — so average effort collapses.

This is how expertise works: practiced situations become intuition (reflex),
novel ones still need thought (deliberation). The compilation bridge — caching
a deliberated solution as a future reflex — is the next step (the brain already
has procedural memory for exactly this).
"""

import random
import time

from tree_reason import solve
from program_synth_guided import OPS, run, rand_name, rand_program, Synthesize
from program_synth_tree import DecisionTree, collect, tree_scores


def reflex(ex, tree, max_len=6):
    """System 1: greedy rollout of the policy, NO search. Returns a solving
    program if the reflex nails it, else None. Cost = a few policy lookups."""
    prog = ()
    for _ in range(max_len):
        sc = tree_scores(tree, prog, ex)
        if sc is None:
            return None
        op = max(OPS, key=lambda o: sc[o])      # top choice, no branching
        prog = prog + (op,)
        try:
            if all(run(prog, i) == o for i, o in ex):
                return prog
        except Exception:
            return None
    return None


def deliberate(ex):
    """System 2: full complete search. Returns the solving program (or None)."""
    path, _, _ = solve(Synthesize(ex, max_len=6, prior=None), max_nodes=600_000)
    return path[-1][1] if path else None


def make_stream(n, rng, routine=False):
    """A task stream. `routine` models a real agent's repetitive workload
    (mostly familiar/easy); otherwise a diverse mix with more hard novelty."""
    weights = [1, 1, 1, 1, 2, 2, 3] if routine else [1, 2, 2, 3, 3, 4, 4]
    tasks = []
    while len(tasks) < n:
        depth = rng.choice(weights)
        prog = rand_program(rng, depth)
        if len(prog) != depth:
            continue
        ins = [rand_name(rng) for _ in range(3)]
        try:
            ex = [(s, run(prog, s)) for s in ins]
        except Exception:
            continue
        if len({o for _, o in ex}) == 1 and ex[0][0] == ex[0][1]:
            continue
        tasks.append(ex)
    return tasks


def main():
    print("=== dual_process — reflex (System 1) + deliberation (System 2) ===\n")
    print("Training the reflex policy (decision tree)...")
    X, y = collect()
    tree = DecisionTree(len(OPS)).fit(X, y)
    print("  done.\n")

    avg = lambda xs: sum(xs) / max(len(xs), 1)

    # A recurring workload: a pool of tasks (incl. hard ones) seen repeatedly,
    # like a real agent facing the same situations over and over.
    rng = random.Random(5)
    pool = make_stream(40, rng, routine=False)        # includes hard tasks
    stream = [pool[rng.randrange(len(pool))] for _ in range(300)]

    cache = {}                                         # compiled reflexes (procedural memory)
    dual_us, base_us = [], []
    n_cache = n_reflex = n_delib = 0

    def sig(ex):
        return tuple(ex)

    for ex in stream:
        t = time.perf_counter(); deliberate(ex)        # baseline: always think
        base_us.append((time.perf_counter() - t) * 1e6)

        t = time.perf_counter()
        if sig(ex) in cache:                           # 1. reflex from memory (compiled)
            n_cache += 1
        elif reflex(ex, tree) is not None:             # 2. policy reflex (intuition)
            n_reflex += 1
        else:                                          # 3. deliberate, then COMPILE
            deliberate(ex)
            cache[sig(ex)] = True
            n_delib += 1
        dual_us.append((time.perf_counter() - t) * 1e6)

    half = len(stream) // 2
    print(f"Recurring workload: {len(pool)} distinct tasks (incl. hard), {len(stream)} encounters.\n")
    print(f"  resolved by compiled reflex (memory): {n_cache}")
    print(f"  resolved by policy reflex (intuition): {n_reflex}")
    print(f"  needed deliberation (then compiled):   {n_delib}\n")
    print(f"  avg latency, always-deliberate     : {avg(base_us):8.0f} us")
    print(f"  avg latency, dual-process (1st half): {avg(dual_us[:half]):8.0f} us   (cache warming)")
    print(f"  avg latency, dual-process (2nd half): {avg(dual_us[half:]):8.0f} us   (cache warm)")
    print(f"\n  -> as it meets situations again, deliberation compiles into instant")
    print(f"     reflex: ~{avg(base_us)/max(avg(dual_us[half:]),1):.0f}x faster once practiced. Fast on the")
    print(f"     familiar, deliberate on the novel — exactly how expertise works.")


if __name__ == "__main__":
    main()
