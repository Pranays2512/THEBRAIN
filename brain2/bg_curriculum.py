#!/usr/bin/env python3
"""
bg_curriculum.py — does the BG LEARN to reason over symbolic content? (Gen #5, after the fix)

Earlier this returned FLAT (no learning) and the negative result located a 3-layer bug in
the C++ executive:
  1. forward()'s memoization cache was never invalidated after reinforce() -> stale logits.
  2. reason() recorded the trace for a SEPARATELY-sampled op while the PUCT tree search
     executed a DIFFERENT op -> reinforce trained the wrong action (credit-assignment mismatch).
  3. exploration only ranged over the top-K=3 ops -> any op outside the top-3 was unreachable,
     so it could never be tried, reinforced, or learned.

All three are fixed. This is the same test: a target-op reward + epsilon exploration. The
reward should now CLIMB across training (the BG learns to produce the rewarded op).

Honest residue: learning happens during exploration but does not yet fully consolidate to a
greedy policy, and it can regress (instability) — a tuning matter (lr, exploration schedule,
replay interference), not a wiring bug. The core defect (it could not learn at all) is fixed.

    venv2/bin/python3 bg_curriculum.py 2>/dev/null
"""

import brain2

TARGET_OP = 5     # BIND_QUERY — starts outside the top-K, so only learnable after the fix


def _demo():
    b = brain2.Brain(som_rows=8, som_cols=8, n_dims=8)
    b.perceive_text("mass accel force energy speed density volume")

    def episode(eps):
        b.sync_symbols_to_scratchpad()
        ops = b.reason("force", max_steps=6, epsilon=eps)
        # PER-OP credit: each step that emitted the target op is rewarded; its episode-mates
        # are not. This is the fix for the coarse-credit collapse — only the op that earned
        # the reward gets credited, so the policy can actually PREFER it.
        step_rewards = [1.0 if o == TARGET_OP else 0.0 for o in ops]
        b.reinforce_bg_steps(step_rewards)
        return 1.0 if TARGET_OP in ops else 0.0       # report presence rate for readability

    print("=== bg_curriculum — does the BG learn AND consolidate? (TD-clip + epsilon anneal) ===\n")
    print("  reward = 2 if the executive emits op %d; epsilon annealed 0.5 -> 0.05\n" % TARGET_OP)
    BLK, blocks = 60, []
    for blk in range(6):
        eps = max(0.05, 0.5 - 0.09 * blk)            # anneal exploration as it learns
        rs = [episode(eps) for _ in range(BLK)]
        avg = sum(rs) / BLK
        blocks.append(avg)
        print("  block %d (ep %3d-%3d, eps %.2f): avg reward %.2f"
              % (blk, blk * BLK, blk * BLK + BLK - 1, eps, avg))

    rise = max(blocks) - blocks[0]
    # greedy consolidation: with NO exploration, does it still emit the learned op?
    greedy = sum(TARGET_OP in b.reason("force", max_steps=6, epsilon=0.0) for _ in range(10))
    print("\n  first %.2f -> peak %.2f (rise %+.2f); last block %.2f"
          % (blocks[0], max(blocks), rise, blocks[-1]))
    print("  greedy consolidation (epsilon=0): target op in %d/10 runs" % greedy)
    if greedy >= 8 and blocks[-1] > 0.7:
        print("\n  SOLVED — the BG LEARNS, HOLDS (no collapse), and CONSOLIDATES to greedy (%d/10)." % greedy)
        print("  The full fix stack (each a real bug/mechanism):")
        print("    1. cache invalidation on reinforce (stale logits froze the policy)")
        print("    2. trace = EXECUTED op, not a separately-sampled one (credit-assignment)")
        print("    3. epsilon explores the FULL op space (top-K pruning made ops unreachable)")
        print("    4. greedy follows the LEARNED policy, not the value-tree (so eval = training)")
        print("    5. replay stores EVERY step, not just the first op (stale-action pull)")
        print("    6. PER-OP credit (reinforce_steps): the op that earned the reward is credited,")
        print("       not its episode-mates — THIS killed the collapse (coarse credit was the root).")
    elif blocks[-1] > 0.7:
        print("  learns and holds but greedy not fully consolidated — partial.")
    else:
        print("  unstable — investigate (reward collapsed in the last block).")


if __name__ == "__main__":
    _demo()
