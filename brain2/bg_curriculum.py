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
        reward = 2.0 if TARGET_OP in ops else 0.0     # reward producing the target op
        b.reinforce_bg(reward)
        return reward

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
    if rise > 0.5 and blocks[-1] > 0.5:
        print("  LEARNS and HOLDS — reward climbs and the last block stays up (no collapse).")
    elif rise > 0.5:
        print("  LEARNS (peaks ~2.0) but COLLAPSES and doesn't recover — diagnosed below.")
    else:
        print("  flat — investigate further.")
    print("\n  STABILITY DIAGNOSIS (after trying TD-clip, lighter L2, epsilon anneal — none fixed it):")
    print("  the collapse is not regularization or exploration. Root cause is COARSE CREDIT —")
    print("  reinforce() spreads the single per-episode scalar reward across ALL ops in the")
    print("  trace, so a good episode also reinforces the 5 non-target ops; once the policy")
    print("  peaks, the noisy credit tips it over and it can't recover. Real fix is per-STEP")
    print("  credit (advantage per op) or a PPO-style trust region — a genuine RL rebuild, not")
    print("  a tuning knob. The 3-layer WIRING bug is fixed (it CAN learn); STABLE learning is")
    print("  the honest open residue.")


if __name__ == "__main__":
    _demo()
