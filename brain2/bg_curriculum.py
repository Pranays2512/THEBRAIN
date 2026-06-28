#!/usr/bin/env python3
"""
bg_curriculum.py — does the BG actually LEARN to reason over symbolic content? (Gen #5, real test)

bg_symbolic_loop wired the path; with a constant reward the BG never improved (no gradient).
This is the real RL test: a SHAPED reward + epsilon-greedy exploration, many episodes, measured.

  each episode: bridge symbols -> reason(goal, epsilon) (explore) -> shaped reward -> reinforce_bg
  reward = number of DISTINCT ops the executive used (degenerate looping of one op scores low;
           engaging the symbol-bearing scratchpad with varied ops scores high)

If the BG learns from the reward, average reward should climb over training. This script
reports the trend honestly — whatever it is. A flat trend means the wiring is sound but the
reward/representation needs more work; a rising trend means the executive learned.
"""

import brain2


def _demo():
    b = brain2.Brain(som_rows=8, som_cols=8, n_dims=8)
    b.perceive_text("mass accel force energy speed momentum density volume")

    def episode(eps):
        b.sync_symbols_to_scratchpad()
        ops = b.reason("force", max_steps=6, epsilon=eps)
        reward = float(len(set(ops)))          # diversity: engage content, don't loop one op
        b.reinforce_bg(reward)
        return reward

    print("=== bg_curriculum — does the BG learn to use symbolic content? ===\n")
    print("  reward = distinct ops used; epsilon-greedy exploration; 200 episodes\n")
    WIN = 40
    windows = []
    win = []
    for ep in range(200):
        win.append(episode(0.3))
        if len(win) == WIN:
            avg = sum(win) / WIN
            windows.append(avg)
            print("  episodes %3d-%3d : avg reward %.2f" % (ep - WIN + 1, ep, avg))
            win = []

    first, last = windows[0], windows[-1]
    trend = last - first
    print("\n  trend: first window %.2f -> last window %.2f  (%+.2f)" % (first, last, trend))
    if trend > 0.3:
        print("  RISING — the BG learned to use the symbol-bearing scratchpad more richly.")
    elif trend < -0.3:
        print("  FALLING — reward signal is pushing the wrong way; reward design needs work.")
    else:
        print("  FLAT — the learning PATH is sound (reward applied every episode) but this")
        print("  reward/representation doesn't yet move the policy. Honest residue: the reward")
        print("  shaping + op semantics need design work. The plumbing is correct; mastery isn't free.")


if __name__ == "__main__":
    _demo()
