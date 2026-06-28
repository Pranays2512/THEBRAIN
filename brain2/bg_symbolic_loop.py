#!/usr/bin/env python3
"""
bg_symbolic_loop.py — close the BG <-> Symbolic LEARNING path (Gen #5, deeper step).

The earlier commit added the data BRIDGE (symbolic content <-> scratchpad). This wires the
LEARNING loop on top of it, end to end:

  bind symbols  ->  sync_symbols_to_scratchpad (bridge symbolic onto the BG's surface)
                ->  reason(goal)  (the BG selects ops over a scratchpad now holding symbols)
                ->  reward         (did the reasoning engage the symbolic content?)
                ->  reinforce_bg   (TD update — the BG learns from the outcome)

So the executive's policy can now be trained to USE symbolic knowledge mid-reasoning,
which the islands architecture made impossible. This script demonstrates the loop runs end
to end and reinforcement applies across episodes.

Honest scope: this wires and exercises the loop; it does NOT claim the BG MASTERS symbolic
reasoning. Real mastery needs a designed task, a shaped reward, many episodes, and held-out
evaluation — a full RL curriculum. That is the remaining ML project; the path is now connected.

    venv2/bin/python3 bg_symbolic_loop.py
"""

import brain2


def _demo():
    print("=== bg_symbolic_loop — the BG<->Symbolic learning path, end to end ===\n")
    b = brain2.Brain(som_rows=8, som_cols=8, n_dims=8)
    # populate the symbolic store + SOM with some content (perceive_text binds symbols)
    b.perceive_text("mass accel force energy speed momentum")

    moved = b.sync_symbols_to_scratchpad()
    print("  bridged %d symbols onto the scratchpad (the BG's working surface)\n" % moved)

    rewards = []
    for ep in range(12):
        b.sync_symbols_to_scratchpad()                 # symbolic content present each episode
        ops = b.reason("force", max_steps=6)           # BG selects ops over symbol-bearing scratchpad
        # reward: engaging the content (a non-trivial op sequence) rather than halting immediately
        reward = 1.0 if len(ops) >= 2 else 0.0
        b.reinforce_bg(reward)                          # TD update — the BG learns from the outcome
        rewards.append(reward)
        if ep < 3 or ep == 11:
            print("  ep %2d: reason -> %d ops %s  reward %.0f  -> reinforced"
                  % (ep, len(ops), ops[:6], reward))

    print("\n  loop ran end to end: symbolic content reached the BG, it acted, and TD")
    print("  reinforcement applied every episode (avg reward %.2f over %d eps)."
          % (sum(rewards) / len(rewards), len(rewards)))
    print("  The path that makes 'the BG learns to use symbolic mid-reasoning' trainable is")
    print("  now connected. A full curriculum (shaped reward, many eps, eval) is the next ML step.")


if __name__ == "__main__":
    _demo()
