"""
BRAIN IN WORLD — 100k step run with all fixes + Q-spread pos_gate
==================================================================
All four fixes applied:
  Fix 1: M56 random tie-breaking
  Fix 2: Wall penalty -0.05
  Fix 3: L3 → Brain → M57 zone scoring
  Fix 4: Q-spread epsilon gate (positive-only)

Extended to 100k closed-loop steps.
Adds per-node Q-direction tracking to see policy convergence.
"""

import time
import numpy as np
from collections import deque, Counter

from m50_neuron import (
    run_sim, build_reverse_lookup, make_blocks,
    compute_stability_plv, decode_resonance,
    DivergenceCUSUM, stabilization_time,
)
from brain import Brain
from world import World, NODES, FREQUENCIES, FOOD_NODES, ADJACENCY, ACTIONS

OPEN_STEPS_PER_NODE = 1_500
CLOSED_LOOP_STEPS   = 200_000
REPORT_EVERY        = 10_000
CAL_BLOCK_DUR       = 40.0
SIGNAL_BLOCK_DUR_S  = 60.0
PLV_STAB_WINDOW     = 20
SEED                = 42

# Optimal actions per node (ground truth for policy evaluation)
OPTIMAL_ACTION = {
    'A': 1,   # East → B → C → E★
    'B': 1,   # East → C → E★
    'C': 2,   # South → E★
    'D': 2,   # South → F → G → H★
    'E': 0,   # North → C (return; E is food, any exit is fine)
    'F': 1,   # East → G → H★
    'G': 1,   # East → H★
    'H': 3,   # West → G (return; H is food)
}
ACTION_NAMES = ['North', 'East', 'South', 'West']


def calibrate():
    print("=" * 64)
    print("  CALIBRATING M50 EAR")
    print("=" * 64)
    sig, _ = make_blocks(FREQUENCIES, block_dur=CAL_BLOCK_DUR)
    total  = stabilization_time + 2 * len(FREQUENCIES) * CAL_BLOCK_DUR + 10.0
    print(f"  Sim time {total:.0f}s — running...", end="", flush=True)
    t0 = time.time()
    np.random.seed(1)
    data = run_sim(sig, total_time=total, sweep_mode=False,
                   dynamic_settle=True, verbose=False, collect_calib=True)
    print(f" done ({time.time()-t0:.1f}s)")
    rx_slow, ry_slow = build_reverse_lookup(
        sorted(data['calib_plv_slow'].keys()),
        data['calib_plv_slow'], data['calib_energy_slow'])
    rx_fast, ry_fast = build_reverse_lookup(
        sorted(data['calib_plv_fast'].keys()),
        data['calib_plv_fast'], data['calib_energy_fast'])
    print(f"  Slow: {len(rx_slow)} pts  Fast: {len(rx_fast)} pts")
    return rx_slow, ry_slow, rx_fast, ry_fast


def build_signal_library(rx_slow, ry_slow, rx_fast, ry_fast):
    print("\n" + "=" * 64)
    print("  BUILDING SIGNAL LIBRARY")
    print("=" * 64)
    library = {}
    for fi, freq in enumerate(FREQUENCIES):
        node = list(NODES.keys())[fi]
        print(f"  {node} ({freq:.1f}Hz)...", end="", flush=True)
        sig, _ = make_blocks([freq], block_dur=SIGNAL_BLOCK_DUR_S)
        total  = stabilization_time + SIGNAL_BLOCK_DUR_S + 5.0
        np.random.seed(200 + fi)
        data = run_sim(sig, total_time=total, sweep_mode=False,
                       dynamic_settle=False, verbose=False)
        plv_hist = deque(maxlen=PLV_STAB_WINDOW)
        cusum    = DivergenceCUSUM()
        steps    = []
        for i in range(len(data['Y'])):
            plv_hist.append(float(np.max(data['plv_slow'][i])))
            w  = compute_stability_plv(plv_hist)
            df = decode_resonance(data['plv_fast'][i], data['energy_fast'][i], rx_fast, ry_fast)
            ds = decode_resonance(data['plv_slow'][i], data['energy_slow'][i], rx_slow, ry_slow)
            _, is_novel = cusum.update(df, ds, data['T'][i], w=w)
            if is_novel: w = 0.0
            decoded = float(w * ds + (1.0 - w) * df)
            if w > 0.5:
                steps.append((decoded, float(w), float(is_novel), data['plv_slow'][i].copy()))
        library[freq] = steps
        print(f" {len(steps)} steps")
    return library


def get_step(library, freq_hz, counters):
    steps = library[freq_hz]
    if not steps:
        return 1.0, 0.5, 0.0, np.zeros(64)
    idx = counters.get(freq_hz, 0) % len(steps)
    counters[freq_hz] = idx + 1
    return steps[idx]


def run_open_loop(brain, library):
    total_open = OPEN_STEPS_PER_NODE * len(FREQUENCIES)
    print(f"\n  PHASE 1: OPEN LOOP  ({total_open:,} steps)")
    counters = {}
    for fi, freq in enumerate(FREQUENCIES):
        node = list(NODES.keys())[fi]
        print(f"  {node} ({freq:.1f}Hz)...", end="", flush=True)
        for _ in range(OPEN_STEPS_PER_NODE):
            decoded, w, nov, plv = get_step(library, freq, counters)
            brain.step(decoded_freq=decoded, stability_w=w, novelty_flag=nov,
                       plv_vector=plv, reward=0.0, freq_idx=fi)
        print(" done")
    brain.l3.assign_zones_from_counters(brain._freq_bmu_counters)
    n = int((brain.l3._bmu_to_zone >= 0).sum())
    print(f"  Zones assigned: {n}/64 BMUs mapped.")

    # Reset ALL navigation Q-learning state after open loop.
    #
    # The open loop runs brain.step() with world_moved=True (default) and reward=0,
    # building BOTH the BMU Q table (_Q) and Q_f from random exploration driven
    # by intrinsic RPE. These values have no navigational meaning:
    # - C.North=+0.30 (wall) after open loop because the brain correctly predicted
    #   it would stay when hitting the north wall — intrinsic reward was high.
    # - F.West=+0.147 (wall) for the same reason.
    # These wrong-direction values persist into closed loop and prevent correct
    # actions from dominating even after 100k food-driven training steps.
    #
    # Resetting both _Q and _Q_f makes the closed loop build Q purely from
    # food/wall signals, with no directional contamination from exploration.
    #
    # Also reset reward_ema: open loop drives it to ~0.985. At ema=0.985,
    # food RPE ≈ +0.015 (invisible) and wall RPE ≈ -1.0 (maximum punishment).
    # Starting from ema=0.0 gives food RPE ≈ +1.0 and wall RPE ≈ -0.05.
    import numpy as np
    brain.action._Q[:]   = 0.0   # BMU Q: reset open-loop directional contamination
    brain.action._Q_f[:] = 0.0  # freq-idx Q: reset open-loop intrinsic RPE values
    brain.action._e[:]   = 0.0  # BMU eligibility trace: no stale credit from open loop
    brain.action._e_f[:] = 0.0  # Q_f eligibility trace
    brain.action._transition_action = -1
    brain.action._prev_freq_idx     = -1
    brain.valence._reward_ema = 0.0
    print(f"  Navigation state reset (_Q, _Q_f, _e, _e_f, reward_ema).\n")


def get_policy_snapshot(brain, world, library):
    """
    Greedy policy evaluation using marginal BMU Q for all nodes except
    G and H (which share modal BMU=4 and need Q_f disambiguation).
    """
    Q_SIGNAL_THRESHOLD = 1e-3
    ALIASED = {'G', 'H'}
    snapshot = {}
    Q_f = brain.action._Q_f

    for fi, freq in enumerate(FREQUENCIES):
        node      = list(NODES.keys())[fi]
        ctr       = brain._freq_bmu_counters[fi]
        if not ctr:
            snapshot[node] = '?'
            continue
        modal_bmu = max(ctr, key=ctr.get)
        q_bmu_row = brain.action._Q[:, modal_bmu, :].max(axis=0)

        if node in ALIASED:
            q_f_row = Q_f[fi]
            q_row = q_f_row if float(q_f_row.max()) > Q_SIGNAL_THRESHOLD else q_bmu_row
        else:
            q_row = q_bmu_row

        if float(q_row.max()) <= Q_SIGNAL_THRESHOLD:
            snapshot[node] = '?'
            continue

        noise = brain.action._rng.uniform(0, 1e-6, size=4)
        a     = int(np.argmax(q_row + noise))
        snapshot[node] = ACTION_NAMES[a]

    return snapshot


def run_closed_loop(brain, world, library):
    print(f"  PHASE 2: CLOSED LOOP  ({CLOSED_LOOP_STEPS:,} steps)")
    print(f"  Food: E(1.3Hz)+1.0  H(2.0Hz)+1.0  |  Wall: -0.05\n")

    world.reset()
    counters      = {}
    food_per_win  = []
    wall_per_win  = []
    wfood = wwalls = 0
    node_visits   = Counter()
    action_counts = Counter()
    policy_history = []   # policy snapshot per window

    freq_hz  = world.current_freq
    freq_idx = world.current_freq_idx
    pending_reward = 0.0
    pending_moved  = True   # world_moved for the upcoming brain.step call.
                            # True initially (no prior action to be wrong about).
                            # Updated each step from info['wall_hit'].

    for step in range(CLOSED_LOOP_STEPS):
        decoded, w, nov, plv = get_step(library, freq_hz, counters)

        out = brain.step(
            decoded_freq = decoded,
            stability_w  = w,
            novelty_flag = nov,
            plv_vector   = plv,
            reward       = pending_reward,
            freq_idx     = freq_idx,
            world_moved  = pending_moved,  # was the PREVIOUS action a real move?
        )
        pending_reward = 0.0
        pending_moved  = True   # reset; will be set below from this step's world result

        action = int(out['action'])
        action_counts[action] += 1
        next_freq_hz, next_freq_idx, reward, info = world.step(action)

        node_visits[info['node']] += 1
        pending_reward = reward
        pending_moved  = not info['wall_hit']   # carry forward to next brain.step call

        # On food: trigger hippocampal replay, scaled by node unfamiliarity.
        # Pass food_freq_idx so replay only credits Q_f entries for this node.
        if info['is_food']:
            food_familiarity = out['familiarity']
            brain.action.replay_on_reward(
                reward=reward,
                familiarity=food_familiarity,
                food_freq_idx=next_freq_idx,   # freq_idx of the food node
            )
            wfood += 1

        if info['wall_hit']:
            wwalls += 1

        freq_hz  = next_freq_hz
        freq_idx = next_freq_idx

        if (step + 1) % REPORT_EVERY == 0:
            food_rate = wfood / REPORT_EVERY * 100
            wall_rate = wwalls / REPORT_EVERY
            snap      = get_policy_snapshot(brain, world, library)
            epsilon   = out['action_epsilon']

            # Score policy: correct/known (exclude ? nodes — no signal yet)
            known   = {n: a for n, a in snap.items() if a != '?'}
            correct = sum(1 for node, act in known.items()
                         if act == ACTION_NAMES[OPTIMAL_ACTION[node]])
            n_known = len(known)
            n_total = len(snap)

            print(f"  Step {step+1:7d} | food/100={food_rate:5.2f} | wall={wall_rate:.1%} | "
                  f"policy={correct}/{n_known} known correct ({n_total-n_known} unvisited) | ε={epsilon:.3f}")

            # Print current policy
            for node in list(NODES.keys()):
                act     = snap.get(node, '?')
                optimal = ACTION_NAMES[OPTIMAL_ACTION[node]]
                if act == '?':
                    check = '?'
                else:
                    check = "✓" if act == optimal else "✗"
                star    = "★" if node in FOOD_NODES else " "
                print(f"    {node}{star}: {act:5s} {check}  (optimal: {optimal})")
            print()

            food_per_win.append(wfood)
            wall_per_win.append(wwalls)
            policy_history.append((step+1, correct, n_known, snap.copy()))
            wfood = wwalls = 0

    return {
        'food_per_win':    food_per_win,
        'wall_per_win':    wall_per_win,
        'node_visits':     node_visits,
        'action_counts':   action_counts,
        'total_food':      world.food_count,
        'total_wall':      world.wall_count,
        'policy_history':  policy_history,
    }


def print_final_report(brain, world, results):
    fpw = results['food_per_win']
    wpw = results['wall_per_win']
    nv  = results['node_visits']
    ac  = results['action_counts']
    ph  = results['policy_history']
    n_w = len(fpw)

    print("\n" + "╔" + "═"*62 + "╗")
    print("║  BRAIN IN WORLD — 100k FINAL REPORT                      ║")
    print("╚" + "═"*62 + "╝")

    print(f"\n  FOOD RATE TRAJECTORY ({REPORT_EVERY:,}-step windows):")
    max_rate = max((f / REPORT_EVERY * 100 for f in fpw), default=1)
    for i, count in enumerate(fpw):
        rate = count / REPORT_EVERY * 100
        bar  = "█" * max(0, int(rate / max(max_rate, 0.01) * 30))
        correct = ph[i][1] if i < len(ph) else '?'
        total_n = ph[i][2] if i < len(ph) else 8
        print(f"    W{i+1:02d}: {rate:6.2f}/100  {bar}  (policy {correct}/{total_n})")

    first  = sum(fpw[:n_w//2]) / max(1, REPORT_EVERY * (n_w//2)) * 100
    second = sum(fpw[n_w//2:]) / max(1, REPORT_EVERY * (n_w - n_w//2)) * 100
    print(f"\n  First-half avg:  {first:.2f}/100 steps")
    print(f"  Second-half avg: {second:.2f}/100 steps")
    print(f"  Improvement:     {second/max(first,0.001):.2f}×")

    wall_rate = results['total_wall'] / CLOSED_LOOP_STEPS
    print(f"\n  WALL RATE: {wall_rate:.1%}  (random baseline 56.2%)")
    if wall_rate < 0.562:
        print(f"  ✓ BELOW random baseline — brain is learning valid directions")
    else:
        print(f"  Still above random — more training needed")

    print(f"\n  FINAL POLICY vs OPTIMAL:")
    snap = ph[-1][3] if ph else {}
    for node in list(NODES.keys()):
        act     = snap.get(node, '?')
        optimal = ACTION_NAMES[OPTIMAL_ACTION[node]]
        check   = "✓" if act == optimal else "✗"
        star    = "★" if node in FOOD_NODES else " "
        freq    = NODES[node][0]
        valid   = list(ADJACENCY[node].keys())
        print(f"    {node}{star} ({freq:.1f}Hz): {act:5s} {check}  optimal={optimal}  valid={valid}")

    final_correct = ph[-1][1] if ph else 0
    print(f"\n  Final policy score: {final_correct}/8 nodes correct")

    print(f"\n  L3 ZONE REWARD EMA:")
    for zi, val in enumerate(brain.l3._zone_reward_ema):
        node = list(NODES.keys())[zi]
        star = "★" if node in FOOD_NODES else " "
        bar  = "█" * max(0, int(val * 40))
        print(f"    Z{zi} {node}{star} ({FREQUENCIES[zi]:.1f}Hz): {val:.4f}  {bar}")

    print(f"\n  TOTAL: {results['total_food']} food events / {CLOSED_LOOP_STEPS:,} steps "
          f"({results['total_food']/CLOSED_LOOP_STEPS*100:.2f}/100 steps)\n")


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  BRAIN IN WORLD — 100k STEPS, ALL FIXES + POS-GATE         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    rx_slow, ry_slow, rx_fast, ry_fast = calibrate()
    library = build_signal_library(rx_slow, ry_slow, rx_fast, ry_fast)

    brain = Brain(seed=SEED)
    world = World(seed=SEED)

    run_open_loop(brain, library)
    results = run_closed_loop(brain, world, library)
    print_final_report(brain, world, results)