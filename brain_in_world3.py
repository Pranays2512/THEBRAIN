"""
BRAIN IN WORLD 3 — GENERALISATION TEST
=======================================

Same brain. Different world.

World 3 has 12 nodes but only 8 frequencies — nodes A/I, B/J, E/K, D/L
share frequencies. The brain must use sequence context (what did I hear
before this sound?) to disambiguate identical-sounding locations.

This tests whether the brain genuinely learned to navigate from sound,
or whether it just memorised the 8-node world.

THREE HARDER CHALLENGES vs World 2:
  1. Frequency reuse: identical sounds at different locations
     → M54 fires same BMU for A and I → M56 Q-table aliased
     → L2 sequence context is the ONLY disambiguation signal
  2. Longer path to second food (K★ is 6 steps from home)
     → M57 planning depth (3 steps) is insufficient alone
     → Requires M56 Q-learning to compound across the full path
  3. Dead end (L): brain must learn L has no value
     → Q(K, South) should be negative → brain avoids L

WHAT SUCCESS LOOKS LIKE:
  - Food rate > 8.3/100 (random baseline for 12-node world)
  - Wall rate < 58.3% (random baseline)
  - Brain finds E★ path (A→B→C→E) — same as World 2
  - Brain makes progress toward K★ path (A→D→F→G→H→I→J→K)
  - Brain does not get stuck in L dead end

WHAT FAILURE LOOKS LIKE:
  - Food rate ≈ 8.3/100 — brain couldn't generalise
  - Brain finds E★ but treats I as A (wrong action at I)
  - Brain loops in dead end L

BRAIN: unchanged — exact same Brain() as world 2.
WORLD: World3 — 12 nodes, frequency reuse, two food sources.
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
from world3 import (
    World3, NODES, FREQUENCIES, FOOD_NODES, ADJACENCY,
    ACTIONS, OPTIMAL_ACTION,
)
from l4_position import L4_CTM_WARMUP

# ── Run parameters ────────────────────────────────────────────
OPEN_STEPS_PER_NODE  = 1_500   # same as world 2
CLOSED_LOOP_STEPS    = 200_000
REPORT_EVERY         = 10_000
CAL_BLOCK_DUR        = 40.0
SIGNAL_BLOCK_DUR_S   = 60.0
PLV_STAB_WINDOW      = 20
SEED                 = 42

# Wall inference — same threshold as world 2
FREQ_SAME_THRESHOLD  = 0.08

# Known frequencies for bucketing — same 8 as world 2
KNOWN_FREQUENCIES    = np.array(FREQUENCIES)

ACTION_NAMES = ['North', 'East', 'South', 'West']

# Random baselines for 12-node world
# 12 nodes, 4 actions, 2 food nodes
# P(food | random) ≈ 2/12 × average_valid_moves/4 ≈ lower than 8-node
RANDOM_FOOD_BASELINE = 6.2
RANDOM_WALL_BASELINE = 0.583


# ══════════════════════════════════════════════════════════════
# UTILITIES (same as world 2)
# ══════════════════════════════════════════════════════════════

def bucket_freq(decoded_hz: float) -> int:
    diffs = np.abs(KNOWN_FREQUENCIES - decoded_hz)
    idx   = int(np.argmin(diffs))
    return -1 if diffs[idx] > 0.4 else idx


def infer_moved(prev_freq: float, curr_freq: float) -> bool:
    return abs(curr_freq - prev_freq) >= FREQ_SAME_THRESHOLD


# ══════════════════════════════════════════════════════════════
# CALIBRATION — same M50 calibration as world 2
# ══════════════════════════════════════════════════════════════

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
        # Library is keyed by frequency — for shared-frequency nodes,
        # BOTH nodes (e.g. A and I) draw from the same library entry.
        # This is correct: they truly sound identical. The brain must
        # use sequence context, not signal shape, to tell them apart.
        node_labels = [n for n, (f, _, _) in NODES.items() if f == freq]
        print(f"  {'/'.join(node_labels)} ({freq:.1f}Hz)...", end="", flush=True)
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
            df = decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                                  rx_fast, ry_fast)
            ds = decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                                  rx_slow, ry_slow)
            _, is_novel = cusum.update(df, ds, data['T'][i], w=w)
            if is_novel: w = 0.0
            decoded = float(w * ds + (1.0 - w) * df)
            if w > 0.5:
                steps.append((decoded, float(w), float(is_novel),
                               data['plv_slow'][i].copy()))
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


# ══════════════════════════════════════════════════════════════
# OPEN LOOP — same structure as world 2
# Brain hears all 8 frequencies. No rewards. No labels.
# ══════════════════════════════════════════════════════════════

def run_open_loop(brain, library):
    total_open = OPEN_STEPS_PER_NODE * len(FREQUENCIES)
    print(f"\n  PHASE 1: OPEN LOOP  ({total_open:,} steps)")
    print(f"  Brain hears all 8 frequencies. No rewards. No location labels.\n")
    counters = {}

    for fi, freq in enumerate(FREQUENCIES):
        node_labels = [n for n, (f, _, _) in NODES.items() if f == freq]
        print(f"  {'/'.join(node_labels)} ({freq:.1f}Hz)...", end="", flush=True)
        for _ in range(OPEN_STEPS_PER_NODE):
            decoded, w, nov, plv = get_step(library, freq, counters)
            bucketed_fi = bucket_freq(decoded)
            brain.step(
                decoded_freq = decoded,
                stability_w  = w,
                novelty_flag = nov,
                plv_vector   = plv,
                reward       = 0.0,
                freq_idx     = bucketed_fi,
                world_moved  = True,
            )
        print(" done")

    brain.l3.assign_zones_from_counters(brain._freq_bmu_counters)
    n = int((brain.l3._bmu_to_zone >= 0).sum())
    print(f"\n  L3 zones assigned: {n}/64 BMUs mapped (from sound alone).")

    # Reset action state — same as world 2
    brain.action._e[:]   = 0.0
    brain.action._e_f[:] = 0.0
    brain.action._transition_action = -1
    brain.valence._reward_ema = 0.0
    brain.l3._zone_visit_ema[:] = 0.0
    brain.l3._recompute_zone_values()
    print(f"  Action state reset.\n")


# ══════════════════════════════════════════════════════════════
# CLOSED LOOP
# ══════════════════════════════════════════════════════════════

def run_closed_loop(brain, world, library):
    print(f"  PHASE 2: CLOSED LOOP  ({CLOSED_LOOP_STEPS:,} steps)")
    print(f"  Food: E(1.3Hz)+1.0  K(1.3Hz)+1.0  |  Wall: -0.05")
    print(f"  NOTE: E and K share 1.3Hz — brain must use sequence context\n")

    world.reset()
    counters       = {}
    food_per_win   = []
    wall_per_win   = []
    wfood = wwalls = 0
    policy_history = []
    win_action_counts = {fi: Counter() for fi in range(8)}

    # L4 accuracy tracking — how often does L4's top_node match ground truth?
    l4_correct_win  = 0
    l4_total_win    = 0
    l4_correct_total = 0
    l4_total_total   = 0
    l4_acc_per_win   = []

    freq_hz  = world.current_freq
    pending_reward = 0.0
    prev_decoded   = freq_hz

    for step in range(CLOSED_LOOP_STEPS):
        decoded, w, nov, plv = get_step(library, freq_hz, counters)

        bucketed_fi   = bucket_freq(decoded)
        inferred_move = infer_moved(prev_decoded, decoded)

        out = brain.step(
            decoded_freq = decoded,
            stability_w  = w,
            novelty_flag = nov,
            plv_vector   = plv,
            reward       = pending_reward,
            freq_idx     = bucketed_fi,
            world_moved  = inferred_move,
        )
        pending_reward = 0.0

        action = int(out['action'])

        if bucketed_fi >= 0:
            win_action_counts[bucketed_fi][action] += 1

        next_freq_hz, next_freq_idx, reward, info = world.step(action)

        pending_reward = reward

        if info['is_food']:
            brain.action.replay_on_reward(
                reward        = reward,
                familiarity   = out['familiarity'],
                food_freq_idx = -1,
            )
            wfood += 1

        if info['wall_hit']:
            wwalls += 1

        # L4 accuracy: compare top_node belief to ground truth current node
        if brain.l4 is not None:
            true_node = world.current_node
            l4_top    = out.get('l4_top_node')
            if l4_top is not None:
                l4_total_win  += 1
                l4_total_total += 1
                if l4_top == true_node:
                    l4_correct_win  += 1
                    l4_correct_total += 1

        prev_decoded = decoded
        freq_hz      = next_freq_hz

        if (step + 1) % REPORT_EVERY == 0:
            food_rate = wfood / REPORT_EVERY * 100
            wall_rate = wwalls / REPORT_EVERY
            epsilon   = out['action_epsilon']
            pa_ready  = brain.pred.pa_ready() if hasattr(brain.pred, 'pa_ready') else False
            plan_rate = brain.planner.planning_rate()

            # Policy scoring
            # Note: bucketed_fi maps to freq_index (0-7), not node.
            # For shared-frequency nodes (A/I=0, B/J=1, E/K=4, D/L=3),
            # we report the modal action at that freq_index and compare
            # to BOTH nodes that share it.
            snap    = {}
            correct = 0
            known   = 0
            scored_nodes = set()

            for fi in range(8):
                ctr = win_action_counts[fi]
                if not ctr:
                    continue
                modal_a = max(ctr, key=ctr.get)
                # Find which nodes share this freq_index
                nodes_at_fi = [n for n, (f, idx, _) in NODES.items() if idx == fi]
                for node in nodes_at_fi:
                    snap[node] = ACTION_NAMES[modal_a]
                    known += 1
                    scored_nodes.add(node)
                    if modal_a == OPTIMAL_ACTION[node]:
                        correct += 1

            print(f"  Step {step+1:7d} | food/100={food_rate:5.2f} | "
                  f"wall={wall_rate:.1%} | policy={correct}/{known} correct | "
                  f"eps={epsilon:.3f} | PA={'ok' if pa_ready else '..'} | "
                  f"plan={plan_rate:.0%} | "
                  f"L4={l4_correct_win/max(1,l4_total_win):.0%}")

            # Print all 12 nodes
            for node in list(NODES.keys()):
                act     = snap.get(node, '?')
                optimal = ACTION_NAMES[OPTIMAL_ACTION[node]]
                check   = '?' if act == '?' else ('✓' if act == optimal else '✗')
                star    = '★' if node in FOOD_NODES else ' '
                freq    = NODES[node][0]
                # Mark aliased nodes
                alias   = '*' if len([n for n, (f,_,_) in NODES.items() if f == freq]) > 1 else ' '
                print(f"    {node}{star}{alias}({freq:.1f}Hz): {act:5s} {check}  "
                      f"(optimal: {optimal})")
            print(f"    * = frequency shared with another node")
            print()

            food_per_win.append(wfood)
            wall_per_win.append(wwalls)
            l4_acc_per_win.append(l4_correct_win / max(1, l4_total_win))
            policy_history.append((step + 1, correct, known, snap.copy()))
            wfood = wwalls = 0
            l4_correct_win = l4_total_win = 0
            win_action_counts = {fi: Counter() for fi in range(8)}

    return {
        'food_per_win':   food_per_win,
        'wall_per_win':   wall_per_win,
        'total_food':     world.food_count,
        'total_wall':     world.wall_count,
        'policy_history': policy_history,
        'node_visits':    world.node_visit_counts,
        'l4_acc_per_win': l4_acc_per_win,
        'l4_correct_total': l4_correct_total,
        'l4_total_total':   l4_total_total,
    }


# ══════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════

def print_final_report(brain, world, results):
    fpw = results['food_per_win']
    ph  = results['policy_history']
    n_w = len(fpw)

    print("\n" + "╔" + "═"*62 + "╗")
    print("║  BRAIN IN WORLD 3 — GENERALISATION TEST REPORT           ║")
    print("╚" + "═"*62 + "╝")

    print(f"\n  FOOD RATE TRAJECTORY ({REPORT_EVERY:,}-step windows):")
    max_rate = max((f / REPORT_EVERY * 100 for f in fpw), default=1)
    for i, count in enumerate(fpw):
        rate    = count / REPORT_EVERY * 100
        bar     = "█" * max(0, int(rate / max(max_rate, 0.01) * 30))
        correct = ph[i][1] if i < len(ph) else '?'
        n_known = ph[i][2] if i < len(ph) else 12
        print(f"    W{i+1:02d}: {rate:6.2f}/100  {bar}  (policy {correct}/{n_known})")

    first  = sum(fpw[:n_w//2]) / max(1, REPORT_EVERY * (n_w//2)) * 100
    second = sum(fpw[n_w//2:]) / max(1, REPORT_EVERY * (n_w - n_w//2)) * 100
    print(f"\n  First-half avg:  {first:.2f}/100 steps")
    print(f"  Second-half avg: {second:.2f}/100 steps")
    print(f"  Improvement:     {second/max(first,0.001):.2f}×")

    wall_rate_total = results['total_wall'] / CLOSED_LOOP_STEPS
    food_rate_total = results['total_food'] / CLOSED_LOOP_STEPS * 100

    print(f"\n  WALL RATE: {wall_rate_total:.1%}  "
          f"(random baseline ~{RANDOM_WALL_BASELINE:.1%})")
    if wall_rate_total < RANDOM_WALL_BASELINE:
        print(f"  ✓ BELOW random baseline")
    else:
        print(f"  ✗ At/above random")

    print(f"\n  FOOD RATE: {food_rate_total:.2f}/100 steps  "
          f"(random baseline ~{RANDOM_FOOD_BASELINE:.1f}/100)")
    if food_rate_total > RANDOM_FOOD_BASELINE:
        print(f"  ✓ ABOVE random baseline — brain navigates toward food")
    else:
        print(f"  ✗ Below/at random — generalisation failed")

    # Node visit distribution — did brain explore beyond E★ corridor?
    nv = results['node_visits']
    total_v = sum(nv.values()) + 1
    print(f"\n  NODE VISIT DISTRIBUTION:")
    for node in list(NODES.keys()):
        freq  = NODES[node][0]
        star  = '★' if node in FOOD_NODES else ' '
        alias = '*' if len([n for n, (f,_,_) in NODES.items() if f == freq]) > 1 else ' '
        pct   = nv[node] / total_v * 100
        bar   = "█" * max(0, int(pct / 2))
        print(f"    {node}{star}{alias}: {pct:5.1f}%  {bar}")

    print(f"\n  ALIASED NODES (share frequency — disambiguation required):")
    for freq in FREQUENCIES:
        shared = [n for n, (f,_,_) in NODES.items() if f == freq]
        if len(shared) > 1:
            print(f"    {freq:.1f}Hz → {shared}")
            for node in shared:
                act = ph[-1][3].get(node, '?') if ph else '?'
                opt = ACTION_NAMES[OPTIMAL_ACTION[node]]
                check = '?' if act == '?' else ('✓' if act == opt else '✗')
                print(f"      {node}: policy={act} {check}  optimal={opt}")

    print(f"\n  DEAD END TEST (node L — should have low visits):")
    l_pct = nv['L'] / total_v * 100
    k_pct = nv['K'] / total_v * 100
    print(f"    K★ visits: {k_pct:.1f}%  L visits: {l_pct:.1f}%")
    if l_pct < k_pct:
        print(f"    ✓ Brain spends less time in dead end than food node")
    else:
        print(f"    ✗ Brain stuck in dead end")

    print(f"\n  L3 ZONE REWARD EMA:")
    for zi, val in enumerate(brain.l3._zone_reward_ema):
        # Zone index = freq_index. Multiple nodes share same zone.
        nodes_here = [n for n, (f, idx, _) in NODES.items() if idx == zi]
        star  = '★' if any(n in FOOD_NODES for n in nodes_here) else ' '
        bar   = "█" * max(0, int(val * 40))
        freq  = FREQUENCIES[zi]
        print(f"    Z{zi} {'/'.join(nodes_here)}{star} ({freq:.1f}Hz): {val:.4f}  {bar}")

    print(f"\n  M57 PLANNER RATE: {brain.planner.planning_rate()*100:.1f}%")

    # L4 position belief accuracy
    if brain.l4 is not None:
        l4_total = results.get('l4_total_total', 0)
        l4_corr  = results.get('l4_correct_total', 0)
        l4_acc   = l4_corr / max(1, l4_total)
        l4_wins  = results.get('l4_acc_per_win', [])
        print(f"\n  L4 POSITION BELIEF ACCURACY: {l4_acc:.1%}  "
              f"({l4_corr}/{l4_total} steps)")
        print(f"  (% of steps where L4's top-node belief = ground truth)")
        tm_cov = brain.l4.tm_coverage()
        print(f"  Transition model coverage: "
              f"{tm_cov['tm_covered']}/{tm_cov['tm_total']} "
              f"(fi,action) pairs  |  "
              f"{tm_cov['ctm_covered']}/{tm_cov['ctm_total']} "
              f"context pairs with ≥{L4_CTM_WARMUP} observations")
        # Show trajectory of L4 accuracy per window
        if l4_wins:
            print(f"  L4 accuracy trajectory:")
            for i, acc in enumerate(l4_wins):
                bar = "█" * int(acc * 20)
                print(f"    W{i+1:02d}: {acc:.0%}  {bar}")

    print(f"\n  GENERALISATION VERDICT:")
    if food_rate_total > RANDOM_FOOD_BASELINE and wall_rate_total < RANDOM_WALL_BASELINE:
        print(f"  ✓ Brain navigates World 3 above chance from sound alone.")
        print(f"  ✓ Architecture generalises beyond the training environment.")
        if food_rate_total > 15.0:
            print(f"  ✓ Strong generalisation — found efficient paths.")
        elif food_rate_total > RANDOM_FOOD_BASELINE:
            print(f"  ~ Partial generalisation — found some paths, missed others.")
    else:
        print(f"  ✗ Brain did not generalise to World 3.")
        print(f"  ✗ Performance at or below random baseline.")

    print(f"\n  TOTAL: {results['total_food']} food / "
          f"{CLOSED_LOOP_STEPS:,} steps "
          f"({food_rate_total:.2f}/100)\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  BRAIN IN WORLD 3 — GENERALISATION TEST                      ║")
    print("║                                                              ║")
    print("║  12 nodes, 8 frequencies — frequency reuse forces            ║")
    print("║  the brain to use sequence context for disambiguation.       ║")
    print("║                                                              ║")
    print("║  Same brain. New world. Does it generalise?                  ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    print("  WHAT IS NEW vs WORLD 2:")
    print("  + 12 nodes (was 8) — 50% larger map")
    print("  + Frequency reuse: A=I=0.5Hz, B=J=0.7Hz, E=K=1.3Hz, D=L=1.1Hz")
    print("  + Brain CANNOT tell A from I by sound alone — needs sequence context")
    print("  + K★ food is 6 steps from home (was max 4 steps in World 2)")
    print("  + Dead end at L — brain must learn to avoid it")
    print("  + Random baseline is lower: ~6.2/100 (was ~12.5/100)\n")

    rx_slow, ry_slow, rx_fast, ry_fast = calibrate()
    library = build_signal_library(rx_slow, ry_slow, rx_fast, ry_fast)

    brain = Brain(seed=SEED, node_fi={n: v[1] for n, v in NODES.items()})
    world = World3(seed=SEED)

    brain.action._Q_f[:] = 0.0

    run_open_loop(brain, library)
    results = run_closed_loop(brain, world, library)
    print_final_report(brain, world, results)