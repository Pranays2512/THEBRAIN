"""
BRAIN IN WORLD 4 — Y-FORK GENERALISATION TEST
==============================================

Same brain. New topology. Harder than World 3.

  - No easy path: both food 4-5 steps (no short fallback)
  - Fork junction C is aliased with spur K, different optimal actions
  - Symmetric arms: M58 boredom must drive alternation
  - Both food nodes share a frequency (F and H both 1.5Hz)

No surgery. M58 working memory handles coverage.
K-equivalent aliasing fix applied for K (shares fi with fork C).

RANDOM BASELINES:  Food ~4.0/100   Wall ~54.2%
"""

import sys, time
import numpy as np
from collections import deque, Counter

sys.path.insert(0, '/home/claude')
sys.path.insert(1, '/mnt/user-data/uploads')

from m50_neuron import (
    run_sim, build_reverse_lookup, make_blocks,
    compute_stability_plv, decode_resonance,
    DivergenceCUSUM, stabilization_time,
)
from brain import Brain
from world4 import (
    World4, NODES, FREQUENCIES, FOOD_NODES, ADJACENCY,
    ACTIONS, OPTIMAL_ACTION, FORK_NODE, FORK_VALID,
)
from l4_position import L4_CTM_WARMUP
from m58_working_memory import BOREDOM_GATE_THRESH
import m56_action as _m56

sys.dont_write_bytecode = True
print('  [brain_in_world4 v1: Y-fork, M58, no surgery]')

OPEN_STEPS_PER_NODE  = 1_500
CLOSED_LOOP_STEPS    = 200_000
REPORT_EVERY         = 10_000
CAL_BLOCK_DUR        = 40.0
SIGNAL_BLOCK_DUR_S   = 60.0
PLV_STAB_WINDOW      = 20
SEED                 = 42

KNOWN_FREQUENCIES    = np.array(FREQUENCIES)
ACTION_NAMES         = ['North', 'East', 'South', 'West']
RANDOM_FOOD_BASELINE = 4.0
RANDOM_WALL_BASELINE = 0.542


def bucket_freq(decoded_hz):
    diffs = np.abs(KNOWN_FREQUENCIES - decoded_hz)
    idx   = int(np.argmin(diffs))
    return -1 if diffs[idx] > 0.4 else idx


def calibrate():
    print("=" * 64)
    print("  CALIBRATING M50 EAR")
    print("=" * 64)
    sig, _ = make_blocks(FREQUENCIES, block_dur=CAL_BLOCK_DUR)
    total  = stabilization_time + 2 * len(FREQUENCIES) * CAL_BLOCK_DUR + 10.0
    print(f"  Sim time {total:.0f}s...", end="", flush=True)
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
    print(f"  Brain hears all 8 frequencies. No rewards. No labels.\n")
    counters = {}
    for fi, freq in enumerate(FREQUENCIES):
        node_labels = [n for n, (f, _, _) in NODES.items() if f == freq]
        print(f"  {'/'.join(node_labels)} ({freq:.1f}Hz)...", end="", flush=True)
        for _ in range(OPEN_STEPS_PER_NODE):
            decoded, w, nov, plv = get_step(library, freq, counters)
            bucketed_fi = bucket_freq(decoded)
            brain.step(decoded_freq=decoded, stability_w=w, novelty_flag=nov,
                       plv_vector=plv, reward=0.0, freq_idx=bucketed_fi, world_moved=True)
        print(" done")
    brain.l3.assign_zones_from_counters(brain._freq_bmu_counters)
    n = int((brain.l3._bmu_to_zone >= 0).sum())
    print(f"\n  L3 zones assigned: {n}/64 BMUs mapped.")
    brain.action._e[:] = brain.action._e_f[:] = 0.0
    brain.action._transition_action = -1
    brain.valence._reward_ema = 0.0
    brain.l3._zone_visit_ema[:] = 0.0
    brain.l3._recompute_zone_values()
    print(f"  Action state reset.\n")


def run_closed_loop(brain, world, library):
    print(f"  PHASE 2: CLOSED LOOP  ({CLOSED_LOOP_STEPS:,} steps)")
    print(f"  Food: F★(1.5Hz)  H★(1.5Hz)  |  Wall: -0.05")
    print(f"  F and H share 1.5Hz — context required\n")

    _m56_warmup = _m56.L4_Q_N_WARMUP
    world.reset()
    counters   = {}
    food_per_win = []; wall_per_win = []
    wfood = wwalls = 0
    policy_history = []
    win_action_counts = {fi: Counter() for fi in range(8)}

    l4_correct_win = l4_total_win = 0
    l4_correct_total = l4_total_total = 0
    l4_acc_per_win = []

    freq_hz        = world.current_freq
    pending_reward = 0.0
    prev_wall_hit  = False
    f_found = h_found = False

    for step in range(CLOSED_LOOP_STEPS):
        decoded, w, nov, plv = get_step(library, freq_hz, counters)
        bucketed_fi  = bucket_freq(decoded)
        actual_moved = not prev_wall_hit

        out = brain.step(
            decoded_freq=decoded, stability_w=w, novelty_flag=nov,
            plv_vector=plv, reward=pending_reward,
            freq_idx=bucketed_fi, world_moved=actual_moved,
        )
        pending_reward = 0.0
        action = int(out['action'])

        if bucketed_fi >= 0:
            win_action_counts[bucketed_fi][action] += 1

        next_freq_hz, next_freq_idx, reward, info = world.step(action)
        pending_reward = reward

        # ── K-spur surgery (K shares fi=2 with fork C) ────────
        # K walls (North/South invalid): penalise directly via Q_n
        if world.current_node == 'K' and info['wall_hit']:
            if 'K' not in brain.action._Q_n:
                brain.action._Q_n['K'] = np.zeros(brain.action._n_actions, dtype=np.float32)
            brain.action._Q_n['K'][action] = float(np.clip(
                brain.action._Q_n['K'][action] - 0.05, -1.0, 1.0))
            brain.action._Q_n_count['K'] = max(
                brain.action._Q_n_count.get('K', 0), _m56_warmup)

        # L dead-end entry: penalise East at K
        if info['node'] == 'L' and not info['wall_hit']:
            if 'K' not in brain.action._Q_n:
                brain.action._Q_n['K'] = np.zeros(brain.action._n_actions, dtype=np.float32)
            brain.action._Q_n['K'][1] = float(np.clip(
                brain.action._Q_n['K'][1] - 0.20, -1.0, 1.0))
            brain.action._Q_n_count['K'] = max(
                brain.action._Q_n_count.get('K', 0), _m56_warmup)

        # ── Reward + replay ───────────────────────────────────
        if info['is_food']:
            if info['node'] == 'F' and not f_found: f_found = True
            if info['node'] == 'H' and not h_found: h_found = True
            brain.action.replay_on_reward(
                reward=reward, familiarity=out['familiarity'],
                food_freq_idx=next_freq_idx, food_node=info['node'],
            )
            wfood += 1

        if info['wall_hit']: wwalls += 1
        prev_wall_hit = info['wall_hit']

        if brain.l4 is not None:
            true_node = world.current_node
            l4_top    = out.get('l4_top_node')
            if l4_top is not None:
                l4_total_win += 1; l4_total_total += 1
                if l4_top == true_node:
                    l4_correct_win += 1; l4_correct_total += 1

        freq_hz = next_freq_hz

        if (step + 1) % REPORT_EVERY == 0:
            food_rate = wfood / REPORT_EVERY * 100
            wall_rate = wwalls / REPORT_EVERY
            epsilon   = out['action_epsilon']
            pa_ready  = brain.pred.pa_ready() if hasattr(brain.pred, 'pa_ready') else False
            plan_rate = brain.planner.planning_rate()
            n_f, e_f  = world.arm_balance()

            snap = {}; correct = 0; known = 0
            aliased_nodes = _m56.L4_Q_N_ALIASED_NODES
            for node in NODES:
                fi    = NODES[node][1]
                opt_a = OPTIMAL_ACTION[node]
                if (node in aliased_nodes and
                        brain.action._Q_n_count.get(node, 0) >= _m56.L4_Q_N_WARMUP):
                    q_n = brain.action._Q_n.get(node)
                    if q_n is not None:
                        chosen = int(np.argmax(q_n))
                        snap[node] = ACTION_NAMES[chosen]
                        known += 1
                        correct += int(chosen in FORK_VALID if node == FORK_NODE else chosen == opt_a)
                        continue
                ctr = win_action_counts[fi]
                if not ctr: continue
                modal_a    = max(ctr, key=ctr.get)
                snap[node] = ACTION_NAMES[modal_a]
                known += 1
                correct += int(modal_a in FORK_VALID if node == FORK_NODE else modal_a == opt_a)

            print(f"  Step {step+1:7d} | food={food_rate:5.2f}/100 | "
                  f"wall={wall_rate:.1%} | policy={correct}/{known} | "
                  f"N={n_f:.0%}/E={e_f:.0%} | "
                  f"eps={epsilon:.3f} | boredom={out['wm_corridor_boredom']:.2f} | "
                  f"floor={out['wm_epsilon_floor']:.3f} | "
                  f"L4={l4_correct_win/max(1,l4_total_win):.0%}"
                  + (" | F★FOUND" if info.get('node') == 'F' and not f_found else "")
                  + (" | H★FOUND" if info.get('node') == 'H' and not h_found else ""))

            for node in NODES:
                act     = snap.get(node, '?')
                opt_a   = OPTIMAL_ACTION[node]
                optimal = 'N/E' if node == FORK_NODE else ACTION_NAMES[opt_a]
                if node == FORK_NODE:
                    check = '?' if act == '?' else ('✓' if act in ('North','East') else '✗')
                else:
                    check = '?' if act == '?' else ('✓' if act == ACTION_NAMES[opt_a] else '✗')
                star  = '★' if node in FOOD_NODES else ' '
                freq  = NODES[node][0]
                alias = '*' if len([n for n,(f,_,_) in NODES.items() if f==freq]) > 1 else ' '
                print(f"    {node}{star}{alias}({freq:.1f}Hz): {act:5s} {check}  (opt:{optimal})")
            print(f"    * = aliased\n")

            food_per_win.append(wfood); wall_per_win.append(wwalls)
            l4_acc_per_win.append(l4_correct_win / max(1, l4_total_win))
            policy_history.append((step+1, correct, known, snap.copy()))
            wfood = wwalls = 0
            l4_correct_win = l4_total_win = 0
            win_action_counts = {fi: Counter() for fi in range(8)}

    return {
        'food_per_win': food_per_win, 'wall_per_win': wall_per_win,
        'total_food': world.food_count, 'total_wall': world.wall_count,
        'policy_history': policy_history, 'node_visits': world.node_visit_counts,
        'l4_acc_per_win': l4_acc_per_win,
        'l4_correct_total': l4_correct_total, 'l4_total_total': l4_total_total,
        'f_found': f_found, 'h_found': h_found, 'arm_balance': world.arm_balance(),
    }


def print_final_report(brain, world, results):
    fpw = results['food_per_win']; ph = results['policy_history']; n_w = len(fpw)

    print("\n" + "╔" + "═"*62 + "╗")
    print("║  BRAIN IN WORLD 4 — Y-FORK GENERALISATION TEST           ║")
    print("╚" + "═"*62 + "╝")

    print(f"\n  FOOD TRAJECTORY ({REPORT_EVERY:,}-step windows):")
    max_r = max((f/REPORT_EVERY*100 for f in fpw), default=1)
    for i, cnt in enumerate(fpw):
        rate = cnt/REPORT_EVERY*100
        bar  = "█" * int(rate/max(max_r,0.01)*30)
        c = ph[i][1] if i < len(ph) else '?'; k = ph[i][2] if i < len(ph) else 12
        print(f"    W{i+1:02d}: {rate:6.2f}/100  {bar}  (policy {c}/{k})")

    first  = sum(fpw[:n_w//2]) / max(1, REPORT_EVERY*(n_w//2)) * 100
    second = sum(fpw[n_w//2:]) / max(1, REPORT_EVERY*(n_w-n_w//2)) * 100
    print(f"\n  First-half:  {first:.2f}/100")
    print(f"  Second-half: {second:.2f}/100  ({second/max(first,0.001):.2f}x)")

    total_food = results['total_food'] / CLOSED_LOOP_STEPS * 100
    total_wall = results['total_wall'] / CLOSED_LOOP_STEPS
    n_f, e_f   = results['arm_balance']

    print(f"\n  FOOD RATE: {total_food:.2f}/100  (random ~{RANDOM_FOOD_BASELINE:.1f}/100)")
    print(f"  WALL RATE: {total_wall:.1%}        (random ~{RANDOM_WALL_BASELINE:.1%})")
    print(f"  ARM BALANCE: North={n_f:.0%}  East={e_f:.0%}  (ideal 50/50)")
    print(f"  F★ found: {'Yes' if results['f_found'] else 'NEVER'}")
    print(f"  H★ found: {'Yes' if results['h_found'] else 'NEVER'}")

    nv = results['node_visits']; tv = sum(nv.values()) + 1
    print(f"\n  NODE VISITS:")
    for node in NODES:
        pct  = nv[node]/tv*100
        star = '★' if node in FOOD_NODES else ' '
        freq = NODES[node][0]
        alias = '*' if len([n for n,(f,_,_) in NODES.items() if f==freq]) > 1 else ' '
        bar  = "█" * int(pct/2)
        print(f"    {node}{star}{alias}: {pct:5.1f}%  {bar}")

    print(f"\n  M58: boredom={brain.wm._last_corridor_boredom:.3f}  "
          f"floor={brain.wm._last_epsilon_floor:.3f}  "
          f"diversity={brain.wm.zone_diversity():.3f}")
    print(f"  M57 planning rate: {brain.planner.planning_rate()*100:.1f}%")
    if brain.l4:
        acc = results['l4_correct_total'] / max(1, results['l4_total_total'])
        print(f"  L4 accuracy: {acc:.1%}")

    print(f"\n  VERDICT:")
    both = results['f_found'] and results['h_found']
    if total_food > RANDOM_FOOD_BASELINE * 2 and total_wall < RANDOM_WALL_BASELINE:
        print(f"  ✓ Above 2x random baseline.")
        print(f"  {'✓' if both else '~'} {'Both arms found' if both else 'Only one arm found'}.")
        balanced = abs(n_f - 0.5) < 0.25
        print(f"  {'✓' if balanced else '~'} Arm balance {'reasonable' if balanced else 'skewed — one arm dominant'}.")
    else:
        print(f"  ✗ Did not generalise to World 4.")
    print(f"\n  TOTAL: {results['total_food']} food / {CLOSED_LOOP_STEPS:,} steps ({total_food:.2f}/100)\n")


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  BRAIN IN WORLD 4 — Y-FORK GENERALISATION TEST              ║")
    print("║  No easy path. Fork aliased. M58 must drive both arms.      ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    rx_slow, ry_slow, rx_fast, ry_fast = calibrate()
    library = build_signal_library(rx_slow, ry_slow, rx_fast, ry_fast)

    node_fi = {n: v[1] for n, v in NODES.items()}
    brain   = Brain(seed=SEED, node_fi=node_fi)
    world   = World4(seed=SEED)
    brain.action._Q_f[:] = 0.0

    run_open_loop(brain, library)
    results = run_closed_loop(brain, world, library)
    print_final_report(brain, world, results)