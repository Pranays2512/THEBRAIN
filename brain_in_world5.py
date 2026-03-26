"""
BRAIN IN WORLD 5 — 4×4 GRID SCALING TEST
=========================================

16 nodes. 8 frequencies. Every frequency shared by exactly 2 nodes.
Zero unique-frequency nodes — Q_f cannot give unambiguous signal alone.

Same brain. Largest world yet. Tests whether the architecture scales.

RANDOM BASELINES:  Food ~7.8/100   Wall ~25.1%
Success threshold: > 15.6/100 (2× random) sustained second half.
"""

import sys, time
import numpy as np
from collections import deque, Counter

# sys.path.insert(0, '/mnt/user-data/uploads')   # cloud-only, not needed locally
# sys.path.insert(0, '/home/claude')              # cloud-only, not needed locally

from m50_neuron import (
    run_sim, build_reverse_lookup, make_blocks,
    compute_stability_plv, decode_resonance,
    DivergenceCUSUM, stabilization_time,
)
from brain import Brain
from world5 import (
    World5, NODES, FREQUENCIES, FOOD_NODES, ADJACENCY,
    ACTIONS, OPTIMAL_ACTION,
)
from l4_position import L4_CTM_WARMUP
from m58_working_memory import BOREDOM_GATE_THRESH
import m56_action as _m56

sys.dont_write_bytecode = True
print('  [brain_in_world5 v1: 4x4 grid, 16 nodes, all fi aliased]')

OPEN_STEPS_PER_NODE  = 1_500
CLOSED_LOOP_STEPS    = 200_000
REPORT_EVERY         = 10_000
CAL_BLOCK_DUR        = 40.0
SIGNAL_BLOCK_DUR_S   = 60.0
PLV_STAB_WINDOW      = 20
SEED                 = 42

KNOWN_FREQUENCIES    = np.array(FREQUENCIES)
ACTION_NAMES         = ['North', 'East', 'South', 'West']
RANDOM_FOOD_BASELINE = 7.8
RANDOM_WALL_BASELINE = 0.251


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
    brain.action.t = 0   # reset step counter so EPSILON_WARMUP_STEPS applies
                         # to closed-loop steps, not open-loop steps.
                         # Without this: open loop consumes 12k of the 15k warmup
                         # budget, leaving only 3k steps of exploration before
                         # epsilon drops to EPSILON_MIN on a cold Q table.
    brain.valence._reward_ema = 0.0
    brain.l3._zone_visit_ema[:] = 0.0
    brain.l3._recompute_zone_values()
    print(f"  Action state reset.\n")


def run_closed_loop(brain, world, library):
    print(f"  PHASE 2: CLOSED LOOP  ({CLOSED_LOOP_STEPS:,} steps)")
    print(f"  Food: H★(2.0Hz) + P★(2.0Hz)  |  Both fi=7, context required\n")

    _m56_warmup = _m56.L4_Q_N_WARMUP
    world.reset()
    if brain.l4 is not None:
        brain.l4.reset_to_node('A')
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
    pending_l4_reset = None
    prev_wall_hit  = False
    h_found = p_found = False

    for step in range(CLOSED_LOOP_STEPS):
        decoded, w, nov, plv = get_step(library, freq_hz, counters)
        bucketed_fi  = bucket_freq(decoded)
        actual_moved = not prev_wall_hit

        out = brain.step(
            decoded_freq=decoded, stability_w=w, novelty_flag=nov,
            plv_vector=plv, reward=pending_reward,
            freq_idx=bucketed_fi, world_moved=actual_moved,
        )
        if pending_l4_reset is not None and brain.l4 is not None:
            brain.l4.replay_from_anchor(pending_l4_reset)
            brain.l4.reset_to_node(pending_l4_reset)
        pending_l4_reset = None

        pending_reward = 0.0
        action = int(out['action'])

        # L4 evaluates its belief of where the brain is CURRENTLY, before the action.
        true_node_at_t = world.current_node

        if bucketed_fi >= 0:
            win_action_counts[bucketed_fi][action] += 1

        next_freq_hz, next_freq_idx, reward, info = world.step(action)
        pending_reward = reward

        # No dead-end surgery needed — 4x4 grid has no dead ends.
        # All corner/edge nodes have 2+ valid exits.
        # L aliasing fix: L shares fi=3 with D.
        # D→South→H is correct. L→South→P is also correct.
        # Both go South — same action — so Q_f[3,South] works for both.
        # No surgery needed here either.

        if info['is_food']:
            if info['node'] == 'H' and not h_found: h_found = True
            if info['node'] == 'P' and not p_found: p_found = True
            brain.action.replay_on_reward(
                reward=reward, familiarity=out['familiarity'],
                food_freq_idx=next_freq_idx, food_node=info['node'],
            )
            # ── FIX 3: L4 anchor reset on food collection ──────
            # Deferred until after brain.step() so we don't corrupt the incoming transition
            pending_l4_reset = info['node']
            # Set the Q_f override hint for the food node step itself.
            brain._node_fi_override_hint = info['node']
            wfood += 1
        else:
            # ── FIX 4: Proactive override hint via L4 confidence ──
            # Previously the override hint was ONLY set at food collection,
            # meaning all approach steps to H/P still wrote to shared Q_f[7]
            # and corrupted each other. Now: when L4 is confident (>0.55)
            # that the brain is AT a registered food node (H or P), set the
            # hint proactively so those steps read/write the dedicated override
            # row — not the shared fi=7 row. This stops H-approach and
            # P-approach steps from cancelling each other in Q_f[7].
            l4_node = out.get('l4_top_node')
            l4_prob = out.get('l4_top_prob', 0.0)
            if (l4_node is not None
                    and l4_prob >= 0.55
                    and l4_node in brain.action._node_fi_override_row):
                brain._node_fi_override_hint = l4_node
            else:
                brain._node_fi_override_hint = None

        if info['wall_hit']: wwalls += 1
        prev_wall_hit = info['wall_hit']

        if brain.l4 is not None:
            l4_top    = out.get('l4_top_node')
            if l4_top is not None:
                l4_total_win += 1; l4_total_total += 1
                if l4_top == true_node_at_t:
                    l4_correct_win += 1; l4_correct_total += 1

        freq_hz = next_freq_hz

        if (step + 1) % REPORT_EVERY == 0:
            food_rate = wfood / REPORT_EVERY * 100
            wall_rate = wwalls / REPORT_EVERY
            epsilon   = out['action_epsilon']
            pa_ready  = brain.pred.pa_ready() if hasattr(brain.pred, 'pa_ready') else False
            plan_rate = brain.planner.planning_rate()
            h_f, p_f  = world.food_balance()

            snap = {}; correct = 0; known = 0
            aliased = _m56.L4_Q_N_ALIASED_NODES
            for node in NODES:
                fi    = NODES[node][1]
                opt_a = OPTIMAL_ACTION[node]
                if node in aliased and brain.action._Q_n_count.get(node,0) >= _m56.L4_Q_N_WARMUP:
                    q_n = brain.action._Q_n.get(node)
                    if q_n is not None:
                        chosen = int(np.argmax(q_n))
                        snap[node] = ACTION_NAMES[chosen]
                        known += 1; correct += int(chosen == opt_a)
                        continue
                ctr = win_action_counts[fi]
                if not ctr: continue
                modal_a = max(ctr, key=ctr.get)
                snap[node] = ACTION_NAMES[modal_a]
                known += 1; correct += int(modal_a == opt_a)

            print(f"  Step {step+1:7d} | food={food_rate:5.2f}/100 | "
                  f"wall={wall_rate:.1%} | policy={correct}/{known} | "
                  f"H={h_f:.0%}/P={p_f:.0%} | "
                  f"eps={epsilon:.3f} | boredom={out['wm_corridor_boredom']:.2f} | "
                  f"floor={out['wm_epsilon_floor']:.3f} | "
                  f"L4={l4_correct_win/max(1,l4_total_win):.0%}"
                  + (" | H★FOUND" if info.get('node')=='H' and not h_found else "")
                  + (" | P★FOUND" if info.get('node')=='P' and not p_found else ""))

            # Print all 16 nodes in grid layout
            grid = [['A','B','C','D'],['E','F','G','H'],
                    ['I','J','K','L'],['M','N','O','P']]
            for row in grid:
                parts = []
                for node in row:
                    act = snap.get(node,'?')
                    opt = ACTION_NAMES[OPTIMAL_ACTION[node]]
                    chk = '?' if act=='?' else ('✓' if act==opt else '✗')
                    star = '★' if node in FOOD_NODES else ' '
                    parts.append(f"{node}{star}:{act[0]}{chk}")
                print(f"    {'  '.join(parts)}")

            # Q_f diagnostics — show best action and value per frequency
            qf = brain.action._Q_f
            qf_diag = []
            for fi in range(8):
                ba = int(np.argmax(qf[fi]))
                bv = float(qf[fi, ba])
                qf_diag.append(f"fi{fi}:{ACTION_NAMES[ba][0]}({bv:+.3f})")
            print(f"    Q_f: {' '.join(qf_diag)}")
            # Show Q_f[7] in detail (food frequency — both H★ and P★)
            print(f"    Q_f[7] N={qf[7,0]:+.3f} E={qf[7,1]:+.3f} "
                  f"S={qf[7,2]:+.3f} W={qf[7,3]:+.3f}")
            # M57 stats
            print(f"    M57: active={out['planning_active']} "
                  f"weight={out['planning_weight']:.3f} "
                  f"rate={brain.planner.planning_rate()*100:.1f}%")
            print()

            food_per_win.append(wfood); wall_per_win.append(wwalls)
            l4_acc_per_win.append(l4_correct_win/max(1,l4_total_win))
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
        'h_found': h_found, 'p_found': p_found, 'food_balance': world.food_balance(),
    }


def print_final_report(brain, world, results):
    fpw = results['food_per_win']; ph = results['policy_history']; n_w = len(fpw)

    print("\n" + "╔" + "═"*62 + "╗")
    print("║  BRAIN IN WORLD 5 — 4×4 GRID SCALING TEST                ║")
    print("╚" + "═"*62 + "╝")

    print(f"\n  FOOD TRAJECTORY ({REPORT_EVERY:,}-step windows):")
    max_r = max((f/REPORT_EVERY*100 for f in fpw), default=1)
    for i,cnt in enumerate(fpw):
        rate = cnt/REPORT_EVERY*100
        bar  = "█"*int(rate/max(max_r,0.01)*30)
        c = ph[i][1] if i<len(ph) else '?'; k = ph[i][2] if i<len(ph) else 16
        print(f"    W{i+1:02d}: {rate:6.2f}/100  {bar}  (policy {c}/{k})")

    first  = sum(fpw[:n_w//2])/max(1,REPORT_EVERY*(n_w//2))*100
    second = sum(fpw[n_w//2:])/max(1,REPORT_EVERY*(n_w-n_w//2))*100
    print(f"\n  First-half:  {first:.2f}/100")
    print(f"  Second-half: {second:.2f}/100  ({second/max(first,0.001):.2f}×)")

    total_food = results['total_food']/CLOSED_LOOP_STEPS*100
    total_wall = results['total_wall']/CLOSED_LOOP_STEPS
    h_f, p_f   = results['food_balance']

    print(f"\n  FOOD RATE: {total_food:.2f}/100  (random ~{RANDOM_FOOD_BASELINE:.1f}/100)")
    print(f"  WALL RATE: {total_wall:.1%}        (random ~{RANDOM_WALL_BASELINE:.1%})")
    print(f"  FOOD BALANCE: H={h_f:.0%}  P={p_f:.0%}  (ideal ~60/40 given distances)")
    print(f"  H★ found: {'Yes' if results['h_found'] else 'NEVER'}")
    print(f"  P★ found: {'Yes' if results['p_found'] else 'NEVER'}")

    nv = results['node_visits']; tv = sum(nv.values())+1
    print(f"\n  NODE VISITS (grid layout):")
    grid = [['A','B','C','D'],['E','F','G','H'],['I','J','K','L'],['M','N','O','P']]
    for row in grid:
        parts = []
        for node in row:
            pct = nv[node]/tv*100
            star = '★' if node in FOOD_NODES else ' '
            parts.append(f"{node}{star}:{pct:4.1f}%")
        print(f"    {'  '.join(parts)}")

    print(f"\n  M58: boredom={brain.wm._last_corridor_boredom:.3f}  "
          f"floor={brain.wm._last_epsilon_floor:.3f}  "
          f"diversity={brain.wm.zone_diversity():.3f}")
    print(f"  M57 planning rate: {brain.planner.planning_rate()*100:.1f}%")
    if brain.l4:
        acc = results['l4_correct_total']/max(1,results['l4_total_total'])
        print(f"  L4 accuracy: {acc:.1%}  ({results['l4_correct_total']}/{results['l4_total_total']} steps)")

    print(f"\n  VERDICT:")
    both = results['h_found'] and results['p_found']
    if total_food > RANDOM_FOOD_BASELINE*2 and total_wall < RANDOM_WALL_BASELINE*2:
        print(f"  ✓ Above 2× random baseline — architecture scales to 16 nodes.")
        print(f"  {'✓' if both else '~'} {'Both food sources found.' if both else 'Only one food source found.'}")
    else:
        print(f"  ✗ Did not scale — below 2× random baseline.")
    print(f"\n  TOTAL: {results['total_food']} food / {CLOSED_LOOP_STEPS:,} steps ({total_food:.2f}/100)\n")


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  BRAIN IN WORLD 5 — 4×4 GRID SCALING TEST                   ║")
    print("║  16 nodes. All fi aliased. Q_f alone insufficient.          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    rx_slow, ry_slow, rx_fast, ry_fast = calibrate()
    library = build_signal_library(rx_slow, ry_slow, rx_fast, ry_fast)

    node_fi = {n: v[1] for n, v in NODES.items()}
    brain   = Brain(seed=SEED, node_fi=node_fi)
    world   = World5(seed=SEED)
    brain.action._Q_f[:] = 0.0

    # ── FIX 1: Register dedicated Q_f rows for aliased food nodes ──
    # H and P share fi=7. Without this, every H visit and every P visit
    # both write to Q_f[7], causing the row to average across both nodes
    # and lose all directional signal. After this call, replay_on_reward
    # and wall penalties for H write to Q_f[8], for P to Q_f[9].
    brain.action.set_node_fi_override(list(FOOD_NODES))

    # ── FIX 2: initialise node override hint to None ──
    # brain.step() reads this each step via getattr.
    # The closed-loop runner sets it to the current food node when the
    # brain is at a food node (so Q_f reads the right row), and clears
    # it when the brain moves away.
    brain._node_fi_override_hint = None

    run_open_loop(brain, library)
    results = run_closed_loop(brain, world, library)
    print_final_report(brain, world, results)