"""
BRAIN IN WORLD 6 — 8×8 GRID (DYNAMIC FOOD + HUNGER + BUTTON→DOOR)
===================================================================

Why this world is harder than World 5
--------------------------------------
  • 64 nodes, 8-aliased per frequency  → Q-table thrashes; L4 + M65 load-bearing
  • Dynamic food (moves every 500 steps) → stale Q-values actively wrong; M60+M63 needed
  • Hunger scaling  → reward = base × hunger; internal state must modulate behaviour
  • Button→Door sequence (15-step window) → M57 + M63 must hold the sub-plan

What "success" looks like here vs World 5
------------------------------------------
  World 5 success: high food/100 rate, L4 accuracy > 80%, Q_n converges.
  World 6 success:
    • food/100 IMPROVES after each food-move event (brain re-adapts)
    • door_rate > 0 (brain learned button→door)
    • hunger_at_eat > 0.6 on average (brain waits until hungry)
    • L4 accuracy > 65% (harder with 8-way aliasing, no locked transitions)

Report printed every REPORT_EVERY steps:
  step | food/100 | wall% | door_rate | hunger_avg | L4% | tex_w | eps
"""

import sys, time
import numpy as np
from collections import deque, Counter

from m50_neuron import (
    run_sim, build_reverse_lookup, make_blocks,
    compute_stability_plv, decode_resonance,
    DivergenceCUSUM, stabilization_time,
)
from brain import Brain
from world6 import (
    World6, NODES, FREQUENCIES, NODE_TEXTURES,
    ADJACENCY, ACTIONS, DOOR_NODES, BUTTON_NODES,
    ALL_NODES, bfs_optimal_actions,
    FOOD_MOVE_INTERVAL,
)
from l4_position import L4_CTM_WARMUP
from m58_workingmemory import BOREDOM_GATE_THRESH
import m56_action as _m56

sys.dont_write_bytecode = True
print('  [brain_in_world6: 8×8 grid — dynamic food + hunger + button→door]')

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

OPEN_STEPS_PER_NODE  = 1_500   # same per-freq calibration as W5
CLOSED_LOOP_STEPS    = 500_000  # longer run — world is harder
REPORT_EVERY         = 10_000
CAL_BLOCK_DUR        = 40.0
SIGNAL_BLOCK_DUR_S   = 60.0
PLV_STAB_WINDOW      = 20
SEED                 = 42

KNOWN_FREQUENCIES    = np.array(FREQUENCIES)
ACTION_NAMES         = ['North', 'East', 'South', 'West']

# Reward nodes that get Q_n override rows in M56
# = all door nodes (they give food reward) + food candidate nodes
from world6 import DOOR_NODES, FOOD_CANDIDATES
REWARD_NODES = list(DOOR_NODES) + list(FOOD_CANDIDATES)

# ═══════════════════════════════════════════════════════════════
# HELPERS  (identical to brain_in_world5)
# ═══════════════════════════════════════════════════════════════

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
        node_labels = [n for n, (f, idx, _) in NODES.items() if idx == fi]
        print(f"  fi={fi} ({freq:.1f}Hz) [{len(node_labels)} nodes]...",
              end="", flush=True)
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


# ═══════════════════════════════════════════════════════════════
# OPEN LOOP  (unchanged from World 5 — calibrates M54/SOM)
# ═══════════════════════════════════════════════════════════════

def run_open_loop(brain, library):
    total_open = OPEN_STEPS_PER_NODE * len(FREQUENCIES)
    print(f"\n  PHASE 1: OPEN LOOP  ({total_open:,} steps)")
    print(f"  Brain hears all 8 frequencies. No rewards. No texture.\n")
    counters = {}
    for fi, freq in enumerate(FREQUENCIES):
        node_labels = [n for n, (f, idx, _) in NODES.items() if idx == fi]
        print(f"  fi={fi} ({freq:.1f}Hz, {len(node_labels)} nodes)...",
              end="", flush=True)
        for _ in range(OPEN_STEPS_PER_NODE):
            decoded, w, nov, plv = get_step(library, freq, counters)
            bucketed_fi = bucket_freq(decoded)
            brain.step(
                decoded_freq   = decoded,
                stability_w    = w,
                novelty_flag   = nov,
                plv_vector     = plv,
                reward         = 0.0,
                freq_idx       = bucketed_fi,
                world_moved    = True,
                texture_active = False,
            )
        print(" done")

    brain.l3.assign_zones_from_counters(brain._freq_bmu_counters)
    n = int((brain.l3._bmu_to_zone >= 0).sum())
    print(f"\n  L3 zones assigned: {n}/64 BMUs mapped.")
    brain.action._e[:] = brain.action._e_f[:] = 0.0
    brain.action._transition_action = -1
    brain.action.t = 0
    brain.valence._reward_ema = 0.0
    brain.l3._zone_visit_ema[:] = 0.0
    brain.l3._recompute_zone_values()
    print(f"  Action state reset.\n")


# ═══════════════════════════════════════════════════════════════
# CLOSED LOOP
# ═══════════════════════════════════════════════════════════════

def run_closed_loop(brain, world, library):
    print(f"  PHASE 2: CLOSED LOOP  ({CLOSED_LOOP_STEPS:,} steps)")
    print(f"  Food moves every {FOOD_MOVE_INTERVAL} steps. "
          f"Button→Door window: 15 steps.\n")

    world.reset()
    if brain.l4 is not None:
        brain.l4.reset_to_node('A0')
    counters = {}

    # Per-window counters
    wfood = wwalls = wdoor_hit = wdoor_open = 0
    wreplay_steps = 0
    hunger_sum_at_eat = 0.0
    n_eats_win = 0
    food_move_events = 0

    # Per-window action tracking (indexed by fi)
    win_action_counts = {fi: Counter() for fi in range(8)}

    # L4 accuracy
    l4_correct_win = l4_total_win = 0
    l4_correct_total = l4_total_total = 0

    # BFS policy for evaluation (recomputed on each food move)
    opt_policy = bfs_optimal_actions(world._active_food, BUTTON_NODES, DOOR_NODES)

    freq_hz         = world.current_freq
    current_texture = world.current_texture
    pending_reward  = 0.0
    pending_food_info = None
    prev_wall_hit   = False
    prev_food_nodes = list(world._active_food)   # track for retirement detection

    for step in range(CLOSED_LOOP_STEPS):
        decoded, w, nov, plv = get_step(library, freq_hz, counters)
        bucketed_fi  = bucket_freq(decoded)
        actual_moved = not prev_wall_hit

        out = brain.step(
            decoded_freq   = decoded,
            stability_w    = w,
            novelty_flag   = nov,
            plv_vector     = plv,
            reward         = pending_reward,
            freq_idx       = bucketed_fi,
            world_moved    = actual_moved,
            texture_val    = current_texture,
            texture_active = True,
        )

        # Replay on food reward (one step delayed, same as W5)
        if pending_food_info is not None:
            brain.action.replay_on_reward(
                reward           = pending_food_info['reward'],
                familiarity      = out['familiarity'],
                food_freq_idx    = pending_food_info['freq_idx'],
                food_node        = pending_food_info['node'],
            )
            pending_food_info = None

        pending_reward = 0.0
        action         = int(out['action'])
        true_node      = world.current_node

        if bucketed_fi >= 0:
            win_action_counts[bucketed_fi][action] += 1

        # ── World step ────────────────────────────────────────
        hunger_before_step = world._hunger   # capture BEFORE step (resets inside on food)
        next_freq_hz, next_freq_idx, reward, info = world.step(action)
        pending_reward  = reward
        current_texture = info['texture']

        # ── Food / anchor logic ───────────────────────────────
        if info['is_food']:
            wfood += 1
            hunger_sum_at_eat += hunger_before_step   # hunger at moment of eating
            n_eats_win += 1
            brain.record_food_trajectory(info['node'], reward)
            pending_food_info = {
                'reward':   reward,
                'freq_idx': next_freq_idx,
                'node':     info['node'],
            }
            if brain.l4 is not None:
                brain.l4.anchor_at_node(info['node'])
            brain._node_fi_override_hint = info['node']

        elif info['is_button']:
            # Button activated — no reward, but log it
            wdoor_open += 1

        else:
            # Standard L4-guided override hint
            l4_node = out.get('l4_top_node')
            l4_prob = out.get('l4_top_prob', 0.0)
            if (l4_node is not None
                    and l4_prob >= 0.55
                    and l4_node in brain.action._node_fi_override_row):
                brain._node_fi_override_hint = l4_node
            else:
                brain._node_fi_override_hint = None

        if info['wall_hit']:
            wwalls += 1
        if info['is_door']:
            wdoor_hit += 1

        prev_wall_hit = info['wall_hit']

        # ── Food moved — recompute BFS policy + clear stale Q_n ──
        if info['food_moved']:
            food_move_events += 1
            opt_policy = bfs_optimal_actions(
                info['food_nodes'], BUTTON_NODES, DOOR_NODES)
            # Clear Q_n for the retired food node AND its 1-2 hop neighborhood.
            # Approach nodes (1-2 steps away) learned "go toward old food" and
            # will keep pointing there even after the food node itself is cleared.
            retired = set(prev_food_nodes) - set(info['food_nodes'])
            for old_node in retired:
                brain.action.decay_node_qn(old_node, factor=0.0)
                # 1-hop neighbors: partial decay — they learned "approach old food here"
                for nbr1 in ADJACENCY[old_node].values():
                    brain.action.decay_node_qn(nbr1, factor=0.3)
            # Boost exploration so brain re-finds new food at high epsilon
            # instead of following stale Q values with the decayed floor (~0.15)
            brain.action.boost_exploration(steps=500)
            prev_food_nodes = list(info['food_nodes'])

        # ── L4 accuracy ───────────────────────────────────────
        if brain.l4 is not None:
            l4_top = out.get('l4_top_node')
            if l4_top is not None:
                l4_total_win   += 1
                l4_total_total += 1
                if l4_top == true_node:
                    l4_correct_win   += 1
                    l4_correct_total += 1

        freq_hz = next_freq_hz

        # ── Sleep Cycle (Hippocampal Replay) ──────────────────
        # Every 100 steps, replay recent food trajectories
        # forward through M56's eligibility trace.
        if (step + 1) % 100 == 0:
            n = brain.idle_step()
            wreplay_steps += n

        # ── Periodic report ───────────────────────────────────
        if (step + 1) % REPORT_EVERY == 0:
            food_rate   = wfood / REPORT_EVERY * 100
            wall_rate   = wwalls / REPORT_EVERY
            door_rate   = wdoor_hit / REPORT_EVERY * 100   # door-food hits per 100 steps
            avg_hunger  = hunger_sum_at_eat / max(1, n_eats_win)
            epsilon     = out['action_epsilon']
            tex_w       = out.get('m65_texture_weight', 0.0)
            fus_active  = out.get('m65_fusion_active', False)
            l4_acc      = l4_correct_win / max(1, l4_total_win)
            plan_active = out.get('planning_active', False)
            plan_rate   = brain.planner.planning_rate()

            # Policy quality vs BFS
            known = correct = 0
            for node in ALL_NODES:
                fi_n = NODES[node][1]
                opt_a = opt_policy.get(node, -1)
                if opt_a < 0:
                    continue
                # Use Q_n if warm, else modal from win_action_counts
                aliased = _m56.L4_Q_N_ALIASED_NODES
                if node in aliased and brain.action._Q_n_count.get(node, 0) >= _m56.L4_Q_N_WARMUP:
                    q_n = brain.action._Q_n.get(node)
                    if q_n is not None:
                        chosen = int(np.argmax(q_n))
                        known  += 1
                        correct += int(chosen == opt_a)
                        continue
                ctr = win_action_counts[fi_n]
                if not ctr:
                    continue
                modal_a = max(ctr, key=ctr.get)
                known  += 1
                correct += int(modal_a == opt_a)

            replay_rate = wreplay_steps / REPORT_EVERY * 100
            print(f"  Step {step+1:7d} | "
                  f"food={food_rate:5.2f}/100 | "
                  f"wall={wall_rate:.1%} | "
                  f"door={door_rate:.2f}/100 | "
                  f"hunger@eat={avg_hunger:.2f} | "
                  f"policy={correct}/{known} | "
                  f"L4={l4_acc:.0%} | "
                  f"M57={'ON' if plan_active else '--'}({plan_rate:.0%}) | "                  f"tex_w={tex_w:.2f}{'★' if fus_active else '○'} | "
                  f"eps={epsilon:.3f} | "
                  f"replay={replay_rate:.1f}/100 | "
                  f"food_moves={food_move_events}")

            # Reset window counters
            wfood = wwalls = wdoor_hit = wdoor_open = 0
            hunger_sum_at_eat = 0.0
            n_eats_win = 0
            l4_correct_win = l4_total_win = 0
            wreplay_steps = 0

    # ── Final summary ─────────────────────────────────────────
    print(f"\n  Final L4 accuracy: "
          f"{l4_correct_total/max(1,l4_total_total):.1%}"
          f" ({l4_correct_total}/{l4_total_total})")
    print(f"  Total food-move events: {food_move_events}")
    if brain._multimodal:
        print(f"  M54b BMU coverage: "
              f"{brain.texture_cortex.bmu_coverage()}/16 neurons")
        print(f"  M65 tex reliability: "
              f"{brain.fusion._tex_reliability:.3f}")

    # ── Q_n policy audit ──────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  Q_n POLICY AUDIT — learned vs BFS-optimal")
    print(f"  {'─'*70}")
    print(f"  {'Node':>5} | {'Opt':>4} | {'Q_n?':>5} | {'N':>5} | "
          f"{'N':>6} {'E':>6} {'S':>6} {'W':>6} | Status")
    print(f"  {'─'*70}")
    anames = ['N', 'E', 'S', 'W']
    correct_qn = wrong_qn = no_qn = 0
    for node in sorted(ALL_NODES):
        opt_a = opt_policy.get(node, -1)
        if opt_a < 0:
            continue
        q_n   = brain.action._Q_n.get(node)
        n_obs = brain.action._Q_n_count.get(node, 0)
        if q_n is not None and n_obs >= _m56.L4_Q_N_WARMUP:
            chosen = int(np.argmax(q_n))
            vals   = '  '.join(f"{q_n[a]:+.3f}" for a in range(4))
            ok     = '✓' if chosen == opt_a else '✗'
            if chosen == opt_a: correct_qn += 1
            else:               wrong_qn   += 1
            print(f"  {node:>5} | {anames[opt_a]:>4} | {anames[chosen]:>5} | "
                  f"{n_obs:>5} | {vals} | {ok}")
        else:
            no_qn += 1
    print(f"  {'─'*70}")
    print(f"  Correct: {correct_qn}  Wrong: {wrong_qn}  No data: {no_qn}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # node_fi: 64 nodes
    node_fi = {name: data[1] for name, data in NODES.items()}

    rx_slow, ry_slow, rx_fast, ry_fast = calibrate()
    library = build_signal_library(rx_slow, ry_slow, rx_fast, ry_fast)

    # Brain with multimodal enabled — required for 8-way aliasing
    brain = Brain(
        seed       = SEED,
        node_fi    = node_fi,
        multimodal = True,
        node_tex   = NODE_TEXTURES,
    )

    # Register all reward nodes for Q_n override rows in M56
    # (door nodes + food candidate nodes — all nodes that give food reward)
    brain.action.set_node_fi_override(REWARD_NODES)
    brain.action.set_node_fi_map(node_fi)

    world = World6(seed=SEED)

    print("\n" + "=" * 64)
    print("  BRAIN IN WORLD 6")
    print(f"  Nodes: {world.n_nodes}  | Frequencies: {len(FREQUENCIES)}")
    print(f"  Aliasing: 8 nodes per frequency")
    print(f"  Dynamic food: moves every {FOOD_MOVE_INTERVAL} steps")
    print(f"  Button→Door: {len(BUTTON_NODES)} buttons, {len(DOOR_NODES)} doors, "
          f"15-step window")
    print(f"  Hunger: reward × hunger (max={1.0})")
    print(f"  Texture: M51→M54b→M65 (min separation=0.13)")
    print("=" * 64)

    run_open_loop(brain, library)
    run_closed_loop(brain, world, library)


if __name__ == '__main__':
    main()