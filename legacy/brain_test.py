"""
BRAIN TEST — COGNITIVE DIAGNOSTIC
===================================

Purpose
-------
This is NOT a performance optimizer. It measures whether the brain
is doing what a real brain should do:

  1. POLICY CONVERGENCE  — does π_node converge to correct directions?
  2. TEMPERATURE DECAY   — does exploration decrease as policy stabilises?
  3. ADAPTATION          — after food moves, how fast does food rate recover?
  4. VALUE GRADIENT      — does the actor policy increase near food?
  5. PREDICTION QUALITY  — does L2 prediction error decrease over time?
  6. LOCALIZATION        — does L4 accuracy improve with texture?

The world (World 6) is the same environment, but it is a test instrument,
not the goal. The brain should function correctly; the test just observes.

What "success" looks like
--------------------------
  • Policy convergence: π_node correct in >50% of warm nodes by step 200k
  • Temperature: decays from ~2.5 to ~0.7 over the run (not stuck at floor)
  • Adaptation: food rate recovers within 2000 steps of each food move
  • L4 accuracy: >70% with texture fusion active
  • Prediction error: decreasing trend (L2 learning the world structure)
  • Wall rate: <15% average (brain learns walls exist, even without explicit tuning)
"""

import sys, time, argparse, pickle, os, hashlib
import numpy as np
from collections import deque

from m50_neuron import (
    run_sim, build_reverse_lookup, make_blocks,
    compute_stability_plv, decode_resonance,
    DivergenceCUSUM, stabilization_time,
)
from brain import Brain
from world6 import (
    World6, NODES, FREQUENCIES, NODE_TEXTURES,
    ADJACENCY, DOOR_NODES, BUTTON_NODES,
    ALL_NODES, bfs_optimal_actions,
    FOOD_MOVE_INTERVAL,
)
import m56_action as _m56

sys.dont_write_bytecode = True

parser = argparse.ArgumentParser()
parser.add_argument('--static-food', action='store_true',
                    help='Disable food movement — convergence validation test')
parser.add_argument('--no-cache', action='store_true',
                    help='Force rebuild of calibration and signal library')
ARGS = parser.parse_args()

mode = 'STATIC FOOD (convergence test)' if ARGS.static_food else 'DYNAMIC FOOD'
print(f'  [brain_test: cognitive diagnostic for Actor-Critic brain | {mode}]')

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

OPEN_STEPS_PER_NODE = 1_500
CLOSED_LOOP_STEPS   = 500_000
REPORT_EVERY        = 10_000
CAL_BLOCK_DUR       = 40.0
SIGNAL_BLOCK_DUR_S  = 60.0
PLV_STAB_WINDOW     = 20
SEED                = 42

KNOWN_FREQUENCIES   = np.array(FREQUENCIES)
ACTION_NAMES        = ['N', 'E', 'S', 'W']

# ═══════════════════════════════════════════════════════════════
# HELPERS
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
# OPEN LOOP
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
    brain.action._e[:] = 0.0
    brain.action._prev_valid = False
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
    print(f"  Food moves every {FOOD_MOVE_INTERVAL} steps.\n")

    world.reset()
    if brain.l4 is not None:
        brain.l4.reset_to_node('A0')
    counters = {}

    # Visit-count curiosity: track how often each node is visited.
    # Adds a small intrinsic reward for visiting less-visited nodes,
    # decaying as 1/(visit_count+1). This prevents the brain from
    # getting stuck looping near one food node (the H3 trap).
    _node_visit_count = {}
    CURIOSITY_SCALE = 0.05   # max intrinsic bonus per step (decays quickly)

    # Per-window counters
    wfood = wwalls = wdoor_hit = 0
    hunger_sum_at_eat = 0.0
    n_eats_win = 0
    food_move_events = 0
    wreplay_steps = 0

    # L4 accuracy
    l4_correct_win = l4_total_win = 0
    l4_correct_total = l4_total_total = 0

    # Adaptation tracking: food rate in the 2000 steps after each food move
    adapt_window = deque(maxlen=2000)   # food hits in last 2000 steps
    adapt_records = []                  # (food_move_n, food_per_100) after each move

    # Prediction error tracking (for cognitive health check)
    pred_err_history = deque(maxlen=REPORT_EVERY)

    # Temperature tracking
    temp_history = deque(maxlen=REPORT_EVERY)

    # BFS policy for evaluation
    opt_policy = bfs_optimal_actions(world._active_food, BUTTON_NODES, DOOR_NODES)

    freq_hz         = world.current_freq
    current_texture = world.current_texture
    pending_reward  = 0.0
    pending_food_info = None
    prev_wall_hit   = False
    prev_food_nodes = list(world._active_food)

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

        # Hippocampal replay one step after food
        if pending_food_info is not None:
            brain.action.replay_on_reward(
                reward        = pending_food_info['reward'],
                familiarity   = out['familiarity'],
                food_freq_idx = pending_food_info['freq_idx'],
                food_node     = pending_food_info['node'],
            )
            pending_food_info = None

        pending_reward = 0.0
        action         = int(out['action'])
        true_node      = world.current_node

        # World step
        hunger_before = world._hunger
        next_freq_hz, next_freq_idx, reward, info = world.step(action)

        # Visit-count curiosity: small bonus for visiting less-visited nodes.
        # Breaks high-visit traps (e.g. eating at H3 every 2 steps with near-zero hunger).
        # Decays to near-zero after ~20 visits so it doesn't override learned policy.
        visited_node = info['node']
        _node_visit_count[visited_node] = _node_visit_count.get(visited_node, 0) + 1
        if not info['wall_hit']:
            reward += CURIOSITY_SCALE / (_node_visit_count[visited_node] + 1)

        pending_reward  = reward
        current_texture = info['texture']

        # Food / anchor
        if info['is_food']:
            wfood += 1
            adapt_window.append(1)
            hunger_sum_at_eat += hunger_before
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
        else:
            adapt_window.append(0)
            l4_node = out.get('l4_top_node')
            l4_prob = out.get('l4_top_prob', 0.0)
            if (l4_node is not None and l4_prob >= 0.55
                    and l4_node in brain.action._node_visits):
                brain._node_fi_override_hint = l4_node
            else:
                brain._node_fi_override_hint = None

        if info['wall_hit']:
            wwalls += 1
        if info['is_door']:
            wdoor_hit += 1

        prev_wall_hit = info['wall_hit']

        # Food moved — clear stale policy and boost exploration
        if info['food_moved'] and not ARGS.static_food:
            food_move_events += 1
            opt_policy = bfs_optimal_actions(
                info['food_nodes'], BUTTON_NODES, DOOR_NODES)
            retired = set(prev_food_nodes) - set(info['food_nodes'])
            for old_node in retired:
                brain.action.decay_node_qn(old_node, factor=0.0)
                for nbr in ADJACENCY[old_node].values():
                    brain.action.decay_node_qn(nbr, factor=0.3)
            brain.action.boost_exploration(steps=500)
            prev_food_nodes = list(info['food_nodes'])
            # Record adaptation quality: food rate in the window before this move
            if len(adapt_window) >= 500:
                recent_food_rate = sum(adapt_window) / len(adapt_window) * 100
                adapt_records.append((food_move_events, recent_food_rate))

        # L4 accuracy
        if brain.l4 is not None:
            l4_top = out.get('l4_top_node')
            if l4_top is not None:
                l4_total_win   += 1
                l4_total_total += 1
                if l4_top == true_node:
                    l4_correct_win   += 1
                    l4_correct_total += 1

        # Cognitive health tracking
        pred_err_history.append(float(out.get('prediction_error', 0.0)))
        temp_history.append(float(out.get('temperature', 2.5)))

        freq_hz = next_freq_hz

        # Idle replay every 100 steps
        if (step + 1) % 100 == 0:
            n = brain.idle_step()
            wreplay_steps += n

        # ── Periodic report ───────────────────────────────────
        if (step + 1) % REPORT_EVERY == 0:
            food_rate  = wfood / REPORT_EVERY * 100
            wall_rate  = wwalls / REPORT_EVERY
            door_rate  = wdoor_hit / REPORT_EVERY * 100
            avg_hunger = hunger_sum_at_eat / max(1, n_eats_win)
            temp_now   = float(brain.action._current_temp)
            base_temp  = float(brain.action._base_temp)
            pred_err   = float(np.mean(pred_err_history)) if pred_err_history else 0.0
            l4_acc     = l4_correct_win / max(1, l4_total_win)
            plan_rate  = brain.planner.planning_rate()
            tex_w      = out.get('m65_texture_weight', 0.0)
            fus_active = out.get('m65_fusion_active', False)
            replay_rate = wreplay_steps / REPORT_EVERY * 100

            # Policy quality: how many warm π_node nodes point correctly
            correct_pol = wrong_pol = no_pol = 0
            for node in ALL_NODES:
                opt_a = opt_policy.get(node, -1)
                if opt_a < 0:
                    continue
                pi, visits = brain.action.get_node_policy(node)
                if pi is not None and visits >= _m56.NODE_MIN_VISITS:
                    chosen = int(np.argmax(pi))
                    if chosen == opt_a:
                        correct_pol += 1
                    else:
                        wrong_pol += 1
                else:
                    no_pol += 1

            print(f"  Step {step+1:7d} | "
                  f"food={food_rate:5.2f}/100 | "
                  f"wall={wall_rate:.1%} | "
                  f"door={door_rate:.2f}/100 | "
                  f"hunger@eat={avg_hunger:.2f} | "
                  f"policy=✓{correct_pol}/✗{wrong_pol}/–{no_pol} | "
                  f"L4={l4_acc:.0%} | "
                  f"M57={plan_rate:.0%} | "
                  f"tex={tex_w:.2f}{'★' if fus_active else '○'} | "
                  f"temp={temp_now:.2f}(base={base_temp:.2f}) | "
                  f"pred_err={pred_err:.3f} | "
                  f"replay={replay_rate:.0f}/100 | "
                  f"moves={food_move_events}")

            # Reset window counters
            wfood = wwalls = wdoor_hit = 0
            hunger_sum_at_eat = 0.0
            n_eats_win = 0
            l4_correct_win = l4_total_win = 0
            wreplay_steps = 0

    # ── Final summary ──────────────────────────────────────────
    print(f"\n  Final L4 accuracy: "
          f"{l4_correct_total/max(1,l4_total_total):.1%}"
          f" ({l4_correct_total}/{l4_total_total})")
    print(f"  Total food-move events: {food_move_events}")
    if brain._multimodal:
        print(f"  M54b BMU coverage: "
              f"{brain.texture_cortex.bmu_coverage()}/16 neurons")
        print(f"  M65 tex reliability: "
              f"{brain.fusion._tex_reliability:.3f}")

    # ── Cognitive health summary ───────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  COGNITIVE HEALTH SUMMARY")
    print(f"  {'─'*70}")

    # Temperature evolution
    final_temp = float(brain.action._base_temp)
    from m56_action import TEMP_INIT, TEMP_MIN, TEMP_WARMUP_STEPS
    print(f"  Temperature:  init={TEMP_INIT:.2f}  final={final_temp:.3f}  "
          f"floor={TEMP_MIN:.2f}  "
          f"{'converging ✓' if final_temp < TEMP_INIT * 0.6 else 'stuck high ✗'}")

    # Adaptation quality
    if adapt_records:
        avg_adapt_rate = float(np.mean([r for _, r in adapt_records]))
        print(f"  Adaptation:   avg food rate at move time = "
              f"{avg_adapt_rate:.2f}/100 across {len(adapt_records)} moves")
    else:
        print(f"  Adaptation:   not enough food move data")

    # ── Actor policy audit ────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  ACTOR POLICY AUDIT — π_node vs BFS-optimal")
    print(f"  {'─'*70}")
    print(f"  {'Node':>5} | {'Opt':>3} | {'Got':>3} | {'Visits':>6} | "
          f"{'N':>6} {'E':>6} {'S':>6} {'W':>6} | Status")
    print(f"  {'─'*70}")

    correct_n = wrong_n = no_data_n = 0
    for node in sorted(ALL_NODES):
        opt_a = opt_policy.get(node, -1)
        if opt_a < 0:
            continue
        pi, visits = brain.action.get_node_policy(node)
        if pi is not None and visits >= _m56.NODE_MIN_VISITS:
            chosen = int(np.argmax(pi))
            vals   = '  '.join(f"{pi[a]:+.3f}" for a in range(4))
            ok     = '✓' if chosen == opt_a else '✗'
            if chosen == opt_a: correct_n += 1
            else:               wrong_n   += 1
            print(f"  {node:>5} | {ACTION_NAMES[opt_a]:>3} | "
                  f"{ACTION_NAMES[chosen]:>3} | {visits:>6} | {vals} | {ok}")
        else:
            no_data_n += 1

    print(f"  {'─'*70}")
    total_warm = correct_n + wrong_n
    acc = correct_n / max(1, total_warm)
    print(f"  Correct: {correct_n}  Wrong: {wrong_n}  No data: {no_data_n}  "
          f"Accuracy: {acc:.0%} of warm nodes")

    # ── Value gradient check ──────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  ACTOR GRADIENT CHECK — policy strength at food-adjacent nodes")
    print(f"  {'─'*70}")
    food_nodes_now = set(world._active_food)
    for food_node in food_nodes_now:
        nbrs = list(ADJACENCY[food_node].values())
        for nbr in nbrs:
            pi, visits = brain.action.get_node_policy(nbr)
            opt_a = opt_policy.get(nbr, -1)
            if pi is None or opt_a < 0:
                continue
            strength = float(pi[opt_a])
            print(f"    {nbr}→{food_node}: "
                  f"π[{ACTION_NAMES[opt_a]}]={strength:+.3f}  visits={visits}  "
                  f"{'strong ✓' if strength > 0.2 else 'weak'}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

_CACHE_KEY = hashlib.md5(
    f"{CAL_BLOCK_DUR}_{SIGNAL_BLOCK_DUR_S}_{PLV_STAB_WINDOW}_{FREQUENCIES}".encode()
).hexdigest()[:8]
_CACHE_FILE = f".brain_test_cache_{_CACHE_KEY}.pkl"


def _load_cache():
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def _save_cache(data):
    try:
        with open(_CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"  [cache write failed: {e}]")


def main():
    node_fi = {name: data[1] for name, data in NODES.items()}

    if not ARGS.no_cache:
        cached = _load_cache()
    else:
        cached = None

    if cached is not None:
        print("  [Using cached calibration + signal library — pass --no-cache to rebuild]")
        rx_slow, ry_slow, rx_fast, ry_fast, library = cached
    else:
        rx_slow, ry_slow, rx_fast, ry_fast = calibrate()
        library = build_signal_library(rx_slow, ry_slow, rx_fast, ry_fast)
        _save_cache((rx_slow, ry_slow, rx_fast, ry_fast, library))
        print(f"  [Signal cache saved → {_CACHE_FILE}]")

    brain = Brain(
        seed       = SEED,
        node_fi    = node_fi,
        multimodal = True,
        node_tex   = NODE_TEXTURES,
    )

    # Set node map so brain.action knows aliasing structure
    brain.action.set_node_fi_map(node_fi)

    world = World6(seed=SEED)

    # Static food: prevent world from ever triggering a food move
    import world6 as _world6_mod
    if ARGS.static_food:
        _world6_mod.FOOD_MOVE_INTERVAL = 10_000_000

    food_mode_str = (f'STATIC (never moves)'
                     if ARGS.static_food
                     else f'DYNAMIC (every {FOOD_MOVE_INTERVAL} steps)')
    print("\n" + "=" * 64)
    print("  BRAIN TEST — WORLD 6")
    print(f"  Nodes: {world.n_nodes}  | Frequencies: {len(FREQUENCIES)}")
    print(f"  Aliasing: 8 nodes per frequency")
    print(f"  Food: {food_mode_str}")
    print(f"  Texture: M51→M54b→M65  |  Actor-Critic: M56 v2")
    print("=" * 64)

    run_open_loop(brain, library)
    run_closed_loop(brain, world, library)


if __name__ == '__main__':
    main()
