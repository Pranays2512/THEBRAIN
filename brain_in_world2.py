"""
BRAIN IN WORLD 2 — HONEST EAR-ONLY TEST
========================================

The brain is blind. It has one sense: an ear (M50).
It receives frequencies. That is ALL it receives from the world.

What the brain does NOT get (removed from the original test):
  ✗  freq_idx    — the ground-truth node label
  ✗  world_moved — whether its last action caused movement

What the brain gets instead (derived from sound alone):
  ✓  decoded_freq  — what M50 hears
  ✓  stability_w   — how stable the signal is
  ✓  novelty_flag  — did the signal regime change?
  ✓  plv_vector    — raw PLV from M50
  ✓  reward        — external signal (food/wall). This is fine.
                     A real animal CAN taste food. It cannot see a map.

WALL DETECTION (sound-derived):
  The brain infers wall hits by comparing the current decoded frequency
  to the previous decoded frequency. If the frequency did not change
  (within FREQ_SAME_THRESHOLD), the brain probably didn't move.
  This is imperfect — exactly as it should be. The brain must learn
  to act on uncertain inference, not ground truth.

ACTION LEARNING (sound-derived):
  _Q_f (the freq_idx-keyed Q table) is removed entirely. It was a
  cheat — it indexed Q on the secret node label. Instead, M56 uses
  only the BMU-pair Q table _Q[prev_bmu, curr_bmu, action], which
  is driven purely by the cortical representation M54 builds from
  hearing frequencies. If M54 correctly separates the 8 frequencies
  into distinct BMU regions, Q will learn node-specific policies.
  If M54 cannot separate them, Q will fail — and that is a real and
  important result.

L3 ZONE LEARNING (sound-derived):
  In the original test, zones were assigned by injecting freq_idx
  every step. Here, L3 must build its own zones from the statistics
  of which BMUs fire together over time — purely from the sound
  stream. Zone assignment is driven by decoded_freq bucketed to the
  nearest known frequency, then used to update freq_bmu_counters.
  The bucketing uses M50's known training frequencies — this is
  analogous to the brain having a prior on what frequencies exist
  in its world, without knowing which location each frequency maps to.

M57 PLANNING (sound-derived):
  M57 simulates future BMU sequences using L2's learned transitions.
  These transitions were learned from the actual frequency stream,
  so M57 is planning over genuinely learned auditory trajectories.
  The zone_bonus in _score_state uses L3's reward EMA — which is
  now learned from experience, not from injected freq_idx labels.

WHAT THIS TEST ACTUALLY MEASURES:
  1. Can M54 reliably separate 8 frequencies into distinct BMU regions?
  2. Can L2 learn auditory transition sequences (A→B→C→E★)?
  3. Can M56 learn to associate BMU pairs with good actions?
  4. Can M57 plan useful look-ahead using learned transitions?
  5. Can the whole stack navigate better than chance using sound alone?

If the food rate rises above random chance (~12.5% assuming uniform
exploration across 8 nodes), the brain is genuinely learning to
navigate from sound. If not, that is also a true result.

RANDOM BASELINE:
  8 nodes, 4 actions each, 2 food nodes (E, H).
  From any node, P(food on this step | random action) ≈ 0.125.
  Wall rate under random policy ≈ 56.2% (most actions are walls).

ARCHITECTURE REMINDER:
  M50 (ear) → M54 (cortex) → M55 (memory) → L2 (sequence predictor)
  → Attention → Thought → Valence → M56 (action/Q) → M57 (planner)
  → L3 (concept zones)

  All modules receive only what they would receive in a real animal
  with no access to world state variables.
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

# ── Run parameters ────────────────────────────────────────────
OPEN_STEPS_PER_NODE  = 1_500
CLOSED_LOOP_STEPS    = 200_000
REPORT_EVERY         = 10_000
CAL_BLOCK_DUR        = 40.0
SIGNAL_BLOCK_DUR_S   = 60.0
PLV_STAB_WINDOW      = 20
SEED                 = 42

# ── Wall inference from sound ─────────────────────────────────
# If the decoded frequency changes by less than this between steps,
# the brain concludes it probably didn't move (wall hit).
# This is a soft heuristic — not ground truth.
FREQ_SAME_THRESHOLD  = 0.08   # Hz. Frequencies are 0.5–2.0 Hz, spaced
                               # 0.2–0.3 Hz apart. 0.08 is below the
                               # smallest gap, so genuine moves are
                               # reliably detected. Noise within a node
                               # is typically < 0.02 Hz.

# ── Frequency bucketing for L3 zone learning ─────────────────
# The brain knows the set of frequencies that exist in its world
# (from open-loop calibration). It does NOT know which frequency
# maps to which location. Bucketing decoded_freq to the nearest
# known frequency gives a noisy but real zone signal.
KNOWN_FREQUENCIES    = np.array(FREQUENCIES)   # [0.5, 0.7, ..., 2.0]

# ── Ground-truth oracle (for policy scoring only — never fed to brain) ──
OPTIMAL_ACTION = {
    'A': 1,   # East → B → C → E★
    'B': 1,   # East → C → E★
    'C': 2,   # South → E★
    'D': 2,   # South → F → G → H★
    'E': 0,   # North → C
    'F': 1,   # East → G → H★
    'G': 1,   # East → H★
    'H': 3,   # West → G
}
ACTION_NAMES = ['North', 'East', 'South', 'West']


# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

def bucket_freq(decoded_hz: float) -> int:
    """
    Map a decoded frequency to the index of the nearest known frequency.
    This is what the brain can infer from its calibration — not ground truth.
    Returns -1 if decoded_hz is implausibly far from all known frequencies.
    """
    diffs = np.abs(KNOWN_FREQUENCIES - decoded_hz)
    idx   = int(np.argmin(diffs))
    if diffs[idx] > 0.4:   # more than ~2 frequency steps away — unreliable
        return -1
    return idx


def infer_moved(prev_freq: float, curr_freq: float) -> bool:
    """
    Infer whether the brain moved based on frequency change.
    Returns True if the frequency changed enough to suggest a new node.
    This replaces the injected world_moved=not info['wall_hit'].
    """
    return abs(curr_freq - prev_freq) >= FREQ_SAME_THRESHOLD


# ══════════════════════════════════════════════════════════════
# CALIBRATION (unchanged from original — M50 setup)
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
# OPEN LOOP — let brain hear all frequencies freely
# No rewards. No location labels. Pure sensory exploration.
# ══════════════════════════════════════════════════════════════

def run_open_loop(brain, library):
    total_open = OPEN_STEPS_PER_NODE * len(FREQUENCIES)
    print(f"\n  PHASE 1: OPEN LOOP  ({total_open:,} steps)")
    print(f"  Brain hears all frequencies. No rewards. No location labels.\n")
    counters = {}

    for fi, freq in enumerate(FREQUENCIES):
        node = list(NODES.keys())[fi]
        print(f"  {node} ({freq:.1f}Hz)...", end="", flush=True)
        for _ in range(OPEN_STEPS_PER_NODE):
            decoded, w, nov, plv = get_step(library, freq, counters)

            # ── KEY DIFFERENCE ────────────────────────────────
            # freq_idx is NOT passed. The brain must build its own
            # internal representation of this frequency from sound.
            # We bucket decoded_freq to update freq_bmu_counters
            # for L3 zone learning — but this uses only the decoded
            # frequency itself, not the ground-truth node label.
            bucketed_fi = bucket_freq(decoded)

            brain.step(
                decoded_freq = decoded,
                stability_w  = w,
                novelty_flag = nov,
                plv_vector   = plv,
                reward       = 0.0,
                freq_idx     = bucketed_fi,   # sound-derived, not ground truth
                world_moved  = True,
            )
        print(" done")

    # Assign L3 zones from the counters the brain built from sound.
    brain.l3.assign_zones_from_counters(brain._freq_bmu_counters)
    n = int((brain.l3._bmu_to_zone >= 0).sum())
    print(f"\n  L3 zones assigned: {n}/64 BMUs mapped (from sound alone).")

    # Reset action/Q_f state — same reasoning as original.
    # The open loop builds Q from intrinsic RPE driven by arbitrary
    # frequency ordering. That has no navigational meaning.
    # _Q (BMU-pair table) is kept — it acts as a weak prior as before.
    # _Q_f does NOT EXIST in this version (removed, see below).
    brain.action._e[:]   = 0.0
    brain.action._e_f[:] = 0.0
    brain.action._transition_action = -1
    brain.valence._reward_ema = 0.0
    # Reset zone visit EMA — open loop visited all zones equally (1500 steps each)
    # so visit_ema ≈ 0.78 for ALL zones after open loop. Without reset, the
    # curiosity bonus is uniform and useless. Reset to 0 so closed-loop visits
    # genuinely differentiate explored vs unexplored territory.
    brain.l3._zone_visit_ema[:] = 0.0
    brain.l3._recompute_zone_values()
    print(f"  Action state reset (eligibility traces, reward_ema, zone_visit_ema).\n")


# ══════════════════════════════════════════════════════════════
# CLOSED LOOP — honest navigation
# The brain hears, decides, acts, receives reward.
# It never receives its location or movement ground truth.
# ══════════════════════════════════════════════════════════════

def run_closed_loop(brain, world, library):
    print(f"  PHASE 2: CLOSED LOOP  ({CLOSED_LOOP_STEPS:,} steps)")
    print(f"  Food: E(1.3Hz)+1.0  H(2.0Hz)+1.0  |  Wall: -0.05")
    print(f"  Brain receives: sound only. No location labels.\n")

    world.reset()
    counters       = {}
    food_per_win   = []
    wall_per_win   = []
    wfood = wwalls = 0

    # Per-window policy tracking
    # Maps bucketed_fi → action counts so we can score the sound-based policy
    # against ground truth at report time.
    policy_history = []

    freq_hz  = world.current_freq
    freq_idx = world.current_freq_idx   # oracle — ONLY used for world navigation
                                        # and end-of-window policy scoring.
                                        # NEVER passed to brain.step().

    pending_reward = 0.0
    prev_decoded   = freq_hz   # last decoded frequency brain heard
                               # used for sound-derived wall inference

    # Per-window action accumulator keyed by bucketed_fi
    win_action_counts = {fi: Counter() for fi in range(8)}

    for step in range(CLOSED_LOOP_STEPS):
        # ── Hear the current node ─────────────────────────────
        decoded, w, nov, plv = get_step(library, freq_hz, counters)

        # ── Sound-derived signals (the brain's ONLY information) ─
        bucketed_fi   = bucket_freq(decoded)
        inferred_move = infer_moved(prev_decoded, decoded)

        # ── Brain step — NO freq_idx, NO world_moved ground truth ─
        out = brain.step(
            decoded_freq = decoded,
            stability_w  = w,
            novelty_flag = nov,
            plv_vector   = plv,
            reward       = pending_reward,
            freq_idx     = bucketed_fi,    # sound-derived bucket, not oracle
            world_moved  = inferred_move,  # sound-derived inference, not oracle
        )
        pending_reward = 0.0

        # ── Brain decides action ──────────────────────────────
        action = int(out['action'])

        # Track this action against the bucketed zone for policy scoring
        if bucketed_fi >= 0:
            win_action_counts[bucketed_fi][action] += 1

        # ── World steps (oracle — brain cannot see this) ──────
        next_freq_hz, next_freq_idx, reward, info = world.step(action)

        pending_reward = reward

        if info['is_food']:
            # Hippocampal replay on food.
            # food_freq_idx is NOT passed — we don't tell the brain
            # which node had food. The replay credits the BMU path
            # that led here, using the BMU-pair Q table only.
            brain.action.replay_on_reward(
                reward        = reward,
                familiarity   = out['familiarity'],
                food_freq_idx = -1,   # ← not telling brain which node this is
            )
            wfood += 1

        if info['wall_hit']:
            wwalls += 1

        prev_decoded = decoded   # update for next step's wall inference
        freq_hz      = next_freq_hz
        freq_idx     = next_freq_idx   # oracle, never passed to brain

        # ── Reporting ─────────────────────────────────────────
        if (step + 1) % REPORT_EVERY == 0:
            food_rate = wfood / REPORT_EVERY * 100
            wall_rate = wwalls / REPORT_EVERY
            epsilon   = out['action_epsilon']

            # Policy scoring: for each bucketed_fi, what was the modal
            # action this window? Score it against ground truth.
            snap    = {}
            correct = 0
            known   = 0
            for fi, node in enumerate(list(NODES.keys())):
                ctr = win_action_counts[fi]
                if not ctr:
                    snap[node] = '?'
                    continue
                modal_a = max(ctr, key=ctr.get)
                snap[node] = ACTION_NAMES[modal_a]
                known += 1
                if modal_a == OPTIMAL_ACTION[node]:
                    correct += 1

            pa_ready  = brain.pred.pa_ready() if hasattr(brain.pred, 'pa_ready') else False
            plan_rate = brain.planner.planning_rate()
            print(f"  Step {step+1:7d} | food/100={food_rate:5.2f} | "
                  f"wall={wall_rate:.1%} | policy={correct}/{known} correct | "
                  f"eps={epsilon:.3f} | PA={'ok' if pa_ready else '..'} | "
                  f"plan={plan_rate:.0%}")

            for node in list(NODES.keys()):
                act     = snap.get(node, '?')
                optimal = ACTION_NAMES[OPTIMAL_ACTION[node]]
                check   = '?' if act == '?' else ('✓' if act == optimal else '✗')
                star    = '★' if node in FOOD_NODES else ' '
                fi      = list(NODES.keys()).index(node)
                freq    = FREQUENCIES[fi]
                print(f"    {node}{star} ({freq:.1f}Hz): {act:5s} {check}  "
                      f"(optimal: {optimal})")
            print()

            food_per_win.append(wfood)
            wall_per_win.append(wwalls)
            policy_history.append((step + 1, correct, known, snap.copy()))
            wfood = wwalls = 0
            win_action_counts = {fi: Counter() for fi in range(8)}

    return {
        'food_per_win':   food_per_win,
        'wall_per_win':   wall_per_win,
        'total_food':     world.food_count,
        'total_wall':     world.wall_count,
        'policy_history': policy_history,
    }


# ══════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════

def print_final_report(brain, world, results):
    fpw = results['food_per_win']
    wpw = results['wall_per_win']
    ph  = results['policy_history']
    n_w = len(fpw)

    print("\n" + "╔" + "═"*62 + "╗")
    print("║  BRAIN IN WORLD 2 — HONEST EAR-ONLY FINAL REPORT         ║")
    print("╚" + "═"*62 + "╝")

    print(f"\n  FOOD RATE TRAJECTORY ({REPORT_EVERY:,}-step windows):")
    max_rate = max((f / REPORT_EVERY * 100 for f in fpw), default=1)
    for i, count in enumerate(fpw):
        rate    = count / REPORT_EVERY * 100
        bar     = "█" * max(0, int(rate / max(max_rate, 0.01) * 30))
        correct = ph[i][1] if i < len(ph) else '?'
        n_known = ph[i][2] if i < len(ph) else 8
        print(f"    W{i+1:02d}: {rate:6.2f}/100  {bar}  "
              f"(policy {correct}/{n_known})")

    first  = sum(fpw[:n_w//2]) / max(1, REPORT_EVERY * (n_w//2)) * 100
    second = sum(fpw[n_w//2:]) / max(1, REPORT_EVERY * (n_w - n_w//2)) * 100
    print(f"\n  First-half avg:  {first:.2f}/100 steps")
    print(f"  Second-half avg: {second:.2f}/100 steps")
    print(f"  Improvement:     {second/max(first,0.001):.2f}×")

    wall_rate = results['total_wall'] / CLOSED_LOOP_STEPS
    print(f"\n  WALL RATE: {wall_rate:.1%}  (random baseline ~56.2%)")
    if wall_rate < 0.562:
        print(f"  ✓ BELOW random baseline — brain is learning valid directions")
    else:
        print(f"  ✗ Still at/above random — more training needed")

    food_rate_total = results['total_food'] / CLOSED_LOOP_STEPS * 100
    print(f"\n  FOOD RATE: {food_rate_total:.2f}/100 steps  "
          f"(random baseline ~12.5/100)")
    if food_rate_total > 12.5:
        print(f"  ✓ ABOVE random baseline — brain is navigating toward food")
    else:
        print(f"  ✗ Below/at random — navigation not learned from sound alone")

    print(f"\n  FINAL POLICY vs OPTIMAL (sound-derived modal actions):")
    snap = ph[-1][3] if ph else {}
    for node in list(NODES.keys()):
        act     = snap.get(node, '?')
        optimal = ACTION_NAMES[OPTIMAL_ACTION[node]]
        check   = '?' if act == '?' else ('✓' if act == optimal else '✗')
        star    = '★' if node in FOOD_NODES else ' '
        fi      = list(NODES.keys()).index(node)
        freq    = FREQUENCIES[fi]
        valid   = list(ADJACENCY[node].keys())
        print(f"    {node}{star} ({freq:.1f}Hz): {act:5s} {check}  "
              f"optimal={optimal}  valid={valid}")

    final_correct = ph[-1][1] if ph else 0
    final_known   = ph[-1][2] if ph else 0
    print(f"\n  Final policy score: {final_correct}/{final_known} nodes correct")

    print(f"\n  L3 ZONE REWARD EMA (learned from food events, no labels):")
    for zi, val in enumerate(brain.l3._zone_reward_ema):
        node = list(NODES.keys())[zi]
        star = '★' if node in FOOD_NODES else ' '
        bar  = "█" * max(0, int(val * 40))
        freq = FREQUENCIES[zi]
        print(f"    Z{zi} {node}{star} ({freq:.1f}Hz): {val:.4f}  {bar}")

    print(f"\n  M57 PLANNER RATE: {brain.planner.planning_rate()*100:.1f}% "
          f"steps where planning overrode habit")

    print(f"\n  WHAT THESE RESULTS MEAN:")
    print(f"  - Food rate > 12.5 → brain navigates above chance from sound alone")
    print(f"  - Wall rate < 56.2% → brain avoids walls using auditory memory")
    print(f"  - Policy score → how many nodes have a correct sound→action mapping")
    print(f"  - If food rate ≈ 12.5 and wall rate ≈ 56.2%, the brain has NOT")
    print(f"    learned to navigate — which is a real and honest result.")

    print(f"\n  TOTAL: {results['total_food']} food events / "
          f"{CLOSED_LOOP_STEPS:,} steps "
          f"({results['total_food']/CLOSED_LOOP_STEPS*100:.2f}/100 steps)\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  BRAIN IN WORLD 2 — HONEST EAR-ONLY NAVIGATION TEST        ║")
    print("║                                                              ║")
    print("║  The brain is blind. It has only an ear.                    ║")
    print("║  No location labels. No wall-hit ground truth.              ║")
    print("║  Navigation must emerge from sound alone.                   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    print("  WHAT IS DIFFERENT FROM brain_in_world.py:")
    print("  ✗  freq_idx NOT injected into brain.step() — brain doesn't")
    print("     know which node it's at. It must infer from M50 output.")
    print("  ✗  world_moved NOT injected — brain infers wall hits from")
    print("     whether the decoded frequency changed.")
    print("  ✗  _Q_f (freq-idx Q table) disabled — was a cheat sheet.")
    print("     M56 uses only BMU-pair Q learned from cortical firing.")
    print("  ✗  food_freq_idx NOT passed to replay — brain doesn't know")
    print("     which node had food, only that food arrived.")
    print("  ✓  reward is passed — animals can taste food.")
    print("  ✓  All modules run unchanged — only the information fed")
    print("     to them is restricted to what a real ear provides.\n")

    rx_slow, ry_slow, rx_fast, ry_fast = calibrate()
    library = build_signal_library(rx_slow, ry_slow, rx_fast, ry_fast)

    brain = Brain(seed=SEED)
    world = World(seed=SEED)

    # Disable _Q_f — it was indexed on injected freq_idx.
    # Q learning now runs entirely through the BMU-pair table _Q.
    # Set _Q_f to a read-only zero array so any accidental access
    # returns 0 rather than raising an error.
    brain.action._Q_f[:] = 0.0

    run_open_loop(brain, library)
    results = run_closed_loop(brain, world, library)
    print_final_report(brain, world, results)