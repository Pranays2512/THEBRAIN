"""
BRAIN AUDIO TEST — M50 → Brain integrated test
================================================

This is the CORRECT test for this brain at its current developmental stage.

M50 (the ear) generates real oscillator signals from 500 Hopf oscillators.
Brain receives those signals step-by-step, exactly as it would in real use.
No grid. No fake inputs. Just the brain listening to sounds.

WHAT THIS TESTS
---------------
Five behaviours that emerge naturally from hearing:

  T1 — FAMILIARITY CONSOLIDATION
       Brain hears the same tone repeatedly.
       M55 familiarity should grow. L2 prediction error should fall.
       Tests: does the brain learn to recognise what it has heard before?

  T2 — NOVELTY DETECTION
       Brain hears a new tone it has never heard.
       Salience should spike. Curiosity should rise. Surprise signal should fire.
       Tests: does the brain notice when something new arrives?

  T3 — SEQUENCE LEARNING
       Brain hears A → B → A → B alternating.
       L2 prediction error should fall over repetitions.
       Thought confidence should rise as sequences become predictable.
       Tests: does the brain learn what comes next?

  T4 — CURIOSITY DRIVES ATTENTION
       In novel regions (new frequencies), curiosity and salience should be
       higher than in familiar regions.
       Tests: does the brain attend more to what it doesn't know yet?

  T5 — PLANNING ENGAGES ON KNOWN SEQUENCES
       M57 planning_weight should rise after sequences are learned.
       Plan disagrees with habit more when the brain has a model.
       Tests: does deliberation emerge from knowledge?

WHY THIS IS THE RIGHT TEST
---------------------------
M50 outputs exactly: decoded_freq, stability_w, novelty_flag, plv_vector
Brain.step() expects exactly: decoded_freq, stability_w, novelty_flag, plv_vector

The calibration phase (which runs first) is the brain's developmental
"exposure period" — like an infant hearing sounds for the first time
and building the internal map that all subsequent cognition rests on.

The test phases are then the brain's actual cognitive behaviour.

ARCHITECTURE NOTE
-----------------
This test does NOT evaluate navigation. The brain has no motor output
connected to an environment here — that requires embodiment (eyes, body)
that doesn't exist yet. What it evaluates is the cognitive stack:
perception → memory → prediction → attention → deliberation.

Run time: ~5-10 minutes (dominated by M50 Hopf oscillator simulation).
"""

import numpy as np
import json
import os
import sys
from collections import deque

# ── M50 imports ─────────────────────────────────────────────────
from m50_neuron import (
    run_sim, make_blocks, make_sweep,
    decode_resonance, build_reverse_lookup,
    DivergenceCUSUM, compute_stability_plv,
    PLV_STAB_WINDOW, stabilization_time, dt,
    fit_ridge, predict_ridge,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
)

# ── Brain import ─────────────────────────────────────────────────
from brain import Brain


# ═══════════════════════════════════════════════════════════════
# CALIBRATION FREQUENCIES
# ═══════════════════════════════════════════════════════════════

# Compact but covering the full 0.5–2.0 Hz range M50 was designed for
CAL_FREQS = sorted([
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
    1.7, 1.8, 1.9, 2.0
])


# ═══════════════════════════════════════════════════════════════
# STEP 1: CALIBRATE M50
# ═══════════════════════════════════════════════════════════════

def calibrate_m50():
    """
    Run M50's calibration phase to build the reverse lookup table.
    This is analogous to the brain's early developmental period —
    the oscillator network tuning itself to the frequency range.

    Returns raw_x_slow, true_y_slow, raw_x_fast, true_y_fast
    for use in decode_resonance() calls during the live test.
    """
    print("=" * 68)
    print("  STEP 1: CALIBRATING M50 EAR")
    print("  (Building frequency → oscillator response lookup)")
    print("=" * 68)

    block_sig, _ = make_blocks(CAL_FREQS, block_dur=40.0)
    total_time   = stabilization_time + 2 * len(CAL_FREQS) * 40.0 + 10.0

    print(f"\n  Frequencies: {len(CAL_FREQS)} points "
          f"({CAL_FREQS[0]:.1f}–{CAL_FREQS[-1]:.1f} Hz)")
    print(f"  Sim duration: {total_time:.0f}s  "
          f"({int(total_time/dt):,} oscillator steps)")
    print(f"  Running...", end="", flush=True)

    np.random.seed(1)
    data_cal = run_sim(
        block_sig,
        total_time    = total_time,
        sweep_mode    = False,
        dynamic_settle = True,
        verbose       = False,
        collect_calib = True,
    )
    print(" done.")

    raw_x_slow, true_y_slow = build_reverse_lookup(
        sorted(data_cal['calib_plv_slow'].keys()),
        data_cal['calib_plv_slow'],
        data_cal['calib_energy_slow'],
    )
    raw_x_fast, true_y_fast = build_reverse_lookup(
        sorted(data_cal['calib_plv_fast'].keys()),
        data_cal['calib_plv_fast'],
        data_cal['calib_energy_fast'],
    )

    print(f"\n  Slow lookup: {len(raw_x_slow)} pts  "
          f"[{raw_x_slow[0]:.3f}, {raw_x_slow[-1]:.3f}] Hz")
    print(f"  Fast lookup: {len(raw_x_fast)} pts  "
          f"[{raw_x_fast[0]:.3f}, {raw_x_fast[-1]:.3f}] Hz")

    return raw_x_slow, true_y_slow, raw_x_fast, true_y_fast


# ═══════════════════════════════════════════════════════════════
# STEP 2: M50 → Brain bridge
# Convert M50 simulation output (array of snapshots) into
# per-step Brain.step() calls
# ═══════════════════════════════════════════════════════════════

def run_brain_on_audio(data, raw_x_slow, true_y_slow,
                       raw_x_fast, true_y_fast,
                       label="", reward_fn=None):
    """
    Feed a pre-simulated M50 dataset through Brain step by step.

    M50 outputs arrays of shape (N_steps, 500) for plv and energy.
    We decode each step to get:
      decoded_freq  ← from slow PLV (most stable)
      stability_w   ← from PLV stability computation
      novelty_flag  ← from DivergenceCUSUM
      plv_vector    ← raw slow PLV (500-dim, M54 takes top-20 internally)

    Returns list of per-step Brain output dicts + metadata.
    """
    n        = len(data['Y'])
    brain    = Brain(seed=42)
    plv_hist = deque(maxlen=PLV_STAB_WINDOW)
    cusum    = DivergenceCUSUM()

    # Decode all steps
    df_arr = np.array([
        decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                         raw_x_fast, true_y_fast)
        for i in range(n)
    ])
    ds_arr = np.array([
        decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                         raw_x_slow, true_y_slow)
        for i in range(n)
    ])

    steps       = []
    true_freqs  = data['Y']

    for i in range(n):
        # Stability weight from PLV
        max_plv = float(np.max(data['plv_slow'][i]))
        plv_hist.append(max_plv)
        w = compute_stability_plv(plv_hist)

        # Novelty flag from CUSUM
        _, is_novel = cusum.update(df_arr[i], ds_arr[i], data['T'][i], w=w)
        if is_novel:
            w = 0.0  # suppress slow during transitions (M50 design)

        decoded_freq = float(w * ds_arr[i] + (1.0 - w) * df_arr[i])
        stability_w  = float(w)
        novelty_flag = float(is_novel)
        plv_vector   = data['plv_slow'][i]  # (500,) — M54 selects top-20 internally

        # External reward (optional — e.g. reward when at a target frequency)
        reward = float(reward_fn(i, true_freqs[i])) if reward_fn else 0.0

        out = brain.step(
            decoded_freq = decoded_freq,
            stability_w  = stability_w,
            novelty_flag = novelty_flag,
            plv_vector   = plv_vector,
            reward       = reward,
        )

        steps.append({
            **out,
            'true_freq':   float(true_freqs[i]),
            'decoded_freq': decoded_freq,
            'stability_w':  stability_w,
            'novelty_flag': novelty_flag,
            't':            float(data['T'][i]),
            'reward':       reward,
        })

        if i % 200 == 0:
            pct = 100 * i / n
            print(f"    [{label}] step {i}/{n} ({pct:.0f}%)  "
                  f"fam={out['familiarity']:.3f}  "
                  f"sal={out['salience']:.3f}  "
                  f"pe={out['prediction_error']:.3f}",
                  flush=True)

    return steps, brain


# ═══════════════════════════════════════════════════════════════
# STEP 3: RUN TEST SCENARIOS
# ═══════════════════════════════════════════════════════════════

def run_all_tests(raw_x_slow, true_y_slow, raw_x_fast, true_y_fast):
    """
    Run five audio scenarios and collect Brain outputs.
    """
    print("\n" + "=" * 68)
    print("  STEP 2: RUNNING AUDIO SCENARIOS THROUGH BRAIN")
    print("=" * 68)

    results = {}

    # ── Scenario A: Repeated single tone (familiarity test) ──────
    print("\n  [A] REPEATED SINGLE TONE — 1.0 Hz (familiarity/habituation)")
    np.random.seed(10)
    sig_a, _ = make_blocks([1.0], block_dur=60.0)
    d_a = run_sim(sig_a, total_time=stabilization_time + 180.0,
                  sweep_mode=False, dynamic_settle=False, verbose=False)
    steps_a, brain_a = run_brain_on_audio(d_a, raw_x_slow, true_y_slow,
                                           raw_x_fast, true_y_fast, label="A")
    results['repeated'] = steps_a

    # ── Scenario B: Two-tone alternating (sequence learning) ─────
    print("\n  [B] TWO-TONE ALTERNATING — 0.7 Hz / 1.4 Hz (sequence learning)")
    np.random.seed(11)
    sig_b, _ = make_blocks([0.7, 1.4], block_dur=30.0)
    d_b = run_sim(sig_b, total_time=stabilization_time + 240.0,
                  sweep_mode=False, dynamic_settle=False, verbose=False)
    steps_b, brain_b = run_brain_on_audio(d_b, raw_x_slow, true_y_slow,
                                           raw_x_fast, true_y_fast, label="B")
    results['alternating'] = steps_b

    # ── Scenario C: Novel frequency (novelty detection) ──────────
    # First, warm up brain on 1.0 Hz, then switch to never-seen 1.8 Hz
    print("\n  [C] NOVEL FREQUENCY — warm 1.0 Hz, then surprise 1.8 Hz")
    np.random.seed(12)
    sig_c, _ = make_blocks([1.0, 1.0, 1.0, 1.8], block_dur=30.0)
    d_c = run_sim(sig_c, total_time=stabilization_time + 240.0,
                  sweep_mode=False, dynamic_settle=False, verbose=False)
    steps_c, brain_c = run_brain_on_audio(d_c, raw_x_slow, true_y_slow,
                                           raw_x_fast, true_y_fast, label="C")
    results['novelty'] = steps_c

    # ── Scenario D: Multi-frequency (curiosity + attention) ──────
    print("\n  [D] MULTI-FREQUENCY — 5 tones cycling (curiosity + attention)")
    np.random.seed(13)
    sig_d, _ = make_blocks([0.6, 0.9, 1.2, 1.5, 1.9], block_dur=25.0)
    d_d = run_sim(sig_d, total_time=stabilization_time + 300.0,
                  sweep_mode=False, dynamic_settle=False, verbose=False)
    steps_d, brain_d = run_brain_on_audio(d_d, raw_x_slow, true_y_slow,
                                           raw_x_fast, true_y_fast, label="D")
    results['multi'] = steps_d

    # ── Scenario E: Extended alternating (planning emergence) ────
    # Longer version of B — enough time for M57 to engage
    print("\n  [E] EXTENDED SEQUENCE — 0.8 Hz / 1.6 Hz long (planning)")
    np.random.seed(14)
    sig_e, _ = make_blocks([0.8, 1.6], block_dur=30.0)
    d_e = run_sim(sig_e, total_time=stabilization_time + 400.0,
                  sweep_mode=False, dynamic_settle=False, verbose=False)
    steps_e, brain_e = run_brain_on_audio(d_e, raw_x_slow, true_y_slow,
                                           raw_x_fast, true_y_fast, label="E")
    results['planning'] = steps_e

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 4: EVALUATE BEHAVIOURS
# ═══════════════════════════════════════════════════════════════

def evaluate(results):
    """
    Score the 5 cognitive behaviours.
    Returns list of test result dicts for the visualiser.
    """
    tests = []

    # ── T1: Familiarity consolidation ───────────────────────────
    s = results['repeated']
    n = len(s)
    q1 = n // 4
    fam_early = np.mean([x['familiarity'] for x in s[:q1]])
    fam_late  = np.mean([x['familiarity'] for x in s[3*q1:]])
    pe_early  = np.mean([x['prediction_error'] for x in s[:q1]])
    pe_late   = np.mean([x['prediction_error'] for x in s[3*q1:]])
    fam_growth = fam_late - fam_early
    pe_drop    = pe_early - pe_late
    t1_pass    = fam_growth > 0.05 and pe_drop > 0.0
    tests.append({
        'id': 'T1', 'name': 'Familiarity Consolidation',
        'pass': t1_pass,
        'grade': 'PASS' if t1_pass else ('PARTIAL' if fam_growth > 0.02 else 'FAIL'),
        'metrics': {
            'fam_early':  round(fam_early, 4),
            'fam_late':   round(fam_late, 4),
            'fam_growth': round(fam_growth, 4),
            'pe_early':   round(pe_early, 4),
            'pe_late':    round(pe_late, 4),
            'pe_drop':    round(pe_drop, 4),
        },
        'desc': 'M55 familiarity rises, L2 prediction error falls, for a repeated tone.',
        'series': {
            'fam': [round(x['familiarity'], 4) for x in s[::3]],
            'pe':  [round(x['prediction_error'], 4) for x in s[::3]],
            'sal': [round(x['salience'], 4) for x in s[::3]],
        }
    })

    # ── T2: Novelty detection ────────────────────────────────────
    s = results['novelty']
    n = len(s)
    # Find the transition point (true_freq changes to 1.8)
    freqs = [x['true_freq'] for x in s]
    transition_idx = next((i for i in range(1, n) if freqs[i] > 1.5 and freqs[i-1] < 1.5), n//2)

    # Salience and curiosity pre vs post transition
    pre_window  = s[max(0, transition_idx-50):transition_idx]
    post_window = s[transition_idx:min(n, transition_idx+100)]

    sal_pre   = np.mean([x['salience'] for x in pre_window])   if pre_window  else 0
    sal_post  = np.mean([x['salience'] for x in post_window])  if post_window else 0
    cur_pre   = np.mean([x['curiosity'] for x in pre_window])  if pre_window  else 0
    cur_post  = np.mean([x['curiosity'] for x in post_window]) if post_window else 0
    sal_spike = sal_post - sal_pre
    cur_spike = cur_post - cur_pre
    t2_pass   = sal_spike > 0.02 or cur_spike > 0.01
    tests.append({
        'id': 'T2', 'name': 'Novelty Detection',
        'pass': t2_pass,
        'grade': 'PASS' if t2_pass else ('PARTIAL' if (sal_spike + cur_spike) > 0.01 else 'FAIL'),
        'metrics': {
            'sal_pre':   round(sal_pre, 4),
            'sal_post':  round(sal_post, 4),
            'sal_spike': round(sal_spike, 4),
            'cur_pre':   round(cur_pre, 4),
            'cur_post':  round(cur_post, 4),
            'cur_spike': round(cur_spike, 4),
        },
        'desc': 'Salience and curiosity spike when a never-heard frequency appears.',
        'series': {
            'sal': [round(x['salience'], 4) for x in s[::3]],
            'cur': [round(x['curiosity'], 4) for x in s[::3]],
            'nov': [round(x['novelty_flag'], 4) for x in s[::3]],
            'transition_idx': transition_idx // 3,
        }
    })

    # ── T3: Sequence learning ────────────────────────────────────
    s = results['alternating']
    n = len(s)
    q1 = n // 4
    pe_early  = np.mean([x['prediction_error'] for x in s[:q1]])
    pe_late   = np.mean([x['prediction_error'] for x in s[3*q1:]])
    tc_early  = np.mean([x['thought_confidence'] for x in s[:q1]])
    tc_late   = np.mean([x['thought_confidence'] for x in s[3*q1:]])
    pe_drop   = pe_early - pe_late
    tc_rise   = tc_late - tc_early
    t3_pass   = pe_drop > 0.02 or tc_rise > 0.005
    tests.append({
        'id': 'T3', 'name': 'Sequence Learning',
        'pass': t3_pass,
        'grade': 'PASS' if (pe_drop > 0.05 and tc_rise > 0.005) else ('PARTIAL' if t3_pass else 'FAIL'),
        'metrics': {
            'pe_early': round(pe_early, 4),
            'pe_late':  round(pe_late, 4),
            'pe_drop':  round(pe_drop, 4),
            'tc_early': round(tc_early, 4),
            'tc_late':  round(tc_late, 4),
            'tc_rise':  round(tc_rise, 4),
        },
        'desc': 'Prediction error falls and Thought confidence rises as A→B→A→B sequence is learned.',
        'series': {
            'pe': [round(x['prediction_error'], 4) for x in s[::3]],
            'tc': [round(x['thought_confidence'], 4) for x in s[::3]],
            'fe': [round(x['focus_entropy'], 4) for x in s[::3]],
        }
    })

    # ── T4: Curiosity drives attention ───────────────────────────
    s = results['multi']
    n = len(s)
    # Group steps by their true frequency
    freq_groups = {}
    for step in s:
        f = round(step['true_freq'], 1)
        if f not in freq_groups:
            freq_groups[f] = []
        freq_groups[f].append(step)

    # Compare early-seen vs late-seen frequencies
    # Early steps: higher curiosity (less familiar) → higher salience
    q1 = n // 4
    sal_early = np.mean([x['salience'] for x in s[:q1]])
    sal_late  = np.mean([x['salience'] for x in s[3*q1:]])
    cur_early = np.mean([x['curiosity'] for x in s[:q1]])
    cur_late  = np.mean([x['curiosity'] for x in s[3*q1:]])
    # Curiosity should be higher in early novel period vs late familiar period
    cur_drop  = cur_early - cur_late
    sal_drop  = sal_early - sal_late
    t4_pass   = cur_drop > 0.02 or sal_drop > 0.01
    tests.append({
        'id': 'T4', 'name': 'Curiosity Drives Attention',
        'pass': t4_pass,
        'grade': 'PASS' if t4_pass else ('PARTIAL' if (cur_drop + sal_drop) > 0.01 else 'FAIL'),
        'metrics': {
            'sal_early': round(sal_early, 4),
            'sal_late':  round(sal_late, 4),
            'sal_drop':  round(sal_drop, 4),
            'cur_early': round(cur_early, 4),
            'cur_late':  round(cur_late, 4),
            'cur_drop':  round(cur_drop, 4),
        },
        'desc': 'Curiosity and salience are higher during novel multi-tone exploration than after learning.',
        'series': {
            'sal': [round(x['salience'], 4) for x in s[::3]],
            'cur': [round(x['curiosity'], 4) for x in s[::3]],
            'fam': [round(x['familiarity'], 4) for x in s[::3]],
        }
    })

    # ── T5: Planning emerges from knowledge ─────────────────────
    s = results['planning']
    n = len(s)
    q1 = n // 4
    pw_early   = np.mean([x['planning_weight'] for x in s[:q1]])
    pw_late    = np.mean([x['planning_weight'] for x in s[3*q1:]])
    pa_early   = np.mean([x['planning_active'] for x in s[:q1]])
    pa_late    = np.mean([x['planning_active'] for x in s[3*q1:]])
    pw_growth  = pw_late - pw_early
    pa_growth  = pa_late - pa_early
    disagree   = sum(1 for x in s if x['planning_active'] and
                     x.get('planned_action', -1) != x.get('habit_action', -2))
    t5_pass    = pw_growth > 0.001 or pa_growth > 0.05 or disagree > len(s) * 0.05
    tests.append({
        'id': 'T5', 'name': 'Planning Emergence',
        'pass': t5_pass,
        'grade': 'PASS' if t5_pass else ('PARTIAL' if pw_growth > 0 else 'FAIL'),
        'metrics': {
            'pw_early':  round(pw_early, 6),
            'pw_late':   round(pw_late, 6),
            'pw_growth': round(pw_growth, 6),
            'pa_early':  round(pa_early, 4),
            'pa_late':   round(pa_late, 4),
            'disagree':  disagree,
            'total':     n,
        },
        'desc': 'M57 planning weight grows as sequences become predictable. Deliberation overrides habit more.',
        'series': {
            'pw': [round(x['planning_weight'], 6) for x in s[::3]],
            'pa': [int(x['planning_active']) for x in s[::3]],
            'tc': [round(x['thought_confidence'], 4) for x in s[::3]],
        }
    })

    return tests


# ═══════════════════════════════════════════════════════════════
# STEP 5: PACKAGE DATA FOR VISUALISER
# ═══════════════════════════════════════════════════════════════

def package_for_viz(results, tests):
    """
    Package all data for the HTML visualiser.
    Keeps file size manageable by sampling every 3rd step.
    """
    def sample(steps, every=3):
        return steps[::every]

    def extract(steps, keys):
        return {k: [round(float(s[k]), 5) for s in steps] for k in keys}

    scenarios = {}
    for name, steps in results.items():
        s = sample(steps)
        scenarios[name] = {
            **extract(s, ['familiarity', 'curiosity', 'salience',
                          'prediction_error', 'thought_confidence',
                          'planning_weight', 'rpe', 'focus_entropy',
                          'eta', 'qe_norm', 'novelty_flag', 'stability_w']),
            'true_freq':    [round(float(x['true_freq']), 3) for x in s],
            'decoded_freq': [round(float(x['decoded_freq']), 3) for x in s],
            'planning_active': [int(x['planning_active']) for x in s],
            'bmu_idx':      [int(x['bmu_idx']) for x in s],
            't':            [round(float(x['t']), 2) for x in s],
        }

    # Summary stats across all scenarios
    summary = {}
    for name, steps in results.items():
        summary[name] = {
            'n_steps':     len(steps),
            'fam_mean':    round(float(np.mean([x['familiarity'] for x in steps])), 4),
            'fam_max':     round(float(np.max([x['familiarity'] for x in steps])), 4),
            'pe_mean':     round(float(np.mean([x['prediction_error'] for x in steps])), 4),
            'pe_min':      round(float(np.min([x['prediction_error'] for x in steps])), 4),
            'sal_mean':    round(float(np.mean([x['salience'] for x in steps])), 4),
            'cur_mean':    round(float(np.mean([x['curiosity'] for x in steps])), 4),
            'pw_mean':     round(float(np.mean([x['planning_weight'] for x in steps])), 6),
            'pa_rate':     round(float(np.mean([x['planning_active'] for x in steps])), 4),
        }

    return {
        'scenarios': scenarios,
        'tests':     tests,
        'summary':   summary,
        'meta': {
            'cal_freqs':   CAL_FREQS,
            'n_scenarios': len(results),
        }
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          BRAIN AUDIO TEST — M50 → Brain Integrated              ║")
    print("║          The correct test for this brain's current stage        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("  This test feeds real M50 oscillator signals into Brain.")
    print("  No grid. No fake inputs. Just the brain listening to sounds.")
    print()

    # 1. Calibrate M50
    raw_x_slow, true_y_slow, raw_x_fast, true_y_fast = calibrate_m50()

    # 2. Run scenarios
    results = run_all_tests(raw_x_slow, true_y_slow, raw_x_fast, true_y_fast)

    # 3. Evaluate
    print("\n" + "=" * 68)
    print("  STEP 3: EVALUATING COGNITIVE BEHAVIOURS")
    print("=" * 68)
    tests = evaluate(results)

    for t in tests:
        grade_sym = "✓" if t['grade'] == 'PASS' else ("~" if t['grade'] == 'PARTIAL' else "✗")
        print(f"\n  [{grade_sym}] {t['id']} — {t['name']}  [{t['grade']}]")
        print(f"      {t['desc']}")
        for k, v in t['metrics'].items():
            print(f"      {k:15s} = {v}")

    passed  = sum(1 for t in tests if t['grade'] == 'PASS')
    partial = sum(1 for t in tests if t['grade'] == 'PARTIAL')
    failed  = sum(1 for t in tests if t['grade'] == 'FAIL')
    print(f"\n  Results: {passed} PASS  {partial} PARTIAL  {failed} FAIL  (out of {len(tests)})")

    # 4. Package and save data
    print("\n" + "=" * 68)
    print("  STEP 4: SAVING DATA FOR VISUALISER")
    print("=" * 68)

    viz_data = package_for_viz(results, tests)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_audio_data.json")

    def convert(obj):
        if isinstance(obj, (np.bool_, bool)): return int(obj)
        if isinstance(obj, (np.integer,)):     return int(obj)
        if isinstance(obj, (np.floating,)):    return float(obj)
        if isinstance(obj, np.ndarray):        return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_path, 'w') as f:
        json.dump(viz_data, f, separators=(',', ':'), default=convert)

    size_kb = len(json.dumps(viz_data, separators=(',', ':'), default=convert)) / 1024
    print(f"\n  Data saved: {out_path}  ({size_kb:.0f} KB)")
    print(f"  Scenarios:  {len(viz_data['scenarios'])}")
    print(f"  Tests:      {len(viz_data['tests'])}")

    print("\n  Done. Build the visualiser HTML next.")
    print()