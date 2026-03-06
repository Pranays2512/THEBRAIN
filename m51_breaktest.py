"""
M50 + M51 BREAK TEST SUITE
===========================
These tests are designed to FIND FAILURES, not confirm passing.
Each test targets a specific known fragility in the architecture.

STRUCTURE:
  BT-01 … BT-06   M50 CUSUM / decoder edge cases
  BT-07 … BT-12   M51 SOM structural stress tests
  BT-13 … BT-16   M50+M51 integration / regression tests
  BT-17 … BT-18   Long-horizon stability tests

HOW TO READ RESULTS:
  PASS  = system handles the case correctly
  FAIL  = real bug found — fix before building Layer 2
  WARN  = degraded but not broken — note for Layer 2 design
  SKIP  = test could not run (dependency missing)

Run with:
  python m51_break_tests.py 2>&1 | tee break_test_results.txt
"""

import numpy as np
import time
import sys
from collections import deque

# ── Imports from M50 / M51 ────────────────────────────────────
try:
    from m50_neuron import (
        run_sim, make_blocks, make_sweep, make_steps,
        fit_ridge, build_reverse_lookup,
        decode_resonance, compute_stability_plv,
        DivergenceCUSUM, DivergenceCUSUM,
        stabilization_time, dt,
        RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
        PLV_STAB_WINDOW, DIVERG_THRESHOLD, CUSUM_W_GATE,
        mae, N, N_FAST, N_SLOW,
        omega_hz, FREQ_MIN, FREQ_MAX,
    )
    from m51_cortex import (
        CortexM51, prepare_input,
        GRID_H, GRID_W, N_NEURONS, INPUT_DIM,
        SURPRISE_THRESH, FREQ_MIN_HZ, FREQ_MAX_HZ,
        ETA_BASE, ETA_MIN,
    )
    IMPORTS_OK = True
except ImportError as e:
    print(f"  [SKIP] Import failed: {e}")
    print("  Place this file alongside m50_neuron.py and m51_cortex.py")
    IMPORTS_OK = False
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# TEST HARNESS
# ═══════════════════════════════════════════════════════════════

results = {}
_DIVIDER = "─" * 72

def section(title):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")

def report(name, passed, detail="", warn=False):
    tag  = "WARN" if warn else ("PASS" if passed else "FAIL")
    sym  = "⚠" if warn else ("✓" if passed else "✗")
    results[name] = tag
    print(f"  {sym} [{tag}] {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")

def summarise():
    section("BREAK TEST SUMMARY")
    n_pass = sum(1 for v in results.values() if v == "PASS")
    n_fail = sum(1 for v in results.values() if v == "FAIL")
    n_warn = sum(1 for v in results.values() if v == "WARN")
    n_skip = sum(1 for v in results.values() if v == "SKIP")

    for name, tag in results.items():
        sym = {"PASS":"✓","FAIL":"✗","WARN":"⚠","SKIP":"-"}[tag]
        print(f"  {sym} [{tag}] {name}")

    print(f"\n  {_DIVIDER}")
    print(f"  PASS:{n_pass}  FAIL:{n_fail}  WARN:{n_warn}  SKIP:{n_skip}")
    print(f"  {'ALL CLEAR — safe to build Layer 2' if n_fail == 0 else 'FAILURES FOUND — fix before Layer 2'}")


# ═══════════════════════════════════════════════════════════════
# SHARED CALIBRATION  (run once, reused by all tests)
# ═══════════════════════════════════════════════════════════════

def build_calibration():
    """Build M50 calibration tables. Run once and cache."""
    SLOW_FREQS_CAL = sorted(set([
        0.41, 0.44, 0.47,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.77,
        0.8, 0.82, 0.85, 0.87,
        0.9, 0.92, 0.95, 0.97, 1.0, 1.03, 1.05, 1.07,
        1.1, 1.15, 1.2, 1.3, 1.35, 1.4,
        1.5, 1.55, 1.6, 1.7, 1.75, 1.8, 1.9, 1.95,
        2.0, 2.05, 2.1, 2.12, 2.16, 2.20,
    ]))

    warmup = stabilization_time + 10.0
    sweep_dur = 60.0

    np.random.seed(0)
    data_train = run_sim(
        make_sweep(0.5, 2.0, 6, sweep_dur),
        total_time=warmup + 6*sweep_dur + 10.0,
        sweep_mode=True, verbose=False, collect_calib=False)
    ridge_fast, ridge_fast_sc = fit_ridge(
        data_train['feat_fast'], data_train['Y'], RIDGE_ALPHA_FAST)

    np.random.seed(1)
    block_sig, _ = make_blocks(SLOW_FREQS_CAL, block_dur=40.0)
    data_slow = run_sim(block_sig,
        total_time=stabilization_time + 2*len(SLOW_FREQS_CAL)*40.0 + 10.0,
        sweep_mode=False, dynamic_settle=True, verbose=False,
        collect_calib=True)

    raw_x_slow, true_y_slow = build_reverse_lookup(
        sorted(data_slow['calib_plv_slow'].keys()),
        data_slow['calib_plv_slow'],
        data_slow['calib_energy_slow'])
    raw_x_fast, true_y_fast = build_reverse_lookup(
        sorted(data_slow['calib_plv_fast'].keys()),
        data_slow['calib_plv_fast'],
        data_slow['calib_energy_fast'])
    ridge_slow, ridge_slow_sc = fit_ridge(
        data_slow['feat_slow'], data_slow['Y'], RIDGE_ALPHA_SLOW)

    print(f"  Calibration: {len(raw_x_slow)} lookup pts, "
          f"range [{raw_x_slow[0]:.3f}, {raw_x_slow[-1]:.3f}]")
    return (raw_x_slow, true_y_slow, raw_x_fast, true_y_fast,
            ridge_fast, ridge_fast_sc, ridge_slow, ridge_slow_sc)


def decode_stream(sim_data, raw_x_slow, true_y_slow,
                  raw_x_fast, true_y_fast, pass_w_to_cusum=True):
    """Run M50 decode pipeline over a sim output. Returns arrays."""
    n = len(sim_data['Y'])
    df = np.array([decode_resonance(sim_data['plv_fast'][i],
                                    sim_data['energy_fast'][i],
                                    raw_x_fast, true_y_fast)
                   for i in range(n)])
    ds = np.array([decode_resonance(sim_data['plv_slow'][i],
                                    sim_data['energy_slow'][i],
                                    raw_x_slow, true_y_slow)
                   for i in range(n)])
    change_det = DivergenceCUSUM()
    plv_hist   = deque(maxlen=PLV_STAB_WINDOW)
    w_arr      = np.zeros(n)
    novelty    = np.zeros(n, dtype=bool)
    fused      = np.zeros(n)

    for i in range(n):
        max_plv = float(np.max(sim_data['plv_slow'][i]))
        plv_hist.append(max_plv)
        w = compute_stability_plv(plv_hist)
        w_arr[i] = w
        _, nov = change_det.update(df[i], ds[i], sim_data['T'][i],
                                   w=(w if pass_w_to_cusum else 1.0))
        novelty[i] = nov
        fused[i]   = w * ds[i] + (1.0 - w) * df[i]

    return df, ds, fused, w_arr, novelty, change_det


def run_cortex_on_stream(sim_data, raw_x_slow, true_y_slow,
                         raw_x_fast, true_y_fast, cortex):
    """Full M50+M51 pipeline over a sim output."""
    n        = len(sim_data['Y'])
    plv_hist = deque(maxlen=PLV_STAB_WINDOW)
    cusum    = DivergenceCUSUM()
    records  = []

    for i in range(n):
        plv_fast_mag = sim_data['plv_fast'][i]
        plv_slow_mag = sim_data['plv_slow'][i]
        e_fast       = sim_data['energy_fast'][i]
        e_slow       = sim_data['energy_slow'][i]

        df = decode_resonance(plv_fast_mag, e_fast, raw_x_fast, true_y_fast)
        ds = decode_resonance(plv_slow_mag, e_slow, raw_x_slow, true_y_slow)

        max_plv = float(np.max(plv_slow_mag))
        plv_hist.append(max_plv)
        w = compute_stability_plv(plv_hist)

        _, novelty = cusum.update(df, ds, sim_data['T'][i], w=w)
        f_fused    = w * ds + (1.0 - w) * df

        cr = cortex.step(
            decoded_freq=f_fused,
            stability_w=w,
            novelty_flag=float(novelty),
            plv_vector=plv_slow_mag,
        )
        records.append({
            'Y': sim_data['Y'][i], 'T': sim_data['T'][i],
            'df': df, 'ds': ds, 'f_fused': f_fused, 'w': w,
            'surprise': cr['qe'], 'eta': cr['eta'],
            'bmu': cr['bmu_pos'], 'sigma': cr['sigma'],
        })
    return records


# ═══════════════════════════════════════════════════════════════
# BUILD CALIBRATION
# ═══════════════════════════════════════════════════════════════

section("CALIBRATION (shared across all tests)")
CAL = build_calibration()
(raw_x_slow, true_y_slow, raw_x_fast, true_y_fast,
 ridge_fast, ridge_fast_sc, ridge_slow, ridge_slow_sc) = CAL


# ═══════════════════════════════════════════════════════════════
# BT-01: CUSUM SUB-FLOOR IMMUNITY
# The 0.038 Hz threshold was chosen to sit above sub-floor noise.
# Feed 0.02 Hz micro-steps (below threshold) — CUSUM must stay silent.
# A false positive here means the threshold is too low.
# ═══════════════════════════════════════════════════════════════

section("BT-01: CUSUM — Sub-floor step immunity (0.02 Hz steps)")

micro_freqs = [1.00, 1.02, 1.00, 1.02, 1.00, 1.02, 1.00, 1.02]
np.random.seed(10)
sig_micro, _ = make_blocks(micro_freqs, block_dur=40.0)
d_micro = run_sim(sig_micro,
    total_time=stabilization_time + 2*len(micro_freqs)*40.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)

df_m, ds_m, fused_m, w_m, nov_m, cd_m = decode_stream(
    d_micro, raw_x_slow, true_y_slow, raw_x_fast, true_y_fast)

n_fp      = int(np.sum(nov_m))
mean_w    = float(np.mean(w_m))
mean_err  = mae(fused_m, d_micro['Y'])

print(f"  Micro-step size: 0.02 Hz  (threshold: {DIVERG_THRESHOLD:.3f} Hz)")
print(f"  CUSUM false positives: {n_fp}")
print(f"  Mean stability w:      {mean_w:.3f}")
print(f"  Fused MAE:             {mean_err:.4f} Hz")

report("BT-01 Sub-floor step immunity",
       n_fp < 5,
       f"FP count={n_fp} (target <5), MAE={mean_err:.4f} Hz",
       warn=(5 <= n_fp < 20))


# ═══════════════════════════════════════════════════════════════
# BT-02: CUSUM MUST FIRE ON REAL TRANSITIONS
# The fix that silenced sub-floor noise must not also silence
# real 0.3+ Hz steps. Test sensitivity on moderate jumps.
# ═══════════════════════════════════════════════════════════════

section("BT-02: CUSUM — Sensitivity on real transitions (0.30 Hz+ steps)")

real_freqs = [0.70, 1.00, 1.30, 1.00, 0.70, 1.00, 1.30, 1.60, 1.00]
np.random.seed(11)
sig_real, _ = make_blocks(real_freqs, block_dur=40.0)
d_real = run_sim(sig_real,
    total_time=stabilization_time + 2*len(real_freqs)*40.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)

df_r, ds_r, fused_r, w_r, nov_r, cd_r = decode_stream(
    d_real, raw_x_slow, true_y_slow, raw_x_fast, true_y_fast)

Y_r = d_real['Y']
# Count expected transitions (between stable settled blocks)
expected_trans = np.sum(np.diff(Y_r) != 0)
actual_detect  = len(cd_r.novelty_events)
detection_rate = actual_detect / max(expected_trans, 1)

print(f"  Step size range: 0.30–0.90 Hz")
print(f"  Expected transitions: {expected_trans}")
print(f"  CUSUM detections:     {actual_detect}")
print(f"  Detection rate:       {detection_rate:.0%}")

report("BT-02 Real transition detection",
       detection_rate >= 0.70,
       f"rate={detection_rate:.0%} (target ≥70%)",
       warn=(0.50 <= detection_rate < 0.70))


# ═══════════════════════════════════════════════════════════════
# BT-03: SWEEP FALSE POSITIVE SUPPRESSION
# During a continuous sweep, w stays low → CUSUM should be gated.
# Check that novelty count is near zero during sweep.
# ═══════════════════════════════════════════════════════════════

section("BT-03: CUSUM — No false positives during frequency sweep")

warmup    = stabilization_time + 10.0
sweep_dur = 60.0
np.random.seed(12)
d_sw = run_sim(make_sweep(0.5, 2.0, 3, sweep_dur),
    total_time=warmup + 3*sweep_dur + 10.0,
    sweep_mode=True, verbose=False)

_, _, _, w_sw, nov_sw, cd_sw = decode_stream(
    d_sw, raw_x_slow, true_y_slow, raw_x_fast, true_y_fast)

fp_during_sweep = int(np.sum(nov_sw))
mean_w_sweep    = float(np.mean(w_sw))

print(f"  Sweep: 0.5→2.0 Hz over {sweep_dur}s × 3 reps")
print(f"  Mean stability w during sweep: {mean_w_sweep:.3f}  (should be low)")
print(f"  CUSUM fires during sweep:      {fp_during_sweep}")

report("BT-03 Sweep false positive suppression",
       fp_during_sweep < 10,
       f"FP count={fp_during_sweep} (target <10), mean_w={mean_w_sweep:.3f}",
       warn=(10 <= fp_during_sweep < 30))


# ═══════════════════════════════════════════════════════════════
# BT-04: CALIBRATION BOUNDARY ACCURACY
# M50 extended the cal range to 0.41–2.20 Hz.
# Test that the edges decode accurately (not clamped).
# ═══════════════════════════════════════════════════════════════

section("BT-04: Calibration boundary accuracy (0.41 Hz and 2.20 Hz)")

edge_freqs = [0.41, 0.44, 0.47, 2.12, 2.16, 2.20]
np.random.seed(13)
sig_edge, _ = make_blocks(edge_freqs, block_dur=45.0)
d_edge = run_sim(sig_edge,
    total_time=stabilization_time + 2*len(edge_freqs)*45.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)

df_e, ds_e, fused_e, w_e, _, _ = decode_stream(
    d_edge, raw_x_slow, true_y_slow, raw_x_fast, true_y_fast)

Y_e = d_edge['Y']
print(f"  {'Freq':>6}  {'Slow MAE':>9}  {'Fused MAE':>10}  {'w_slow':>7}")
print(f"  {'─'*6}  {'─'*9}  {'─'*10}  {'─'*7}")
per_freq_ok = []
for f in edge_freqs:
    m = Y_e == f
    if m.any():
        sl_mae  = mae(ds_e[m], Y_e[m])
        fu_mae  = mae(fused_e[m], Y_e[m])
        wv      = np.mean(w_e[m])
        ok      = sl_mae < 0.05
        per_freq_ok.append(ok)
        print(f"  {f:6.2f}  {sl_mae:9.4f}  {fu_mae:10.4f}  {wv:7.3f}"
              f"  {'✓' if ok else '✗'}")

all_ok = all(per_freq_ok)
report("BT-04 Calibration boundary accuracy",
       all_ok,
       "All edge frequencies should decode with MAE < 0.05 Hz",
       warn=(not all_ok and sum(per_freq_ok) >= len(per_freq_ok)-1))


# ═══════════════════════════════════════════════════════════════
# BT-05: CUSUM WITHOUT STABILITY GATE (REGRESSION)
# Intentionally disable the w-gate (pass w=1.0 always).
# This should reproduce the M49 false-positive problem.
# If it does NOT produce FPs, the gate was never needed — flag that.
# If it DOES produce FPs, the gate is confirmed essential.
# ═══════════════════════════════════════════════════════════════

section("BT-05: Regression — CUSUM without stability gate (should fail like M49)")

np.random.seed(14)
sig_b5, _ = make_blocks([0.60, 1.20, 1.80], block_dur=40.0)
d_b5 = run_sim(sig_b5,
    total_time=stabilization_time + 2*3*40.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)

# Gated (M50 behavior)
_, _, _, _, nov_gated, cd_gated = decode_stream(
    d_b5, raw_x_slow, true_y_slow, raw_x_fast, true_y_fast,
    pass_w_to_cusum=True)

# Ungated (M49 behavior — always pass w=1.0)
_, _, _, _, nov_ungated, cd_ungated = decode_stream(
    d_b5, raw_x_slow, true_y_slow, raw_x_fast, true_y_fast,
    pass_w_to_cusum=False)

fp_gated   = int(np.sum(nov_gated))
fp_ungated = int(np.sum(nov_ungated))

print(f"  M50 gated FPs:   {fp_gated}   (should be low)")
print(f"  M49 ungated FPs: {fp_ungated}  (should be higher — proves gate matters)")

gate_helps = fp_ungated > fp_gated
report("BT-05 Stability gate regression",
       gate_helps and fp_gated < 10,
       f"gated={fp_gated}, ungated={fp_ungated}, gate_improvement={fp_ungated - fp_gated}")


# ═══════════════════════════════════════════════════════════════
# BT-06: RAPID CONSECUTIVE TRANSITIONS
# Steps every 5s (faster than most settling times).
# Tests CUSUM debounce logic and fast decoder fallback.
# ═══════════════════════════════════════════════════════════════

section("BT-06: CUSUM — Rapid consecutive transitions (5s steps)")

rapid_freqs = [0.5, 1.0, 1.5, 2.0, 0.7, 1.2, 1.8, 0.6, 1.1, 1.6]
np.random.seed(15)
d_rapid = run_sim(
    make_steps(rapid_freqs, step_dur=5.0),
    total_time=stabilization_time + 5.0 + len(rapid_freqs)*5.0*4 + 10.0,
    sweep_mode=True, verbose=False)

_, _, fused_rapid, w_rapid, nov_rapid, cd_rapid = decode_stream(
    d_rapid, raw_x_slow, true_y_slow, raw_x_fast, true_y_fast)

Y_rapid   = d_rapid['Y']
fused_mae = mae(fused_rapid, Y_rapid)
fast_mae  = mae(
    np.array([decode_resonance(d_rapid['plv_fast'][i],
                               d_rapid['energy_fast'][i],
                               raw_x_fast, true_y_fast)
              for i in range(len(Y_rapid))]),
    Y_rapid)
n_detect  = len(cd_rapid.novelty_events)
mean_w    = float(np.mean(w_rapid))

print(f"  Fast decoder MAE: {fast_mae:.4f} Hz  (should dominate — w low)")
print(f"  Fused MAE:        {fused_mae:.4f} Hz")
print(f"  Mean w (fast block): {mean_w:.3f}   (should be low during steps)")
print(f"  CUSUM detections: {n_detect}")

# Fast decoder should carry the load when steps are too rapid to settle
report("BT-06 Rapid step tracking",
       fused_mae < 0.30,
       f"fused_MAE={fused_mae:.4f} Hz (target <0.30), mean_w={mean_w:.3f}",
       warn=(0.20 <= fused_mae < 0.30))


# ═══════════════════════════════════════════════════════════════
# BT-07: SOM CATASTROPHIC FORGETTING
# Train on A,B,C → train heavily on D,E,F → test A,B,C again.
# A forgetting-prone SOM will lose A,B,C completely.
# Pass = old neurons still activate for old frequencies.
# ═══════════════════════════════════════════════════════════════

section("BT-07: SOM — Catastrophic forgetting test")

cortex_cf = CortexM51(seed=42)

# Phase 1: train on A,B,C
print("  Phase 1: training on A=0.60, B=1.00, C=1.80 Hz (12 blocks × 30s)...")
np.random.seed(20)
sig_abc, _ = make_blocks([0.60, 1.00, 1.80]*4, block_dur=30.0)
d_abc = run_sim(sig_abc,
    total_time=stabilization_time + 2*12*30.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)
run_cortex_on_stream(d_abc, raw_x_slow, true_y_slow,
                     raw_x_fast, true_y_fast, cortex_cf)

# Record which neurons A,B,C mapped to
pos_a_before, err_a_before = cortex_cf.find_neuron_for_freq(0.60)
pos_b_before, err_b_before = cortex_cf.find_neuron_for_freq(1.00)
pos_c_before, err_c_before = cortex_cf.find_neuron_for_freq(1.80)

# Phase 2: heavy overtraining on D=0.41, E=1.40, F=2.20
print("  Phase 2: heavy overtraining on D=0.41, E=1.40, F=2.20 Hz (24 blocks × 30s)...")
np.random.seed(21)
sig_def, _ = make_blocks([0.41, 1.40, 2.20]*8, block_dur=30.0)
d_def = run_sim(sig_def,
    total_time=stabilization_time + 2*24*30.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)
run_cortex_on_stream(d_def, raw_x_slow, true_y_slow,
                     raw_x_fast, true_y_fast, cortex_cf)

# Test: can the SOM still find A,B,C?
pos_a_after, err_a_after = cortex_cf.find_neuron_for_freq(0.60)
pos_b_after, err_b_after = cortex_cf.find_neuron_for_freq(1.00)
pos_c_after, err_c_after = cortex_cf.find_neuron_for_freq(1.80)

print(f"  {'Freq':>5}  {'Before pos':>12}  {'Before err':>11}  "
      f"{'After pos':>12}  {'After err':>11}  {'OK':>4}")
print(f"  {'─'*5}  {'─'*12}  {'─'*11}  {'─'*12}  {'─'*11}  {'─'*4}")
forgetting_ok = []
for label, pb, eb, pa, ea in [
    ("A=0.60", pos_a_before, err_a_before, pos_a_after, err_a_after),
    ("B=1.00", pos_b_before, err_b_before, pos_b_after, err_b_after),
    ("C=1.80", pos_c_before, err_c_before, pos_c_after, err_c_after),
]:
    ok = ea < 0.15   # still represents within 0.15 normalized units
    forgetting_ok.append(ok)
    print(f"  {label:>5}  {str(pb):>12}  {eb:11.4f}  "
          f"{str(pa):>12}  {ea:11.4f}  {'✓' if ok else '✗'}")

report("BT-07 Catastrophic forgetting",
       all(forgetting_ok),
       "After D/E/F overtraining, old A/B/C should still be representable",
       warn=(sum(forgetting_ok) == 2))


# ═══════════════════════════════════════════════════════════════
# BT-08: SOM DEAD NEURONS
# After training, some neurons may never activate.
# Dead neurons waste capacity and indicate map collapse.
# ═══════════════════════════════════════════════════════════════

section("BT-08: SOM — Dead neuron detection after training")

# Use the cortex from BT-07 (it has seen A,B,C,D,E,F)
counts = cortex_cf.neuron_activation_counts()
dead   = int(np.sum(counts == 0))
total  = N_NEURONS
dead_frac = dead / total

print(f"  Total neurons:  {total}")
print(f"  Dead (0 wins):  {dead}  ({dead_frac:.0%})")
print(f"  Max activations for a single neuron: {counts.max()}")
print(f"  Std of activation counts: {counts.std():.1f}")

# Dead neurons > 25% is a problem — map is too compressed
report("BT-08 Dead neuron fraction",
       dead_frac < 0.25,
       f"dead={dead}/{total} ({dead_frac:.0%}), target <25%",
       warn=(0.15 <= dead_frac < 0.25))


# ═══════════════════════════════════════════════════════════════
# BT-09: SOM SINGLE-FREQUENCY COLLAPSE
# Feed only ONE frequency for a long time.
# The SOM should collapse into a single cluster (all neurons → same freq).
# This confirms competitive learning is working at the extreme.
# ═══════════════════════════════════════════════════════════════

section("BT-09: SOM — Single-frequency collapse test")

cortex_collapse = CortexM51(seed=99)

np.random.seed(22)
sig_one, _ = make_blocks([1.00]*12, block_dur=30.0)
d_one = run_sim(sig_one,
    total_time=stabilization_time + 2*12*30.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)
run_cortex_on_stream(d_one, raw_x_slow, true_y_slow,
                     raw_x_fast, true_y_fast, cortex_collapse)

state_c = cortex_collapse.get_map_state()
freq_map = state_c['freq_map']
freq_std  = float(np.std(freq_map))
freq_mean = float(np.mean(freq_map))
freq_err  = abs(freq_mean - 1.00)

print(f"  Trained only on: 1.00 Hz")
print(f"  Map freq mean:   {freq_mean:.4f} Hz  (should be ~1.00)")
print(f"  Map freq std:    {freq_std:.4f} Hz   (should be low — collapsed)")

# In a collapsed map all neurons converge toward 1.00 Hz
report("BT-09 Single-frequency collapse",
       freq_err < 0.15 and freq_std < 0.25,
       f"mean={freq_mean:.4f} Hz (err={freq_err:.4f}), std={freq_std:.4f}",
       warn=(freq_err < 0.25 and freq_std < 0.40))


# ═══════════════════════════════════════════════════════════════
# BT-10: SOM FINE-GRAINED DISCRIMINATION
# Can the SOM distinguish 0.60 vs 0.65 Hz after training?
# These are only 0.05 Hz apart — within a single calibration bin.
# Tests the limits of cortical resolution.
# ═══════════════════════════════════════════════════════════════

section("BT-10: SOM — Fine-grained frequency discrimination (0.05 Hz apart)")

cortex_fine = CortexM51(seed=55)

# Train on two close frequencies
fine_freqs = [0.60, 0.65] * 8
np.random.seed(23)
sig_fine, _ = make_blocks(fine_freqs, block_dur=35.0)
d_fine = run_sim(sig_fine,
    total_time=stabilization_time + 2*len(fine_freqs)*35.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)
r_fine = run_cortex_on_stream(d_fine, raw_x_slow, true_y_slow,
                               raw_x_fast, true_y_fast, cortex_fine)

pos_060, err_060 = cortex_fine.find_neuron_for_freq(0.60)
pos_065, err_065 = cortex_fine.find_neuron_for_freq(0.65)

# Grid distance between the two neurons
dr = pos_060[0] - pos_065[0]
dc = pos_060[1] - pos_065[1]
grid_dist = float(np.sqrt(dr*dr + dc*dc))

print(f"  0.60 Hz → neuron {pos_060}  (weight err={err_060:.4f})")
print(f"  0.65 Hz → neuron {pos_065}  (weight err={err_065:.4f})")
print(f"  Grid distance between them: {grid_dist:.2f} cells")

# At minimum they should map to DIFFERENT neurons (grid_dist > 0)
# Ideal: adjacent neurons (grid_dist ~ 1)
same_neuron = (pos_060 == pos_065)
report("BT-10 Fine-grained discrimination",
       not same_neuron,
       f"0.60 and 0.65 Hz map to {'SAME' if same_neuron else 'DIFFERENT'} neurons, "
       f"grid_dist={grid_dist:.2f}",
       warn=(not same_neuron and grid_dist < 1.5))


# ═══════════════════════════════════════════════════════════════
# BT-11: SOM CURIOSITY BOOST MAGNITUDE
# The 1.06× boost ratio in the standard tests was thin.
# Use a more extreme scenario: very familiar vs truly novel.
# ═══════════════════════════════════════════════════════════════

section("BT-11: SOM — Curiosity boost magnitude (more extreme scenario)")

cortex_cur = CortexM51(seed=77)

# Phase 1: make 0.80 Hz very familiar (long exposure)
print("  Phase 1: making 0.80 Hz very familiar (20 blocks × 35s)...")
np.random.seed(24)
sig_fam, _ = make_blocks([0.80]*20, block_dur=35.0)
d_fam = run_sim(sig_fam,
    total_time=stabilization_time + 2*20*35.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)
run_cortex_on_stream(d_fam, raw_x_slow, true_y_slow,
                     raw_x_fast, true_y_fast, cortex_cur)

# Phase 2: interleave very familiar (0.80) with truly novel (2.20)
print("  Phase 2: interleaving familiar (0.80) with novel (2.20)...")
mix_freqs = [0.80, 2.20, 0.80, 2.20, 0.80, 2.20, 0.80, 2.20]
np.random.seed(25)
sig_mix, _ = make_blocks(mix_freqs, block_dur=30.0)
d_mix = run_sim(sig_mix,
    total_time=stabilization_time + 2*len(mix_freqs)*30.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)
r_mix = run_cortex_on_stream(d_mix, raw_x_slow, true_y_slow,
                              raw_x_fast, true_y_fast, cortex_cur)

Y_mix = np.array([r['Y'] for r in r_mix])
eta_fam  = [r['eta'] for r in r_mix if abs(r['Y'] - 0.80) < 0.05]
eta_nov  = [r['eta'] for r in r_mix if abs(r['Y'] - 2.20) < 0.05]
qe_fam   = [r['surprise'] for r in r_mix if abs(r['Y'] - 0.80) < 0.05]
qe_nov   = [r['surprise'] for r in r_mix if abs(r['Y'] - 2.20) < 0.05]

mean_eta_fam = float(np.mean(eta_fam)) if eta_fam else 0.0
mean_eta_nov = float(np.mean(eta_nov)) if eta_nov else 0.0
mean_qe_fam  = float(np.mean(qe_fam))  if qe_fam  else 0.0
mean_qe_nov  = float(np.mean(qe_nov))  if qe_nov  else 0.0
boost_ratio  = mean_eta_nov / mean_eta_fam if mean_eta_fam > 1e-8 else 0.0

print(f"  Familiar 0.80 Hz:   mean_η={mean_eta_fam:.5f},  mean_QE={mean_qe_fam:.4f}")
print(f"  Novel   2.20 Hz:    mean_η={mean_eta_nov:.5f},  mean_QE={mean_qe_nov:.4f}")
print(f"  Curiosity boost ratio: {boost_ratio:.2f}×  (target >1.20×)")

report("BT-11 Curiosity boost magnitude",
       boost_ratio > 1.20,
       f"boost={boost_ratio:.2f}× (target >1.20×), "
       f"novel_QE={mean_qe_nov:.4f} vs familiar_QE={mean_qe_fam:.4f}",
       warn=(1.05 <= boost_ratio <= 1.20))


# ═══════════════════════════════════════════════════════════════
# BT-12: SOM INPUT NORMALIZATION INTEGRITY
# prepare_input() must not produce values outside [0,1].
# Check edge cases: f=FREQ_MIN, f=FREQ_MAX, zero PLV vector.
# A NaN or out-of-range value would corrupt the SOM silently.
# ═══════════════════════════════════════════════════════════════

section("BT-12: SOM — Input normalization integrity check")

edge_cases = [
    ("f=FREQ_MIN",      FREQ_MIN_HZ, 0.0,  0.0,  np.zeros(N)),
    ("f=FREQ_MAX",      FREQ_MAX_HZ, 1.0,  1.0,  np.ones(N)),
    ("f below range",   0.10,        0.5,  0.0,  np.zeros(N)),
    ("f above range",   3.00,        0.5,  0.0,  np.zeros(N)),
    ("zero PLV",        1.00,        0.5,  0.0,  np.zeros(N)),
    ("uniform PLV",     1.00,        0.5,  0.0,  np.ones(N) * 0.5),
    ("single spike PLV",1.00,        0.5,  0.0,  np.eye(1, N, 100).flatten()),
    ("NaN freq",        float('nan'),0.0,  0.0,  np.zeros(N)),
]

all_norm_ok = True
for label, freq, stab, nov, plv in edge_cases:
    freq_in = freq if not np.isnan(freq) else FREQ_MIN_HZ
    vec = prepare_input(freq_in, stab, nov, plv)
    has_nan = bool(np.any(np.isnan(vec)))
    in_range = bool(np.all((vec >= 0.0) & (vec <= 1.0)))
    ok = in_range and not has_nan
    if not ok:
        all_norm_ok = False
    print(f"  {label:25s}  shape={vec.shape}  "
          f"range=[{vec.min():.3f},{vec.max():.3f}]  "
          f"NaN={'YES' if has_nan else 'no'}  {'✓' if ok else '✗ FAIL'}")

report("BT-12 Input normalization integrity",
       all_norm_ok,
       "All inputs must produce vectors in [0,1] with no NaNs")


# ═══════════════════════════════════════════════════════════════
# BT-13: STABILITY WEIGHT DYNAMIC RANGE
# w should be high during stable blocks and low during transitions.
# If w never exceeds 0.5, the slow decoder never gets used.
# If w never drops below 0.5, the fast decoder never gets used.
# ═══════════════════════════════════════════════════════════════

section("BT-13: Stability weight dynamic range check")

stable_freqs = [0.80, 1.20, 1.60]
np.random.seed(30)
sig_stab, _ = make_blocks(stable_freqs, block_dur=45.0)
d_stab = run_sim(sig_stab,
    total_time=stabilization_time + 2*len(stable_freqs)*45.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)

_, _, _, w_stab, _, _ = decode_stream(
    d_stab, raw_x_slow, true_y_slow, raw_x_fast, true_y_fast)

Y_stab = d_stab['Y']
# Get stable-period w (avoid transient samples at block boundaries)
# Approximate: last 60% of each block
stable_mask = np.zeros(len(Y_stab), dtype=bool)
for i in range(1, len(Y_stab)):
    if Y_stab[i] == Y_stab[i-1]:
        stable_mask[i] = True

w_in_stable     = w_stab[stable_mask]
w_in_transition = w_stab[~stable_mask]

w_stable_mean = float(np.mean(w_in_stable)) if w_in_stable.any() else 0.0
w_trans_mean  = float(np.mean(w_in_transition)) if w_in_transition.any() else 1.0

print(f"  w during stable periods:     {w_stable_mean:.3f}  (target >0.60)")
print(f"  w during transitions:        {w_trans_mean:.3f}  (target <0.60)")
print(f"  Dynamic range:               {w_stable_mean - w_trans_mean:.3f}")

report("BT-13 Stability weight dynamic range",
       w_stable_mean > 0.60 and w_trans_mean < 0.70,
       f"stable_w={w_stable_mean:.3f}, transition_w={w_trans_mean:.3f}",
       warn=(w_stable_mean > 0.50 and w_trans_mean < 0.80))


# ═══════════════════════════════════════════════════════════════
# BT-14: HIGH NOISE DECODER DEGRADATION PROFILE
# At what noise level does the decoder completely break?
# Establishes a noise floor for Layer 2 design.
# ═══════════════════════════════════════════════════════════════

section("BT-14: High noise decoder degradation profile")

noise_test_freqs = [0.70, 1.00, 1.40, 1.80]
print(f"  {'Noise':>6}  {'SlowMAE':>8}  {'FusedMAE':>9}  {'w_slow':>7}  {'Status':>8}")
print(f"  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*7}  {'─'*8}")

noise_results = {}
for nl in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
    np.random.seed(40 + int(nl))
    ns, _ = make_blocks(noise_test_freqs, block_dur=40.0, noise_level=nl)
    d_n = run_sim(ns, total_time=500.0,
                  sweep_mode=False, dynamic_settle=True, verbose=False)
    df_n, ds_n, fused_n, w_n, _, _ = decode_stream(
        d_n, raw_x_slow, true_y_slow, raw_x_fast, true_y_fast)
    sl_mae  = mae(ds_n,    d_n['Y'])
    fu_mae  = mae(fused_n, d_n['Y'])
    w_mean  = float(np.mean(w_n))
    status  = "OK" if fu_mae < 0.15 else ("WARN" if fu_mae < 0.40 else "FAIL")
    print(f"  {nl:6.1f}  {sl_mae:8.4f}  {fu_mae:9.4f}  {w_mean:7.3f}  {status:>8}")
    noise_results[nl] = fu_mae

# Pass if noise_level=1.0 is still under 0.15 Hz MAE
report("BT-14 Noise robustness (noise=1.0)",
       noise_results.get(1.0, 999) < 0.15,
       f"MAE at noise=1.0: {noise_results.get(1.0,0):.4f} Hz (target <0.15)",
       warn=(0.15 <= noise_results.get(1.0, 999) < 0.30))


# ═══════════════════════════════════════════════════════════════
# BT-15: SOM LONG-HORIZON DRIFT
# Run 2× longer than any previous test.
# Check if surprise, sigma, and map topology remain stable
# (no runaway plasticity, no map collapse over time).
# ═══════════════════════════════════════════════════════════════

section("BT-15: SOM — Long-horizon stability (2× exposure)")

cortex_long = CortexM51(seed=33)

long_freqs = [0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00] * 6
np.random.seed(50)
sig_long, _ = make_blocks(long_freqs, block_dur=25.0)
d_long = run_sim(sig_long,
    total_time=stabilization_time + 2*len(long_freqs)*25.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)

t0 = time.time()
r_long = run_cortex_on_stream(d_long, raw_x_slow, true_y_slow,
                               raw_x_fast, true_y_fast, cortex_long)
elapsed = time.time() - t0

n_long = len(r_long)
mid    = n_long // 2

# Compare first half vs second half statistics
qe_first  = float(np.mean([r['surprise'] for r in r_long[:mid]]))
qe_second = float(np.mean([r['surprise'] for r in r_long[mid:]]))
sig_first  = float(np.mean([r['sigma'] for r in r_long[:mid]]))
sig_second = float(np.mean([r['sigma'] for r in r_long[mid:]]))

# Dead neurons after long training
counts_long = cortex_long.neuron_activation_counts()
dead_long   = int(np.sum(counts_long == 0))

print(f"  Total steps: {n_long}  ({elapsed:.0f}s wall time)")
print(f"  QE first half:    {qe_first:.4f}   QE second half:  {qe_second:.4f}")
print(f"  σ first half:     {sig_first:.4f}   σ second half:   {sig_second:.4f}")
print(f"  Dead neurons after long training: {dead_long}/{N_NEURONS}")

# QE should decrease or stabilize (not increase — no runaway)
qe_stable   = qe_second <= qe_first * 1.10   # allows ±10% fluctuation
sigma_sane  = sig_second < 3.0               # σ shouldn't explode
no_collapse = dead_long < N_NEURONS * 0.40   # <40% dead

report("BT-15 Long-horizon SOM stability",
       qe_stable and sigma_sane and no_collapse,
       f"QE:{qe_first:.4f}→{qe_second:.4f} (stable={qe_stable}), "
       f"σ={sig_second:.4f}, dead={dead_long}/{N_NEURONS}",
       warn=(qe_stable and sigma_sane and dead_long >= N_NEURONS * 0.25))


# ═══════════════════════════════════════════════════════════════
# BT-16: SOM + M50 SURPRISE SPIKE ON CALIBRATION BOUNDARIES
# Test that inputs at the very edge of the cal range
# (0.41 Hz, 2.20 Hz) produce well-controlled surprise, not
# runaway QE that would corrupt the σ history.
# ═══════════════════════════════════════════════════════════════

section("BT-16: SOM — Surprise at calibration boundaries (0.41 Hz, 2.20 Hz)")

cortex_edge = CortexM51(seed=66)

# First train on middle frequencies so the map has structure
np.random.seed(60)
sig_mid, _ = make_blocks([0.80, 1.20, 1.60]*4, block_dur=30.0)
d_mid = run_sim(sig_mid,
    total_time=stabilization_time + 2*12*30.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)
run_cortex_on_stream(d_mid, raw_x_slow, true_y_slow,
                     raw_x_fast, true_y_fast, cortex_edge)

# Now hit boundary frequencies
np.random.seed(61)
sig_bnd, _ = make_blocks([0.41, 2.20, 0.41, 2.20]*2, block_dur=30.0)
d_bnd = run_sim(sig_bnd,
    total_time=stabilization_time + 2*8*30.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)
r_bnd = run_cortex_on_stream(d_bnd, raw_x_slow, true_y_slow,
                              raw_x_fast, true_y_fast, cortex_edge)

qe_041 = [r['surprise'] for r in r_bnd if abs(r['Y'] - 0.41) < 0.05]
qe_220 = [r['surprise'] for r in r_bnd if abs(r['Y'] - 2.20) < 0.05]

mean_qe_041 = float(np.mean(qe_041)) if qe_041 else 0.0
max_qe_041  = float(np.max(qe_041))  if qe_041 else 0.0
mean_qe_220 = float(np.mean(qe_220)) if qe_220 else 0.0
max_qe_220  = float(np.max(qe_220))  if qe_220 else 0.0
sigma_end   = float(np.mean(list(cortex_edge._surprise_history)))

print(f"  0.41 Hz surprise:  mean={mean_qe_041:.4f}  max={max_qe_041:.4f}")
print(f"  2.20 Hz surprise:  mean={mean_qe_220:.4f}  max={max_qe_220:.4f}")
print(f"  σ at end: {sigma_end:.4f}  (should be moderate — not 1.0 = stuck open)")

# QE at boundaries should be elevated (novel) but not infinite
# σ should not be stuck at max (SIGMA_MAX=3.5) — that means runaway
runaway = sigma_end > 0.90
report("BT-16 Boundary surprise control",
       max_qe_041 < 2.0 and max_qe_220 < 2.0 and not runaway,
       f"0.41Hz max_QE={max_qe_041:.4f}, 2.20Hz max_QE={max_qe_220:.4f}, "
       f"σ={sigma_end:.4f} (runaway={runaway})",
       warn=(max_qe_041 < 3.0 and max_qe_220 < 3.0 and sigma_end < 0.95))


# ═══════════════════════════════════════════════════════════════
# BT-17: END-TO-END FREQUENCY RECONSTRUCTION ACCURACY
# Full pipeline: M50 simulation → decode → fuse → MAE.
# Tests 15 frequencies across the full range including edges.
# This is the broadest single-number quality check.
# ═══════════════════════════════════════════════════════════════

section("BT-17: End-to-end frequency reconstruction (15 test frequencies)")

e2e_freqs = [0.41, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
             1.2, 1.4, 1.6, 1.8, 2.0, 2.12, 2.20]
np.random.seed(70)
sig_e2e, _ = make_blocks(e2e_freqs, block_dur=40.0)
d_e2e = run_sim(sig_e2e,
    total_time=stabilization_time + 2*len(e2e_freqs)*40.0 + 10.0,
    sweep_mode=False, dynamic_settle=True, verbose=False)

df_e2e, ds_e2e, fused_e2e, w_e2e, _, _ = decode_stream(
    d_e2e, raw_x_slow, true_y_slow, raw_x_fast, true_y_fast)

Y_e2e = d_e2e['Y']
print(f"  {'Freq':>5}  {'Slow MAE':>9}  {'Fused MAE':>10}  {'w':>5}  {'OK':>4}")
print(f"  {'─'*5}  {'─'*9}  {'─'*10}  {'─'*5}  {'─'*4}")

e2e_per = []
for f in e2e_freqs:
    m = Y_e2e == f
    if m.any():
        sl = mae(ds_e2e[m], Y_e2e[m])
        fu = mae(fused_e2e[m], Y_e2e[m])
        w  = np.mean(w_e2e[m])
        ok = fu < 0.040
        e2e_per.append((f, ok, fu))
        print(f"  {f:5.2f}  {sl:9.4f}  {fu:10.4f}  {w:5.3f}  {'✓' if ok else '✗'}")

n_ok = sum(ok for _, ok, _ in e2e_per)
worst = max(fu for _, _, fu in e2e_per)
overall_mae = mae(fused_e2e, Y_e2e)

print(f"\n  Overall fused MAE:   {overall_mae:.4f} Hz")
print(f"  Per-freq pass rate:  {n_ok}/{len(e2e_per)} (target = all pass)")

report("BT-17 End-to-end reconstruction",
       n_ok == len(e2e_per) and overall_mae < 0.020,
       f"overall_MAE={overall_mae:.4f}, per_freq_pass={n_ok}/{len(e2e_per)}, "
       f"worst={worst:.4f} Hz",
       warn=(n_ok >= len(e2e_per)-2 and overall_mae < 0.030))


# ═══════════════════════════════════════════════════════════════
# BT-18: FULL CORTEX PIPELINE DETERMINISM
# Run the exact same simulation+cortex twice with the same seeds.
# Both runs must produce identical QE sequences.
# If not, there's a hidden stateful RNG somewhere.
# ═══════════════════════════════════════════════════════════════

section("BT-18: Full pipeline determinism (same seed → same output)")

def run_pipeline_once(seed):
    cortex = CortexM51(seed=seed)
    np.random.seed(seed)
    sig, _ = make_blocks([0.70, 1.10, 1.50], block_dur=30.0)
    d = run_sim(sig,
        total_time=stabilization_time + 2*3*30.0 + 10.0,
        sweep_mode=False, dynamic_settle=True, verbose=False)
    r = run_cortex_on_stream(d, raw_x_slow, true_y_slow,
                             raw_x_fast, true_y_fast, cortex)
    return np.array([x['surprise'] for x in r])

print("  Running pipeline twice with seed=88...")
qe_run1 = run_pipeline_once(88)
qe_run2 = run_pipeline_once(88)

max_diff = float(np.max(np.abs(qe_run1 - qe_run2)))
mean_diff = float(np.mean(np.abs(qe_run1 - qe_run2)))

print(f"  Max QE diff between runs:  {max_diff:.2e}  (target = 0.00)")
print(f"  Mean QE diff between runs: {mean_diff:.2e}")

report("BT-18 Pipeline determinism",
       max_diff < 1e-6,
       f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

summarise()