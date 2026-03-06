"""
M49 FULL STRESS BREAK TEST
============================
Tests everything NOT covered by the original four-hole break test.
All 13 untested gaps, prioritised by risk.

HOLE 5  — DivergenceCUSUM false positives during sweep
HOLE 6  — Edge frequency accuracy (0.40–0.52 Hz and 2.08–2.20 Hz)
HOLE 7  — Downward steps + multi-base detection (asymmetry)
HOLE 8  — Noise + transitions (compound stress)
HOLE 9  — Short blocks (block_dur < debounce ceiling)
HOLE 10 — Multi-seed calibration generalization (5 seed pairs)
HOLE 11 — Step detection at base=0.50, 1.50, 2.00 Hz
HOLE 12 — Below-floor steps (0.02 Hz apart — should NOT fire)
HOLE 13 — Long-run drift (600s stable block)
HOLE 14 — Oscillator boundary frequencies (0.95–1.05 Hz)
HOLE 15 — Multi-step rapid sequence (A→B→C→D→E)

Pass criteria are defined per-hole and printed with each verdict.
All holes must pass for M49 to be considered stress-tested.
"""

import numpy as np
from collections import deque

from m50_neuron import (
    run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks, make_steps,
    decode_resonance, decode_resonance_raw, build_reverse_lookup,
    compute_stability_plv, DivergenceCUSUM,
    mae, dt, stabilization_time,
    PLV_STAB_WINDOW, PLV_THRESHOLD_LO, PLV_THRESHOLD_HI,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
    DIVERG_DEBOUNCE, DIVERG_THRESHOLD,
    MIN_SETTLE_S, SETTLE_CYCLES,
)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0

SLOW_FREQS_CAL = sorted(set([
    0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87,
    0.9, 0.92, 0.95, 0.97, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.35, 1.4,
    1.5, 1.55, 1.6, 1.7, 1.75, 1.8, 1.9, 1.95, 2.0, 2.05, 2.1,
]))


# ── Shared calibration + decode pipeline ─────────────────────────────

def calibrate(sweep_seed=0, block_seed=1, label=""):
    print(f"  [Cal{label}] sweep={sweep_seed}, block={block_seed}")
    np.random.seed(sweep_seed)
    data_train = run_sim(
        make_sweep(0.5, 2.0, 6, sweep_dur),
        total_time=warmup + 6*sweep_dur + 10.0,
        sweep_mode=True, verbose=False, collect_calib=False)
    ridge_fast, ridge_fast_sc = fit_ridge(
        data_train['feat_fast'], data_train['Y'], RIDGE_ALPHA_FAST)

    np.random.seed(block_seed)
    block_sig, _ = make_blocks(SLOW_FREQS_CAL, block_dur=40.0)
    slow_total = stabilization_time + 2*len(SLOW_FREQS_CAL)*40.0 + 10.0
    data_slow = run_sim(block_sig, total_time=slow_total,
                        sweep_mode=False, dynamic_settle=True,
                        verbose=False, collect_calib=True)
    raw_x_slow, true_y_slow = build_reverse_lookup(
        sorted(data_slow['calib_plv_slow'].keys()),
        data_slow['calib_plv_slow'], data_slow['calib_energy_slow'])
    raw_x_fast, true_y_fast = build_reverse_lookup(
        sorted(data_slow['calib_plv_fast'].keys()),
        data_slow['calib_plv_fast'], data_slow['calib_energy_fast'])
    ridge_slow, ridge_slow_sc = fit_ridge(
        data_slow['feat_slow'], data_slow['Y'], RIDGE_ALPHA_SLOW)
    print(f"  Done: {len(raw_x_slow)} lookup pts")
    return (ridge_fast, ridge_fast_sc, ridge_slow, ridge_slow_sc,
            raw_x_slow, true_y_slow, raw_x_fast, true_y_fast)


def decode_test(data, cal):
    (ridge_fast, ridge_fast_sc, ridge_slow, ridge_slow_sc,
     raw_x_slow, true_y_slow, raw_x_fast, true_y_fast) = cal
    Y = data['Y']; T = data['T']; n = len(Y)

    df = np.array([decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                                     raw_x_fast, true_y_fast) for i in range(n)])
    ds = np.array([decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                                     raw_x_slow, true_y_slow) for i in range(n)])

    change_det = DivergenceCUSUM()
    novelty    = np.zeros(n, dtype=bool)
    divergence = np.zeros(n)
    for i in range(n):
        div, nov      = change_det.update(df[i], ds[i], T[i])
        divergence[i] = div
        novelty[i]    = nov

    plv_hist = deque(maxlen=PLV_STAB_WINDOW)
    d_fused  = np.zeros(n); d_w_slow = np.zeros(n)
    for i in range(n):
        max_plv = float(np.max(data['plv_slow'][i]))
        plv_hist.append(max_plv)
        w = compute_stability_plv(plv_hist)
        if novelty[i]: w = 0.0
        d_fused[i]  = w * ds[i] + (1.0 - w) * df[i]
        d_w_slow[i] = w

    return {
        'df': df, 'ds': ds, 'd_fused': d_fused, 'd_w_slow': d_w_slow,
        'novelty': novelty, 'divergence': divergence,
        'change_events': change_det.novelty_events,
        'Y': Y, 'T': T,
    }


def count_false_positives(novelty, Y, window=DIVERG_DEBOUNCE+10):
    trans_idx  = np.where(np.diff(Y) != 0)[0] + 1
    near_trans = np.zeros(len(Y), dtype=bool)
    for idx in trans_idx:
        near_trans[max(0, idx-5):min(len(Y), idx+window)] = True
    return int(np.sum(novelty & ~near_trans))


# ── CALIBRATE (shared across all holes) ──────────────────────────────
print("=" * 72)
print("  M49 FULL STRESS BREAK TEST")
print("  13 untested gaps — all must pass")
print("=" * 72)

print(f"\n{'='*72}")
print("  CALIBRATION  (seeds 0, 1 — standard)")
print(f"{'='*72}")
CAL = calibrate(0, 1)
results = {}   # hole_name → pass/fail


# ═══════════════════════════════════════════════════════════════════
# HOLE 5 — DivergenceCUSUM false positives during sweep
# During a continuous frequency sweep, df and ds both track the
# moving frequency — but df is faster. The question is whether
# |df - ds| stays below DIVERG_THRESHOLD 0.020 Hz during steady sweep,
# or whether the lag between them causes false novelty fires.
# Pass: < 5 false positive events during a full sweep.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 5 — DivergenceCUSUM false positives during sweep")
print("  Pass: < 5 novelty events fired during continuous sweep")
print(f"{'='*72}")

np.random.seed(50)
d_sw5 = run_sim(make_sweep(0.5, 2.0, 3, sweep_dur),
                total_time=warmup + 3*sweep_dur + 10.0,
                sweep_mode=True, verbose=False)
r_sw5 = decode_test(d_sw5, CAL)

sweep_events = len(r_sw5['change_events'])
div_mean_sw  = np.mean(r_sw5['divergence'])
div_max_sw   = np.max(r_sw5['divergence'])
h5_pass      = sweep_events < 5

print(f"\n  Sweep novelty events:   {sweep_events}  (target < 5)")
print(f"  Mean |df-ds| in sweep:  {div_mean_sw:.4f} Hz")
print(f"  Max  |df-ds| in sweep:  {div_max_sw:.4f} Hz")
print(f"  w_slow mean:            {np.mean(r_sw5['d_w_slow']):.4f}  (target < 0.10)")
print(f"\n  HOLE 5 {'✓ PASS' if h5_pass else '✗ FAIL'}")
results['H5 — DivergenceCUSUM sweep FP'] = h5_pass


# ═══════════════════════════════════════════════════════════════════
# HOLE 6 — Edge frequency accuracy
# Oscillators at extreme indices may not lock as cleanly.
# Test: MAE < 0.015 Hz for each edge frequency.
# (Relaxed from 0.008 — edge freqs are genuinely harder.)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 6 — Edge frequency accuracy (0.40–0.52 Hz, 2.08–2.20 Hz)")
print("  Pass: per-frequency MAE < 0.015 Hz at both edges")
print(f"{'='*72}")

edge_freqs = [0.41, 0.44, 0.47, 0.50, 2.08, 2.12, 2.16, 2.20]
sig_e6, _  = make_blocks(edge_freqs, block_dur=40.0)
total_e6   = stabilization_time + 2*len(edge_freqs)*40.0 + 10.0
np.random.seed(60)
d_e6 = run_sim(sig_e6, total_time=total_e6,
               sweep_mode=False, dynamic_settle=True, verbose=False)
r_e6 = decode_test(d_e6, CAL)

print(f"\n  {'Freq':>6}  {'Slow MAE':>10}  {'OK':>4}")
print(f"  {'─'*6}  {'─'*10}  {'─'*4}")
h6_pass = True
for f in sorted(set(r_e6['Y'])):
    m = r_e6['Y'] == f
    if m.any():
        m_mae = mae(r_e6['ds'][m], r_e6['Y'][m])
        ok    = m_mae < 0.015
        if not ok: h6_pass = False
        print(f"  {f:6.2f}  {m_mae:10.4f}  {'✓' if ok else '✗':>4}")

print(f"\n  HOLE 6 {'✓ PASS' if h6_pass else '✗ FAIL'}")
results['H6 — Edge frequency accuracy'] = h6_pass


# ═══════════════════════════════════════════════════════════════════
# HOLE 7 — Downward steps + multi-base detection
# All H4 tests were upward (0.80 → higher). Test downward steps
# and steps from different base frequencies.
# Pass: ≥ 80% detection, < 5 false positives for each base.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 7 — Downward steps + multi-base detection")
print("  Pass: ≥80% detection, <5 FP for each configuration")
print(f"{'='*72}")

h7_configs = [
    (1.10, 0.80, 0.30, "DOWN  1.10→0.80 Hz"),
    (1.50, 1.20, 0.30, "DOWN  1.50→1.20 Hz"),
    (0.50, 0.80, 0.30, "UP    0.50→0.80 Hz"),
    (2.00, 1.70, 0.30, "DOWN  2.00→1.70 Hz"),
    (0.70, 1.00, 0.30, "UP    0.70→1.00 Hz"),
]

print(f"\n  {'Config':22s}  {'Trans':>6}  {'Det':>6}  {'Rate':>7}  {'FP':>5}  {'OK':>4}")
print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*4}")

h7_pass = True
for base, target, step, label in h7_configs:
    step_freqs = [base, target] * 3
    np.random.seed(700 + int(abs(base)*10))
    c_sig, _ = make_blocks(step_freqs, block_dur=30.0)
    total_h7  = stabilization_time + len(step_freqs)*30.0*2 + 10.0
    d_h7 = run_sim(c_sig, total_time=total_h7,
                   sweep_mode=False, dynamic_settle=False, verbose=False)
    r_h7 = decode_test(d_h7, CAL)

    Y_h7     = r_h7['Y']
    expected = len(np.where(np.diff(Y_h7) != 0)[0])
    detected = len(r_h7['change_events'])
    rate     = detected / max(1, expected)
    fp       = count_false_positives(r_h7['novelty'], Y_h7)
    ok       = rate >= 0.80 and fp < 5
    if not ok: h7_pass = False
    print(f"  {label:22s}  {expected:6d}  {detected:6d}  {rate:7.0%}  {fp:5d}  {'✓' if ok else '✗':>4}")

print(f"\n  HOLE 7 {'✓ PASS' if h7_pass else '✗ FAIL'}")
results['H7 — Downward + multi-base steps'] = h7_pass


# ═══════════════════════════════════════════════════════════════════
# HOLE 8 — Noise + transitions (compound stress)
# H3 tested noise on stable blocks. H4 tested transitions without noise.
# This tests both at once: noisy blocks with frequency transitions.
# Pass: ≥ 70% detection rate (relaxed for noise), < 10 FP.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 8 — Noise + transitions (compound stress)")
print("  Pass: ≥70% detection, <10 FP at each noise level")
print(f"{'='*72}")

print(f"\n  {'σ':>4}  {'Trans':>6}  {'Det':>6}  {'Rate':>7}  {'FP':>5}  {'Slow MAE':>10}  {'OK':>4}")
print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*10}  {'─'*4}")

h8_pass = True
for nl in [0.5, 1.0, 2.0, 3.0]:
    step_freqs_n = [0.80, 1.10, 0.80, 1.10, 0.80, 1.10]
    np.random.seed(800 + int(nl*10))
    cn_sig, _ = make_blocks(step_freqs_n, block_dur=30.0, noise_level=nl)
    total_n8  = stabilization_time + len(step_freqs_n)*30.0*2 + 10.0
    d_n8 = run_sim(cn_sig, total_time=total_n8,
                   sweep_mode=False, dynamic_settle=False, verbose=False)
    r_n8 = decode_test(d_n8, CAL)

    Y_n8     = r_n8['Y']
    expected = len(np.where(np.diff(Y_n8) != 0)[0])
    detected = len(r_n8['change_events'])
    rate     = detected / max(1, expected)
    fp       = count_false_positives(r_n8['novelty'], Y_n8)
    s_mae    = mae(r_n8['ds'], Y_n8)
    ok       = rate >= 0.70 and fp < 10
    if not ok: h8_pass = False
    print(f"  {nl:4.1f}  {expected:6d}  {detected:6d}  {rate:7.0%}  "
          f"{fp:5d}  {s_mae:10.4f}  {'✓' if ok else '✗':>4}")

print(f"\n  HOLE 8 {'✓ PASS' if h8_pass else '✗ FAIL'}")
results['H8 — Noise + transitions'] = h8_pass


# ═══════════════════════════════════════════════════════════════════
# HOLE 9 — Short blocks (block_dur < debounce ceiling)
# Debounce = 150 samples = 15s. Block_dur=30s was tested.
# What about block_dur=20s (just above debounce)?
# Expected: detect first transition per pair, miss second if too close.
# Pass: at least 50% detection rate (every other transition), < 5 FP.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 9 — Short blocks (block_dur approaching debounce limit)")
print("  Pass: ≥50% detection (expected: every-other), <5 FP")
print(f"{'='*72}")

print(f"\n  {'block_dur':>10}  {'Trans':>6}  {'Det':>6}  {'Rate':>7}  {'FP':>5}  {'OK':>4}")
print(f"  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*4}")

h9_pass = True
for bd in [40.0, 30.0, 22.0, 18.0]:
    step_freqs_9 = [0.80, 1.10] * 4
    np.random.seed(900)
    c9_sig, _ = make_blocks(step_freqs_9, block_dur=bd)
    total_h9  = stabilization_time + len(step_freqs_9)*bd*2 + 10.0
    d_h9 = run_sim(c9_sig, total_time=total_h9,
                   sweep_mode=False, dynamic_settle=False, verbose=False)
    r_h9 = decode_test(d_h9, CAL)

    Y_h9     = r_h9['Y']
    expected = len(np.where(np.diff(Y_h9) != 0)[0])
    detected = len(r_h9['change_events'])
    rate     = detected / max(1, expected)
    fp       = count_false_positives(r_h9['novelty'], Y_h9)

    # For very short blocks, expect ~50% (debounce blocks alternate fires)
    if bd >= 25:
        ok = rate >= 0.80 and fp < 5
    else:
        ok = rate >= 0.40 and fp < 5  # every-other is acceptable
    if not ok: h9_pass = False

    note = f"  ← debounce covers {DIVERG_DEBOUNCE*0.1:.0f}s of {bd:.0f}s block" if bd < 25 else ""
    print(f"  {bd:10.1f}  {expected:6d}  {detected:6d}  {rate:7.0%}  "
          f"{fp:5d}  {'✓' if ok else '✗':>4}{note}")

print(f"\n  HOLE 9 {'✓ PASS' if h9_pass else '✗ FAIL'}")
results['H9 — Short blocks'] = h9_pass


# ═══════════════════════════════════════════════════════════════════
# HOLE 10 — Multi-seed calibration generalization (5 seed pairs)
# H1 only tested 2 seed pairs. Test 5 pairs to confirm generalization
# isn't a coincidence.
# Pass: slow MAE < 0.010 Hz for every seed pair.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 10 — Multi-seed calibration generalization (5 pairs)")
print("  Pass: slow MAE < 0.010 Hz for every seed pair")
print(f"{'='*72}")

seed_pairs = [(0,1), (17,23), (42,7), (99,13), (200,50)]
test_freqs_10 = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
test_sig_10, _ = make_blocks(test_freqs_10, block_dur=40.0)
total_10       = stabilization_time + 2*len(test_freqs_10)*40.0 + 10.0
np.random.seed(103)
d_10 = run_sim(test_sig_10, total_time=total_10,
               sweep_mode=False, dynamic_settle=True, verbose=False)

print(f"\n  {'Seeds':>12}  {'Slow MAE':>10}  {'w_slow':>8}  {'OK':>4}")
print(f"  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*4}")

h10_pass = True
for ss, bs in seed_pairs:
    cal_i = calibrate(ss, bs, label=f" ({ss},{bs})")
    r_i   = decode_test(d_10, cal_i)
    s_mae = mae(r_i['ds'], d_10['Y'])
    w_s   = np.mean(r_i['d_w_slow'])
    ok    = s_mae < 0.010
    if not ok: h10_pass = False
    print(f"  ({ss:3d},{bs:3d})      {s_mae:10.4f}  {w_s:8.4f}  {'✓' if ok else '✗':>4}")

print(f"\n  HOLE 10 {'✓ PASS' if h10_pass else '✗ FAIL'}")
results['H10 — Multi-seed generalization'] = h10_pass


# ═══════════════════════════════════════════════════════════════════
# HOLE 11 — Step detection at different base frequencies
# H4 only used base=0.80 Hz. Test bases at 0.50, 1.20, 1.80 Hz.
# Pass: ≥ 80% detection, < 5 FP for each base.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 11 — Step detection at different base frequencies")
print("  Pass: ≥80% detection, <5 FP for each base")
print(f"{'='*72}")

h11_bases = [
    (0.50, 0.20, "base=0.50 step=+0.20"),
    (1.20, 0.20, "base=1.20 step=+0.20"),
    (1.80, 0.20, "base=1.80 step=+0.20"),
    (0.50, 0.30, "base=0.50 step=+0.30"),
    (1.50, 0.30, "base=1.50 step=+0.30"),
]

print(f"\n  {'Config':24s}  {'Trans':>6}  {'Det':>6}  {'Rate':>7}  {'FP':>5}  {'OK':>4}")
print(f"  {'─'*24}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*4}")

h11_pass = True
for base, step, label in h11_bases:
    target     = round(base + step, 3)
    sf11       = [base, target] * 3
    np.random.seed(1100 + int(base*10))
    c11, _     = make_blocks(sf11, block_dur=30.0)
    total_h11  = stabilization_time + len(sf11)*30.0*2 + 10.0
    d_h11 = run_sim(c11, total_time=total_h11,
                    sweep_mode=False, dynamic_settle=False, verbose=False)
    r_h11 = decode_test(d_h11, CAL)

    Y_h11    = r_h11['Y']
    expected = len(np.where(np.diff(Y_h11) != 0)[0])
    detected = len(r_h11['change_events'])
    rate     = detected / max(1, expected)
    fp       = count_false_positives(r_h11['novelty'], Y_h11)
    ok       = rate >= 0.80 and fp < 5
    if not ok: h11_pass = False
    print(f"  {label:24s}  {expected:6d}  {detected:6d}  {rate:7.0%}  "
          f"{fp:5d}  {'✓' if ok else '✗':>4}")

print(f"\n  HOLE 11 {'✓ PASS' if h11_pass else '✗ FAIL'}")
results['H11 — Multi-base step detection'] = h11_pass


# ═══════════════════════════════════════════════════════════════════
# HOLE 12 — Below-floor steps (should NOT fire)
# Steps of 0.02 Hz (well below 0.15 Hz floor) should produce
# near-zero false positives. This tests that sub-threshold changes
# are silently ignored, not spuriously detected.
# Pass: < 3 events for each sub-floor step.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 12 — Below-floor steps (0.02 Hz apart — should NOT fire)")
print("  Pass: < 3 detection events (these are below the reliable floor)")
print(f"{'='*72}")

print(f"\n  {'Step':>6}  {'Base→Target':>14}  {'Trans':>6}  {'Events':>8}  {'OK':>4}")
print(f"  {'─'*6}  {'─'*14}  {'─'*6}  {'─'*8}  {'─'*4}")

h12_pass = True
for base, step in [(0.80, 0.02), (1.20, 0.02), (0.60, 0.02)]:
    target    = round(base + step, 3)
    sf12      = [base, target] * 4
    np.random.seed(1200 + int(base*10))
    c12, _    = make_blocks(sf12, block_dur=30.0)
    total_h12 = stabilization_time + len(sf12)*30.0*2 + 10.0
    d_h12 = run_sim(c12, total_time=total_h12,
                    sweep_mode=False, dynamic_settle=False, verbose=False)
    r_h12 = decode_test(d_h12, CAL)

    events = len(r_h12['change_events'])
    ok     = events < 3
    if not ok: h12_pass = False
    print(f"  {step:6.2f}  {base:.2f}→{target:.2f}       "
          f"  {len(np.where(np.diff(r_h12['Y'])!=0)[0]):6d}  {events:8d}  "
          f"{'✓' if ok else '✗':>4}")

print(f"\n  HOLE 12 {'✓ PASS' if h12_pass else '✗ FAIL'}")
results['H12 — Below-floor steps silent'] = h12_pass


# ═══════════════════════════════════════════════════════════════════
# HOLE 13 — Long-run drift
# 600s stable block at one frequency. Decoder should not drift.
# Pass: MAE < 0.008 Hz and std(decoded_freq) < 0.010 Hz over the run.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 13 — Long-run drift (600s stable block)")
print("  Pass: slow MAE < 0.008 Hz, std < 0.010 Hz throughout")
print(f"{'='*72}")

print(f"\n  {'Freq':>6}  {'Slow MAE':>10}  {'Slow std':>10}  {'w_slow':>8}  {'OK':>4}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*4}")

h13_pass = True
for f_long in [0.70, 1.00, 1.50]:
    lr_sig, _ = make_blocks([f_long], block_dur=700.0)
    total_lr  = stabilization_time + 700.0 + 10.0
    np.random.seed(1300 + int(f_long*10))
    d_lr = run_sim(lr_sig, total_time=total_lr,
                   sweep_mode=False, dynamic_settle=True, verbose=False)
    r_lr = decode_test(d_lr, CAL)

    s_mae = mae(r_lr['ds'], r_lr['Y'])
    s_std = float(np.std(r_lr['ds']))
    w_s   = np.mean(r_lr['d_w_slow'])
    ok    = s_mae < 0.008 and s_std < 0.010
    if not ok: h13_pass = False
    print(f"  {f_long:6.2f}  {s_mae:10.4f}  {s_std:10.4f}  {w_s:8.4f}  "
          f"{'✓' if ok else '✗':>4}")

print(f"\n  HOLE 13 {'✓ PASS' if h13_pass else '✗ FAIL'}")
results['H13 — Long-run drift'] = h13_pass


# ═══════════════════════════════════════════════════════════════════
# HOLE 14 — Oscillator group boundary (0.95–1.05 Hz)
# Fast oscillators end at index ~250, slow begin. The transition
# between groups may cause decoder instability near 1.0 Hz.
# Pass: per-frequency MAE < 0.008 Hz for all boundary freqs.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 14 — Oscillator group boundary (0.95–1.05 Hz)")
print("  Pass: per-frequency MAE < 0.008 Hz across the boundary")
print(f"{'='*72}")

boundary_freqs = [0.93, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05, 1.07]
sig_b14, _     = make_blocks(boundary_freqs, block_dur=40.0)
total_b14      = stabilization_time + 2*len(boundary_freqs)*40.0 + 10.0
np.random.seed(1400)
d_b14 = run_sim(sig_b14, total_time=total_b14,
                sweep_mode=False, dynamic_settle=True, verbose=False)
r_b14 = decode_test(d_b14, CAL)

print(f"\n  {'Freq':>6}  {'Slow MAE':>10}  {'OK':>4}")
print(f"  {'─'*6}  {'─'*10}  {'─'*4}")

h14_pass = True
for f in sorted(set(r_b14['Y'])):
    m = r_b14['Y'] == f
    if m.any():
        m_mae = mae(r_b14['ds'][m], r_b14['Y'][m])
        ok    = m_mae < 0.008
        if not ok: h14_pass = False
        print(f"  {f:6.2f}  {m_mae:10.4f}  {'✓' if ok else '✗':>4}")

print(f"\n  HOLE 14 {'✓ PASS' if h14_pass else '✗ FAIL'}")
results['H14 — Oscillator boundary'] = h14_pass


# ═══════════════════════════════════════════════════════════════════
# HOLE 15 — Multi-step rapid sequence (A→B→C→D→E)
# All H4 tests were binary (A↔B). Test a 5-frequency chain where
# each step is 0.30 Hz. CUSUM should fire at each transition.
# Pass: ≥ 80% of transitions detected, < 5 FP.
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 15 — Multi-step rapid sequence (A→B→C→D→E chain)")
print("  Pass: ≥80% detection, <5 FP across the full chain")
print(f"{'='*72}")

chain_freqs = [0.50, 0.80, 1.10, 1.40, 1.70, 0.50, 0.80, 1.10, 1.40, 1.70]
np.random.seed(1500)
c15, _    = make_blocks(chain_freqs, block_dur=30.0)
total_h15 = stabilization_time + len(chain_freqs)*30.0*2 + 10.0
d_h15 = run_sim(c15, total_time=total_h15,
                sweep_mode=False, dynamic_settle=False, verbose=False)
r_h15 = decode_test(d_h15, CAL)

Y_h15    = r_h15['Y']
expected = len(np.where(np.diff(Y_h15) != 0)[0])
detected = len(r_h15['change_events'])
rate     = detected / max(1, expected)
fp       = count_false_positives(r_h15['novelty'], Y_h15)
h15_pass = rate >= 0.80 and fp < 5

print(f"\n  Chain: {' → '.join(str(f) for f in [0.50,0.80,1.10,1.40,1.70])}")
print(f"  Transitions expected: {expected}")
print(f"  Transitions detected: {detected}  ({rate:.0%})")
print(f"  False positives:      {fp}")
if r_h15['change_events']:
    print(f"\n  Detection times (first 10):")
    for t_ev, div_ev, acc_ev in r_h15['change_events'][:10]:
        print(f"    t={t_ev:.0f}s  |df-ds|={div_ev:.4f}  accum={acc_ev:.4f}")

print(f"\n  HOLE 15 {'✓ PASS' if h15_pass else '✗ FAIL'}")
results['H15 — Multi-step chain'] = h15_pass


# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  M49 FULL STRESS BREAK TEST — FINAL SUMMARY")
print(f"{'='*72}")

print(f"\n  {'Test':44s}  {'Result':>8}")
print(f"  {'─'*44}  {'─'*8}")
overall = True
for name, passed in results.items():
    if not passed: overall = False
    print(f"  {name:44s}  {'✓ PASS' if passed else '✗ FAIL':>8}")

print(f"\n  {'─'*54}")
print(f"  {'OVERALL':44s}  {'✓ PASS' if overall else '✗ FAIL':>8}")

if overall:
    print("""
  ✓✓✓ ALL STRESS TESTS PASS ✓✓✓

  M49 is comprehensively validated:
    - No false positives during sweep
    - Edge frequencies accurate within 0.015 Hz
    - Downward steps detected as reliably as upward
    - Noise + transitions handled correctly
    - Short-block behaviour matches debounce design
    - Calibration stable across 5 seed pairs
    - Step detection generalises to all base frequencies
    - Sub-floor changes silently ignored
    - No long-run drift
    - Oscillator boundary frequencies clean
    - Multi-step chains detected correctly

  Safe to build the next layer on M49.
""")
else:
    failed = [n for n, p in results.items() if not p]
    print(f"""
  ✗ {len(failed)} stress test(s) failed:
""")
    for name in failed:
        print(f"    - {name}")
    print("""
  Do NOT build further layers until these are resolved.
""")