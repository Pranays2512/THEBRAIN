"""
M47 BREAK TEST — Reproducibility Across 5 Random Seeds
========================================================
Tests whether M47's results are consistent or a fluke.

For each of 5 different random seeds, runs:
  1. Calibration (Ridge + block bias)
  2. Sweep → w_slow, fused MAE
  3. Blocks → w_slow, fused MAE
  4. Noise (σ=3.0) → fused MAE
  5. Curiosity → CUSUM detection count, JS ratio

Prints per-seed results + mean/std across seeds.
"""

import numpy as np
import sys
import time

# Import everything from m47_neuron
from m47_neuron import (
    build_network, run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks, make_steps,
    decode_resonance, decode_resonance_raw,
    build_bias_table, compute_stability, TwoWindowChangeDetector,
    mae, N, N_FAST, dt, stabilization_time,
    STABILITY_WINDOW, CHANGE_WINDOW_K,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
)
from collections import deque

SEEDS = [10, 20, 30, 40, 50]  # 5 different seeds (none from M47 main)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0
n_sweeps  = 6

results = []

print("=" * 72)
print("  M47 BREAK TEST — 5 SEEDS, FULL PIPELINE EACH")
print("=" * 72)

for run_idx, seed_base in enumerate(SEEDS):
    t0 = time.time()
    print(f"\n{'─'*72}")
    print(f"  RUN {run_idx+1}/5  (seed_base={seed_base})")
    print(f"{'─'*72}")

    # ── 1. Calibration ────────────────────────────────────────────────
    # Sweep for Ridge fast
    np.random.seed(seed_base)
    train_time = warmup + n_sweeps*sweep_dur + 10.0
    data_train = run_sim(
        make_sweep(0.5, 2.0, n_sweeps, sweep_dur),
        total_time=train_time, sweep_mode=True, verbose=False,
        collect_calib=False
    )
    ridge_fast, ridge_fast_sc = fit_ridge(
        data_train['feat_fast'], data_train['Y'], RIDGE_ALPHA_FAST)

    # Blocks for bias tables + Ridge slow
    slow_freqs = sorted(set([
        0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,
        0.55,0.75,0.95,1.15,1.35,1.55,1.75,1.95,2.05,
    ]))
    block_sig_train, _ = make_blocks(slow_freqs, block_dur=40.0)
    slow_total = stabilization_time + 2*len(slow_freqs)*40.0 + 10.0
    np.random.seed(seed_base + 1)
    data_slow = run_sim(
        block_sig_train, total_time=slow_total,
        sweep_mode=False, dynamic_settle=True, verbose=False,
        collect_calib=True
    )

    block_freqs_slow = sorted(data_slow['calib_plv_slow'].keys())
    bias_freqs_slow, bias_vals_slow = build_bias_table(
        np.array(block_freqs_slow),
        data_slow['calib_plv_slow'], data_slow['calib_energy_slow'])
    block_freqs_fast = sorted(data_slow['calib_plv_fast'].keys())
    bias_freqs_fast, bias_vals_fast = build_bias_table(
        np.array(block_freqs_fast),
        data_slow['calib_plv_fast'], data_slow['calib_energy_fast'])

    ridge_slow, ridge_slow_sc = fit_ridge(
        data_slow['feat_slow'], data_slow['Y'], RIDGE_ALPHA_SLOW)

    print(f"  Calibrated: {len(bias_freqs_slow)} bias pts")

    # ── decode helper ─────────────────────────────────────────────────
    def decode_test(data):
        Y = data['Y']; T = data['T']; n = len(Y)
        df = np.array([decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                                        bias_freqs_fast, bias_vals_fast) for i in range(n)])
        ds = np.array([decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                                        bias_freqs_slow, bias_vals_slow) for i in range(n)])

        change_det = TwoWindowChangeDetector()
        js_raw = np.zeros(n)
        novelty = np.zeros(n, dtype=bool)
        for i in range(n):
            js, nov = change_det.update(data['energy_fast'][i], T[i])
            js_raw[i] = js
            novelty[i] = nov

        slow_hist = deque(maxlen=STABILITY_WINDOW)
        d_fused = np.zeros(n); d_w_slow = np.zeros(n)
        for i in range(n):
            slow_hist.append(ds[i])
            w = compute_stability(slow_hist)
            if novelty[i]: w = 0.0
            d_fused[i] = w*ds[i] + (1.-w)*df[i]
            d_w_slow[i] = w

        return {'df':df,'ds':ds,'d_fused':d_fused,'d_w_slow':d_w_slow,
                'js_raw':js_raw,'novelty':novelty,
                'change_events':change_det.novelty_events,
                'js_threshold':change_det.threshold,
                'Y':Y,'T':T}

    # ── 2. Sweep ──────────────────────────────────────────────────────
    np.random.seed(seed_base + 2)
    d_sw = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
                   total_time=warmup+2*sweep_dur+10., sweep_mode=True, verbose=False)
    r_sw = decode_test(d_sw)
    sw_fused = mae(r_sw['d_fused'], r_sw['Y'])
    sw_wslow = np.mean(r_sw['d_w_slow'])
    sw_fast  = mae(r_sw['df'], r_sw['Y'])

    # ── 3. Blocks ─────────────────────────────────────────────────────
    test_freqs = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
    test_sig, _ = make_blocks(test_freqs, block_dur=40.0)
    test_total = stabilization_time + 2*len(test_freqs)*40. + 10.
    np.random.seed(seed_base + 3)
    d_bl = run_sim(test_sig, total_time=test_total,
                   sweep_mode=False, dynamic_settle=True, verbose=False)
    r_bl = decode_test(d_bl)
    bl_fused = mae(r_bl['d_fused'], r_bl['Y'])
    bl_slow  = mae(r_bl['ds'], r_bl['Y'])
    bl_wslow = np.mean(r_bl['d_w_slow'])
    bl_events = int(np.sum(r_bl['novelty']))

    # ── 4. Noise ──────────────────────────────────────────────────────
    np.random.seed(seed_base + 4)
    ns, _ = make_blocks([0.5,1.0,1.5,2.0], block_dur=40., noise_level=3.0)
    d_n = run_sim(ns, total_time=500., sweep_mode=False,
                  dynamic_settle=True, verbose=False)
    r_n = decode_test(d_n)
    noise_fused = mae(r_n['d_fused'], r_n['Y'])

    # ── 5. Curiosity ──────────────────────────────────────────────────
    curiosity_freqs = [0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
    c_sig, _ = make_blocks(curiosity_freqs, block_dur=30.)
    np.random.seed(seed_base + 5)
    d_c = run_sim(c_sig, total_time=stabilization_time+len(curiosity_freqs)*30.*2+10.,
                  sweep_mode=False, dynamic_settle=False, verbose=False)
    r_c = decode_test(d_c)

    Y_c = r_c['Y']
    freq_changes = np.where(np.diff(Y_c) != 0)[0] + 1
    near = np.zeros(len(Y_c), dtype=bool)
    for idx in freq_changes:
        near[max(0,idx-5):min(len(Y_c),idx+CHANGE_WINDOW_K+5)] = True
    js_trans = np.mean(r_c['js_raw'][near]) if near.any() else 0
    js_calm  = np.mean(r_c['js_raw'][~near]) if (~near).any() else 0
    js_ratio = js_trans / (js_calm + 1e-12)
    cusum_events = len(r_c['change_events'])

    elapsed = time.time() - t0
    r = {
        'sw_fused': sw_fused, 'sw_wslow': sw_wslow, 'sw_fast': sw_fast,
        'bl_fused': bl_fused, 'bl_slow': bl_slow, 'bl_wslow': bl_wslow,
        'bl_events': bl_events,
        'noise_fused': noise_fused,
        'js_ratio': js_ratio, 'cusum_events': cusum_events,
        'elapsed': elapsed,
    }
    results.append(r)

    print(f"  w_slow sweep:   {sw_wslow:.3f}  (target <0.30)")
    print(f"  w_slow blocks:  {bl_wslow:.3f}  (target >0.80)")
    print(f"  Block fused:    {bl_fused:.4f} Hz")
    print(f"  Block events:   {bl_events}  (target ~0)")
    print(f"  Noise fused:    {noise_fused:.4f} Hz")
    print(f"  CUSUM events:   {cusum_events}/12")
    print(f"  JS ratio:       {js_ratio:.1f}×")
    print(f"  Time:           {elapsed:.0f}s")


# ── SUMMARY ───────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  BREAK TEST SUMMARY — M47 ACROSS 5 SEEDS")
print(f"{'='*72}")

keys = [
    ('w_slow sweep',   'sw_wslow',    '<0.30'),
    ('w_slow blocks',  'bl_wslow',    '>0.80'),
    ('Block fused MAE','bl_fused',    '-'),
    ('Block slow MAE', 'bl_slow',     '<0.02'),
    ('Block events',   'bl_events',   '~0'),
    ('Noise fused',    'noise_fused', '-'),
    ('CUSUM events',   'cusum_events','=12'),
    ('JS ratio',       'js_ratio',    '>3.0'),
]

print(f"\n  {'Metric':18s}", end="")
for i in range(len(SEEDS)):
    print(f"  {'S'+str(SEEDS[i]):>6}", end="")
print(f"  {'Mean':>7}  {'Std':>7}  {'Target':>7}  {'Pass':>5}")
print(f"  {'─'*18}", end="")
for _ in range(len(SEEDS)):
    print(f"  {'─'*6}", end="")
print(f"  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}")

all_pass = True
for label, key, target in keys:
    vals = [r[key] for r in results]
    mn = np.mean(vals)
    sd = np.std(vals)
    print(f"  {label:18s}", end="")
    for v in vals:
        print(f"  {v:6.3f}", end="")

    if target.startswith('<'):
        thr = float(target[1:])
        ok = all(v < thr for v in vals)
    elif target.startswith('>'):
        thr = float(target[1:])
        ok = all(v > thr for v in vals)
    elif target.startswith('='):
        thr = int(target[1:])
        ok = all(int(v) == thr for v in vals)
    elif target == '~0':
        ok = all(v < 20 for v in vals)  # allow some false positives
    else:
        ok = True  # no hard target

    if not ok: all_pass = False
    print(f"  {mn:7.4f}  {sd:7.4f}  {target:>7}  {'✓' if ok else '✗':>5}")

print(f"\n  Total time: {sum(r['elapsed'] for r in results):.0f}s")

if all_pass:
    print("\n  ✓ ALL METRICS PASS ACROSS ALL 5 SEEDS — M47 IS ROBUST")
else:
    print("\n  Some metrics fail — see details above")
    print("  Check if failures are borderline or systematic")

print()
