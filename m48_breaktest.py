"""
M48 BREAK TEST — Multi-seed reproducibility + stress test
=========================================================
Run M48 across 3 different random seeds to verify:
  1. Results are consistent (not a fluke)
  2. All metrics pass on every seed
  3. PLV-based stability works reliably
"""

import numpy as np
from collections import deque
from m48_neuron import (
    run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks,
    decode_resonance, decode_resonance_raw, build_reverse_lookup,
    compute_stability_plv, compute_stability_variance,
    TwoWindowChangeDetector,
    mae, N, dt, stabilization_time,
    STABILITY_WINDOW, PLV_STAB_WINDOW,
    PLV_THRESHOLD_LO, PLV_THRESHOLD_HI,
    CHANGE_WINDOW_K,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0

SEEDS = [42, 99, 7]

print("=" * 72)
print("  M48 BREAK TEST — 3 seeds")
print("=" * 72)

# ── Calibrate ONCE (shared across seeds) ──────────────────────
print("\n  [Cal] Sweep + block calibration (shared)...")
np.random.seed(0)
data_train = run_sim(
    make_sweep(0.5, 2.0, 6, sweep_dur),
    total_time=warmup + 6*sweep_dur + 10.0,
    sweep_mode=True, verbose=False, collect_calib=False)
ridge_fast, ridge_fast_sc = fit_ridge(
    data_train['feat_fast'], data_train['Y'], RIDGE_ALPHA_FAST)

np.random.seed(1)
slow_freqs = sorted(set([
    0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87,
    0.9, 0.92, 0.95, 0.97, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.35, 1.4,
    1.5, 1.55, 1.6, 1.7, 1.75, 1.8, 1.9, 1.95, 2.0, 2.05, 2.1,
]))
block_sig, _ = make_blocks(slow_freqs, block_dur=40.0)
slow_total = stabilization_time + 2*len(slow_freqs)*40.0 + 10.0
data_slow = run_sim(block_sig, total_time=slow_total,
                    sweep_mode=False, dynamic_settle=True, verbose=False,
                    collect_calib=True)
raw_x_slow, true_y_slow = build_reverse_lookup(
    sorted(data_slow['calib_plv_slow'].keys()),
    data_slow['calib_plv_slow'], data_slow['calib_energy_slow'])
raw_x_fast, true_y_fast = build_reverse_lookup(
    sorted(data_slow['calib_plv_fast'].keys()),
    data_slow['calib_plv_fast'], data_slow['calib_energy_fast'])
ridge_slow, ridge_slow_sc = fit_ridge(
    data_slow['feat_slow'], data_slow['Y'], RIDGE_ALPHA_SLOW)
print(f"  Calibration done: {len(raw_x_slow)} lookup pts")


def decode_test(data):
    Y = data['Y']; T = data['T']; n = len(Y)
    df = np.array([decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                                     raw_x_fast, true_y_fast) for i in range(n)])
    ds = np.array([decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                                     raw_x_slow, true_y_slow) for i in range(n)])

    change_det = TwoWindowChangeDetector()
    js_raw = np.zeros(n); novelty = np.zeros(n, dtype=bool)
    for i in range(n):
        js, nov = change_det.update(data['energy_fast'][i], T[i])
        js_raw[i] = js; novelty[i] = nov

    plv_hist = deque(maxlen=PLV_STAB_WINDOW)
    d_fused = np.zeros(n); d_w_slow = np.zeros(n)
    for i in range(n):
        max_plv = float(np.max(data['plv_slow'][i]))
        plv_hist.append(max_plv)
        w = compute_stability_plv(plv_hist)
        if novelty[i]: w = 0.0
        d_fused[i] = w*ds[i] + (1.-w)*df[i]
        d_w_slow[i] = w

    rf = predict_ridge(data['feat_fast'], ridge_fast, ridge_fast_sc)
    rs = predict_ridge(data['feat_slow'], ridge_slow, ridge_slow_sc)

    return {'df':df,'ds':ds,'d_fused':d_fused,'d_w_slow':d_w_slow,
            'rf':rf,'rs':rs,'novelty':novelty,
            'change_events':change_det.novelty_events,
            'Y':Y,'T':T}


# ── Run break test ────────────────────────────────────────────
all_results = []
for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*72}")
    print(f"  SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'='*72}")

    # Sweep test
    np.random.seed(seed)
    d_sw = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
                   total_time=warmup+2*sweep_dur+10., sweep_mode=True, verbose=False)
    r_sw = decode_test(d_sw)

    # Block test
    np.random.seed(seed + 100)
    test_freqs = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
    test_sig, _ = make_blocks(test_freqs, block_dur=40.0)
    test_total = stabilization_time + 2*len(test_freqs)*40. + 10.
    d_bl = run_sim(test_sig, total_time=test_total,
                   sweep_mode=False, dynamic_settle=True, verbose=False)
    r_bl = decode_test(d_bl)

    # Curiosity test
    np.random.seed(seed + 200)
    curiosity_freqs = [0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
    c_sig, _ = make_blocks(curiosity_freqs, block_dur=30.)
    d_c = run_sim(c_sig, total_time=stabilization_time+len(curiosity_freqs)*30.*2+10.,
                  sweep_mode=False, dynamic_settle=False, verbose=False)
    r_c = decode_test(d_c)
    freq_changes = np.where(np.diff(r_c['Y']) != 0)[0] + 1
    detected = len(r_c['change_events'])
    block_false = int(np.sum(r_bl['novelty']))

    # Noise test
    np.random.seed(seed + 300)
    ns, _ = make_blocks([0.5,1.0,1.5,2.0], block_dur=40., noise_level=3.0)
    d_n = run_sim(ns, total_time=500., sweep_mode=False,
                  dynamic_settle=True, verbose=False)
    r_n = decode_test(d_n)

    # Collect metrics
    metrics = {
        'block_slow_mae': mae(r_bl['ds'], r_bl['Y']),
        'block_fused_mae': mae(r_bl['d_fused'], r_bl['Y']),
        'w_blocks': np.mean(r_bl['d_w_slow']),
        'w_sweep': np.mean(r_sw['d_w_slow']),
        'cusum_rate': detected / max(1, len(freq_changes)),
        'block_false': float(block_false),
        'noise_fused': mae(r_n['d_fused'], r_n['Y']),
    }
    all_results.append(metrics)

    print(f"  Block slow MAE:  {metrics['block_slow_mae']:.4f}  (<0.008)")
    print(f"  Block fused MAE: {metrics['block_fused_mae']:.4f}")
    print(f"  w_slow blocks:   {metrics['w_blocks']:.4f}  (>0.80)")
    print(f"  w_slow sweep:    {metrics['w_sweep']:.4f}  (<0.30)")
    print(f"  CUSUM:           {detected}/{len(freq_changes)} ({metrics['cusum_rate']:.0%})")
    print(f"  Block false pos: {metrics['block_false']:.0f}")
    print(f"  Noise σ=3 fused: {metrics['noise_fused']:.4f}")


# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  BREAK TEST SUMMARY")
print(f"{'='*72}")

targets = {
    'block_slow_mae': ('<', 0.008),
    'block_fused_mae': ('<', 0.008),
    'w_blocks': ('>', 0.80),
    'w_sweep': ('<', 0.30),
    'cusum_rate': ('>', 0.80),
    'block_false': ('<', 20.0),
    'noise_fused': ('<', 0.050),
}

print(f"\n  {'Metric':18s}", end="")
for seed in SEEDS:
    print(f"  {'S='+str(seed):>8}", end="")
print(f"  {'Range':>10}  {'Target':>8}  {'ALL OK':>7}")
print(f"  {'─'*18}", end="")
for _ in SEEDS:
    print(f"  {'─'*8}", end="")
print(f"  {'─'*10}  {'─'*8}  {'─'*7}")

all_pass = True
for metric, (op, target) in targets.items():
    vals = [r[metric] for r in all_results]
    rng = max(vals) - min(vals)
    ok_each = [(v < target if op == '<' else v > target) for v in vals]
    ok_all = all(ok_each)
    if not ok_all: all_pass = False

    print(f"  {metric:18s}", end="")
    for v, ok in zip(vals, ok_each):
        mark = "✓" if ok else "✗"
        print(f"  {v:7.4f}{mark}", end="")
    print(f"  {rng:10.6f}  {op}{target:<7}  {'✓ ALL' if ok_all else '✗ FAIL':>7}")

print(f"\n  Seeds tested: {SEEDS}")
if all_pass:
    print("  ✓✓✓ ALL SEEDS PASS ALL METRICS ✓✓✓")
else:
    print("  ✗ Some seeds/metrics fail")
print()
