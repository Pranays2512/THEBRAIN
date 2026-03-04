"""
M48 S-FIX TEST — Calibrate S from first 2 sweeps only
=====================================================
Tests the fix in isolation: calibrate STABILITY_SCALE from only
the first 2 training sweeps (before decoder converges), then run
all key metrics.
"""

import numpy as np
from collections import deque
from m48_neuron import (
    build_network, run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks,
    decode_resonance, decode_resonance_raw, build_reverse_lookup,
    compute_stability, TwoWindowChangeDetector,
    mae, N, dt, stabilization_time,
    STABILITY_WINDOW, CHANGE_WINDOW_K,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
    TAU_FAST_S, TAU_SLOW_S,
)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0

print("=" * 72)
print("  M48 S-FIX TEST — first-2-sweeps calibration")
print("=" * 72)

# ── 1. Training sweep (6 sweeps for Ridge, first 2 for S) ────
print("\n  [1] Training sweep...")
np.random.seed(0)
data_train = run_sim(
    make_sweep(0.5, 2.0, 6, sweep_dur),
    total_time=warmup + 6*sweep_dur + 10.0,
    sweep_mode=True, verbose=False, collect_calib=False
)
ridge_fast, ridge_fast_sc = fit_ridge(
    data_train['feat_fast'], data_train['Y'], RIDGE_ALPHA_FAST)
print(f"  Training samples: {len(data_train['Y'])}")

# ── 2. Block calibration (reverse lookup) ─────────────────────
print("\n  [2] Block calibration...")
slow_freqs = sorted(set([
    0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87,
    0.9, 0.92, 0.95, 0.97, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.35, 1.4,
    1.5, 1.55, 1.6, 1.7, 1.75, 1.8, 1.9, 1.95, 2.0, 2.05, 2.1,
]))
block_sig, _ = make_blocks(slow_freqs, block_dur=40.0)
slow_total = stabilization_time + 2*len(slow_freqs)*40.0 + 10.0
np.random.seed(1)
data_slow = run_sim(block_sig, total_time=slow_total,
                    sweep_mode=False, dynamic_settle=True, verbose=False,
                    collect_calib=True)

raw_x_slow, true_y_slow = build_reverse_lookup(
    sorted(data_slow['calib_plv_slow'].keys()),
    data_slow['calib_plv_slow'], data_slow['calib_energy_slow'])
raw_x_fast, true_y_fast = build_reverse_lookup(
    sorted(data_slow['calib_plv_fast'].keys()),
    data_slow['calib_plv_fast'], data_slow['calib_energy_fast'])
print(f"  Reverse lookup: {len(raw_x_slow)} pts")

ridge_slow, ridge_slow_sc = fit_ridge(
    data_slow['feat_slow'], data_slow['Y'], RIDGE_ALPHA_SLOW)

# ── 3. Auto-S from FIRST 2 SWEEPS ONLY ───────────────────────
print("\n  [3] Auto-S from first 2 sweeps...")
ds_cal = np.array([
    decode_resonance(data_train['plv_slow'][i], data_train['energy_slow'][i],
                     raw_x_slow, true_y_slow)
    for i in range(len(data_train['Y']))
])
T_train = data_train['T']

# Only use samples from first 2 sweeps (warmup to warmup+2*sweep_dur)
t_cutoff = warmup + 2 * sweep_dur
mask_first2 = T_train < t_cutoff
ds_first2 = ds_cal[mask_first2]

cal_vars = []
for i in range(STABILITY_WINDOW, len(ds_first2)):
    cal_vars.append(np.var(ds_first2[i-STABILITY_WINDOW:i]))
var_sweep = np.mean(cal_vars)

# Target w=0.10 (data showed w_block=0.947 at this setting)
STABILITY_SCALE = -var_sweep / np.log(0.10)
print(f"  First-2-sweep var mean: {var_sweep:.6f}")
print(f"  S = {STABILITY_SCALE:.6f}")
print(f"  Expected w_sweep: {np.exp(-var_sweep/STABILITY_SCALE):.3f}")

# ── 4. decode_test helper ─────────────────────────────────────
def decode_test(data):
    Y = data['Y']; T = data['T']; n = len(Y)
    df = np.array([decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                                     raw_x_fast, true_y_fast) for i in range(n)])
    ds = np.array([decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                                     raw_x_slow, true_y_slow) for i in range(n)])

    change_det = TwoWindowChangeDetector()
    js_raw = np.zeros(n)
    novelty = np.zeros(n, dtype=bool)
    for i in range(n):
        js, nov = change_det.update(data['energy_fast'][i], T[i])
        js_raw[i] = js; novelty[i] = nov

    slow_hist = deque(maxlen=STABILITY_WINDOW)
    d_fused = np.zeros(n); d_w_slow = np.zeros(n)
    for i in range(n):
        slow_hist.append(ds[i])
        w = compute_stability(slow_hist)
        if novelty[i]: w = 0.0
        d_fused[i] = w*ds[i] + (1.-w)*df[i]
        d_w_slow[i] = w

    rf = predict_ridge(data['feat_fast'], ridge_fast, ridge_fast_sc)
    rs = predict_ridge(data['feat_slow'], ridge_slow, ridge_slow_sc)
    ridge_hist = deque(maxlen=STABILITY_WINDOW)
    r_fused = np.zeros(n); r_w_slow = np.zeros(n)
    for i in range(n):
        ridge_hist.append(rs[i])
        w = compute_stability(ridge_hist)
        r_fused[i] = w*rs[i] + (1.-w)*rf[i]
        r_w_slow[i] = w

    return {'df':df,'ds':ds,'d_fused':d_fused,'d_w_slow':d_w_slow,
            'rf':rf,'rs':rs,'r_fused':r_fused,'r_w_slow':r_w_slow,
            'js_raw':js_raw,'novelty':novelty,
            'change_events':change_det.novelty_events,
            'Y':Y,'T':T}

# ── 5. TEST: Sweep ────────────────────────────────────────────
print(f"\n{'='*72}")
print("  TEST 1: SWEEP")
print(f"{'='*72}")
np.random.seed(2)
d_sw = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
               total_time=warmup+2*sweep_dur+10., sweep_mode=True, verbose=False)
r_sw = decode_test(d_sw)
print(f"  w_slow sweep:  {np.mean(r_sw['d_w_slow']):.4f}  (target <0.30)")
print(f"  Fused MAE:     {mae(r_sw['d_fused'], r_sw['Y']):.4f}")

# ── 6. TEST: Blocks ───────────────────────────────────────────
print(f"\n{'='*72}")
print("  TEST 2: BLOCKS")
print(f"{'='*72}")
test_freqs = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
test_sig, _ = make_blocks(test_freqs, block_dur=40.0)
test_total = stabilization_time + 2*len(test_freqs)*40. + 10.
np.random.seed(3)
d_bl = run_sim(test_sig, total_time=test_total,
               sweep_mode=False, dynamic_settle=True, verbose=False)
r_bl = decode_test(d_bl)
print(f"  w_slow blocks: {np.mean(r_bl['d_w_slow']):.4f}  (target >0.80)")
print(f"  Slow MAE:      {mae(r_bl['ds'], r_bl['Y']):.4f}  (target <0.008)")
print(f"  Fused MAE:     {mae(r_bl['d_fused'], r_bl['Y']):.4f}")
print(f"  Block events:  {int(np.sum(r_bl['novelty']))}")

# Per-freq
print(f"\n  {'Freq':>6}  {'Slow MAE':>9}  {'Fused MAE':>10}  {'w_d':>5}")
for f in sorted(set(r_bl['Y'])):
    m = r_bl['Y'] == f
    if m.any():
        print(f"  {f:6.2f}  {mae(r_bl['ds'][m], r_bl['Y'][m]):9.4f}  "
              f"{mae(r_bl['d_fused'][m], r_bl['Y'][m]):10.4f}  "
              f"{np.mean(r_bl['d_w_slow'][m]):5.2f}")

# ── 7. TEST: Noise ────────────────────────────────────────────
print(f"\n{'='*72}")
print("  TEST 3: NOISE σ=3.0")
print(f"{'='*72}")
np.random.seed(5)
ns, _ = make_blocks([0.5,1.0,1.5,2.0], block_dur=40., noise_level=3.0)
d_n = run_sim(ns, total_time=500., sweep_mode=False, dynamic_settle=True, verbose=False)
r_n = decode_test(d_n)
print(f"  Fused MAE:     {mae(r_n['d_fused'], r_n['Y']):.4f}")
print(f"  w_slow:        {np.mean(r_n['d_w_slow']):.4f}")

# ── 8. TEST: Curiosity ────────────────────────────────────────
print(f"\n{'='*72}")
print("  TEST 4: CURIOSITY")
print(f"{'='*72}")
curiosity_freqs = [0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
c_sig, _ = make_blocks(curiosity_freqs, block_dur=30.)
np.random.seed(6)
d_c = run_sim(c_sig, total_time=stabilization_time+len(curiosity_freqs)*30.*2+10.,
              sweep_mode=False, dynamic_settle=False, verbose=False)
r_c = decode_test(d_c)
freq_changes = np.where(np.diff(r_c['Y']) != 0)[0] + 1
detected = len(r_c['change_events'])
print(f"  Transitions:   {len(freq_changes)}")
print(f"  CUSUM events:  {detected}")
print(f"  Detection:     {detected/max(1,len(freq_changes)):.0%}")

# ── VERDICT ───────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  VERDICT")
print(f"{'='*72}")
block_false = int(np.sum(r_bl['novelty']))
results = {
    'Block slow MAE':  (mae(r_bl['ds'], r_bl['Y']),       '<0.008'),
    'Block fused MAE': (mae(r_bl['d_fused'], r_bl['Y']),  '-'),
    'w_slow blocks':   (np.mean(r_bl['d_w_slow']),        '>0.80'),
    'w_slow sweep':    (np.mean(r_sw['d_w_slow']),         '<0.30'),
    'CUSUM detection': (detected/max(1,len(freq_changes)), '>0.80'),
    'Block false pos': (float(block_false),                '<20'),
    'Noise fused':     (mae(r_n['d_fused'], r_n['Y']),     '-'),
}
print(f"\n  {'Metric':20s}  {'Value':>8}  {'Target':>8}  {'OK':>4}")
print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*4}")
all_pass = True
for name, (val, target) in results.items():
    if target.startswith('<'):
        ok = val < float(target[1:])
    elif target.startswith('>'):
        ok = val > float(target[1:])
    else:
        ok = True
    if not ok: all_pass = False
    print(f"  {name:20s}  {val:8.4f}  {target:>8}  {'✓' if ok else '✗':>4}")

print(f"\n  STABILITY_SCALE = {STABILITY_SCALE:.6f}")
if all_pass:
    print("  ✓ ALL PASS — Fix confirmed!")
else:
    print("  Some metrics fail — see above")
print()
