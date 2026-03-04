"""
M48 MAX-PLV STABILITY TEST — Verify the fix flows correctly
=============================================================
Signal D (max PLV slow) won the search:
  Sweep: mean=0.187, p75=0.176
  Block: mean=0.977, p25=0.971
  Gap: 0.795 (zero overlap)

This script tests the COMPLETE flow:
  1. Compute max_plv_slow for each sample
  2. Use it as w_slow directly (clipped/scaled)
  3. Fuse: fused = w * slow + (1-w) * fast
  4. Verify ALL metrics end-to-end
"""

import numpy as np
from collections import deque
from m48_neuron import (
    run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks, make_steps,
    decode_resonance, decode_resonance_raw, build_reverse_lookup,
    TwoWindowChangeDetector,
    mae, N, dt, stabilization_time,
    STABILITY_WINDOW, CHANGE_WINDOW_K,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0

# ────────────────────────────────────────────────────────────
# NEW: PLV-based stability
# ────────────────────────────────────────────────────────────
# The idea: max(PLV_slow) measures how strongly the network
# is phase-locked to ANY frequency. During stable blocks it's
# ~0.98 (strong lock). During sweeps it's ~0.10 (no lock).
# We use a sliding window of max_plv values and take the min
# over the window (conservative — needs sustained high PLV).
# Then apply a sigmoid to map to [0,1].

PLV_WINDOW = 20  # samples (~2s)
PLV_THRESHOLD_LO = 0.30   # below this → w=0 (sweep)
PLV_THRESHOLD_HI = 0.90   # above this → w=1 (block)

def compute_stability_plv(plv_history):
    """
    Stability from max PLV over a sliding window.
    Uses the MINIMUM of recent max-PLV values (conservative).
    Maps linearly between LO and HI thresholds.
    """
    if len(plv_history) < 3:
        return 0.0
    recent = np.array(list(plv_history))
    # Take min over window — needs sustained phase lock
    plv_min = np.min(recent)
    # Linear ramp between thresholds
    w = (plv_min - PLV_THRESHOLD_LO) / (PLV_THRESHOLD_HI - PLV_THRESHOLD_LO)
    return float(np.clip(w, 0.0, 1.0))


print("=" * 72)
print("  M48 MAX-PLV STABILITY — FLOW TEST")
print("=" * 72)

# ── Calibrate ─────────────────────────────────────────────────
print("\n  [Cal] Building reverse lookup + Ridge...")
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
print(f"  Done: {len(raw_x_slow)} cal pts")

# ── Full decode+fuse pipeline ─────────────────────────────────
def decode_test(data):
    Y = data['Y']; T = data['T']; n = len(Y)

    # Decode
    df = np.array([decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                                     raw_x_fast, true_y_fast) for i in range(n)])
    ds = np.array([decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                                     raw_x_slow, true_y_slow) for i in range(n)])

    # Change detection
    change_det = TwoWindowChangeDetector()
    js_raw = np.zeros(n)
    novelty = np.zeros(n, dtype=bool)
    for i in range(n):
        js, nov = change_det.update(data['energy_fast'][i], T[i])
        js_raw[i] = js; novelty[i] = nov

    # NEW: PLV-based stability
    plv_hist = deque(maxlen=PLV_WINDOW)
    d_fused = np.zeros(n)
    d_w_slow = np.zeros(n)
    for i in range(n):
        max_plv = np.max(data['plv_slow'][i])
        plv_hist.append(max_plv)
        w = compute_stability_plv(plv_hist)
        # Suppress slow during novelty events
        if novelty[i]:
            w = 0.0
        d_fused[i] = w * ds[i] + (1.0 - w) * df[i]
        d_w_slow[i] = w

    # Ridge benchmark (using old variance method for comparison)
    rf = predict_ridge(data['feat_fast'], ridge_fast, ridge_fast_sc)
    rs = predict_ridge(data['feat_slow'], ridge_slow, ridge_slow_sc)
    ridge_hist = deque(maxlen=STABILITY_WINDOW)
    r_fused = np.zeros(n); r_w_slow = np.zeros(n)
    for i in range(n):
        ridge_hist.append(rs[i])
        v = np.var(list(ridge_hist)) if len(ridge_hist) >= 5 else 1.0
        w = float(np.clip(np.exp(-v / 0.0002), 0, 1))
        r_fused[i] = w*rs[i] + (1.-w)*rf[i]
        r_w_slow[i] = w

    return {'df':df,'ds':ds,'d_fused':d_fused,'d_w_slow':d_w_slow,
            'rf':rf,'rs':rs,'r_fused':r_fused,'r_w_slow':r_w_slow,
            'js_raw':js_raw,'novelty':novelty,
            'change_events':change_det.novelty_events,
            'Y':Y,'T':T}


# ── TEST 1: SWEEP ─────────────────────────────────────────────
print(f"\n{'='*72}")
print("  TEST 1: SWEEP TRACKING")
print(f"{'='*72}")
np.random.seed(2)
d_sw = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
               total_time=warmup+2*sweep_dur+10., sweep_mode=True, verbose=False)
r_sw = decode_test(d_sw)

# Show w_slow flow every 20 samples
print(f"\n  FLOW: w_slow during sweep (every 20th sample)")
print(f"  {'Idx':>5}  {'Time':>7}  {'True':>6}  {'Slow':>7}  {'Fast':>7}  {'Fused':>7}  "
      f"{'maxPLV':>7}  {'w':>6}")
print(f"  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}")
Y_sw = r_sw['Y']
for i in range(0, len(Y_sw), 20):
    max_plv = np.max(d_sw['plv_slow'][i])
    print(f"  {i:5d}  {r_sw['T'][i]:7.1f}  {Y_sw[i]:6.3f}  {r_sw['ds'][i]:7.4f}  "
          f"{r_sw['df'][i]:7.4f}  {r_sw['d_fused'][i]:7.4f}  "
          f"{max_plv:7.4f}  {r_sw['d_w_slow'][i]:6.3f}")

print(f"\n  w_slow sweep mean: {np.mean(r_sw['d_w_slow']):.4f} (target <0.30)")
print(f"  Fused MAE:         {mae(r_sw['d_fused'], r_sw['Y']):.4f}")
print(f"  Fast MAE:          {mae(r_sw['df'], r_sw['Y']):.4f}")


# ── TEST 2: BLOCKS ────────────────────────────────────────────
print(f"\n{'='*72}")
print("  TEST 2: STEADY-STATE BLOCKS")
print(f"{'='*72}")
test_freqs = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
test_sig, _ = make_blocks(test_freqs, block_dur=40.0)
test_total = stabilization_time + 2*len(test_freqs)*40. + 10.
np.random.seed(3)
d_bl = run_sim(test_sig, total_time=test_total,
               sweep_mode=False, dynamic_settle=True, verbose=False)
r_bl = decode_test(d_bl)

# Show w_slow flow at block transitions
print(f"\n  FLOW: w_slow at block transitions")
Y_bl = r_bl['Y']
freq_changes_bl = np.where(np.diff(Y_bl) != 0)[0] + 1
for idx in freq_changes_bl[:3]:  # show first 3 transitions
    lo = max(0, idx-5)
    hi = min(len(Y_bl), idx+15)
    print(f"\n  Transition at idx={idx}: {Y_bl[idx-1]:.2f} → {Y_bl[idx]:.2f} Hz")
    print(f"    {'Off':>5}  {'True':>6}  {'Slow':>7}  {'Fused':>7}  {'maxPLV':>7}  {'w':>6}")
    for i in range(lo, hi):
        max_plv = np.max(d_bl['plv_slow'][i])
        print(f"    {i-idx:+5d}  {Y_bl[i]:6.3f}  {r_bl['ds'][i]:7.4f}  "
              f"{r_bl['d_fused'][i]:7.4f}  {max_plv:7.4f}  {r_bl['d_w_slow'][i]:6.3f}")

print(f"\n  w_slow blocks mean: {np.mean(r_bl['d_w_slow']):.4f} (target >0.80)")
print(f"  Slow MAE:           {mae(r_bl['ds'], r_bl['Y']):.4f} (target <0.008)")
print(f"  Fused MAE:          {mae(r_bl['d_fused'], r_bl['Y']):.4f}")

print(f"\n  Per-frequency:")
print(f"  {'Freq':>6}  {'Slow MAE':>9}  {'Fused MAE':>10}  {'w':>6}  {'maxPLV':>7}")
for f in sorted(set(Y_bl)):
    m = Y_bl == f
    if m.any():
        plvs = [np.max(d_bl['plv_slow'][i]) for i in range(len(Y_bl)) if m[i]]
        print(f"  {f:6.2f}  {mae(r_bl['ds'][m], r_bl['Y'][m]):9.4f}  "
              f"{mae(r_bl['d_fused'][m], r_bl['Y'][m]):10.4f}  "
              f"{np.mean(r_bl['d_w_slow'][m]):6.3f}  {np.mean(plvs):7.4f}")


# ── TEST 3: NOISE σ=3.0 ──────────────────────────────────────
print(f"\n{'='*72}")
print("  TEST 3: NOISE ROBUSTNESS (σ=3.0)")
print(f"{'='*72}")
np.random.seed(5)
ns, _ = make_blocks([0.5,1.0,1.5,2.0], block_dur=40., noise_level=3.0)
d_n = run_sim(ns, total_time=500., sweep_mode=False,
              dynamic_settle=True, verbose=False)
r_n = decode_test(d_n)
print(f"  w_slow blocks: {np.mean(r_n['d_w_slow']):.4f} (target >0.80)")
print(f"  Fused MAE:     {mae(r_n['d_fused'], r_n['Y']):.4f}")

# Show w values during noisy blocks
Y_n = r_n['Y']
print(f"\n  Per-frequency:")
for f in sorted(set(Y_n)):
    m = Y_n == f
    if m.any():
        plvs = [np.max(d_n['plv_slow'][i]) for i in range(len(Y_n)) if m[i]]
        print(f"  {f:6.2f}  fused_MAE={mae(r_n['d_fused'][m], r_n['Y'][m]):.4f}  "
              f"w={np.mean(r_n['d_w_slow'][m]):.3f}  maxPLV={np.mean(plvs):.4f}")


# ── TEST 4: CURIOSITY ────────────────────────────────────────
print(f"\n{'='*72}")
print("  TEST 4: CURIOSITY DETECTION")
print(f"{'='*72}")
curiosity_freqs = [0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
c_sig, _ = make_blocks(curiosity_freqs, block_dur=30.)
np.random.seed(6)
d_c = run_sim(c_sig, total_time=stabilization_time+len(curiosity_freqs)*30.*2+10.,
              sweep_mode=False, dynamic_settle=False, verbose=False)
r_c = decode_test(d_c)
freq_changes = np.where(np.diff(r_c['Y']) != 0)[0] + 1
detected = len(r_c['change_events'])
block_false = int(np.sum(r_bl['novelty']))
print(f"  Transitions:      {len(freq_changes)}")
print(f"  CUSUM detections: {detected}")
print(f"  Detection rate:   {detected/max(1,len(freq_changes)):.0%}")
print(f"  Block false pos:  {block_false}")


# ── VERDICT ───────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  VERDICT")
print(f"{'='*72}")
detection_rate = detected / max(1, len(freq_changes))
results = [
    ('Block slow MAE',  mae(r_bl['ds'], r_bl['Y']),       '<', 0.008),
    ('Block fused MAE', mae(r_bl['d_fused'], r_bl['Y']),  '<', 0.008),
    ('w_slow blocks',   np.mean(r_bl['d_w_slow']),        '>', 0.80),
    ('w_slow sweep',    np.mean(r_sw['d_w_slow']),        '<', 0.30),
    ('CUSUM detection', detection_rate,                    '>', 0.80),
    ('Block false pos', float(block_false),                '<', 20.0),
    ('Noise fused',     mae(r_n['d_fused'], r_n['Y']),    '<', 0.050),
]
print(f"\n  {'Metric':20s}  {'Value':>8}  {'Target':>8}  {'OK':>4}")
print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*4}")
passed = 0
for name, val, op, target in results:
    ok = val < target if op == '<' else val > target
    if ok: passed += 1
    print(f"  {name:20s}  {val:8.4f}  {op}{target:<7}  {'✓' if ok else '✗':>4}")

print(f"\n  Passed: {passed}/{len(results)}")
print(f"  PLV thresholds: lo={PLV_THRESHOLD_LO}, hi={PLV_THRESHOLD_HI}")
print(f"  PLV window: {PLV_WINDOW} samples")
if passed == len(results):
    print("\n  ✓✓✓ ALL PASS — MAX PLV STABILITY CONFIRMED ✓✓✓")
print()
