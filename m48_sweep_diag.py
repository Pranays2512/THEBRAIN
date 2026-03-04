"""
M48 SWEEP DIAGNOSTIC — Why does auto-S fail?
=============================================
Question: Auto-S calibrated S=0.002368 from training sweep (var_mean=0.0045),
predicted w=0.15. But test sweep gave w=0.573, meaning test variance ≈ 0.0013.

This script traces EXACTLY:
  1. What the training sweep decoder output looks like (sample by sample)
  2. What the test sweep decoder output looks like (sample by sample)
  3. The variance distributions in both
  4. WHY they differ
"""

import numpy as np
from collections import deque
from m48_neuron import (
    build_network, run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks,
    decode_resonance, decode_resonance_raw, build_reverse_lookup,
    compute_stability, TwoWindowChangeDetector,
    mae, N, dt, stabilization_time,
    STABILITY_WINDOW,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
    TAU_FAST_S, TAU_SLOW_S,
)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0

print("=" * 72)
print("  M48 SWEEP DIAGNOSTIC")
print("=" * 72)

# ─── Step 1: Run training sweep (same as m48 main) ─────────
print("\n  [1] Training sweep (6 sweeps, seed=0)...")
np.random.seed(0)
data_train = run_sim(
    make_sweep(0.5, 2.0, 6, sweep_dur),
    total_time=warmup + 6*sweep_dur + 10.0,
    sweep_mode=True, verbose=False,
    collect_calib=False
)
print(f"  Training samples: {len(data_train['Y'])}")
print(f"  Time range: {data_train['T'][0]:.1f} - {data_train['T'][-1]:.1f}s")

# ─── Step 2: Run block calibration (same as m48 main) ──────
print("\n  [2] Block calibration (35 freqs, seed=1)...")
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

# Build reverse lookup
raw_x_slow, true_y_slow = build_reverse_lookup(
    sorted(data_slow['calib_plv_slow'].keys()),
    data_slow['calib_plv_slow'], data_slow['calib_energy_slow'])
print(f"  Reverse lookup: {len(raw_x_slow)} pts")

# ─── Step 3: Decode TRAINING sweep with reverse lookup ──────
print("\n  [3] Decoding TRAINING sweep with reverse lookup...")
ds_train = np.array([
    decode_resonance(data_train['plv_slow'][i], data_train['energy_slow'][i],
                     raw_x_slow, true_y_slow)
    for i in range(len(data_train['Y']))
])
Y_train = data_train['Y']
T_train = data_train['T']

# ─── Step 4: Run TEST sweep (same seed as m48 main test) ────
print("\n  [4] Running TEST sweep (2 sweeps, seed=2)...")
np.random.seed(2)
d_test = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
                 total_time=warmup+2*sweep_dur+10.,
                 sweep_mode=True, verbose=False)

ds_test = np.array([
    decode_resonance(d_test['plv_slow'][i], d_test['energy_slow'][i],
                     raw_x_slow, true_y_slow)
    for i in range(len(d_test['Y']))
])
Y_test = d_test['Y']
T_test = d_test['T']

# ─── Step 5: Compare variance distributions ─────────────────
print(f"\n{'─'*72}")
print("  VARIANCE COMPARISON: TRAINING vs TEST sweep")
print(f"{'─'*72}")

def compute_variances(ds, label):
    """Compute variance in sliding windows."""
    vars_list = []
    w_list = []
    for i in range(STABILITY_WINDOW, len(ds)):
        v = np.var(ds[i-STABILITY_WINDOW:i])
        vars_list.append(v)
        # Compute w at S=0.002368 (the auto-calibrated value)
        w_list.append(np.exp(-v / 0.002368))
    return np.array(vars_list), np.array(w_list)

train_vars, train_w = compute_variances(ds_train, "TRAIN")
test_vars, test_w = compute_variances(ds_test, "TEST")

print(f"\n  {'Statistic':25s}  {'Training':>10}  {'Test':>10}")
print(f"  {'─'*25}  {'─'*10}  {'─'*10}")
print(f"  {'Samples':25s}  {len(train_vars):10d}  {len(test_vars):10d}")
print(f"  {'Variance mean':25s}  {np.mean(train_vars):10.6f}  {np.mean(test_vars):10.6f}")
print(f"  {'Variance median':25s}  {np.median(train_vars):10.6f}  {np.median(test_vars):10.6f}")
print(f"  {'Variance std':25s}  {np.std(train_vars):10.6f}  {np.std(test_vars):10.6f}")
print(f"  {'Variance min':25s}  {np.min(train_vars):10.6f}  {np.min(test_vars):10.6f}")
print(f"  {'Variance max':25s}  {np.max(train_vars):10.6f}  {np.max(test_vars):10.6f}")
print(f"  {'Variance p25':25s}  {np.percentile(train_vars,25):10.6f}  {np.percentile(test_vars,25):10.6f}")
print(f"  {'Variance p75':25s}  {np.percentile(train_vars,75):10.6f}  {np.percentile(test_vars,75):10.6f}")
print(f"  {'w mean (S=0.002368)':25s}  {np.mean(train_w):10.4f}  {np.mean(test_w):10.4f}")
print(f"  {'w median':25s}  {np.median(train_w):10.4f}  {np.median(test_w):10.4f}")

# ─── Step 6: Breakdown by sweep phase ───────────────────────
print(f"\n{'─'*72}")
print("  VARIANCE BY SWEEP PHASE (training sweep)")
print(f"{'─'*72}")

# Training sweep: warmup ends, then 6 sweeps of 60s each
# Warmup = stabilization_time + 10 = 70s
# Sweep starts at ~70s
# Sample rate ≈ 1/(dt * feature_sample_interval) = 1/(0.05*2) = 10 Hz

for sweep_idx in range(min(6, int((T_train[-1]-warmup)/sweep_dur))):
    t_start = warmup + sweep_idx * sweep_dur
    t_end = t_start + sweep_dur
    mask = (T_train >= t_start) & (T_train < t_end)
    if mask.sum() < STABILITY_WINDOW + 5:
        continue
    
    ds_this = ds_train[mask]
    Y_this = Y_train[mask]
    
    # Variance within this sweep
    v_list = []
    for i in range(STABILITY_WINDOW, len(ds_this)):
        v_list.append(np.var(ds_this[i-STABILITY_WINDOW:i]))
    v_arr = np.array(v_list)
    
    # Decoder accuracy on this sweep
    the_mae = mae(ds_this, Y_this)
    
    direction = "UP" if sweep_idx % 2 == 0 else "DOWN"
    freq_range = f"{Y_this[0]:.2f}-{Y_this[-1]:.2f}"
    
    print(f"  Sweep {sweep_idx+1} ({direction:4s}) {freq_range:>12s}: "
          f"var_mean={np.mean(v_arr):.6f}  var_max={np.max(v_arr):.6f}  "
          f"MAE={the_mae:.4f}")

# ─── Step 7: Same for test sweep ────────────────────────────
print(f"\n{'─'*72}")
print("  VARIANCE BY SWEEP PHASE (test sweep)")
print(f"{'─'*72}")

for sweep_idx in range(min(2, int((T_test[-1]-warmup)/sweep_dur))):
    t_start = warmup + sweep_idx * sweep_dur
    t_end = t_start + sweep_dur
    mask = (T_test >= t_start) & (T_test < t_end)
    if mask.sum() < STABILITY_WINDOW + 5:
        continue
    
    ds_this = ds_test[mask]
    Y_this = Y_test[mask]
    
    v_list = []
    for i in range(STABILITY_WINDOW, len(ds_this)):
        v_list.append(np.var(ds_this[i-STABILITY_WINDOW:i]))
    v_arr = np.array(v_list)
    
    the_mae = mae(ds_this, Y_this)
    direction = "UP" if sweep_idx % 2 == 0 else "DOWN"
    freq_range = f"{Y_this[0]:.2f}-{Y_this[-1]:.2f}"
    
    print(f"  Sweep {sweep_idx+1} ({direction:4s}) {freq_range:>12s}: "
          f"var_mean={np.mean(v_arr):.6f}  var_max={np.max(v_arr):.6f}  "
          f"MAE={the_mae:.4f}")

# ─── Step 8: Per-frequency-band variance in training sweep ──
print(f"\n{'─'*72}")
print("  DECODER OUTPUT BY FREQUENCY BAND (training sweep)")
print(f"{'─'*72}")

print(f"  {'Band':>12}  {'Decoded mean':>12}  {'True mean':>10}  {'Error':>8}  {'Std':>8}  {'N':>5}")
print(f"  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*5}")
for blo, bhi in [(0.50,0.65),(0.65,0.80),(0.80,0.95),(0.95,1.10),
                  (1.10,1.25),(1.25,1.40),(1.40,1.55),(1.55,1.70),
                  (1.70,1.85),(1.85,2.00)]:
    m = (Y_train >= blo) & (Y_train < bhi)
    if m.sum() > 3:
        err = ds_train[m] - Y_train[m]
        print(f"  {blo:.2f}-{bhi:.2f} Hz  {np.mean(ds_train[m]):12.4f}  "
              f"{np.mean(Y_train[m]):10.4f}  {np.mean(err):+8.4f}  "
              f"{np.std(ds_train[m]):8.4f}  {m.sum():5d}")

# ─── Step 9: Sample-by-sample around the high-variance zone ─
print(f"\n{'─'*72}")
print("  SAMPLE-BY-SAMPLE: First 50 samples of training sweep")
print(f"{'─'*72}")
print(f"  {'Idx':>5}  {'Time':>7}  {'True':>6}  {'Decoded':>8}  {'Error':>8}  {'Var30':>10}  {'w':>6}")
print(f"  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*6}")
slow_hist = deque(maxlen=STABILITY_WINDOW)
for i in range(min(100, len(ds_train))):
    slow_hist.append(ds_train[i])
    if len(slow_hist) >= 5:
        v = np.var(list(slow_hist))
        w = np.exp(-v / 0.002368) if v > 0 else 1.0
    else:
        v = 0
        w = 0
    if i < 50 or (i >= 200 and i < 230):
        print(f"  {i:5d}  {T_train[i]:7.1f}  {Y_train[i]:6.3f}  {ds_train[i]:8.4f}  "
              f"{ds_train[i]-Y_train[i]:+8.4f}  {v:10.8f}  {w:6.3f}")

# ─── Step 10: What S value would give w_test=0.20? ──────────
print(f"\n{'─'*72}")
print("  WHAT S VALUE WOULD WORK?")
print(f"{'─'*72}")

for target_w in [0.30, 0.20, 0.10, 0.05]:
    needed_S = -np.mean(test_vars) / np.log(target_w)
    # Check what w_block would be
    # Block test at seed=3
    np.random.seed(3)
    test_freqs = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
    test_sig, _ = make_blocks(test_freqs, block_dur=40.0)
    test_total = stabilization_time + 2*len(test_freqs)*40. + 10.
    d_bl = run_sim(test_sig, total_time=test_total,
                   sweep_mode=False, dynamic_settle=True, verbose=False)
    ds_bl = np.array([decode_resonance(d_bl['plv_slow'][i], d_bl['energy_slow'][i],
                                        raw_x_slow, true_y_slow)
                       for i in range(len(d_bl['Y']))])
    bl_vars = []
    for i in range(STABILITY_WINDOW, len(ds_bl)):
        bl_vars.append(np.var(ds_bl[i-STABILITY_WINDOW:i]))
    w_block = np.mean([np.exp(-v / needed_S) for v in bl_vars])
    print(f"  Target w_sweep={target_w:.2f}: need S={needed_S:.6f}, w_block={w_block:.3f}")
    break  # Only need one block run (expensive)

# Also compute for different target_w using test_vars directly
test_var_mean = np.mean(test_vars)
for target_w in [0.30, 0.20, 0.10, 0.05]:
    needed_S = -test_var_mean / np.log(target_w)
    # Estimate w_block from bl_vars
    w_block_est = np.mean([np.exp(-v / needed_S) for v in bl_vars])
    print(f"  Target w_sweep={target_w:.2f}: need S={needed_S:.6f}, "
          f"w_block≈{w_block_est:.3f}")

print(f"\n{'='*72}")
print("  DIAGNOSIS COMPLETE")
print(f"{'='*72}")
print()
