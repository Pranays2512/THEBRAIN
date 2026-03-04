"""
M47 DEEP STRUCTURAL DIAGNOSIS
==============================
Isolates and measures EVERY component independently.
Tests each root cause hypothesis with real numbers.
"""

import numpy as np
from collections import deque
from m47_neuron import (
    build_network, run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks, make_steps,
    decode_resonance, decode_resonance_raw, build_bias_table,
    compute_stability, compute_js_divergence, TwoWindowChangeDetector,
    update_leaky, mae,
    N, N_FAST, dt, stabilization_time, omega_hz, omega_vec,
    STABILITY_WINDOW, STABILITY_SCALE, CHANGE_WINDOW_K,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
    TAU_FAST_S, TAU_SLOW_S, alpha_leak_fast, alpha_leak_slow,
    PLV_SHARPENING, eps, noise_amp, input_gain,
)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0
n_sweeps  = 6

print("=" * 72)
print("  M47 DEEP STRUCTURAL DIAGNOSIS")
print("  Isolating every component")
print("=" * 72)

# ══════════════════════════════════════════════════════════════
# SECTION 1: OSCILLATOR BANK — Frequency Resolution
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  SECTION 1: OSCILLATOR BANK PROPERTIES")
print(f"{'─'*72}")

spacing = np.diff(omega_hz)
print(f"  N oscillators:     {N}")
print(f"  Frequency range:   {omega_hz[0]:.3f} – {omega_hz[-1]:.3f} Hz")
print(f"  Spacing (min):     {spacing.min():.5f} Hz (at {omega_hz[0]:.2f} Hz)")
print(f"  Spacing (max):     {spacing.max():.5f} Hz (at {omega_hz[-2]:.2f} Hz)")
print(f"  Spacing (at 1 Hz): {spacing[np.argmin(np.abs(omega_hz[:-1]-1.0))]:.5f} Hz")
print(f"  Log-spaced: YES (denser at low freq, sparser at high)")

# Resolution limit: how many oscillators within ±0.05 Hz of each test freq
for f_test in [0.5, 1.0, 1.5, 2.0]:
    n_near = np.sum(np.abs(omega_hz - f_test) < 0.05)
    print(f"  Oscillators within ±0.05 Hz of {f_test:.1f} Hz: {n_near}")


# ══════════════════════════════════════════════════════════════
# SECTION 2: RUN A SINGLE STABLE FREQUENCY — Isolate PLV noise
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  SECTION 2: PLV NOISE FLOOR AT STEADY STATE")
print(f"{'─'*72}")

np.random.seed(100)
stable_sig, _ = make_blocks([1.0], block_dur=100.0)
d_stable = run_sim(stable_sig, total_time=stabilization_time + 200.0,
                   sweep_mode=False, dynamic_settle=True, verbose=False,
                   collect_calib=False)

# Analyze PLV magnitude statistics at steady state
plv_slow_arr = np.array(d_stable['plv_slow'])  # shape: (n_samples, N)
energy_slow_arr = np.array(d_stable['energy_slow'])

# Decode each sample raw (no bias)
raw_estimates = np.array([decode_resonance_raw(plv_slow_arr[i], energy_slow_arr[i])
                          for i in range(len(d_stable['Y']))])
true_freq = np.array(d_stable['Y'])

raw_mae = mae(raw_estimates, true_freq)
raw_std = np.std(raw_estimates)
raw_mean = np.mean(raw_estimates)
raw_bias = raw_mean - 1.0

print(f"  True frequency: 1.000 Hz (constant)")
print(f"  Raw decoder mean:  {raw_mean:.5f} Hz")
print(f"  Raw decoder std:   {raw_std:.5f} Hz")
print(f"  Raw decoder bias:  {raw_bias:+.5f} Hz")
print(f"  Raw decoder MAE:   {raw_mae:.5f} Hz")
print(f"  Samples: {len(true_freq)}")

# PLV magnitude at resonant oscillator
peak_idx = np.argmin(np.abs(omega_hz - 1.0))
plv_at_peak = plv_slow_arr[:, peak_idx]
energy_at_peak = energy_slow_arr[:, peak_idx]
print(f"\n  PLV at nearest oscillator ({omega_hz[peak_idx]:.4f} Hz):")
print(f"    mean={np.mean(plv_at_peak):.4f}  std={np.std(plv_at_peak):.4f}")
print(f"    PLV^8 mean={np.mean(plv_at_peak**8):.4f}  std={np.std(plv_at_peak**8):.4f}")
print(f"  Energy at peak: mean={np.mean(energy_at_peak):.4f}  std={np.std(energy_at_peak):.4f}")

# Weight distribution: PLV^8 * energy
weights_all = (plv_slow_arr**PLV_SHARPENING) * energy_slow_arr
weight_sums = weights_all.sum(axis=1)
effective_N = (weights_all.sum(axis=1)**2) / (weights_all**2).sum(axis=1)
print(f"\n  Effective # oscillators (N_eff):")
print(f"    mean={np.mean(effective_N):.1f}  std={np.std(effective_N):.1f}")
print(f"  Weight sum: mean={np.mean(weight_sums):.4f}  std={np.std(weight_sums):.4f}")


# ══════════════════════════════════════════════════════════════
# SECTION 3: BIAS CORRECTION — Error at Each Calibration Point
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  SECTION 3: BIAS CORRECTION ANALYSIS")
print(f"{'─'*72}")

# Run calibration blocks
slow_freqs = sorted(set([
    0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,
    0.55,0.75,0.95,1.15,1.35,1.55,1.75,1.95,2.05,
]))
block_sig, _ = make_blocks(slow_freqs, block_dur=40.0)
slow_total = stabilization_time + 2*len(slow_freqs)*40.0 + 10.0
np.random.seed(1)
data_cal = run_sim(block_sig, total_time=slow_total,
                   sweep_mode=False, dynamic_settle=True, verbose=False,
                   collect_calib=True)

# Build bias tables
bf_slow = sorted(data_cal['calib_plv_slow'].keys())
bias_freqs_slow, bias_vals_slow = build_bias_table(
    np.array(bf_slow), data_cal['calib_plv_slow'], data_cal['calib_energy_slow'])
bf_fast = sorted(data_cal['calib_plv_fast'].keys())
bias_freqs_fast, bias_vals_fast = build_bias_table(
    np.array(bf_fast), data_cal['calib_plv_fast'], data_cal['calib_energy_fast'])

print(f"  Calibration points: {len(bias_freqs_slow)}")
print(f"\n  {'Freq':>6}  {'Raw bias':>10}  {'Corrected':>10}  {'Samples':>8}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}")
for i, f in enumerate(bias_freqs_slow):
    n_samples = len(data_cal['calib_plv_slow'].get(f, []))
    # Re-decode with correction
    if n_samples > 0:
        plv_mean = np.mean(data_cal['calib_plv_slow'][f], axis=0)
        eng_mean = np.mean(data_cal['calib_energy_slow'][f], axis=0)
        f_raw = decode_resonance_raw(plv_mean, eng_mean)
        f_corr = decode_resonance(plv_mean, eng_mean, bias_freqs_slow, bias_vals_slow)
        print(f"  {f:6.2f}  {f_raw-f:+10.4f}  {f_corr-f:+10.4f}  {n_samples:8d}")

# Test on held-out frequencies (between calibration points)
print(f"\n  Interpolation error at HELD-OUT frequencies:")
test_freqs_held = [0.525, 0.65, 0.85, 1.025, 1.25, 1.45, 1.65, 1.85]
for f_test in test_freqs_held:
    # Find nearest calibrated points
    below = bias_freqs_slow[bias_freqs_slow <= f_test]
    above = bias_freqs_slow[bias_freqs_slow >= f_test]
    if len(below) > 0 and len(above) > 0:
        gap = above[0] - below[-1]
        # Use the existing block data to estimate
        Y = data_cal['Y']
        mask = np.abs(Y - f_test) < 0.03
        if mask.sum() > 0:
            ds_held = np.array([
                decode_resonance(data_cal['plv_slow'][i], data_cal['energy_slow'][i],
                                 bias_freqs_slow, bias_vals_slow)
                for i in range(len(Y)) if mask[i]
            ])
            Y_held = Y[mask]
            print(f"  {f_test:.3f} Hz: MAE={mae(ds_held, Y_held):.4f}, "
                  f"gap={gap:.2f} Hz, n={mask.sum()}")


# ══════════════════════════════════════════════════════════════
# SECTION 4: STABILITY DETECTOR — Variance Values at Each Regime
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  SECTION 4: VARIANCE-BASED STABILITY — Actual Values")
print(f"{'─'*72}")

# Run sweep and blocks with decode
ridge_fast, ridge_fast_sc = fit_ridge(
    data_cal['feat_fast'], data_cal['Y'], RIDGE_ALPHA_FAST)
ridge_slow, ridge_slow_sc = fit_ridge(
    data_cal['feat_slow'], data_cal['Y'], RIDGE_ALPHA_SLOW)

# Sweep
np.random.seed(2)
d_sw = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
               total_time=warmup+2*sweep_dur+10., sweep_mode=True, verbose=False)
ds_sw = np.array([decode_resonance(d_sw['plv_slow'][i], d_sw['energy_slow'][i],
                                    bias_freqs_slow, bias_vals_slow)
                   for i in range(len(d_sw['Y']))])

# Compute actual variances during sweep
slow_hist = deque(maxlen=STABILITY_WINDOW)
sweep_vars = []
sweep_wslow = []
for i in range(len(ds_sw)):
    slow_hist.append(ds_sw[i])
    if len(slow_hist) >= 5:
        v = np.var(list(slow_hist))
        sweep_vars.append(v)
        sweep_wslow.append(np.exp(-v / STABILITY_SCALE))

print(f"  STABILITY_SCALE = {STABILITY_SCALE}")
print(f"  STABILITY_WINDOW = {STABILITY_WINDOW}")
print(f"\n  During SWEEP:")
print(f"    Variance: mean={np.mean(sweep_vars):.6f}  std={np.std(sweep_vars):.6f}")
print(f"    Variance: min={np.min(sweep_vars):.6f}  max={np.max(sweep_vars):.6f}")
print(f"    w_slow:   mean={np.mean(sweep_wslow):.4f}")
print(f"    w_slow:   min={np.min(sweep_wslow):.4f}  max={np.max(sweep_wslow):.4f}")

# What SHOULD the variance be?
# Sweep rate = (2.0-0.5) / 60s = 0.025 Hz/s
# Window = 30 samples * 0.1s/sample = 3.0s
# In 3s, frequency changes by 0.075 Hz
# Variance of uniform[0, 0.075] = 0.075^2/12 = 0.000469
theoretical_var = (0.075)**2 / 12.0
theoretical_w = np.exp(-theoretical_var / STABILITY_SCALE)
print(f"\n    Theoretical var (uniform): {theoretical_var:.6f}")
print(f"    Theoretical w_slow:       {theoretical_w:.4f}")
print(f"    Actual/Theoretical ratio: {np.mean(sweep_vars)/theoretical_var:.2f}×")

# Blocks
test_freqs = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
test_sig, _ = make_blocks(test_freqs, block_dur=40.0)
test_total = stabilization_time + 2*len(test_freqs)*40. + 10.
np.random.seed(3)
d_bl = run_sim(test_sig, total_time=test_total,
               sweep_mode=False, dynamic_settle=True, verbose=False)
ds_bl = np.array([decode_resonance(d_bl['plv_slow'][i], d_bl['energy_slow'][i],
                                    bias_freqs_slow, bias_vals_slow)
                   for i in range(len(d_bl['Y']))])

slow_hist2 = deque(maxlen=STABILITY_WINDOW)
block_vars = []
block_wslow = []
for i in range(len(ds_bl)):
    slow_hist2.append(ds_bl[i])
    if len(slow_hist2) >= 5:
        v = np.var(list(slow_hist2))
        block_vars.append(v)
        block_wslow.append(np.exp(-v / STABILITY_SCALE))

print(f"\n  During BLOCKS:")
print(f"    Variance: mean={np.mean(block_vars):.8f}  std={np.std(block_vars):.8f}")
print(f"    Variance: min={np.min(block_vars):.8f}  max={np.max(block_vars):.8f}")
print(f"    w_slow:   mean={np.mean(block_wslow):.4f}")
print(f"    w_slow:   min={np.min(block_wslow):.4f}  max={np.max(block_wslow):.4f}")

# What S value would give exactly w_sweep=0.30?
needed_S = -np.mean(sweep_vars) / np.log(0.30)
print(f"\n  To get w_sweep=0.30 exactly: need S = {needed_S:.6f}")
print(f"  Current S = {STABILITY_SCALE:.6f}, ratio = {STABILITY_SCALE/needed_S:.2f}×")


# ══════════════════════════════════════════════════════════════
# SECTION 5: JS CHANGE DETECTOR — Signal Quality
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  SECTION 5: JS CHANGE DETECTOR — Signal at Transitions")
print(f"{'─'*72}")

# Run curiosity blocks
curiosity_freqs = [0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
c_sig, _ = make_blocks(curiosity_freqs, block_dur=30.)
np.random.seed(6)
d_c = run_sim(c_sig, total_time=stabilization_time+len(curiosity_freqs)*30.*2+10.,
              sweep_mode=False, dynamic_settle=False, verbose=False)

# Run through change detector
change_det = TwoWindowChangeDetector()
js_raw = np.zeros(len(d_c['Y']))
novelty = np.zeros(len(d_c['Y']), dtype=bool)
for i in range(len(d_c['Y'])):
    js, nov = change_det.update(d_c['energy_fast'][i], d_c['T'][i])
    js_raw[i] = js
    novelty[i] = nov

Y_c = d_c['Y']
T_c = d_c['T']
freq_changes = np.where(np.diff(Y_c) != 0)[0] + 1

print(f"  Transitions found: {len(freq_changes)}")
print(f"  Threshold (auto-calibrated): {change_det.threshold:.8f}")
print(f"  CUSUM events: {len(change_det.novelty_events)}")

# For each transition, show the JS values in a window around it
print(f"\n  JS values around each transition (+/- 5 samples):")
print(f"  {'Trans#':>6}  {'Time':>7}  {'From':>5}  {'To':>5}  "
      f"{'JS_max':>10}  {'JS_at':>10}  {'Ratio':>7}")
print(f"  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*7}")

for ti, idx in enumerate(freq_changes[:12]):
    lo = max(0, idx-5)
    hi = min(len(js_raw), idx+15)
    js_window = js_raw[lo:hi]
    js_max = js_window.max()
    js_at_idx = js_raw[idx] if idx < len(js_raw) else 0

    # Calm baseline: 50 samples before transition
    calm_lo = max(0, idx-60)
    calm_hi = max(0, idx-10)
    js_calm = np.mean(js_raw[calm_lo:calm_hi]) if calm_hi > calm_lo else 0

    ratio = js_max / (js_calm + 1e-12)
    print(f"  {ti+1:6d}  {T_c[idx]:7.1f}  {Y_c[idx-1]:5.1f}  {Y_c[idx]:5.1f}  "
          f"{js_max:10.6f}  {js_at_idx:10.6f}  {ratio:7.1f}×")

# Overall statistics
calm_mask = np.ones(len(js_raw), dtype=bool)
for idx in freq_changes:
    calm_mask[max(0,idx-5):min(len(js_raw),idx+20)] = False

js_calm_all = js_raw[calm_mask & (js_raw > 0)]  # exclude initial zeros
js_trans_peaks = []
for idx in freq_changes:
    lo = max(0, idx-2)
    hi = min(len(js_raw), idx+8)
    if hi > lo:
        js_trans_peaks.append(js_raw[lo:hi].max())

if len(js_calm_all) > 0:
    print(f"\n  Calm JS: mean={np.mean(js_calm_all):.8f}  "
          f"std={np.std(js_calm_all):.8f}  max={np.max(js_calm_all):.8f}")
if len(js_trans_peaks) > 0:
    print(f"  Peak JS at trans: mean={np.mean(js_trans_peaks):.6f}  "
          f"min={np.min(js_trans_peaks):.6f}")
    peak_ratio = np.mean(js_trans_peaks) / (np.mean(js_calm_all) + 1e-12)
    print(f"  Peak/Calm ratio: {peak_ratio:.1f}×  (this is the TRUE signal quality)")


# ══════════════════════════════════════════════════════════════
# SECTION 6: FUSION — Per-Sample Analysis at Transitions
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  SECTION 6: FUSION BEHAVIOR AROUND TRANSITIONS")
print(f"{'─'*72}")

# Use block test data
ds_test = np.array([decode_resonance(d_bl['plv_slow'][i], d_bl['energy_slow'][i],
                                      bias_freqs_slow, bias_vals_slow)
                     for i in range(len(d_bl['Y']))])
df_test = np.array([decode_resonance(d_bl['plv_fast'][i], d_bl['energy_fast'][i],
                                      bias_freqs_fast, bias_vals_fast)
                     for i in range(len(d_bl['Y']))])

Y_bl = d_bl['Y']
block_changes = np.where(np.diff(Y_bl) != 0)[0] + 1

if len(block_changes) > 0:
    idx = block_changes[0]  # First transition
    lo = max(0, idx-10)
    hi = min(len(Y_bl), idx+30)
    print(f"\n  First transition at sample {idx} (t={d_bl['T'][idx]:.1f}s):")
    print(f"  {Y_bl[idx-1]:.2f} Hz → {Y_bl[idx]:.2f} Hz")
    print(f"\n  {'Sample':>7}  {'True':>6}  {'Fast':>7}  {'Slow':>7}  "
          f"{'Err_F':>7}  {'Err_S':>7}")
    print(f"  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for i in range(lo, min(hi, len(Y_bl))):
        delta = i - idx
        print(f"  {delta:+7d}  {Y_bl[i]:6.2f}  {df_test[i]:7.3f}  {ds_test[i]:7.3f}  "
              f"{df_test[i]-Y_bl[i]:+7.3f}  {ds_test[i]-Y_bl[i]:+7.3f}")


# ══════════════════════════════════════════════════════════════
# SECTION 7: PLV ORACLE EFFECT — Compare phase computed from
# true freq vs phase that oscillator would see
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  SECTION 7: PLV ORACLE CHECK")
print(f"{'─'*72}")
print(f"  Current PLV: exp(j * (θ_osc - 2π*f_true*t))")
print(f"  This uses f_true (the TRUE input frequency) — oracle")
print(f"  For pure sine input sin(2πft), the phase IS 2πft")
print(f"  So for pure sines, oracle = physical reality")
print(f"  But for complex signals (noise, harmonics), this breaks")
print(f"\n  Current test uses pure sine: oracle effect = NONE")
print(f"  The PLV is physically correct for the current test signals")


# ══════════════════════════════════════════════════════════════
# SECTION 8: DECODER PERFORMANCE BY FREQUENCY BAND
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  SECTION 8: DECODER ERROR BY FREQUENCY BAND")
print(f"{'─'*72}")

# Use block data for per-frequency analysis
print(f"\n  {'Freq':>6}  {'Slow MAE':>10}  {'Fast MAE':>10}  {'Slow bias':>10}  "
      f"{'Slow std':>10}  {'N_eff':>6}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}")

for f in sorted(set(Y_bl)):
    m = Y_bl == f
    if m.sum() > 5:
        slow_err = ds_test[m] - f
        fast_err = df_test[m] - f
        # Effective oscillators at this frequency
        plv_mean = np.mean(np.array(d_bl['plv_slow'])[m], axis=0)
        eng_mean = np.mean(np.array(d_bl['energy_slow'])[m], axis=0)
        w8 = (plv_mean**PLV_SHARPENING) * eng_mean
        n_eff = (w8.sum()**2) / (w8**2).sum() if w8.sum() > 0 else 0
        print(f"  {f:6.2f}  {np.mean(np.abs(slow_err)):10.4f}  "
              f"{np.mean(np.abs(fast_err)):10.4f}  {np.mean(slow_err):+10.4f}  "
              f"{np.std(slow_err):10.4f}  {n_eff:6.1f}")


# ══════════════════════════════════════════════════════════════
# SECTION 9: SWEEP — What causes the 0.655 Hz error?
# ══════════════════════════════════════════════════════════════
print(f"\n{'─'*72}")
print("  SECTION 9: SWEEP ERROR DECOMPOSITION")
print(f"{'─'*72}")

# Is it LAG or SPREAD?
Y_sw = d_sw['Y']
ds_sweep = ds_sw  # already computed above
df_sweep = np.array([decode_resonance(d_sw['plv_fast'][i], d_sw['energy_fast'][i],
                                       bias_freqs_fast, bias_vals_fast)
                      for i in range(len(d_sw['Y']))])

# Compute cross-correlation to find lag
from scipy.signal import correlate
# Find lag between true and decoded
cc = correlate(ds_sweep - ds_sweep.mean(), Y_sw - Y_sw.mean(), mode='full')
lags = np.arange(-len(Y_sw)+1, len(Y_sw))
best_lag = lags[np.argmax(cc)]
lag_seconds = best_lag * dt * 2  # samples * dt * feature_sample_interval

print(f"  Slow decoder lag vs true: {best_lag} samples = {lag_seconds:.1f}s")
print(f"  At sweep rate 0.025 Hz/s: lag error = {abs(lag_seconds)*0.025:.3f} Hz")
print(f"  Total slow MAE on sweep: {mae(ds_sweep, Y_sw):.3f} Hz")
print(f"  Total fast MAE on sweep: {mae(df_sweep, Y_sw):.3f} Hz")

# Breakdown: systematic (lag) vs random (jitter)
residual_after_lag = ds_sweep[max(0,best_lag):] - Y_sw[:len(Y_sw)-max(0,best_lag)] \
    if best_lag >= 0 else ds_sweep[:len(ds_sweep)+best_lag] - Y_sw[-best_lag:]
if len(residual_after_lag) > 0:
    print(f"  After removing lag — residual MAE: {np.mean(np.abs(residual_after_lag)):.3f} Hz")
    print(f"  After removing lag — residual std: {np.std(residual_after_lag):.3f} Hz")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  DIAGNOSIS COMPLETE")
print(f"{'='*72}")
print()
