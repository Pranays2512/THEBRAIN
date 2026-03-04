"""
M48 FULL FLOW TRACE — Every sample through the sweep pipeline
==============================================================
No guessing. Trace every value:
  1. True frequency (Y)
  2. Raw decoder output (f_raw)
  3. Reverse-lookup corrected output (f_corrected)
  4. Sliding window variance (var30)
  5. w_slow = exp(-var30 / S)
  6. Time

Print EVERY sample so we can see exactly where w stays high and why.
"""

import numpy as np
from collections import deque
from m48_neuron import (
    build_network, run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks,
    decode_resonance, decode_resonance_raw, build_reverse_lookup,
    mae, N, dt, stabilization_time,
    STABILITY_WINDOW,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
    PLV_SHARPENING, eps, omega_hz,
)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0

# Use S=0.005409 (from first-2-sweep calibration)
S = 0.005409

print("=" * 72)
print("  M48 FULL FLOW TRACE — SWEEP PIPELINE")
print(f"  S = {S:.6f}")
print("=" * 72)

# ── Build calibration (same as m48 main) ──────────────────────
print("\n  Building calibration...")
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

print(f"  Reverse lookup: {len(raw_x_slow)} pts")
print(f"  f_raw range: [{raw_x_slow[0]:.4f}, {raw_x_slow[-1]:.4f}]")
print(f"  f_true range: [{true_y_slow[0]:.4f}, {true_y_slow[-1]:.4f}]")

# Print the full lookup table
print(f"\n  REVERSE LOOKUP TABLE:")
print(f"  {'f_raw':>8}  {'f_true':>8}  {'bias':>8}")
print(f"  {'─'*8}  {'─'*8}  {'─'*8}")
for i in range(len(raw_x_slow)):
    print(f"  {raw_x_slow[i]:8.4f}  {true_y_slow[i]:8.4f}  "
          f"{raw_x_slow[i]-true_y_slow[i]:+8.4f}")

# ── Run test sweep (seed=2, 2 sweeps) ────────────────────────
print(f"\n  Running test sweep (2 sweeps, seed=2)...")
np.random.seed(2)
d = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
            total_time=warmup+2*sweep_dur+10.,
            sweep_mode=True, verbose=False)

Y = d['Y']
T = d['T']
n = len(Y)

# Decode each sample
f_raw_arr = np.zeros(n)
f_corrected_arr = np.zeros(n)
for i in range(n):
    plv = d['plv_slow'][i]
    eng = d['energy_slow'][i]
    f_raw_arr[i] = decode_resonance_raw(plv, eng)
    f_corrected_arr[i] = decode_resonance(plv, eng, raw_x_slow, true_y_slow)

# Compute variance and w for each sample
var_arr = np.zeros(n)
w_arr = np.zeros(n)
slow_hist = deque(maxlen=STABILITY_WINDOW)
for i in range(n):
    slow_hist.append(f_corrected_arr[i])
    if len(slow_hist) >= 5:
        var_arr[i] = np.var(list(slow_hist))
        w_arr[i] = np.exp(-var_arr[i] / S)
    else:
        w_arr[i] = 0.0

# ── Print every 5th sample ───────────────────────────────────
print(f"\n{'─'*100}")
print(f"  SAMPLE-BY-SAMPLE TRACE (every 5th sample)")
print(f"{'─'*100}")
print(f"  {'Idx':>5}  {'Time':>7}  {'True_f':>7}  {'f_raw':>8}  {'f_corr':>8}  "
      f"{'raw_err':>8}  {'cor_err':>8}  {'var30':>12}  {'w':>6}")
print(f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  "
      f"{'─'*8}  {'─'*8}  {'─'*12}  {'─'*6}")
for i in range(0, n, 5):
    print(f"  {i:5d}  {T[i]:7.1f}  {Y[i]:7.3f}  {f_raw_arr[i]:8.4f}  {f_corrected_arr[i]:8.4f}  "
          f"{f_raw_arr[i]-Y[i]:+8.4f}  {f_corrected_arr[i]-Y[i]:+8.4f}  "
          f"{var_arr[i]:12.8f}  {w_arr[i]:6.3f}")

# ── Statistics by frequency band ─────────────────────────────
print(f"\n{'─'*72}")
print("  w DISTRIBUTION BY FREQUENCY BAND (test sweep)")
print(f"{'─'*72}")
print(f"  {'Band':>12}  {'N':>5}  {'w_mean':>7}  {'w>0.5':>6}  {'w>0.8':>6}  "
      f"{'var_mean':>12}  {'cor_mae':>8}")
print(f"  {'─'*12}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*12}  {'─'*8}")
for blo, bhi in [(0.50,0.65),(0.65,0.80),(0.80,0.95),(0.95,1.10),
                  (1.10,1.25),(1.25,1.40),(1.40,1.55),(1.55,1.70),
                  (1.70,1.85),(1.85,2.00)]:
    m = (Y >= blo) & (Y < bhi) & (np.arange(n) >= STABILITY_WINDOW)
    if m.sum() > 3:
        print(f"  {blo:.2f}-{bhi:.2f} Hz  {m.sum():5d}  {np.mean(w_arr[m]):7.3f}  "
              f"{np.sum(w_arr[m]>0.5):6d}  {np.sum(w_arr[m]>0.8):6d}  "
              f"{np.mean(var_arr[m]):12.8f}  {mae(f_corrected_arr[m], Y[m]):8.4f}")

# ── Histogram of w values ────────────────────────────────────
print(f"\n{'─'*72}")
print("  w VALUE HISTOGRAM (test sweep, after warmup)")
print(f"{'─'*72}")
w_valid = w_arr[STABILITY_WINDOW:]
bins = [(0,0.05),(0.05,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.5),
        (0.5,0.7),(0.7,0.9),(0.9,0.95),(0.95,0.99),(0.99,1.001)]
for lo, hi in bins:
    count = np.sum((w_valid >= lo) & (w_valid < hi))
    pct = 100*count/len(w_valid)
    bar = '█' * int(pct)
    print(f"  {lo:5.2f}-{hi:5.3f}: {count:5d} ({pct:5.1f}%) {bar}")

# Overall
print(f"\n  Total samples: {len(w_valid)}")
print(f"  w_mean: {np.mean(w_valid):.4f}")
print(f"  w_median: {np.median(w_valid):.4f}")
print(f"  Samples with w > 0.80: {np.sum(w_valid > 0.80)} ({100*np.sum(w_valid>0.80)/len(w_valid):.1f}%)")
print(f"  Samples with w > 0.95: {np.sum(w_valid > 0.95)} ({100*np.sum(w_valid>0.95)/len(w_valid):.1f}%)")
print(f"  Samples with w < 0.10: {np.sum(w_valid < 0.10)} ({100*np.sum(w_valid<0.10)/len(w_valid):.1f}%)")

# ── What's happening at turnaround? ──────────────────────────
print(f"\n{'─'*72}")
print("  TURNAROUND ANALYSIS")
print(f"{'─'*72}")
# Find where sweep reverses direction
dY = np.diff(Y)
reversals = np.where(np.diff(np.sign(dY)) != 0)[0] + 1
print(f"  Sweep reversals at indices: {reversals}")
for rev in reversals:
    lo = max(STABILITY_WINDOW, rev-15)
    hi = min(n, rev+15)
    print(f"\n  Around reversal at idx={rev}, t={T[rev]:.1f}s, Y={Y[rev]:.3f} Hz:")
    print(f"    {'Offset':>7}  {'True':>7}  {'Corrected':>9}  {'Var':>12}  {'w':>6}")
    for i in range(lo, hi, 2):
        print(f"    {i-rev:+7d}  {Y[i]:7.3f}  {f_corrected_arr[i]:9.4f}  "
              f"{var_arr[i]:12.8f}  {w_arr[i]:6.3f}")

print(f"\n{'='*72}")
print("  TRACE COMPLETE")
print(f"{'='*72}")
print()
