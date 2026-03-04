"""
M48 STABILITY SIGNAL SEARCH
============================
The slow decoder outputs CONSTANT during sweeps → variance=0 → w=1.0.
We need a DIFFERENT signal that separates sweep from block.

Test candidates:
  A. Variance of FAST decoder output (tau=1s, should vary more)
  B. |fast - slow| disagreement
  C. Energy concentration (peak/total — peaked=block, spread=sweep)
  D. Max PLV (high during blocks, low during sweeps)

For each candidate:
  - Measure during sweep and blocks
  - Show the actual values sample by sample
  - Show the separation between regimes
"""

import numpy as np
from collections import deque
from m48_neuron import (
    run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks,
    decode_resonance, decode_resonance_raw, build_reverse_lookup,
    mae, N, dt, stabilization_time,
    STABILITY_WINDOW,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
    PLV_SHARPENING, omega_hz,
)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0

print("=" * 72)
print("  M48 — STABILITY SIGNAL SEARCH")
print("=" * 72)

# ── Calibrate ─────────────────────────────────────────────────
print("\n  [Cal] Building reverse lookup...")
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
print(f"  Done: {len(raw_x_slow)} pts")


# ── Run test sweep ────────────────────────────────────────────
print("\n  [Sweep] Running test sweep...")
np.random.seed(2)
d_sw = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
               total_time=warmup+2*sweep_dur+10.,
               sweep_mode=True, verbose=False)

# ── Run test blocks ───────────────────────────────────────────
print("\n  [Blocks] Running test blocks...")
test_freqs = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
test_sig, _ = make_blocks(test_freqs, block_dur=40.0)
test_total = stabilization_time + 2*len(test_freqs)*40. + 10.
np.random.seed(3)
d_bl = run_sim(test_sig, total_time=test_total,
               sweep_mode=False, dynamic_settle=True, verbose=False)


# ── Compute ALL candidate signals for sweep and blocks ────────
def compute_signals(data, label):
    Y = data['Y']; T = data['T']; n = len(Y)

    # Decode
    df = np.array([decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                                     raw_x_fast, true_y_fast) for i in range(n)])
    ds = np.array([decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                                     raw_x_slow, true_y_slow) for i in range(n)])
    plv_slow = np.array(data['plv_slow'])
    plv_fast = np.array(data['plv_fast'])
    energy_slow = np.array(data['energy_slow'])
    energy_fast = np.array(data['energy_fast'])

    # Candidate A: variance of fast decoder output
    var_fast = np.zeros(n)
    fast_hist = deque(maxlen=STABILITY_WINDOW)
    for i in range(n):
        fast_hist.append(df[i])
        if len(fast_hist) >= 5:
            var_fast[i] = np.var(list(fast_hist))

    # Candidate B: |fast - slow| disagreement
    disagree = np.abs(df - ds)

    # Candidate C: energy concentration (Gini-like)
    # N_eff = (sum e)^2 / sum(e^2) — effective number of oscillators
    concentration = np.zeros(n)
    for i in range(n):
        e = energy_slow[i]
        e_sum = e.sum()
        e_sum2 = (e**2).sum()
        if e_sum2 > 0:
            concentration[i] = e_sum**2 / (e_sum2 * N)  # normalized 0-1

    # Candidate D: max PLV (slow)
    max_plv_slow = np.max(plv_slow, axis=1)
    max_plv_fast = np.max(plv_fast, axis=1)

    # Candidate E: variance of slow decoder (current method — for reference)
    var_slow = np.zeros(n)
    slow_hist = deque(maxlen=STABILITY_WINDOW)
    for i in range(n):
        slow_hist.append(ds[i])
        if len(slow_hist) >= 5:
            var_slow[i] = np.var(list(slow_hist))

    return {
        'Y': Y, 'T': T, 'df': df, 'ds': ds,
        'var_fast': var_fast, 'var_slow': var_slow,
        'disagree': disagree,
        'concentration': concentration,
        'max_plv_slow': max_plv_slow, 'max_plv_fast': max_plv_fast,
    }

print("\n  Computing signals...")
sw = compute_signals(d_sw, "sweep")
bl = compute_signals(d_bl, "blocks")


# ── Show statistics per candidate ─────────────────────────────
print(f"\n{'='*72}")
print("  CANDIDATE SIGNALS — SWEEP vs BLOCK")
print(f"{'='*72}")

candidates = [
    ('A: var(fast decoder)',  'var_fast',      'high=sweep, low=block'),
    ('B: |fast - slow|',     'disagree',      'high=sweep, low=block'),
    ('C: energy conc.',      'concentration', 'low=sweep, high=block'),
    ('D: max PLV (slow)',    'max_plv_slow',  'low=sweep, high=block'),
    ('E: var(slow decoder)', 'var_slow',      'high=sweep, low=block (CURRENT)'),
]

print(f"\n  {'Candidate':30s}  {'Sweep mean':>11}  {'Block mean':>11}  "
      f"{'Ratio':>7}  {'Gap':>10}  {'Overlap?':>9}")
print(f"  {'─'*30}  {'─'*11}  {'─'*11}  {'─'*7}  {'─'*10}  {'─'*9}")

for label, key, desc in candidates:
    sw_vals = sw[key][STABILITY_WINDOW:]
    bl_vals = bl[key][STABILITY_WINDOW:]
    sw_mean = np.mean(sw_vals)
    bl_mean = np.mean(bl_vals)
    ratio = sw_mean / (bl_mean + 1e-12) if sw_mean > bl_mean else bl_mean / (sw_mean + 1e-12)

    # Check overlap: does the signal separate the two regimes?
    sw_p25, sw_p75 = np.percentile(sw_vals, [25, 75])
    bl_p25, bl_p75 = np.percentile(bl_vals, [25, 75])
    # Gap between IQR ranges
    if sw_mean > bl_mean:
        gap = sw_p25 - bl_p75  # positive = no overlap
    else:
        gap = bl_p25 - sw_p75
    overlap = "YES" if gap < 0 else "NO"

    print(f"  {label:30s}  {sw_mean:11.6f}  {bl_mean:11.6f}  "
          f"{ratio:7.1f}×  {gap:+10.6f}  {overlap:>9}")

# ── Detailed per-candidate analysis ───────────────────────────
for label, key, desc in candidates:
    print(f"\n{'─'*72}")
    print(f"  {label} — {desc}")
    print(f"{'─'*72}")

    sw_vals = sw[key][STABILITY_WINDOW:]
    bl_vals = bl[key][STABILITY_WINDOW:]

    print(f"  {'Stat':15s}  {'Sweep':>12}  {'Blocks':>12}")
    print(f"  {'─'*15}  {'─'*12}  {'─'*12}")
    for stat_name, func in [('mean', np.mean), ('median', np.median),
                             ('std', np.std), ('min', np.min), ('max', np.max),
                             ('p10', lambda x: np.percentile(x, 10)),
                             ('p25', lambda x: np.percentile(x, 25)),
                             ('p75', lambda x: np.percentile(x, 75)),
                             ('p90', lambda x: np.percentile(x, 90))]:
        print(f"  {stat_name:15s}  {func(sw_vals):12.6f}  {func(bl_vals):12.6f}")

    # If this signal looks good, show what w would look like
    if key in ['var_fast', 'disagree']:
        # w = exp(-signal / S), need S to give w_sweep ≈ 0.10
        sw_med = np.median(sw_vals[sw_vals > 0]) if np.any(sw_vals > 0) else 0.001
        if sw_med > 0:
            S_test = -sw_med / np.log(0.10)
            w_sweep = np.mean(np.exp(-sw_vals / S_test))
            w_block = np.mean(np.exp(-bl_vals / S_test))
            print(f"\n  If S={S_test:.6f} (target w_sweep_median=0.10):")
            print(f"    w_sweep_mean = {w_sweep:.4f}")
            print(f"    w_block_mean = {w_block:.4f}")
    elif key in ['concentration', 'max_plv_slow']:
        # w = signal directly or clipped
        # For concentration: w = (conc - conc_sweep) / (conc_block - conc_sweep)
        sw_med = np.median(sw_vals)
        bl_med = np.median(bl_vals)
        if bl_med > sw_med:
            # Linear: w = (val - sw_p25) / (bl_p75 - sw_p25)
            lo = np.percentile(sw_vals, 25)
            hi = np.percentile(bl_vals, 75)
            if hi > lo:
                w_sweep = np.mean(np.clip((sw_vals - lo) / (hi - lo), 0, 1))
                w_block = np.mean(np.clip((bl_vals - lo) / (hi - lo), 0, 1))
                print(f"\n  If w = linear scale [{lo:.4f}, {hi:.4f}]:")
                print(f"    w_sweep_mean = {w_sweep:.4f}")
                print(f"    w_block_mean = {w_block:.4f}")

print(f"\n{'='*72}")
print("  SEARCH COMPLETE")
print(f"{'='*72}")
print()
