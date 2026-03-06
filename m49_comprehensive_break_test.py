"""
M49 COMPREHENSIVE BREAK TEST
==============================
The original four-hole break test, updated with two v3 fixes:

  FIX 1 (Hole 2): MIN_SETTLE_S 8 → 20
    make_blocks_fixed replaces make_blocks everywhere.
    Ensures plv_slow ≥ 0.982 before any sample is collected.

  FIX 2 (Hole 4): DivergenceCUSUM replaces TwoWindowChangeDetector
    threshold=0.020 Hz (fixed, from known noise floor)
    debounce=150 samples (15s, covers ds settling time)

Everything else — hole definitions, pass criteria, seeds, structure — 
is identical to the original comprehensive break test.
"""

import numpy as np
from collections import deque

from m49_neuron import (
    run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks,
    decode_resonance, decode_resonance_raw, build_reverse_lookup,
    compute_stability_plv,
    mae, dt, stabilization_time,
    PLV_STAB_WINDOW,
    PLV_THRESHOLD_LO, PLV_THRESHOLD_HI,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
    SETTLE_CYCLES,
)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0

SLOW_FREQS_CAL = sorted(set([
    0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87,
    0.9, 0.92, 0.95, 0.97, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.35, 1.4,
    1.5, 1.55, 1.6, 1.7, 1.75, 1.8, 1.9, 1.95, 2.0, 2.05, 2.1,
]))


# ── FIX 1: make_blocks with MIN_SETTLE=20 ────────────────────────────
MIN_SETTLE_FIXED = 20.0

def make_blocks_fixed(freqs, block_dur=40.0, noise_level=0.0):
    """make_blocks with MIN_SETTLE_S=20 instead of 8."""
    settle_times = {f: max(MIN_SETTLE_FIXED, SETTLE_CYCLES / f) for f in freqs}
    def sig(t):
        block_idx     = int(t / block_dur) % len(freqs)
        block_t0      = int(t / block_dur) * block_dur
        f             = freqs[block_idx]
        time_in_block = t - block_t0
        sig._settled  = (time_in_block >= settle_times[f])
        I = np.sin(2 * np.pi * f * t)
        if noise_level > 0:
            I += noise_level * np.random.randn()
        return I, f, f
    sig._settled = False
    return sig, settle_times


# ── FIX 2: DivergenceCUSUM ───────────────────────────────────────────
DIVERG_THRESHOLD = 0.020   # Hz — fixed from known noise floor (0.0057 Hz × 3.5)
DIVERG_DEBOUNCE  = 150     # samples = 15s — covers ds settling after any transition
DIVERG_RESET_PAT = 15


class DivergenceCUSUM:
    """
    Detects frequency transitions via |df - ds| + CUSUM.
    df (tau=1s) moves fast; ds (tau=5s) holds old value.
    Peak divergence ∝ step size → monotonic detection.
    Fixed threshold immune to startup-transient contamination.
    Long debounce prevents re-firing while ds catches up.
    """
    def __init__(self):
        self.threshold      = DIVERG_THRESHOLD
        self.accumulator    = 0.0
        self.calm_count     = 0
        self.debounce_count = 0
        self.novelty_events = []
        self.divergence_log = []

    def update(self, df, ds, t):
        divergence = abs(df - ds)
        self.divergence_log.append(divergence)

        if self.debounce_count > 0:
            self.debounce_count -= 1
            return divergence, False

        if divergence > self.threshold:
            self.accumulator += divergence - self.threshold
            self.calm_count   = 0
        else:
            self.calm_count  += 1
            if self.calm_count >= DIVERG_RESET_PAT:
                self.accumulator = 0.0

        is_novel = self.accumulator > self.threshold
        if is_novel:
            self.novelty_events.append((t, divergence, self.accumulator))
            self.accumulator    = 0.0
            self.debounce_count = DIVERG_DEBOUNCE

        return divergence, is_novel


# ── Calibration helper ───────────────────────────────────────────────
def calibrate(cal_seed_sweep, cal_seed_block, label=""):
    print(f"\n  [Cal{label}] Sweep seed={cal_seed_sweep}, block seed={cal_seed_block}")

    np.random.seed(cal_seed_sweep)
    data_train = run_sim(
        make_sweep(0.5, 2.0, 6, sweep_dur),
        total_time=warmup + 6*sweep_dur + 10.0,
        sweep_mode=True, verbose=False, collect_calib=False)
    ridge_fast, ridge_fast_sc = fit_ridge(
        data_train['feat_fast'], data_train['Y'], RIDGE_ALPHA_FAST)

    # FIX 1: calibration blocks use MIN_SETTLE=20
    np.random.seed(cal_seed_block)
    block_sig, _ = make_blocks_fixed(SLOW_FREQS_CAL, block_dur=40.0)
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

    print(f"  Done: {len(raw_x_slow)} lookup pts, "
          f"f_raw [{raw_x_slow[0]:.3f}, {raw_x_slow[-1]:.3f}]")
    return (ridge_fast, ridge_fast_sc,
            ridge_slow, ridge_slow_sc,
            raw_x_slow, true_y_slow,
            raw_x_fast, true_y_fast)


# ── Decode + fuse pipeline ───────────────────────────────────────────
def decode_test(data, ridge_fast, ridge_fast_sc,
                ridge_slow, ridge_slow_sc,
                raw_x_slow, true_y_slow,
                raw_x_fast, true_y_fast):
    Y = data['Y']; T = data['T']; n = len(Y)

    df = np.array([decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                                     raw_x_fast, true_y_fast) for i in range(n)])
    ds = np.array([decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                                     raw_x_slow, true_y_slow) for i in range(n)])

    # FIX 2: DivergenceCUSUM on |df - ds|
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

    rf = predict_ridge(data['feat_fast'], ridge_fast, ridge_fast_sc)
    rs = predict_ridge(data['feat_slow'], ridge_slow, ridge_slow_sc)

    return {
        'df': df, 'ds': ds, 'd_fused': d_fused, 'd_w_slow': d_w_slow,
        'rf': rf, 'rs': rs,
        'divergence': divergence, 'novelty': novelty,
        'change_events': change_det.novelty_events,
        'threshold': change_det.threshold,
        'Y': Y, 'T': T,
    }


# ═══════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("  M49 COMPREHENSIVE BREAK TEST")
print("  Four holes — four verdicts — all must pass")
print(f"  MIN_SETTLE={MIN_SETTLE_FIXED}s  |  DivergenceCUSUM thr={DIVERG_THRESHOLD}  deb={DIVERG_DEBOUNCE}")
print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# HOLE 1 — CALIBRATION GENERALIZATION
# Seeds {0,1} vs {17,23}, tested on seed 103.
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 1 — CALIBRATION GENERALIZATION")
print("  Does a fresh-seed calibration perform as well as the original?")
print(f"{'='*72}")

cal_orig  = calibrate(cal_seed_sweep=0,  cal_seed_block=1,  label=" ORIG (seeds 0,1)")
cal_fresh = calibrate(cal_seed_sweep=17, cal_seed_block=23, label=" FRESH (seeds 17,23)")

# FIX 1: test signal also uses MIN_SETTLE=20
test_freqs_h1 = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
test_sig_h1, _ = make_blocks_fixed(test_freqs_h1, block_dur=40.0)
test_total_h1  = stabilization_time + 2*len(test_freqs_h1)*40.0 + 10.0

np.random.seed(103)
d_h1 = run_sim(test_sig_h1, total_time=test_total_h1,
               sweep_mode=False, dynamic_settle=True, verbose=False)

r_orig  = decode_test(d_h1, *cal_orig)
r_fresh = decode_test(d_h1, *cal_fresh)

print(f"\n  {'Metric':22s}  {'Orig (0,1)':>12}  {'Fresh (17,23)':>14}  {'Δ':>8}  {'OK':>4}")
print(f"  {'─'*22}  {'─'*12}  {'─'*14}  {'─'*8}  {'─'*4}")

h1_metrics = [
    ('Slow MAE',   mae(r_orig['ds'],      d_h1['Y']), mae(r_fresh['ds'],      d_h1['Y'])),
    ('Fused MAE',  mae(r_orig['d_fused'], d_h1['Y']), mae(r_fresh['d_fused'], d_h1['Y'])),
    ('w_slow mean',np.mean(r_orig['d_w_slow']),        np.mean(r_fresh['d_w_slow'])),
]
h1_pass = True
for name, v_orig, v_fresh in h1_metrics:
    delta = abs(v_fresh - v_orig)
    if 'MAE' in name:
        ok = v_fresh < 0.010 and delta < 0.005
    else:
        ok = v_fresh > 0.75 and delta < 0.10
    if not ok: h1_pass = False
    print(f"  {name:22s}  {v_orig:12.4f}  {v_fresh:14.4f}  {delta:8.4f}  {'✓' if ok else '✗':>4}")

print(f"\n  Per-frequency (fresh cal):")
print(f"  {'Freq':>6}  {'Orig slow':>10}  {'Fresh slow':>11}  {'Δ':>8}")
Y_h1 = d_h1['Y']
for f in sorted(set(Y_h1)):
    m = Y_h1 == f
    if m.any():
        v_o = mae(r_orig['ds'][m],  Y_h1[m])
        v_f = mae(r_fresh['ds'][m], Y_h1[m])
        flag = " ← FAIL" if v_f > 0.010 else ""
        print(f"  {f:6.2f}  {v_o:10.4f}  {v_f:11.4f}  {abs(v_f-v_o):8.4f}{flag}")

print(f"\n  HOLE 1 {'✓ PASS' if h1_pass else '✗ FAIL'}")


# ═══════════════════════════════════════════════════════════════════════════
# HOLE 2 — TARGETED 0.7–1.0 Hz BAND STRESS TEST
# Per-frequency MAE < 0.008 Hz — not just the global average.
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 2 — TARGETED 0.7–1.0 Hz BAND STRESS TEST")
print("  Per-frequency MAE < 0.008 Hz for every point in the trouble zone")
print(f"{'='*72}")

trouble_freqs = [0.70, 0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.00]
# FIX 1: test signal uses MIN_SETTLE=20
test_sig_h2, _ = make_blocks_fixed(trouble_freqs, block_dur=40.0)
test_total_h2  = stabilization_time + 2*len(trouble_freqs)*40.0 + 10.0

np.random.seed(104)
d_h2 = run_sim(test_sig_h2, total_time=test_total_h2,
               sweep_mode=False, dynamic_settle=True, verbose=False)

r_h2 = decode_test(d_h2, *cal_orig)

print(f"\n  {'Freq':>6}  {'Slow mean':>10}  {'Slow MAE':>10}  {'Fused MAE':>10}  "
      f"{'w_slow':>7}  {'OK':>4}")
print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*7}  {'─'*4}")

Y_h2 = d_h2['Y']
h2_pass = True
h2_worst_mae = 0.0; h2_worst_freq = None
for f in sorted(set(Y_h2)):
    m = Y_h2 == f
    if m.any():
        slow_mae  = mae(r_h2['ds'][m],      Y_h2[m])
        fused_mae = mae(r_h2['d_fused'][m], Y_h2[m])
        w_mean    = np.mean(r_h2['d_w_slow'][m])
        slow_mean = np.mean(r_h2['ds'][m])
        ok = slow_mae < 0.008
        if not ok: h2_pass = False
        if slow_mae > h2_worst_mae:
            h2_worst_mae = slow_mae; h2_worst_freq = f
        print(f"  {f:6.2f}  {slow_mean:10.4f}  {slow_mae:10.4f}  {fused_mae:10.4f}  "
              f"{w_mean:7.3f}  {'✓' if ok else '✗':>4}")

print(f"\n  Global band MAE: {mae(r_h2['ds'], Y_h2):.4f}")
print(f"  Worst frequency: {h2_worst_freq} Hz  (MAE={h2_worst_mae:.4f})")
print(f"\n  HOLE 2 {'✓ PASS' if h2_pass else '✗ FAIL'}")


# ═══════════════════════════════════════════════════════════════════════════
# HOLE 3 — PLV SIGNAL QUALITY UNDER NOISE
# Direct measurement of PLV gap at σ=0,1,2,3.
# Gap ≥ 0.50, leakage < 20% on each side.
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 3 — PLV SIGNAL QUALITY UNDER NOISE")
print("  Direct measurement of PLV gap (block vs sweep) at each noise level")
print(f"{'='*72}")

print(f"\n  {'σ':>4}  {'Block PLV':>10}  {'Sweep PLV':>10}  {'Gap':>8}  "
      f"{'Blk <HI%':>9}  {'Swp >LO%':>9}  {'OK':>4}")
print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*4}")

h3_pass = True
for nl in [0.0, 1.0, 2.0, 3.0]:
    # Blocks: use make_blocks_fixed so PLV is fully settled
    np.random.seed(500 + int(nl*10))
    ns, _ = make_blocks_fixed([0.5, 1.0, 1.5, 2.0], block_dur=40.0, noise_level=nl)
    d_blk = run_sim(ns, total_time=500.0, sweep_mode=False,
                    dynamic_settle=True, verbose=False)

    # Sweep: noise is added on top of the sweep signal (no settle change needed)
    np.random.seed(510 + int(nl*10))
    def noisy_sweep(f_start, f_end, n_sw, sw_dur, nl=nl):
        base = make_sweep(f_start, f_end, n_sw, sw_dur)
        def sig(t):
            I, f, freq = base(t)
            return I + nl * np.random.randn(), f, freq
        sig._settled = True
        return sig

    d_swp = run_sim(noisy_sweep(0.5, 2.0, 2, sweep_dur),
                    total_time=warmup + 2*sweep_dur + 10.0,
                    sweep_mode=True, verbose=False)

    blk_plv = np.array([np.max(d_blk['plv_slow'][i]) for i in range(len(d_blk['Y']))])
    swp_plv = np.array([np.max(d_swp['plv_slow'][i]) for i in range(len(d_swp['Y']))])

    blk_mean = np.mean(blk_plv); swp_mean = np.mean(swp_plv)
    gap      = blk_mean - swp_mean
    blk_leakage_pct = np.mean(blk_plv < PLV_THRESHOLD_HI) * 100
    swp_leakage_pct = np.mean(swp_plv > PLV_THRESHOLD_LO) * 100

    ok = gap >= 0.50 and blk_leakage_pct < 20.0 and swp_leakage_pct < 20.0
    if not ok: h3_pass = False
    print(f"  {nl:4.1f}  {blk_mean:10.4f}  {swp_mean:10.4f}  {gap:8.4f}  "
          f"{blk_leakage_pct:8.1f}%  {swp_leakage_pct:8.1f}%  {'✓' if ok else '✗':>4}")

print(f"\n  Thresholds: LO={PLV_THRESHOLD_LO}, HI={PLV_THRESHOLD_HI}")
print(f"  Pass criteria: gap ≥ 0.50, block leakage < 20%, sweep leakage < 20%")
print(f"\n  HOLE 3 {'✓ PASS' if h3_pass else '✗ FAIL'}")


# ═══════════════════════════════════════════════════════════════════════════
# HOLE 4 — CUSUM RESOLUTION FLOOR (SMALL TRANSITIONS)
# Steps 0.30, 0.20, 0.15, 0.10, 0.05 Hz inside 0.70–1.00 Hz.
# ≥ 0.15 Hz: detection rate ≥ 80%, false positives < 5.
# < 0.15 Hz: false positives < 5 only (below reliable floor).
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  HOLE 4 — CUSUM RESOLUTION FLOOR")
print("  Smallest detectable frequency step inside 0.70–1.00 Hz")
print(f"{'='*72}")

base_freq    = 0.80
step_sizes   = [0.30, 0.20, 0.15, 0.10, 0.05]
block_dur_h4 = 30.0
repeats_h4   = 6

print(f"\n  Base: {base_freq} Hz, block_dur: {block_dur_h4}s, "
      f"debounce: {DIVERG_DEBOUNCE} samples ({DIVERG_DEBOUNCE*dt*2:.0f}s)")
print(f"\n  {'Step':>6}  {'Target':>9}  {'Trans':>6}  "
      f"{'Det':>6}  {'Rate':>7}  {'FalsePos':>9}  {'OK':>4}")
print(f"  {'─'*6}  {'─'*9}  {'─'*6}  "
      f"{'─'*6}  {'─'*7}  {'─'*9}  {'─'*4}")

h4_pass = True; resolution_floor = None

for step in step_sizes:
    target_freq   = round(base_freq + step, 3)
    step_freqs_h4 = [base_freq, target_freq] * (repeats_h4 // 2)

    np.random.seed(600 + int(step * 100))
    # FIX 1: use make_blocks_fixed
    c_sig, _ = make_blocks_fixed(step_freqs_h4, block_dur=block_dur_h4)
    total_h4  = stabilization_time + len(step_freqs_h4)*block_dur_h4*2 + 10.0
    d_h4 = run_sim(c_sig, total_time=total_h4,
                   sweep_mode=False, dynamic_settle=False, verbose=False)

    r_h4 = decode_test(d_h4, *cal_orig)

    Y_h4     = r_h4['Y']
    expected = len(np.where(np.diff(Y_h4) != 0)[0])
    detected = len(r_h4['change_events'])
    rate     = detected / max(1, expected)

    # False positives: novelty outside transition + debounce window
    trans_idx  = np.where(np.diff(Y_h4) != 0)[0] + 1
    near_trans = np.zeros(len(Y_h4), dtype=bool)
    for idx in trans_idx:
        near_trans[max(0, idx-5) : min(len(Y_h4), idx + DIVERG_DEBOUNCE + 10)] = True
    false_pos = int(np.sum(r_h4['novelty'] & ~near_trans))

    if step >= 0.15:
        ok = rate >= 0.80 and false_pos < 5
        if not ok: h4_pass = False
    else:
        ok = false_pos < 5   # sub-floor: just no false alarms

    if rate >= 0.80: resolution_floor = step

    print(f"  {step:6.2f}  {target_freq:9.3f}  {expected:6d}  "
          f"{detected:6d}  {rate:7.0%}  {false_pos:9d}  {'✓' if ok else '✗':>4}")

    # Show divergence trace at first transition
    if len(trans_idx) > 0:
        idx0   = trans_idx[0]
        lo, hi = max(0, idx0-3), min(len(Y_h4), idx0+10)
        trace  = [f"{r_h4['divergence'][i]:.3f}" for i in range(lo, hi)]
        print(f"         |df-ds|: [{', '.join(trace)}]  thr={DIVERG_THRESHOLD:.3f}")

floor_str = (f"≥{resolution_floor:.2f} Hz (≥80% detection)"
             if resolution_floor else "no step reached 80%")
print(f"\n  Resolution floor: {floor_str}")
print(f"\n  HOLE 4 {'✓ PASS' if h4_pass else '✗ FAIL'}")


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("  M49 COMPREHENSIVE BREAK TEST — FINAL SUMMARY")
print(f"{'='*72}")

all_holes = [
    ("HOLE 1 — Calibration generalization",   h1_pass),
    ("HOLE 2 — 0.7–1.0 Hz per-frequency MAE", h2_pass),
    ("HOLE 3 — PLV signal quality under noise",h3_pass),
    ("HOLE 4 — Divergence CUSUM floor",        h4_pass),
]

print(f"\n  {'Test':44s}  {'Result':>8}")
print(f"  {'─'*44}  {'─'*8}")
overall_pass = True
for name, passed in all_holes:
    if not passed: overall_pass = False
    print(f"  {name:44s}  {'✓ PASS' if passed else '✗ FAIL':>8}")

print(f"\n  {'─'*54}")
print(f"  {'OVERALL':44s}  {'✓ PASS' if overall_pass else '✗ FAIL':>8}")

if overall_pass:
    print("""
  ✓✓✓ ALL FOUR HOLES PASS ✓✓✓

  M49 is solid. Safe to fold these two changes into m48_neuron.py:

    1. MIN_SETTLE_S = 8.0  →  20.0   (one line)

    2. Replace TwoWindowChangeDetector with DivergenceCUSUM.
       In decode_test(): call change_det.update(df[i], ds[i], T[i])
       Parameters: threshold=0.020 Hz, debounce=150 samples

  Then run the original m48_break_test.py multi-seed suite
  to confirm stability across seeds before building further.
""")
else:
    print("""
  ✗ One or more holes remain open.
  Do NOT build further layers until fixed.
  See per-hole output above for failure details.
""")