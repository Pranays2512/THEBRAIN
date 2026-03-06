"""
M48 PATCH v3 — HOLE 4 TARGETED FIX
=====================================
Hole 2 is already fixed (MIN_SETTLE_S=20). Do not change that.

Hole 4 has exactly two bugs in DivergenceCUSUM, identified from data:

BUG 1 — Auto-calibration threshold is contaminated by startup transients
  The calibration window collects the first 200 samples of each test run.
  Each run starts at stabilization_time=60s with dynamic_settle=False.
  At t=60s the network just exited stabilization — plv_slow has not yet
  locked to the first block frequency. df and ds are both noisy/drifting.
  |df-ds| during these first 200 samples is large (startup divergence),
  not steady-state noise. This inflates the threshold to 5–29x the
  actual noise floor (0.0057 Hz from Hole 2):
    step=0.20: thr calibrated to 0.1658 Hz, signal peak ~0.16 Hz → misses 50%
    step=0.30: thr calibrated to 0.1549 Hz, fires correctly but barely
  This is not a signal problem — the divergence signal is correct.
  The threshold is wrong because it was measured from the wrong data.

  FIX: use a fixed threshold derived from the known noise floor.
  Noise floor from Hole 2 settled data = 0.0057 Hz.
  Fixed threshold = 0.020 Hz (3.5× noise floor, clear margin).
  This is safe to hardcode because the noise floor is set by
  MIN_SETTLE_S=20, which is now a fixed architectural parameter.

BUG 2 — Debounce (2.5s) is much shorter than |df-ds| settling time (15s)
  After a frequency transition, |df-ds| does NOT return to zero quickly.
  ds has tau_slow=5s. After a step of Δf:
    |df-ds|(t) ≈ Δf × exp(-t / tau_slow)
  Time to drop below threshold=0.020 Hz:
    Δf=0.15: t = 5×ln(0.15/0.020) = 10.2s = 102 samples
    Δf=0.20: t = 5×ln(0.20/0.020) = 11.5s = 115 samples
    Δf=0.30: t = 5×ln(0.30/0.020) = 13.5s = 135 samples
  Current debounce = 25 samples = 2.5s.
  After the correct first fire + 2.5s debounce, |df-ds| is STILL
  well above 0.020 Hz. CUSUM immediately re-accumulates and fires again.
  This repeats ~6 times per transition → all counted as false positives.
  The 0.15 Hz trace confirmed this: divergence stayed at 0.28–0.30 Hz
  for the entire 18-sample trace after the transition (well above threshold).

  FIX: debounce = 150 samples (15s).
  This covers the worst case (Δf=0.30: settle time ≈ 135 samples).
  With block_dur=30s = 300 samples: debounce fits inside one block,
  and consecutive transitions (300 samples apart) are each detected once.

NOTHING ELSE CHANGES from patch v2 (MIN_SETTLE_S=20, DivergenceCUSUM concept).
"""

import numpy as np
from collections import deque

from m48_neuron import (
    run_sim, fit_ridge, predict_ridge,
    make_sweep, make_blocks,
    decode_resonance_raw, build_reverse_lookup,
    compute_stability_plv,
    mae, dt, stabilization_time,
    PLV_STAB_WINDOW,
    RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
    SETTLE_CYCLES,
)

warmup    = stabilization_time + 10.0
sweep_dur = 60.0

# ── Fix 1 (Hole 2, carried forward from v2) ──────────────────────────
MIN_SETTLE_FIXED = 20.0

def make_blocks_fixed(freqs, block_dur=40.0, noise_level=0.0):
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


# ── Fix 2 (Hole 4) — DivergenceCUSUM with correct threshold + debounce ──

# Fixed threshold: 0.020 Hz
#   Derived from Hole 2 settled noise floor = 0.0057 Hz.
#   3.5× margin gives clean separation from steady-state |df-ds| noise.
#   Safe to hardcode because noise floor is determined by MIN_SETTLE=20.
DIVERG_THRESHOLD = 0.020   # Hz  (was: auto-calibrated from startup noise)

# Fixed debounce: 150 samples = 15s
#   Covers ds settling time for all step sizes up to 0.30 Hz.
#   After a true detection, suppresses re-firing until |df-ds| has
#   genuinely returned near zero.
#   Works for block_dur ≥ 20s (300 samples > 150 debounce).
DIVERG_DEBOUNCE  = 150     # samples  (was: 25 samples = 2.5s)

# CUSUM reset patience: unchanged
DIVERG_RESET_PAT = 15


class DivergenceCUSUM:
    """
    Frequency transition detector using |df - ds| + CUSUM.

    df (tau=1s) responds in ~1–3s. ds (tau=5s) holds for ~5–15s.
    |df - ds| rises proportionally to step size at every transition,
    then decays exponentially as ds catches up.

    Fixed threshold (0.020 Hz): set from known noise floor, not runtime
    calibration. Immune to startup transients.

    Long debounce (150 samples = 15s): prevents re-firing while ds
    is still catching up to the new frequency after a detected transition.
    """

    def __init__(self,
                 threshold     = DIVERG_THRESHOLD,
                 debounce      = DIVERG_DEBOUNCE,
                 reset_patience= DIVERG_RESET_PAT):
        self.threshold      = threshold
        self.debounce       = debounce
        self.reset_patience = reset_patience

        self.accumulator    = 0.0
        self.calm_count     = 0
        self.debounce_count = 0
        self.novelty_events = []
        self.divergence_log = []

    def update(self, df, ds, t):
        """
        Args:
            df: scalar decoded frequency from fast stream
            ds: scalar decoded frequency from slow stream
            t:  current time in seconds
        Returns:
            (divergence, is_novel)
        """
        divergence = abs(df - ds)
        self.divergence_log.append(divergence)

        if self.debounce_count > 0:
            self.debounce_count -= 1
            return divergence, False

        # CUSUM
        if divergence > self.threshold:
            self.accumulator += divergence - self.threshold
            self.calm_count   = 0
        else:
            self.calm_count  += 1
            if self.calm_count >= self.reset_patience:
                self.accumulator = 0.0

        is_novel = self.accumulator > self.threshold

        if is_novel:
            self.novelty_events.append((t, divergence, self.accumulator))
            self.accumulator    = 0.0
            self.debounce_count = self.debounce

        return divergence, is_novel


# ── Calibration ──────────────────────────────────────────────────────

SLOW_FREQS_CAL = sorted(set([
    0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87,
    0.9, 0.92, 0.95, 0.97, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.35, 1.4,
    1.5, 1.55, 1.6, 1.7, 1.75, 1.8, 1.9, 1.95, 2.0, 2.05, 2.1,
]))


def calibrate_v3(cal_seed_sweep=0, cal_seed_block=1, label=""):
    print(f"\n  [Cal{label}] sweep={cal_seed_sweep}, block={cal_seed_block}, "
          f"MIN_SETTLE={MIN_SETTLE_FIXED}s")

    np.random.seed(cal_seed_sweep)
    data_train = run_sim(
        make_sweep(0.5, 2.0, 6, sweep_dur),
        total_time=warmup + 6*sweep_dur + 10.0,
        sweep_mode=True, verbose=False, collect_calib=False)
    ridge_fast, ridge_fast_sc = fit_ridge(
        data_train['feat_fast'], data_train['Y'], RIDGE_ALPHA_FAST)

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

    print(f"  Done: {len(raw_x_slow)} lookup pts")
    return (ridge_fast, ridge_fast_sc,
            ridge_slow, ridge_slow_sc,
            raw_x_slow, true_y_slow,
            raw_x_fast, true_y_fast)


def decode_resonance(plv_leaky, energy_leaky, raw_x, true_y):
    f_raw = decode_resonance_raw(plv_leaky, energy_leaky)
    return float(np.interp(f_raw, raw_x, true_y,
                            left=true_y[0], right=true_y[-1]))


def decode_test_v3(data, ridge_fast, ridge_fast_sc,
                   ridge_slow, ridge_slow_sc,
                   raw_x_slow, true_y_slow,
                   raw_x_fast, true_y_fast):
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
    d_fused  = np.zeros(n)
    d_w_slow = np.zeros(n)
    for i in range(n):
        max_plv_s = float(np.max(data['plv_slow'][i]))
        plv_hist.append(max_plv_s)
        w = compute_stability_plv(plv_hist)
        if novelty[i]: w = 0.0
        d_fused[i]  = w * ds[i] + (1.0 - w) * df[i]
        d_w_slow[i] = w

    rf = predict_ridge(data['feat_fast'], ridge_fast, ridge_fast_sc)
    rs = predict_ridge(data['feat_slow'], ridge_slow, ridge_slow_sc)

    return {
        'df': df, 'ds': ds, 'd_fused': d_fused, 'd_w_slow': d_w_slow,
        'rf': rf, 'rs': rs,
        'novelty': novelty, 'divergence': divergence,
        'change_events': change_det.novelty_events,
        'threshold': change_det.threshold,
        'Y': Y, 'T': T,
    }


# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 72)
    print("  M48 PATCH v3 — HOLE 4 TWO-BUG FIX")
    print("=" * 72)
    print(f"\n  DivergenceCUSUM changes only:")
    print(f"    threshold: auto-calibrated from startup noise → fixed {DIVERG_THRESHOLD} Hz")
    print(f"    debounce:  25 samples (2.5s)                 → {DIVERG_DEBOUNCE} samples (15s)")
    print(f"  MIN_SETTLE_S=20 carried forward from v2 (Hole 2 fix).")

    # ── CALIBRATE ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  CALIBRATION")
    print(f"{'='*72}")
    cal = calibrate_v3(0, 1)
    (ridge_fast, ridge_fast_sc,
     ridge_slow, ridge_slow_sc,
     raw_x_slow, true_y_slow,
     raw_x_fast, true_y_fast) = cal

    # ── HOLE 2 CONFIRM (should stay passed) ───────────────────────────
    print(f"\n{'='*72}")
    print("  HOLE 2 CONFIRM — still passing with v3 calibration")
    print(f"{'='*72}")
    trouble_freqs = [0.70, 0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.00]
    test_sig_h2, _ = make_blocks_fixed(trouble_freqs, block_dur=40.0)
    test_total_h2  = stabilization_time + 2*len(trouble_freqs)*40.0 + 10.0
    np.random.seed(104)
    d_h2 = run_sim(test_sig_h2, total_time=test_total_h2,
                   sweep_mode=False, dynamic_settle=True, verbose=False)
    r_h2 = decode_test_v3(d_h2, *cal)
    Y_h2 = d_h2['Y']
    h2_pass = True
    print(f"\n  {'Freq':>6}  {'Slow MAE':>10}  {'OK':>4}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*4}")
    for f in sorted(set(Y_h2)):
        m = Y_h2 == f
        if m.any():
            m_mae = mae(r_h2['ds'][m], Y_h2[m])
            ok = m_mae < 0.008
            if not ok: h2_pass = False
            print(f"  {f:6.2f}  {m_mae:10.4f}  {'✓' if ok else '✗':>4}")
    print(f"\n  Global MAE: {mae(r_h2['ds'], Y_h2):.4f}")
    print(f"  HOLE 2 {'✓ PASS' if h2_pass else '✗ FAIL'}")

    # ── HOLE 4 RETEST ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  HOLE 4 RETEST — DivergenceCUSUM resolution floor")
    print(f"{'='*72}")
    print(f"\n  Fixed threshold={DIVERG_THRESHOLD} Hz  debounce={DIVERG_DEBOUNCE} samples ({DIVERG_DEBOUNCE*0.1:.0f}s)")

    base_freq  = 0.80
    step_sizes = [0.30, 0.20, 0.15, 0.10, 0.05]
    block_dur_h4 = 30.0
    repeats_h4   = 6

    print(f"\n  {'Step':>6}  {'Target':>8}  {'Trans':>6}  "
          f"{'Det':>6}  {'Rate':>7}  {'FalsePos':>9}  {'OK':>4}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*6}  "
          f"{'─'*6}  {'─'*7}  {'─'*9}  {'─'*4}")

    h4_pass = True
    resolution_floor = None

    for step in step_sizes:
        target_freq   = round(base_freq + step, 3)
        step_freqs_h4 = [base_freq, target_freq] * (repeats_h4 // 2)

        np.random.seed(600 + int(step * 100))
        c_sig, _ = make_blocks_fixed(step_freqs_h4, block_dur=block_dur_h4)
        total_h4  = stabilization_time + len(step_freqs_h4)*block_dur_h4*2 + 10.0
        d_h4 = run_sim(c_sig, total_time=total_h4,
                       sweep_mode=False, dynamic_settle=False, verbose=False)
        r_h4 = decode_test_v3(d_h4, *cal)

        Y_h4     = r_h4['Y']
        expected = len(np.where(np.diff(Y_h4) != 0)[0])
        detected = len(r_h4['change_events'])
        rate     = detected / max(1, expected)

        # False positives: novelty outside transition windows
        # Window = debounce length (events inside debounce are the correct single fire)
        trans_idx  = np.where(np.diff(Y_h4) != 0)[0] + 1
        near_trans = np.zeros(len(Y_h4), dtype=bool)
        for idx in trans_idx:
            near_trans[max(0, idx-5) : min(len(Y_h4), idx + DIVERG_DEBOUNCE + 10)] = True
        false_pos = int(np.sum(r_h4['novelty'] & ~near_trans))

        if step >= 0.15:
            ok = rate >= 0.80 and false_pos < 5
            if not ok: h4_pass = False
        else:
            ok = false_pos < 5

        if rate >= 0.80:
            resolution_floor = step

        print(f"  {step:6.2f}  {target_freq:8.3f}  {expected:6d}  "
              f"{detected:6d}  {rate:7.0%}  {false_pos:9d}  {'✓' if ok else '✗':>4}")

        # Show divergence trace at first transition
        if len(trans_idx) > 0:
            idx0   = trans_idx[0]
            lo, hi = max(0, idx0-3), min(len(Y_h4), idx0+12)
            trace  = [f"{r_h4['divergence'][i]:.3f}" for i in range(lo, hi)]
            print(f"         |df-ds|: [{', '.join(trace)}]  thr={DIVERG_THRESHOLD:.3f}")

    floor_str = (f"≥{resolution_floor:.2f} Hz (≥80% detection)"
                 if resolution_floor else "no step reached 80%")
    print(f"\n  Resolution floor: {floor_str}")
    print(f"\n  HOLE 4 {'✓ PASS' if h4_pass else '✗ FAIL'}")

    # ── SANITY CHECKS ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  SANITY — sweep + blocks + noise")
    print(f"{'='*72}")

    np.random.seed(2)
    d_sw = run_sim(make_sweep(0.5, 2.0, 2, sweep_dur),
                   total_time=warmup+2*sweep_dur+10., sweep_mode=True, verbose=False)
    r_sw = decode_test_v3(d_sw, *cal)
    sw_ok = np.mean(r_sw['d_w_slow']) < 0.30
    print(f"\n  Sweep:  w_slow={np.mean(r_sw['d_w_slow']):.4f} (<0.30)  "
          f"fused_MAE={mae(r_sw['d_fused'], r_sw['Y']):.4f}  {'✓' if sw_ok else '✗'}")

    test_freqs_bl = [0.55, 0.75, 0.95, 1.15, 1.35, 1.55, 1.75, 1.95, 2.05]
    sig_bl, _ = make_blocks_fixed(test_freqs_bl, block_dur=40.0)
    np.random.seed(3)
    d_bl = run_sim(sig_bl, total_time=stabilization_time+2*len(test_freqs_bl)*40+10,
                   sweep_mode=False, dynamic_settle=True, verbose=False)
    r_bl = decode_test_v3(d_bl, *cal)
    bl_mae_ok = mae(r_bl['ds'], r_bl['Y']) < 0.008
    bl_w_ok   = np.mean(r_bl['d_w_slow']) > 0.80
    bl_fp     = int(np.sum(r_bl['novelty']))
    bl_fp_ok  = bl_fp < 5
    print(f"  Blocks: slow_MAE={mae(r_bl['ds'], r_bl['Y']):.4f} (<0.008)  "
          f"w_slow={np.mean(r_bl['d_w_slow']):.4f} (>0.80)  "
          f"false_pos={bl_fp} (<5)  "
          f"{'✓' if bl_mae_ok and bl_w_ok and bl_fp_ok else '✗'}")

    np.random.seed(5)
    ns, _ = make_blocks_fixed([0.5,1.0,1.5,2.0], block_dur=40., noise_level=3.0)
    d_n = run_sim(ns, total_time=500., sweep_mode=False, dynamic_settle=True, verbose=False)
    r_n = decode_test_v3(d_n, *cal)
    n_ok = mae(r_n['d_fused'], r_n['Y']) < 0.050
    print(f"  Noise:  fused_MAE={mae(r_n['d_fused'], r_n['Y']):.4f} (<0.050)  "
          f"w_slow={np.mean(r_n['d_w_slow']):.4f}  {'✓' if n_ok else '✗'}")

    # ── FINAL VERDICT ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  FINAL VERDICT")
    print(f"{'='*72}")

    verdicts = [
        ("HOLE 2 — per-freq MAE < 0.008 Hz",          h2_pass),
        ("HOLE 4 — detection ≥80% (steps ≥0.15 Hz)",  h4_pass),
        ("SANITY — sweep w_slow < 0.30",               sw_ok),
        ("SANITY — block slow MAE < 0.008",            bl_mae_ok),
        ("SANITY — block w_slow > 0.80",               bl_w_ok),
        ("SANITY — block false positives < 5",         bl_fp_ok),
        ("SANITY — noise σ=3 fused MAE < 0.050",       n_ok),
    ]
    all_pass = True
    print(f"\n  {'Test':48s}  {'Result':>8}")
    print(f"  {'─'*48}  {'─'*8}")
    for name, ok in verdicts:
        if not ok: all_pass = False
        print(f"  {name:48s}  {'✓ PASS' if ok else '✗ FAIL':>8}")

    print(f"\n  {'─'*58}")
    print(f"  {'OVERALL':48s}  {'✓ PASS' if all_pass else '✗ FAIL':>8}")

    if all_pass:
        print("""
  ✓✓✓ ALL CHECKS PASS ✓✓✓

  Two changes to fold into m48_neuron.py:

  1. MIN_SETTLE_S = 8.0  →  20.0   (one line)

  2. Replace TwoWindowChangeDetector with DivergenceCUSUM (this file).
     In decode_test(): call change_det.update(df[i], ds[i], T[i])
     Key parameters: threshold=0.020, debounce=150
""")
    else:
        print("""
  ✗ Still failing. Check per-test output above.
  If Hole 4 false positives persist: increase DIVERG_DEBOUNCE further.
  If Hole 4 misses transitions: decrease DIVERG_THRESHOLD slightly.
  If Hole 2 regressed: check plv_slow values at collection time.
""")