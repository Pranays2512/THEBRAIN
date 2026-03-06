"""
M55 ASSOCIATIVE MEMORY — BREAK TEST SUITE
==========================================
Adversarial tests designed to find real failure modes, not confirm
happy-path behavior. Every test targets a specific thing that could
silently go wrong in the memory layer.

Each test has a ROOT CAUSE section explaining WHY that failure mode
exists architecturally, not just what we're checking.

Tests
-----
BT-01  Weight explosion under high learning rate / no decay
BT-02  Trace bleed — unrelated neurons shouldn't associate
BT-03  Catastrophic overwrite — heavy retraining destroys old memory
BT-04  Familiarity monotonicity — must grow with exposure, never decrease
BT-05  Forgetting curve — familiarity must drop after long absence
BT-06  Recall under noise — partial/noisy cue still recalls correctly
BT-07  Symmetry preservation under long run (asymmetry accumulation)
BT-08  Dead weight test — no neuron permanently blocked from writing
BT-09  Homeostasis ceiling — W never exceeds W_MAX under any training
BT-10  Trace window adaptation — novel really does extend window
BT-11  Pattern separation — nearby BMUs must not bleed into each other
BT-12  Familiarity pipeline flow — score actually flows in full pipeline
BT-13  Long-run stability — W doesn't drift to degenerate state
BT-14  Determinism — same seed gives identical results
BT-15  Cold start — zero memories, recall returns coherent output
BT-16  Ring buffer capacity — MAX_EPISODES not applicable (W-based),
        but exposure counter must not overflow int32
BT-17  Full pipeline familiarity — second encounter more familiar than first
BT-18  Memory after cortex reset — M55 persists independently of M54
"""

import numpy as np
import sys
import time
from collections import deque

# ── Imports ──────────────────────────────────────────────────────
try:
    from m50_neuron import (
        run_sim, make_blocks, make_sweep,
        fit_ridge, build_reverse_lookup,
        decode_resonance, compute_stability_plv,
        DivergenceCUSUM,
        stabilization_time, dt,
        RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
        PLV_STAB_WINDOW,
        mae, N,
    )
    from m54_cortex import (
        CortexM54,
        GRID_H, GRID_W, N_NEURONS,
        FREQ_MIN_HZ, FREQ_MAX_HZ,
    )
    from m54_experience import ExperienceBuffer
    from m55_memory import (
        AssociativeMemory,
        ETA_HEBB, DECAY_RATE, W_MAX,
        TRACE_DECAY_BASE, TRACE_DECAY_MIN,
        NOVELTY_MODULATION, MIN_TRACE_TO_WRITE,
        RECALL_MAX_STEPS,
    )
    IMPORTS_OK = True
except Exception as e:
    print(f"  [SKIP] Import failed: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# HARNESS
# ═══════════════════════════════════════════════════════════════

results  = {}
_DIVIDER = "─" * 72

def section(title):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")

def report(name, passed, detail="", warn=False):
    tag = "PASS" if passed else ("WARN" if warn else "FAIL")
    sym = "✓" if passed else ("⚠" if warn else "✗")
    results[name] = tag
    print(f"  {sym} [{tag}] {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")

def summarise():
    section("M55 BREAK TEST SUMMARY")
    n_pass = sum(1 for v in results.values() if v == "PASS")
    n_fail = sum(1 for v in results.values() if v == "FAIL")
    n_warn = sum(1 for v in results.values() if v == "WARN")
    for name, tag in results.items():
        sym = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}[tag]
        print(f"  {sym} [{tag}] {name}")
    print(f"\n  {_DIVIDER}")
    print(f"  PASS:{n_pass}  FAIL:{n_fail}  WARN:{n_warn}")
    print(f"  {'ALL CLEAR — green flag for Layer 2' if n_fail == 0 and n_warn == 0 else 'FAILURES FOUND — fix before proceeding'}")


# ═══════════════════════════════════════════════════════════════
# CALIBRATION  (shared pipeline — identical to breaktest pattern)
# ═══════════════════════════════════════════════════════════════

section("CALIBRATION")

def build_calibration():
    SLOW_FREQS_CAL = sorted(set([
        0.41, 0.44, 0.47, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.77,
        0.8, 0.82, 0.85, 0.87, 0.9, 0.92, 0.95, 0.97, 1.0, 1.03, 1.05,
        1.07, 1.1, 1.15, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.75,
        1.8, 1.9, 1.95, 2.0, 2.05, 2.1, 2.12, 2.16, 2.20,
    ]))
    warmup    = stabilization_time + 10.0
    sweep_dur = 60.0

    np.random.seed(0)
    data_train = run_sim(
        make_sweep(0.5, 2.0, 6, sweep_dur),
        total_time=warmup + 6 * sweep_dur + 10.0,
        sweep_mode=True, verbose=False, collect_calib=False,
    )
    ridge_fast, ridge_fast_sc = fit_ridge(
        data_train['feat_fast'], data_train['Y'], RIDGE_ALPHA_FAST)

    np.random.seed(1)
    block_sig, _ = make_blocks(SLOW_FREQS_CAL, block_dur=40.0)
    data_slow = run_sim(
        block_sig,
        total_time=stabilization_time + 2 * len(SLOW_FREQS_CAL) * 40.0 + 10.0,
        sweep_mode=False, dynamic_settle=True, verbose=False,
        collect_calib=True,
    )
    raw_x_slow, raw_y_slow = build_reverse_lookup(
        sorted(data_slow['calib_plv_slow'].keys()),
        data_slow['calib_plv_slow'], data_slow['calib_energy_slow'],
    )
    raw_x_fast, raw_y_fast = build_reverse_lookup(
        sorted(data_slow['calib_plv_fast'].keys()),
        data_slow['calib_plv_fast'], data_slow['calib_energy_fast'],
    )
    print(f"  Calibration: {len(raw_x_slow)} pts, "
          f"[{raw_x_slow[0]:.3f}, {raw_x_slow[-1]:.3f}]")
    return raw_x_slow, raw_y_slow, raw_x_fast, raw_y_fast

print("  Building calibration...")
raw_x_slow, raw_y_slow, raw_x_fast, raw_y_fast = build_calibration()
print("  Done.")

warmup    = stabilization_time + 10.0
sweep_dur = 60.0


def run_full_pipeline(sim_data, cortex, memory, buf=None):
    """
    Run M50 decode → M54 cortex → M55 memory → (optional) ExperienceBuffer.
    Returns list of per-step records.
    """
    n          = len(sim_data['T'])
    plv_hist   = deque(maxlen=PLV_STAB_WINDOW)
    cusum      = DivergenceCUSUM()
    records    = []

    for i in range(n):
        plv_fast_i = sim_data['plv_fast'][i]
        plv_slow_i = sim_data['plv_slow'][i]
        e_fast_i   = sim_data['energy_fast'][i]
        e_slow_i   = sim_data['energy_slow'][i]
        t          = float(sim_data['T'][i])

        df = decode_resonance(plv_fast_i, e_fast_i, raw_x_fast, raw_y_fast)
        ds = decode_resonance(plv_slow_i, e_slow_i, raw_x_slow, raw_y_slow)

        max_plv = float(np.abs(plv_slow_i).max())
        plv_hist.append(max_plv)
        w = compute_stability_plv(plv_hist)

        _, nov = cusum.update(df, ds, t, w=w)
        if nov:
            w = 0.0
        fused = w * ds + (1.0 - w) * df

        cortex_out = cortex.step(
            decoded_freq=fused, stability_w=w,
            novelty_flag=float(nov), plv_vector=plv_slow_i,
        )
        mem_out = memory.step(cortex_out['bmu_idx'], cortex_out['qe_norm'])

        if buf is not None:
            buf.push(t=t, cortex_out=cortex_out, decoded_freq=fused,
                     stability_w=w, transition=nov, cortex_step=cortex.t)

        records.append({
            'Y':          sim_data['Y'][i],
            'T':          t,
            'bmu_idx':    cortex_out['bmu_idx'],
            'qe_norm':    cortex_out['qe_norm'],
            'is_novel':   cortex_out['is_novel'],
            'fused':      fused,
            'w':          w,
            'nov':        nov,
            'mem_wrote':  mem_out['wrote'],
            'w_mean':     mem_out['w_mean'],
        })
    return records


# ═══════════════════════════════════════════════════════════════
# BT-01  Weight explosion under sustained high-rate writing
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: ETA_HEBB=0.04 adds 0.04 per step. Without normalization,
# 1000 co-active steps = W[i,j] += 40.0. Homeostasis must clamp this.
# If normalization has a bug (e.g. applied before decay, or wrong axis),
# weights can spiral unbounded and NaN the entire matrix silently.
# ───────────────────────────────────────────────────────────────
section("BT-01  Weight explosion — sustained high-rate Hebbian writing")

mem_01 = AssociativeMemory()
for _ in range(10_000):
    mem_01.step(10, qe_norm=1.0)
    mem_01.step(11, qe_norm=1.0)

W01     = mem_01.get_state()['W_snapshot']
has_nan = bool(np.any(np.isnan(W01)))
has_inf = bool(np.any(np.isinf(W01)))
w_max   = float(W01.max())
bounded = w_max <= W_MAX + 1e-4

report("BT-01 Weight explosion",
       bounded and not has_nan and not has_inf,
       f"W_max={w_max:.6f} (ceiling={W_MAX})  NaN={has_nan}  Inf={has_inf}",
       warn=(w_max <= W_MAX * 1.1 and not has_nan))


# ═══════════════════════════════════════════════════════════════
# BT-02  Trace bleed — inactive neurons must not gain weight
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: The eligibility trace decays each step but never fully
# reaches zero in finite steps (exponential decay). If MIN_TRACE_TO_WRITE
# is too low, neurons that fired 50 steps ago still have trace ~e^(-50*0.05)
# = 0.082 and participate in Hebb updates. After 10,000 steps this bleeds
# associations across the entire matrix, destroying pattern separation.
# ───────────────────────────────────────────────────────────────
section("BT-02  Trace bleed — inactive neurons must not associate")

mem_02 = AssociativeMemory()
# Fire cluster A (neurons 0-3) for 500 steps
for _ in range(500):
    for n in range(4):
        mem_02.step(n, qe_norm=0.5)

# Now wait 200 steps with NO firing (trace fully decays)
# Use a dummy BMU that we'll check separately
for _ in range(200):
    mem_02.step(63, qe_norm=0.0)   # BMU 63 = "idle" marker

# Now fire cluster B (neurons 32-35)
for _ in range(500):
    for n in range(32, 36):
        mem_02.step(n, qe_norm=0.5)

W02 = mem_02.get_state()['W_snapshot']

# Cross-cluster weight: A neurons (0-3) should NOT associate with B (32-35)
# They were separated by 200 idle steps — trace of A was completely dead
# when B started firing
cross = float(W02[0:4, 32:36].mean())
within_A = float(W02[0:4, 0:4].mean())
within_B = float(W02[32:36, 32:36].mean())

# Cross should be near zero — trace of cluster A was below MIN_TRACE_TO_WRITE
# long before cluster B started
ok = cross < 0.01
print(f"  within_A={within_A:.4f}  within_B={within_B:.4f}  cross_AB={cross:.6f}")
report("BT-02 Trace bleed isolation",
       ok,
       f"cross_AB={cross:.6f} (should be <0.01 — trace dead between clusters)",
       warn=(0.01 <= cross < 0.05))


# ═══════════════════════════════════════════════════════════════
# BT-03  Catastrophic overwrite — heavy retraining on B destroys A
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: This is the M54 BT-07 equivalent for the memory layer.
# If a new pattern is trained 5× more than an old one, the old memory
# should still be recallable. Overtraining B should not erase A because:
# (1) Hebb only strengthens connections between co-active neurons
# (2) A and B neurons are disjoint — B training never touches W[A,A]
# (3) Decay affects ALL weights equally — A decays but so does B
# If this fails, it means B training is somehow writing into A's region.
# ───────────────────────────────────────────────────────────────
section("BT-03  Catastrophic overwrite — heavy B training preserves A memory")

mem_03 = AssociativeMemory()

# Phase 1: train A (neurons 5, 6) — 100 steps
for _ in range(100):
    mem_03.step(5, qe_norm=0.8)
    mem_03.step(6, qe_norm=0.8)

fam_A_before = mem_03.recall(5)['familiarity']
exp_A_before = int(mem_03._bmu_exposure[5])

# Phase 2: HEAVY overtraining on B (neurons 50, 51) — 500 steps (5× A)
for _ in range(500):
    mem_03.step(50, qe_norm=0.8)
    mem_03.step(51, qe_norm=0.8)

fam_A_after = mem_03.recall(5)['familiarity']
fam_B_after = mem_03.recall(50)['familiarity']
exp_A_after = int(mem_03._bmu_exposure[5])

print(f"  A familiarity: before={fam_A_before:.4f}  after={fam_A_after:.4f}")
print(f"  B familiarity: {fam_B_after:.4f}")
print(f"  A exposure count: {exp_A_before} → {exp_A_after} (should be unchanged)")

# A familiarity drops due to weight decay, but should stay above 50% of before
# because exposure count is permanent and W[5,6] only decays, never overwritten
ok = fam_A_after > fam_A_before * 0.5 and exp_A_after == exp_A_before
report("BT-03 Catastrophic overwrite",
       ok,
       f"A: {fam_A_before:.4f}→{fam_A_after:.4f} (must stay >50% of before)  "
       f"B: {fam_B_after:.4f}  A_exposure unchanged: {exp_A_after==exp_A_before}",
       warn=(fam_A_after > fam_A_before * 0.3 and exp_A_after == exp_A_before))


# ═══════════════════════════════════════════════════════════════
# BT-04  Familiarity monotonicity — must never decrease mid-training
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: If familiarity is computed from any combination of W values
# and exposure counts, it could temporarily dip if W is being renormalized
# mid-session while exposure hasn't grown enough yet. The exposure-weighted
# formula should prevent this, but log1p scaling + W decay could create
# a transient dip at a specific training volume. Must verify monotonicity.
# ───────────────────────────────────────────────────────────────
section("BT-04  Familiarity monotonicity — never decreases during training")

mem_04 = AssociativeMemory()
scores_04 = []

for step in range(500):
    mem_04.step(20, qe_norm=0.6)
    mem_04.step(21, qe_norm=0.6)
    if step % 50 == 49:
        scores_04.append(mem_04.recall(20)['familiarity'])

print(f"  Familiarity at steps 50,100,...,500:")
for i, s in enumerate(scores_04):
    print(f"    step {(i+1)*50:4d}: {s:.4f}")

# Check strictly non-decreasing
dips = [(i, scores_04[i-1], scores_04[i])
        for i in range(1, len(scores_04))
        if scores_04[i] < scores_04[i-1] - 1e-6]

ok = len(dips) == 0
report("BT-04 Familiarity monotonicity",
       ok,
       f"{'No dips found' if ok else f'{len(dips)} dip(s): {dips}'}",
       warn=(len(dips) == 1 and dips[0][1] - dips[0][2] < 0.01))


# ═══════════════════════════════════════════════════════════════
# BT-05  Forgetting curve — familiarity drops after long absence
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: Forgetting is implemented via DECAY_RATE on W.
# But the exposure counter never decays — it's permanent.
# So familiarity (which uses exposure as primary signal) will NEVER
# fully forget. This is actually correct biologically — you never
# completely forget something you've seen 500 times, but familiarity
# SHOULD decrease noticeably with absence. The test checks that the
# W-based component contributes enough decay to be measurable,
# even though exposure count anchors a floor.
# ───────────────────────────────────────────────────────────────
section("BT-05  Forgetting curve — familiarity drops after long absence")

mem_05 = AssociativeMemory()

# Train A heavily
for _ in range(300):
    mem_05.step(15, qe_norm=0.9)
    mem_05.step(16, qe_norm=0.9)

fam_peak = mem_05.recall(15)['familiarity']

# Now run 5000 steps firing DIFFERENT neurons entirely (A gets no reinforcement)
for _ in range(5000):
    mem_05.step(40, qe_norm=0.1)
    mem_05.step(41, qe_norm=0.1)

fam_after_absence = mem_05.recall(15)['familiarity']
drop = fam_peak - fam_after_absence

print(f"  Peak familiarity:    {fam_peak:.4f}")
print(f"  After 5000 steps:    {fam_after_absence:.4f}")
print(f"  Drop:                {drop:.4f}")

# Should drop measurably (W decays) but not to zero (exposure count floor)
ok = drop > 0.02 and fam_after_absence > 0.0
report("BT-05 Forgetting curve",
       ok,
       f"peak={fam_peak:.4f} → after_absence={fam_after_absence:.4f}  "
       f"drop={drop:.4f} (must be >0.02, must not reach 0)",
       warn=(0.005 <= drop <= 0.02))


# ═══════════════════════════════════════════════════════════════
# BT-06  Recall under partial/noisy cue
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: Recall seeds from a single BMU. The pattern completion
# loop uses W @ pattern to pull associations. If the recall procedure
# fails to activate any associated neuron (e.g. W is too weak, or the
# softmax temperature is too high and spreads activation too uniformly),
# then recall returns the cue neuron alone — correct BMU but no pattern.
# This test checks that a trained association is actually recalled, not
# just that the cue neuron remains active.
# ───────────────────────────────────────────────────────────────
section("BT-06  Recall — trained association actually activates in recall")

mem_06 = AssociativeMemory()

# Train a strong 3-neuron association: 7 → 8 → 9
for _ in range(400):
    mem_06.step(7, qe_norm=0.9)
    mem_06.step(8, qe_norm=0.9)
    mem_06.step(9, qe_norm=0.9)

# Recall from cue=7 only
r06 = mem_06.recall(7)
pattern = r06['settled_pattern']

# BMU 8 and 9 should be activated in the recalled pattern
act_8 = float(pattern[8])
act_9 = float(pattern[9])
act_7 = float(pattern[7])

print(f"  Recalled activations: BMU7={act_7:.4f}  BMU8={act_8:.4f}  BMU9={act_9:.4f}")
print(f"  Top associations: {r06['top_associations']}")

ok = act_8 > 0.01 and act_9 > 0.01
report("BT-06 Recall activates associations",
       ok,
       f"BMU8 activation={act_8:.4f} (>0.01)  BMU9 activation={act_9:.4f} (>0.01)",
       warn=(act_8 > 0.005 and act_9 > 0.005))


# ═══════════════════════════════════════════════════════════════
# BT-07  Symmetry preservation under long run
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: W is enforced symmetric at each step via (W + W.T)/2.
# But floating point arithmetic accumulates errors differently for W[i,j]
# vs W[j,i] if the averaging is done in float32. Over 100,000 steps,
# tiny per-step asymmetries could accumulate to measurable levels,
# which would make the energy function non-quadratic and recall unstable.
# ───────────────────────────────────────────────────────────────
section("BT-07  Symmetry preservation under long run (100k steps)")

mem_07 = AssociativeMemory()
np.random.seed(77)
for _ in range(100_000):
    mem_07.step(int(np.random.randint(0, 64)), qe_norm=float(np.random.rand()))

W07  = mem_07.get_state()['W_snapshot']
asym = float(np.abs(W07 - W07.T).max())
ok   = asym < 1e-4

print(f"  Steps: 100,000  max asymmetry: {asym:.2e}")
report("BT-07 Long-run symmetry",
       ok,
       f"max |W[i,j]-W[j,i]| = {asym:.2e} (must be <1e-4)",
       warn=(1e-4 <= asym < 1e-3))


# ═══════════════════════════════════════════════════════════════
# BT-08  Dead weight — no neuron permanently blocked from writing
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: The conscience mechanism in M54 prevents BMU monopoly.
# M55 has no conscience — only MIN_TRACE_TO_WRITE as a threshold.
# If a neuron is never activated (never fires as BMU), its row and
# column in W stay at zero indefinitely. This is correct behavior —
# dead neurons in M55 reflect dead neurons in M54, not a M55 bug.
# But we test that neurons which DO fire are never silently blocked
# from writing by the normalization (scale factor going to 0).
# ───────────────────────────────────────────────────────────────
section("BT-08  Dead weight — active neurons can always write to W")

mem_08 = AssociativeMemory()

# Train the first 32 neurons heavily — they should all have nonzero rows
for _ in range(200):
    for n in range(32):
        mem_08.step(n, qe_norm=0.7)

W08 = mem_08.get_state()['W_snapshot']

# Every neuron in 0-31 should have at least one nonzero connection
row_max_active = np.max(W08[:32, :], axis=1)  # (32,) max weight per row
dead_active = int(np.sum(row_max_active < 1e-6))

# Neurons 32-63 were never fired — their rows should be zero (correct)
row_max_inactive = np.max(W08[32:, :], axis=1)
nonzero_inactive = int(np.sum(row_max_inactive > 1e-6))

print(f"  Active neurons (0-31) with zero rows: {dead_active}/32  (should be 0)")
print(f"  Inactive neurons (32-63) with nonzero rows: {nonzero_inactive}/32  (should be 0)")

ok = dead_active == 0 and nonzero_inactive == 0
report("BT-08 Dead weight",
       ok,
       f"active_dead={dead_active}/32 (=0), "
       f"inactive_nonzero={nonzero_inactive}/32 (=0)")


# ═══════════════════════════════════════════════════════════════
# BT-09  Homeostasis ceiling — W_MAX never breached
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: Row normalization clips per-row max to W_MAX. But the
# symmetry enforcement (W + W.T)/2 happens AFTER normalization. If
# W[i,j]=W_MAX and W[j,i]=W_MAX, their average is still W_MAX — fine.
# But if both rows get independently clamped and then averaged, the
# result could briefly exceed W_MAX if there's ordering dependence.
# Also: W is float32. Floating point rounding can push values to
# W_MAX + epsilon (e.g. 1.0000001). Test for strict ceiling.
# ───────────────────────────────────────────────────────────────
section("BT-09  Homeostasis ceiling — W_MAX strictly enforced")

mem_09 = AssociativeMemory()
max_seen = 0.0

# Record max W over 50,000 steps of heavy training
for step in range(50_000):
    mem_09.step(step % 64, qe_norm=1.0)
    if step % 1000 == 999:
        current_max = float(mem_09.get_state()['W_snapshot'].max())
        max_seen = max(max_seen, current_max)

ok = max_seen <= W_MAX + 1e-5
print(f"  Max W ever seen over 50k steps: {max_seen:.8f}  (ceiling={W_MAX})")
report("BT-09 Homeostasis ceiling",
       ok,
       f"max_W={max_seen:.8f} ≤ {W_MAX}+1e-5: {ok}",
       warn=(W_MAX < max_seen <= W_MAX + 1e-3))


# ═══════════════════════════════════════════════════════════════
# BT-10  Trace window adaptation — novel genuinely extends window
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: The adaptive trace formula is:
#   decay = max(TRACE_DECAY_MIN, TRACE_DECAY_BASE - NOVELTY_MODULATION * qe_norm)
# This is a linear interpolation. The test checks both endpoints and
# also that intermediate qe_norm values interpolate correctly (no
# clamping bug that makes the curve flat in the middle).
# ───────────────────────────────────────────────────────────────
section("BT-10  Trace window adaptation — correct interpolation")

mem_10 = AssociativeMemory()

qe_vals     = [0.0, 0.25, 0.5, 0.75, 1.0]
decays      = []
print(f"  {'qe_norm':>8}  {'decay':>8}  {'window_steps':>13}")
for qe in qe_vals:
    mem_10.step(0, qe_norm=qe)
    d = mem_10.get_state()['trace_decay']
    decays.append(d)
    print(f"  {qe:8.2f}  {d:8.4f}  {1/d:13.1f}")

# Monotonically decreasing: higher qe_norm → lower decay → longer window
monotone = all(decays[i] >= decays[i+1] - 1e-6 for i in range(len(decays)-1))
# Floor is respected
floor_ok = all(d >= TRACE_DECAY_MIN - 1e-6 for d in decays)
# Ceiling is TRACE_DECAY_BASE at qe_norm=0
ceil_ok  = abs(decays[0] - TRACE_DECAY_BASE) < 1e-6

ok = monotone and floor_ok and ceil_ok
report("BT-10 Trace window adaptation",
       ok,
       f"monotone={monotone}  floor_ok={floor_ok}  ceil_ok={ceil_ok}",
       warn=(monotone and (not floor_ok or not ceil_ok)))


# ═══════════════════════════════════════════════════════════════
# BT-11  Pattern separation — nearby BMUs stay distinct
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: M54 maps similar frequencies to nearby grid positions.
# E.g. 0.80 Hz → BMU (3,2), 0.85 Hz → BMU (3,3). If those BMUs are
# adjacent, they may fire close together in time and accumulate
# cross-associations. After many exposures, recalling BMU (3,2) would
# activate (3,3) even when 0.85 Hz was never presented. This is the
# memory equivalent of M54's BT-07 (catastrophic forgetting) — but
# at the association layer. Test with BMUs that are grid-adjacent.
# ───────────────────────────────────────────────────────────────
section("BT-11  Pattern separation — adjacent BMUs stay distinct")

mem_11 = AssociativeMemory()

# Train two ADJACENT BMUs alternately but never simultaneously
# BMU 27 = grid (3,3), BMU 28 = grid (3,4) — neighboring cells
# They never fire in the same trace window (200 idle steps between)
for rep in range(20):
    # Train A (BMU 27) for 50 steps
    for _ in range(50):
        mem_11.step(27, qe_norm=0.8)
    # Gap: 200 idle steps (trace of 27 fully decays below MIN_TRACE_TO_WRITE)
    for _ in range(200):
        mem_11.step(63, qe_norm=0.0)
    # Train B (BMU 28) for 50 steps
    for _ in range(50):
        mem_11.step(28, qe_norm=0.8)
    # Gap again
    for _ in range(200):
        mem_11.step(63, qe_norm=0.0)

W11 = mem_11.get_state()['W_snapshot']

within_27 = float(W11[27, 27])   # self = 0 by design
assoc_2728 = float(W11[27, 28])   # cross-association
assoc_2827 = float(W11[28, 27])   # symmetric

print(f"  W[27,28] = {assoc_2728:.6f}  (cross-association — should be ~0)")
print(f"  W[28,27] = {assoc_2827:.6f}  (symmetric)")

# Since they were separated by 200-step gaps, trace of one should be dead
# when the other fires. Cross-weight should be near zero.
ok = assoc_2728 < 0.01
report("BT-11 Pattern separation",
       ok,
       f"W[27,28]={assoc_2728:.6f} (target <0.01 — never co-fired)",
       warn=(0.01 <= assoc_2728 < 0.05))


# ═══════════════════════════════════════════════════════════════
# BT-12  Familiarity pipeline flow — score flows in full pipeline
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: MT-14 of the unit test showed mean_familiarity=0.000
# in the summary. This is because recall() is NEVER called during
# the pipeline — _familiarity_history stays empty. The memory writes
# correctly but the familiarity signal (the whole point of M55 in a
# live system) is never computed. This test explicitly calls recall()
# at each step and verifies the signal is nonzero and meaningful.
# ───────────────────────────────────────────────────────────────
section("BT-12  Familiarity pipeline flow — recall called per step")

print("  Running pipeline with explicit per-step recall()...")
np.random.seed(42)
cortex_12 = CortexM54(seed=12)
memory_12 = AssociativeMemory(seed=12)

freqs_12  = [0.60, 1.20, 1.80]
sig_12, _ = make_blocks(freqs_12 * 4, block_dur=35.0)   # 4 repeats = 12 blocks
d_12      = run_sim(sig_12,
    total_time=stabilization_time + 4*len(freqs_12)*2*35.0 + 10.0,
    sweep_mode=False, dynamic_settle=False, verbose=False)

n          = len(d_12['T'])
plv_hist   = deque(maxlen=PLV_STAB_WINDOW)
cusum      = DivergenceCUSUM()
fam_scores = []

for i in range(n):
    plv_fast_i = d_12['plv_fast'][i]
    plv_slow_i = d_12['plv_slow'][i]
    e_fast_i   = d_12['energy_fast'][i]
    e_slow_i   = d_12['energy_slow'][i]
    t          = float(d_12['T'][i])

    df = decode_resonance(plv_fast_i, e_fast_i, raw_x_fast, raw_y_fast)
    ds = decode_resonance(plv_slow_i, e_slow_i, raw_x_slow, raw_y_slow)

    max_plv = float(np.abs(plv_slow_i).max())
    plv_hist.append(max_plv)
    w = compute_stability_plv(plv_hist)

    _, nov = cusum.update(df, ds, t, w=w)
    if nov:
        w = 0.0
    fused = w * ds + (1.0 - w) * df

    cortex_out = cortex_12.step(
        decoded_freq=fused, stability_w=w,
        novelty_flag=float(nov), plv_vector=plv_slow_i,
    )
    memory_12.step(cortex_out['bmu_idx'], cortex_out['qe_norm'])

    # Explicitly call recall at every step — this is what a real system does
    recall_out = memory_12.recall(cortex_out['bmu_idx'])
    fam_scores.append(recall_out['familiarity'])

mean_fam  = float(np.mean(fam_scores))
final_fam = float(np.mean(fam_scores[-500:]))   # last 500 steps = settled
early_fam = float(np.mean(fam_scores[:500]))    # first 500 steps = fresh

print(f"  Total steps: {n}")
print(f"  Early familiarity (first 500 steps): {early_fam:.4f}")
print(f"  Final familiarity (last 500 steps):  {final_fam:.4f}")
print(f"  Mean familiarity overall:             {mean_fam:.4f}")

ok = mean_fam > 0.0 and final_fam > early_fam
report("BT-12 Familiarity pipeline flow",
       ok,
       f"mean={mean_fam:.4f} (>0)  early={early_fam:.4f} → final={final_fam:.4f}  "
       f"growing: {final_fam > early_fam}",
       warn=(mean_fam > 0 and final_fam <= early_fam))


# ═══════════════════════════════════════════════════════════════
# BT-13  Long-run stability — W doesn't degenerate
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: Over very long runs (M50 runs for hours in deployment),
# several failure modes can emerge:
# (1) W converges to all-ones (homeostasis clamps everything to W_MAX)
# (2) W converges to all-zeros (decay wins, Hebb too weak to compensate)
# (3) A single neuron monopolizes all associations (winner-take-all)
# A healthy W after long runs should be sparse-to-moderate, distributed,
# bounded away from both 0 and W_MAX uniformly.
# ───────────────────────────────────────────────────────────────
section("BT-13  Long-run stability — W doesn't degenerate after 200k steps")

print("  Running 200,000 steps with realistic BMU distribution...")
mem_13 = AssociativeMemory()
np.random.seed(13)

# Realistic: 6 dominant BMUs (like 6 trained frequencies) with noise
dominant = [10, 17, 26, 35, 42, 55]
t0 = time.time()
for step in range(200_000):
    # 80% chance of dominant BMU, 20% random (like real cortex noise)
    if np.random.rand() < 0.8:
        bmu = dominant[step % len(dominant)]
    else:
        bmu = int(np.random.randint(0, 64))
    qe = float(np.random.rand() * 0.3)   # mostly familiar
    mem_13.step(bmu, qe_norm=qe)

elapsed = time.time() - t0
W13     = mem_13.get_state()['W_snapshot']

w_mean       = float(W13.mean())
w_max        = float(W13.max())
w_nonzero    = float((W13 > 1e-4).mean())
all_ones     = w_mean > 0.95    # degenerate: everything maxed
all_zeros    = w_mean < 1e-4    # degenerate: everything decayed
monopoly     = w_nonzero < 0.02  # <2% nonzero = only 1-2 neurons have memory

ok = not all_ones and not all_zeros and not monopoly
print(f"  Steps: 200,000  ({elapsed:.1f}s)")
print(f"  W mean={w_mean:.5f}  max={w_max:.5f}  nonzero={w_nonzero*100:.1f}%")
print(f"  Degenerate all-ones: {all_ones}  all-zeros: {all_zeros}  monopoly: {monopoly}")
report("BT-13 Long-run stability",
       ok,
       f"mean={w_mean:.5f} max={w_max:.5f} nonzero={w_nonzero*100:.1f}%  "
       f"degenerate: all_ones={all_ones} all_zeros={all_zeros} monopoly={monopoly}",
       warn=(not all_ones and not all_zeros and monopoly))


# ═══════════════════════════════════════════════════════════════
# BT-14  Determinism — same seed gives identical results
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: AssociativeMemory uses numpy random state only during
# init (none actually — W starts at zeros). All updates are deterministic
# given the BMU sequence. If the pipeline (M50+M54) is deterministic
# (verified in M54 BT-18), then M55 must also be deterministic.
# Any randomness leak (e.g. from numpy global state affecting float32
# rounding) would manifest as small differences that accumulate.
# ───────────────────────────────────────────────────────────────
section("BT-14  Determinism — identical results with same seed")

def run_determinism_trial(seed):
    np.random.seed(seed)
    cortex = CortexM54(seed=seed)
    memory = AssociativeMemory(seed=seed)
    sig, _ = make_blocks([0.70, 1.10, 1.50], block_dur=30.0)
    d = run_sim(sig,
        total_time=stabilization_time + 2*3*30.0 + 10.0,
        sweep_mode=False, dynamic_settle=False, verbose=False)
    records = run_full_pipeline(d, cortex, memory)
    return np.array([r['w_mean'] for r in records])

print("  Running pipeline twice with seed=88...")
r1 = run_determinism_trial(88)
r2 = run_determinism_trial(88)
max_diff  = float(np.max(np.abs(r1 - r2)))
mean_diff = float(np.mean(np.abs(r1 - r2)))
print(f"  Max diff: {max_diff:.2e}   Mean diff: {mean_diff:.2e}")
report("BT-14 Determinism",
       max_diff < 1e-6,
       f"max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  (both must be <1e-6)")


# ═══════════════════════════════════════════════════════════════
# BT-15  Cold start — recall on empty memory returns valid output
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: On cold start W=0. recall() runs W @ pattern = 0 vector.
# softmax(0/T) = uniform distribution. The cue neuron is anchored to 0.5
# and renormalized. This is mathematically well-defined but needs to
# return a valid dict without NaN, division by zero, or exceptions.
# If the system is deployed from cold and the first query is recall(),
# it must not crash.
# ───────────────────────────────────────────────────────────────
section("BT-15  Cold start — recall on empty memory is safe")

mem_15 = AssociativeMemory()   # no training at all
try:
    r15     = mem_15.recall(32)
    has_nan = bool(np.any(np.isnan(r15['settled_pattern'])))
    sum_ok  = abs(r15['settled_pattern'].sum() - 1.0) < 0.01
    fam_ok  = 0.0 <= r15['familiarity'] <= 1.0
    ok      = not has_nan and sum_ok and fam_ok
    print(f"  familiarity={r15['familiarity']:.4f}  "
          f"depth={r15['depth']:.6f}  "
          f"pattern_sum={r15['settled_pattern'].sum():.4f}  "
          f"NaN={has_nan}")
    report("BT-15 Cold start safety",
           ok,
           f"NaN={has_nan}  pattern_sum≈1: {sum_ok}  familiarity∈[0,1]: {fam_ok}")
except Exception as e:
    report("BT-15 Cold start safety", False, f"Exception: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════
# BT-16  Exposure counter overflow safety
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: _bmu_exposure is int32. Max int32 = 2,147,483,647.
# At dt=0.05s that's ~107M seconds ≈ 3.4 years of continuous running.
# But in accelerated tests or mis-configured loops someone could hit it.
# int32 overflow wraps to negative, making familiarity score go negative
# (log1p of negative = NaN). Test that the familiarity formula handles
# very large exposure counts without NaN or negative scores.
# ───────────────────────────────────────────────────────────────
section("BT-16  Exposure counter — large counts stay valid")

mem_16 = AssociativeMemory()
# Manually inject a very large exposure count
mem_16._bmu_exposure[0] = 2_000_000   # 2 million exposures
r16 = mem_16.recall(0)

has_nan  = bool(np.isnan(r16['familiarity']))
in_range = 0.0 <= r16['familiarity'] <= 1.0
print(f"  exposure=2,000,000  familiarity={r16['familiarity']:.4f}  "
      f"NaN={has_nan}  in_range={in_range}")
report("BT-16 Large exposure count",
       not has_nan and in_range,
       f"familiarity={r16['familiarity']:.4f}  NaN={has_nan}  ∈[0,1]={in_range}")


# ═══════════════════════════════════════════════════════════════
# BT-17  Full pipeline — second encounter more familiar than first
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: This is the most important end-to-end test. It verifies
# that M55 correctly recognizes repetition in a real M50+M54 stream —
# not just in toy isolated training. The same frequency appearing a
# second time should produce a higher familiarity score than the first
# encounter. This is the core behavioral contract of M55.
# ───────────────────────────────────────────────────────────────
section("BT-17  Full pipeline — 2nd encounter more familiar than 1st")

print("  Running: 3 frequencies × 2 repeats with explicit recall per step...")
np.random.seed(17)
cortex_17 = CortexM54(seed=17)
memory_17 = AssociativeMemory(seed=17)

freqs_17  = [0.60, 1.20, 1.80]
# Run 2 complete passes through the same 3 frequencies
sig_17, _ = make_blocks(freqs_17 * 2, block_dur=40.0)
d_17      = run_sim(sig_17,
    total_time=stabilization_time + 2*len(freqs_17)*2*40.0 + 10.0,
    sweep_mode=False, dynamic_settle=False, verbose=False)

n17         = len(d_17['T'])
plv_hist    = deque(maxlen=PLV_STAB_WINDOW)
cusum       = DivergenceCUSUM()
Y17         = d_17['Y']

# Pass assignment: purely by simulation time.
# The run has 2 complete passes through freqs_17 (each block_dur=40s).
# One pass = len(freqs_17) * 40s = 3 * 40 = 120s of active frequency time.
# Everything before the midpoint of active time = pass 1.
# Everything from the midpoint onward = pass 2.
# We use the simulation time range to find the midpoint cleanly.
T17           = d_17['T']
t_start_active = float(T17[0])
t_end_active   = float(T17[-1])
t_mid          = (t_start_active + t_end_active) / 2.0

fam_by_freq_pass = {f: {1: [], 2: []} for f in freqs_17}

for i in range(n17):
    plv_fast_i = d_17['plv_fast'][i]
    plv_slow_i = d_17['plv_slow'][i]
    e_fast_i   = d_17['energy_fast'][i]
    e_slow_i   = d_17['energy_slow'][i]
    t          = float(T17[i])
    y_true     = float(Y17[i])

    df = decode_resonance(plv_fast_i, e_fast_i, raw_x_fast, raw_y_fast)
    ds = decode_resonance(plv_slow_i, e_slow_i, raw_x_slow, raw_y_slow)

    max_plv = float(np.abs(plv_slow_i).max())
    plv_hist.append(max_plv)
    w = compute_stability_plv(plv_hist)

    _, nov = cusum.update(df, ds, t, w=w)
    if nov:
        w = 0.0
    fused = w * ds + (1.0 - w) * df

    cortex_out = cortex_17.step(
        decoded_freq=fused, stability_w=w,
        novelty_flag=float(nov), plv_vector=plv_slow_i,
    )
    memory_17.step(cortex_out['bmu_idx'], cortex_out['qe_norm'])
    recall_out = memory_17.recall(cortex_out['bmu_idx'])

    # Pass is determined by simulation time — first half vs second half.
    # This is the only correct approach: pass is a temporal property of
    # the run, not a property of the BMU's exposure count.
    pass_num = 1 if t < t_mid else 2

    for f in freqs_17:
        if abs(y_true - f) < 0.01:
            fam_by_freq_pass[f][pass_num].append(recall_out['familiarity'])

print(f"\n  {'Freq':>6}  {'Pass1 mean':>12}  {'Pass2 mean':>12}  {'Growing':>8}")
all_growing = True
for f in freqs_17:
    p1 = fam_by_freq_pass[f][1]
    p2 = fam_by_freq_pass[f][2]
    m1 = float(np.mean(p1)) if p1 else 0.0
    m2 = float(np.mean(p2)) if p2 else 0.0
    growing = m2 > m1
    if not growing:
        all_growing = False
    print(f"  {f:6.2f}  {m1:12.4f}  {m2:12.4f}  {'✓' if growing else '✗'}")

report("BT-17 2nd encounter more familiar",
       all_growing,
       f"All frequencies show higher familiarity on 2nd pass: {all_growing}",
       warn=(sum(
           float(np.mean(fam_by_freq_pass[f][2])) >
           float(np.mean(fam_by_freq_pass[f][1]))
           for f in freqs_17
           if fam_by_freq_pass[f][2]
       ) >= 2))


# ═══════════════════════════════════════════════════════════════
# BT-18  M55 persists independently of M54 reset
# ═══════════════════════════════════════════════════════════════
# ROOT CAUSE: M55 owns its own state (W, trace, exposure counters).
# It has no reference to M54's internal state — it only receives
# bmu_idx and qe_norm as inputs. If M54 is replaced with a fresh
# instance (cortex reset), M55's memories should be unaffected.
# This verifies the architectural independence: M54 and M55 are
# truly decoupled layers. A production system might reset the cortex
# (re-learning the frequency map) while keeping the memory intact.
# ───────────────────────────────────────────────────────────────
section("BT-18  M55 persists independently of M54 reset")

mem_18 = AssociativeMemory()

# Train memory with first cortex instance
for _ in range(300):
    mem_18.step(25, qe_norm=0.8)
    mem_18.step(26, qe_norm=0.8)

fam_before = mem_18.recall(25)['familiarity']
W_before   = mem_18.get_state()['W_snapshot'].copy()
exp_before = int(mem_18._bmu_exposure[25])

# "Reset" M54 (create new instance) — M55 is unchanged
cortex_new = CortexM54(seed=999)   # brand new cortex, forgets everything

# M55 state should be identical
fam_after = mem_18.recall(25)['familiarity']
W_after   = mem_18.get_state()['W_snapshot'].copy()
exp_after = int(mem_18._bmu_exposure[25])

w_identical  = float(np.abs(W_before - W_after).max()) < 1e-8
fam_identical = abs(fam_before - fam_after) < 1e-6
exp_identical = exp_before == exp_after

print(f"  Before cortex reset: familiarity={fam_before:.4f}  exposure={exp_before}")
print(f"  After cortex reset:  familiarity={fam_after:.4f}  exposure={exp_after}")
print(f"  W unchanged: {w_identical}  fam unchanged: {fam_identical}")

ok = w_identical and fam_identical and exp_identical
report("BT-18 M55 persists after M54 reset",
       ok,
       f"W unchanged={w_identical}  fam unchanged={fam_identical}  "
       f"exposure unchanged={exp_identical}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

summarise()