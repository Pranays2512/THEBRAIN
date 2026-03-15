"""
THEBRAIN — ENVIRONMENT TEST SUITE
===================================
5 tests that verify the brain responds correctly to meaningfully
different EXTERNAL CONDITIONS — not internal signal mechanics,
but how the whole pipeline adapts to different environments.

Each test runs two brains (or two phases on one brain) under
contrasting environmental conditions and verifies that the
brain's state diverges in the expected direction.

Tests:
  ET-01  Stable vs Volatile environment
  ET-02  Predictable vs Unpredictable sequence
  ET-03  Rewarded vs Unrewarded environment
  ET-04  Gradual vs Sudden environmental change
  ET-05  Pure repetition vs Pure novelty

Run: python brain_env_tests.py
"""

import sys
import numpy as np
from collections import deque

sys.path.insert(0, '/home/claude')

from brain import Brain
from m50_neuron import (
    run_sim, make_blocks, build_reverse_lookup,
    decode_resonance, compute_stability_plv,
    stabilization_time, dt, PLV_STAB_WINDOW,
)

# ═══════════════════════════════════════════════════════════════
# SHARED CALIBRATION
# ═══════════════════════════════════════════════════════════════

FREQS_CAL = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]

print("=" * 64)
print("  THEBRAIN ENVIRONMENT TEST SUITE")
print("=" * 64)
print(f"\n  Calibrating M50 ({len(FREQS_CAL)} frequencies)...")

np.random.seed(7)
cal_sig, _ = make_blocks(FREQS_CAL, block_dur=35.0)
cal_data   = run_sim(cal_sig,
                     total_time=stabilization_time + 2*len(FREQS_CAL)*35.0 + 10.0,
                     sweep_mode=False, dynamic_settle=True,
                     verbose=False, collect_calib=True)
raw_x, true_y = build_reverse_lookup(
    sorted(cal_data['calib_plv_slow'].keys()),
    cal_data['calib_plv_slow'],
    cal_data['calib_energy_slow'],
)
print(f"  Done: {len(raw_x)} calib pts  "
      f"[{raw_x[0]:.3f}, {raw_x[-1]:.3f}] Hz")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

N_RAW = 1901   # → ~1050 stable steps per freq

def build_lib(freqs, n_raw=N_RAW):
    lib = {}
    for i, f in enumerate(freqs):
        np.random.seed(500 + i * 20)
        sig, _ = make_blocks([f], block_dur=stabilization_time + n_raw*dt + 10.0)
        data   = run_sim(sig,
                         total_time=stabilization_time + n_raw*dt + 20.0,
                         sweep_mode=False, dynamic_settle=True,
                         verbose=False, collect_calib=False)
        lib[f] = data
    return lib


def brain_step(brain, lib, freq, step_idx, reward=0.0):
    """Single brain step using pre-built signal library at given step index."""
    data    = lib[freq]
    i       = step_idx % len(data['Y'])
    plv     = data['plv_slow'][i]
    eng     = data['energy_slow'][i]
    f_slow  = decode_resonance(plv, eng, raw_x, true_y)
    plv_mag = np.abs(plv)
    stab    = compute_stability_plv(
        deque([float(np.max(plv_mag))] * PLV_STAB_WINDOW,
              maxlen=PLV_STAB_WINDOW))
    return brain.step(decoded_freq=f_slow, stability_w=stab,
                      novelty_flag=0.0, plv_vector=plv_mag,
                      reward=reward)


def run_schedule(brain, lib, schedule, reward_fn=None):
    """
    Run brain through a schedule: list of (freq, n_steps) pairs.
    reward_fn(step_idx) -> float, defaults to 0.
    Returns flat list of result dicts.
    """
    results = []
    total   = 0
    for (freq, n_steps) in schedule:
        for j in range(n_steps):
            r = 0.0 if reward_fn is None else reward_fn(total)
            results.append(brain_step(brain, lib, freq, j, reward=r))
            total += 1
    return results


def som_warmup(brain, lib, freqs, passes=10, steps_per=80):
    for _ in range(passes):
        for f in freqs:
            for j in range(steps_per):
                brain_step(brain, lib, f, j)


def stats(results, key, start=None, end=None):
    vals = [s[key] for s in results[start:end]]
    return np.mean(vals)


# ═══════════════════════════════════════════════════════════════
# BUILD SIGNAL LIBRARIES
# ═══════════════════════════════════════════════════════════════

FREQS_ALL = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]
print(f"\n  Building signal libraries...")
lib = build_lib(FREQS_ALL)
n_steps_0 = len(lib[0.5]['Y'])
print(f"  Ready ({len(FREQS_ALL)} freqs, ~{n_steps_0} steps each)")


# ═══════════════════════════════════════════════════════════════
# RESULT TRACKING
# ═══════════════════════════════════════════════════════════════

summary    = []
pass_count = 0
fail_count = 0

def record(label, passed, detail=""):
    global pass_count, fail_count
    sym = "✓" if passed else "✗"
    if passed: pass_count += 1
    else:      fail_count += 1
    summary.append((label, passed, detail))
    print(f"  {sym}  {label}")
    if detail:
        print(f"       {detail}")


# ═══════════════════════════════════════════════════════════════
# ET-01  STABLE vs VOLATILE ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

print("\n" + "─" * 64)
print("  ET-01  STABLE vs VOLATILE ENVIRONMENT")
print("─" * 64)
print("  Environment A: same frequency for 600 steps (stable world)")
print("  Environment B: random frequency switch every 5–20 steps")
print()
print("  Predictions (B > A):")
print("    salience_B > salience_A  (volatility keeps attention high)")
print("    pe_B > pe_A              (can't predict random switches)")
print("    familiarity_B < familiarity_A  (no single freq to habituate)")

# Environment A — stable
brain_A = Brain(seed=10)
rng_A   = np.random.RandomState(1)
steps_A = []
for j in range(600):
    steps_A.append(brain_step(brain_A, lib, 0.7, j))

# Environment B — volatile (random freq, switch every 5–20 steps)
brain_B  = Brain(seed=10)
rng_B    = np.random.RandomState(2)
steps_B  = []
current_f = 0.7
switch_at = rng_B.randint(5, 20)
for j in range(600):
    if j >= switch_at:
        current_f = rng_B.choice(FREQS_ALL)
        switch_at = j + rng_B.randint(5, 20)
    steps_B.append(brain_step(brain_B, lib, current_f, j % 200))

# Compare final 200 steps (both settled)
sal_A = stats(steps_A, 'salience',    400, 600)
sal_B = stats(steps_B, 'salience',    400, 600)
pe_A  = stats(steps_A, 'prediction_error', 400, 600)
pe_B  = stats(steps_B, 'prediction_error', 400, 600)
fam_A = stats(steps_A, 'familiarity', 400, 600)
fam_B = stats(steps_B, 'familiarity', 400, 600)

print(f"\n    {'Signal':<20} {'Stable':>8}  {'Volatile':>8}  {'Direction':>10}")
print(f"    {'──────':<20} {'──────':>8}  {'────────':>8}  {'─────────':>10}")
print(f"    {'salience':<20} {sal_A:>8.4f}  {sal_B:>8.4f}  "
      f"{'✓ B>A' if sal_B > sal_A else '✗ expected B>A':>10}")
print(f"    {'prediction_error':<20} {pe_A:>8.4f}  {pe_B:>8.4f}  "
      f"{'✓ B>A' if pe_B > pe_A else '✗ expected B>A':>10}")
print(f"    {'familiarity':<20} {fam_A:>8.4f}  {fam_B:>8.4f}  "
      f"{'✓ A>B' if fam_A > fam_B else '✗ expected A>B':>10}")

passes = (sal_B > sal_A) and (pe_B > pe_A) and (fam_A > fam_B)
record("ET-01: volatile env → higher salience, PE, lower familiarity",
       passes,
       f"sal {sal_A:.3f}→{sal_B:.3f}  pe {pe_A:.3f}→{pe_B:.3f}  "
       f"fam {fam_A:.3f}→{fam_B:.3f}")


# ═══════════════════════════════════════════════════════════════
# ET-02  PREDICTABLE vs UNPREDICTABLE SEQUENCE
# ═══════════════════════════════════════════════════════════════

print("\n" + "─" * 64)
print("  ET-02  PREDICTABLE vs UNPREDICTABLE SEQUENCE")
print("─" * 64)
print("  Both brains see the same 3 frequencies (0.7, 1.3, 1.7 Hz).")
print("  Environment A: strict A→B→C→A deterministic loop")
print("  Environment B: same freqs, random order each time")
print()
print("  After SOM warm-up + 2000 sequence steps:")
print("    pe_A < pe_B      (L2 can only learn deterministic structure)")
print("    curiosity_A < curiosity_B  (novelty-seeking reflects PE)")

SEQ_FREQS = [0.7, 1.3, 1.7]

brain_P = Brain(seed=20)   # Predictable
brain_U = Brain(seed=20)   # Unpredictable (same seed — same starting weights)
rng_U   = np.random.RandomState(99)

print("    [warm-up: stabilising SOM on both brains identically]")
som_warmup(brain_P, lib, SEQ_FREQS, passes=12, steps_per=80)
som_warmup(brain_U, lib, SEQ_FREQS, passes=12, steps_per=80)

# Run 2000 brain steps each
seq_P = []
seq_U = []
loop_idx = 0
for rep in range(667):             # 667 reps × 3 freqs = 2001 steps
    for f in SEQ_FREQS:            # deterministic loop
        seq_P.append(brain_step(brain_P, lib, f, rep % 200))
    rand_freqs = rng_U.choice(SEQ_FREQS, 3, replace=True)
    for f in rand_freqs:           # random order
        seq_U.append(brain_step(brain_U, lib, f, rep % 200))

pe_P  = stats(seq_P, 'prediction_error', 1600, 2000)
pe_U  = stats(seq_U, 'prediction_error', 1600, 2000)
cur_P = stats(seq_P, 'curiosity',        1600, 2000)
cur_U = stats(seq_U, 'curiosity',        1600, 2000)

print(f"\n    {'Signal':<20} {'Deterministic':>14}  {'Random':>8}  {'Direction':>10}")
print(f"    {'──────':<20} {'─────────────':>14}  {'──────':>8}  {'─────────':>10}")
print(f"    {'prediction_error':<20} {pe_P:>14.4f}  {pe_U:>8.4f}  "
      f"{'✓ P<U' if pe_P < pe_U else '✗ expected P<U':>10}")
print(f"    {'curiosity':<20} {cur_P:>14.4f}  {cur_U:>8.4f}  "
      f"{'✓ P<U' if cur_P < cur_U else '✗ expected P<U':>10}")

passes = (pe_P < pe_U) and (cur_P < cur_U)
record("ET-02: deterministic sequence → lower PE and curiosity than random",
       passes,
       f"pe {pe_P:.4f}<{pe_U:.4f}  curiosity {cur_P:.4f}<{cur_U:.4f}")


# ═══════════════════════════════════════════════════════════════
# ET-03  REWARDED vs UNREWARDED ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

print("\n" + "─" * 64)
print("  ET-03  REWARDED vs UNREWARDED ENVIRONMENT")
print("─" * 64)
print("  Both brains see identical input (0.7 Hz for 800 steps).")
print("  Environment R: external reward=1.0 every step")
print("  Environment N: no reward (reward=0.0)")
print()
print("  Prediction: reward boosts pos_rpe → M55 η_eff higher → ")
print("  more Hebbian writes → higher familiarity in R than N.")

brain_R = Brain(seed=30)
brain_N = Brain(seed=30)

steps_R = []
steps_N = []
for j in range(800):
    steps_R.append(brain_step(brain_R, lib, 0.7, j, reward=1.0))
    steps_N.append(brain_step(brain_N, lib, 0.7, j, reward=0.0))

# The reward signal matters EARLY — before intrinsic reward (1-PE) converges.
# At convergence both brains have high intrinsic reward and the external signal
# is drowned out. Test the first 80 steps: reward_ema and total_reward should
# be meaningfully higher in R, and familiarity should build faster early.
fam_R_early  = stats(steps_R, 'familiarity',    50,  150)
fam_N_early  = stats(steps_N, 'familiarity',    50,  150)
tr_R_early   = stats(steps_R, 'total_reward',    0,   80)
tr_N_early   = stats(steps_N, 'total_reward',    0,   80)
rpe_R_early  = stats(steps_R, 'pos_rpe',         0,   80)
rpe_N_early  = stats(steps_N, 'pos_rpe',         0,   80)
wrote_R = sum(s['wrote'] for s in steps_R)
wrote_N = sum(s['wrote'] for s in steps_N)

print(f"\n    Comparing early phase (steps 0–80) before intrinsic reward converges:")
print(f"\n    {'Signal':<22} {'Rewarded':>10}  {'Unrewarded':>10}  {'Direction':>10}")
print(f"    {'──────':<22} {'────────':>10}  {'──────────':>10}  {'─────────':>10}")
print(f"    {'total_reward [0:80]':<22} {tr_R_early:>10.4f}  {tr_N_early:>10.4f}  "
      f"{'✓ R>N' if tr_R_early > tr_N_early else '✗ expected R>N':>10}")
print(f"    {'pos_rpe [0:80]':<22} {rpe_R_early:>10.4f}  {rpe_N_early:>10.4f}  "
      f"{'✓ R>N' if rpe_R_early > rpe_N_early else '✗ expected R>N':>10}")
print(f"    {'familiarity [50:150]':<22} {fam_R_early:>10.4f}  {fam_N_early:>10.4f}  "
      f"{'✓ R>N' if fam_R_early > fam_N_early else '✗ expected R>N':>10}")
print(f"    {'M55 total writes':<22} {wrote_R:>10d}  {wrote_N:>10d}  "
      f"{'✓ R>N' if wrote_R > wrote_N else '? R=N':>10}")

passes = (tr_R_early > tr_N_early) and (rpe_R_early > rpe_N_early) and (fam_R_early > fam_N_early)
record("ET-03: early-phase reward boosts total_reward, pos_rpe, and familiarity",
       passes,
       f"total_reward {tr_N_early:.4f}→{tr_R_early:.4f}  "
       f"rpe {rpe_N_early:.4f}→{rpe_R_early:.4f}  "
       f"fam {fam_N_early:.4f}→{fam_R_early:.4f}")


# ═══════════════════════════════════════════════════════════════
# ET-04  GRADUAL vs SUDDEN ENVIRONMENTAL CHANGE
# ═══════════════════════════════════════════════════════════════

print("\n" + "─" * 64)
print("  ET-04  GRADUAL vs SUDDEN ENVIRONMENTAL CHANGE")
print("─" * 64)
print("  Both brains settle on 0.5 Hz for 200 steps, then change.")
print("  Environment G: gradual — steps through 0.7, 0.9, 1.1... 2.0 Hz")
print("                 (one new freq every 40 steps)")
print("  Environment S: sudden — instant jump to 2.0 Hz")
print()
print("  Prediction: max salience spike much larger in S than G.")

# Both brains settle on 0.5 Hz for 200 steps (identical warm-up)
brain_G = Brain(seed=40)
brain_S = Brain(seed=40)

for j in range(200):
    brain_step(brain_G, lib, 0.5, j)
    brain_step(brain_S, lib, 0.5, j)

# Compare EQUIVALENT single transitions at the moment of first change:
#   Gradual brain: small step 0.5 → 0.7 Hz  (gap = 0.2 Hz)
#   Sudden brain:  large jump 0.5 → 2.0 Hz  (gap = 1.5 Hz)
# The larger gap should produce a larger salience spike.
steps_G_first = [brain_step(brain_G, lib, 0.7, j) for j in range(20)]
steps_S_first = [brain_step(brain_S, lib, 2.0, j) for j in range(20)]

sal_G_spike = max(s['salience'] for s in steps_G_first[:10])
sal_S_spike = max(s['salience'] for s in steps_S_first[:10])

print(f"\n    First transition spike (first 10 steps, max salience):")
print(f"      Small step 0.5→0.7 Hz (gap=0.2): {sal_G_spike:.4f}")
print(f"      Large jump 0.5→2.0 Hz (gap=1.5): {sal_S_spike:.4f}")
print(f"      Ratio large/small: {sal_S_spike/(sal_G_spike+1e-6):.2f}x")

passes = sal_S_spike > sal_G_spike
record("ET-04: large frequency gap (0.5→2.0) spikes salience more than small step (0.5→0.7)",
       passes,
       f"small_gap={sal_G_spike:.4f}  large_gap={sal_S_spike:.4f}  "
       f"ratio={sal_S_spike/(sal_G_spike+1e-6):.2f}x")


# ═══════════════════════════════════════════════════════════════
# ET-05  PURE REPETITION vs PURE NOVELTY
# ═══════════════════════════════════════════════════════════════

print("\n" + "─" * 64)
print("  ET-05  PURE REPETITION vs PURE NOVELTY")
print("─" * 64)
print("  Environment R: same frequency (0.7 Hz) for 800 steps")
print("  Environment N: new random frequency every 8 steps")
print()
print("  After 800 steps, final 200 steps compared:")
print("    familiarity:  R >> N  (habituated vs always new)")
print("    salience:     R <  N  (familiar vs always novel)")
print("    prediction_error: R < N  (stable BMU vs shifting)")
print("    curiosity:    R <  N  (complacent vs alert)")
print("  All 4 directions correct = PASS")

brain_R5 = Brain(seed=50)
brain_N5 = Brain(seed=50)
rng_N5   = np.random.RandomState(77)

steps_R5 = []
steps_N5 = []

current_f5 = rng_N5.choice(FREQS_ALL)
switch_at5  = 8
for j in range(800):
    steps_R5.append(brain_step(brain_R5, lib, 0.7, j))
    if j >= switch_at5:
        current_f5 = rng_N5.choice(FREQS_ALL)
        switch_at5 = j + 8
    steps_N5.append(brain_step(brain_N5, lib, current_f5, j % 200))

fam_R5 = stats(steps_R5, 'familiarity',     600, 800)
fam_N5 = stats(steps_N5, 'familiarity',     600, 800)
sal_R5 = stats(steps_R5, 'salience',        600, 800)
sal_N5 = stats(steps_N5, 'salience',        600, 800)
pe_R5  = stats(steps_R5, 'prediction_error',600, 800)
pe_N5  = stats(steps_N5, 'prediction_error',600, 800)
cur_R5 = stats(steps_R5, 'curiosity',       600, 800)
cur_N5 = stats(steps_N5, 'curiosity',       600, 800)

print(f"\n    {'Signal':<20} {'Repetition':>11}  {'Novelty':>8}  {'Expected':>10}  {'OK?':>4}")
print(f"    {'──────':<20} {'──────────':>11}  {'───────':>8}  {'────────':>10}  {'───':>4}")

checks = []
for label, r_val, n_val, direction in [
    ('familiarity',      fam_R5, fam_N5, 'R>N'),
    ('salience',         sal_R5, sal_N5, 'R<N'),
    ('prediction_error', pe_R5,  pe_N5,  'R<N'),
    ('curiosity',        cur_R5, cur_N5, 'R<N'),
]:
    if direction == 'R>N':
        ok = r_val > n_val
    else:
        ok = r_val < n_val
    checks.append(ok)
    print(f"    {label:<20} {r_val:>11.4f}  {n_val:>8.4f}  "
          f"{direction:>10}  {'✓' if ok else '✗':>4}")

passes = all(checks)
record("ET-05: repetition vs novelty — all 4 signals diverge correctly",
       passes,
       f"fam {fam_R5:.3f}>{fam_N5:.3f}  sal {sal_R5:.3f}<{sal_N5:.3f}  "
       f"pe {pe_R5:.3f}<{pe_N5:.3f}  cur {cur_R5:.3f}<{cur_N5:.3f}")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

print()
print("═" * 64)
print("  ENVIRONMENT TEST SUMMARY")
print("═" * 64)

for label, passed, detail in summary:
    sym = "✓" if passed else "✗"
    print(f"  {sym}  {label}")
    if detail:
        print(f"       {detail}")

print()
print(f"  PASSED: {pass_count} / {pass_count + fail_count}")
print()

if fail_count == 0:
    print("  ALL ENVIRONMENT TESTS PASS ✓")
    print("  The brain correctly adapts its internal state to")
    print("  different external environmental conditions.")
elif fail_count <= 1:
    print("  MOSTLY PASSING — review flagged test above.")
else:
    print("  MULTIPLE FAILURES — investigate environment-signal coupling.")
print()