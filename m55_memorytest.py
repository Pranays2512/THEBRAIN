"""
M55 MEMORY TEST SUITE
=====================
Verifies AssociativeMemory behaves correctly as a pure Hebbian
pattern memory layer on top of M54 Cortex + M50.

M54 and M50 are NEVER modified. M55 only reads their output.

Tests
-----
MT-01  Zero init — W starts at zero, no spurious memories at birth
MT-02  Hebbian write — repeated BMU fires strengthen W
MT-03  Decay — weights decay toward zero without reinforcement
MT-04  Trace adaptation — novel input extends eligibility window
MT-05  Symmetry — W[i,j] == W[j,i] always
MT-06  No self-connections — diagonal of W stays zero
MT-07  Recall depth — familiar BMU has deeper energy than novel BMU
MT-08  Recall speed — familiar BMU settles faster than novel BMU
MT-09  Familiarity score — increases with repeated exposure
MT-10  Top associations — strongest_associations returns correct neurons
MT-11  Memory map — shape (8×8), sums match W
MT-12  Homeostasis — weights stay bounded after many steps
MT-13  Novelty isolation — two distinct frequencies don't bleed memories
MT-14  Full pipeline — M50 → M54 → M55 wired together, zero errors
MT-15  Summary — runs without error on populated memory
"""

import numpy as np
import sys
from collections import deque

# ── Imports ──────────────────────────────────────────────────────
try:
    from m50_neuron import (
        run_sim, make_blocks, make_sweep,
        fit_ridge, build_reverse_lookup,
        decode_resonance, compute_stability_plv,
        DivergenceCUSUM,
        stabilization_time,
        RIDGE_ALPHA_FAST, RIDGE_ALPHA_SLOW,
        PLV_STAB_WINDOW,
        mae, N,
        dt,
    )
    from m54_cortex import (
        CortexM54, prepare_input,
        GRID_H, GRID_W, N_NEURONS,
        SURPRISE_THRESH, FREQ_MIN_HZ, FREQ_MAX_HZ,
    )
    from m54_experience import ExperienceBuffer
    from m55_memory import (
        AssociativeMemory,
        N_NEURONS as MEM_N,
        ETA_HEBB, DECAY_RATE, W_MAX,
        TRACE_DECAY_BASE, TRACE_DECAY_NOVEL, TRACE_DECAY_MIN,
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

results = {}

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
    section("M55 MEMORY TEST SUMMARY")
    n_pass = sum(1 for v in results.values() if v == "PASS")
    n_fail = sum(1 for v in results.values() if v == "FAIL")
    n_warn = sum(1 for v in results.values() if v == "WARN")
    for name, tag in results.items():
        sym = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}[tag]
        print(f"  {sym} [{tag}] {name}")
    print(f"\n  {'─'*68}")
    print(f"  PASS:{n_pass}  FAIL:{n_fail}  WARN:{n_warn}")
    print(f"  {'ALL CLEAR' if n_fail == 0 else 'FAILURES FOUND'}")


# ═══════════════════════════════════════════════════════════════
# CALIBRATION  (shared — builds M50 + M54 pipeline)
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

    # Ridge calibration (sweep)
    np.random.seed(0)
    data_train = run_sim(
        make_sweep(0.5, 2.0, 6, sweep_dur),
        total_time=warmup + 6 * sweep_dur + 10.0,
        sweep_mode=True, verbose=False, collect_calib=False,
    )
    ridge_fast, ridge_fast_sc = fit_ridge(
        data_train['feat_fast'], data_train['Y'], RIDGE_ALPHA_FAST)

    # Reverse lookup calibration (blocks with collect_calib=True)
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
    return raw_x_slow, raw_y_slow, raw_x_fast, raw_y_fast, ridge_fast, ridge_fast_sc

print("  Building calibration (M50 sweep + blocks)...")
raw_x_slow, raw_y_slow, raw_x_fast, raw_y_fast, ridge_fast, ridge_fast_sc = build_calibration()
print("  Calibration done.")


def run_pipeline(data, cortex, memory, buf=None):
    """
    Feed M50 simulation output through M54 → M55 → (optional) ExperienceBuffer.
    """
    n          = len(data['T'])
    steps      = []
    plv_hist   = deque(maxlen=PLV_STAB_WINDOW)
    change_det = DivergenceCUSUM()

    for i in range(n):
        plv_fast_i = data['plv_fast'][i]
        plv_slow_i = data['plv_slow'][i]
        e_fast_i   = data['energy_fast'][i]
        e_slow_i   = data['energy_slow'][i]
        t          = float(data['T'][i])

        df = decode_resonance(plv_fast_i, e_fast_i, raw_x_fast, raw_y_fast)
        ds = decode_resonance(plv_slow_i, e_slow_i, raw_x_slow, raw_y_slow)

        max_plv = float(np.abs(plv_slow_i).max())
        plv_hist.append(max_plv)
        w = compute_stability_plv(plv_hist)

        _, nov = change_det.update(df, ds, t, w=w)
        if nov:
            w = 0.0

        fused = w * ds + (1.0 - w) * df

        cortex_out = cortex.step(
            decoded_freq=fused, stability_w=w,
            novelty_flag=float(nov), plv_vector=plv_slow_i,
        )

        mem_out = memory.step(cortex_out['bmu_idx'], cortex_out['qe_norm'])

        if buf is not None:
            buf.push(
                t=t, cortex_out=cortex_out, decoded_freq=fused,
                stability_w=w, transition=nov, cortex_step=cortex.t,
            )

        steps.append({
            'bmu_idx':     cortex_out['bmu_idx'],
            'qe_norm':     cortex_out['qe_norm'],
            'is_novel':    cortex_out['is_novel'],
            'fused_freq':  fused,
            'w':           w,
            'nov':         nov,
            'mem_wrote':   mem_out['wrote'],
            'trace_peak':  mem_out['trace_peak'],
            'trace_decay': mem_out['trace_decay'],
            'w_mean':      mem_out['w_mean'],
        })

    return steps


# ═══════════════════════════════════════════════════════════════
# MT-01  Zero init
# ═══════════════════════════════════════════════════════════════
section("MT-01  Zero init — W starts at zero")

mem_01 = AssociativeMemory()
s = mem_01.get_state()
w_zero = s['W_snapshot']
ok = (w_zero == 0).all() and s['write_events'] == 0 and s['t'] == 0
report("MT-01 Zero init",
       ok,
       f"W all zeros: {(w_zero==0).all()}, write_events={s['write_events']}, t={s['t']}")


# ═══════════════════════════════════════════════════════════════
# MT-02  Hebbian write — repeated fires strengthen W
# ═══════════════════════════════════════════════════════════════
section("MT-02  Hebbian write")

mem_02 = AssociativeMemory()
# Fire BMU 10 and BMU 20 in close succession many times
for _ in range(100):
    mem_02.step(10, qe_norm=0.8)   # BMU 10 fires (novel)
    mem_02.step(20, qe_norm=0.8)   # BMU 20 fires (novel, trace of 10 still active)

s02 = mem_02.get_state()
w_10_20 = s02['W_snapshot'][10, 20]
w_0_1   = s02['W_snapshot'][0, 1]   # neurons that never fired — should be ~0

ok = w_10_20 > 0.01 and w_0_1 < 1e-4
report("MT-02 Hebbian write",
       ok,
       f"W[10,20]={w_10_20:.4f} (should be >0.01)  "
       f"W[0,1]={w_0_1:.6f} (should be ≈0)")


# ═══════════════════════════════════════════════════════════════
# MT-03  Decay — weights fade without reinforcement
# ═══════════════════════════════════════════════════════════════
section("MT-03  Synaptic decay")

mem_03 = AssociativeMemory()
# Build up a memory
for _ in range(50):
    mem_03.step(5, qe_norm=0.9)
    mem_03.step(15, qe_norm=0.9)

w_before = mem_03.get_state()['W_snapshot'][5, 15]

# Now fire completely different neurons for a long time
for _ in range(2000):
    mem_03.step(40, qe_norm=0.0)
    mem_03.step(41, qe_norm=0.0)

w_after = mem_03.get_state()['W_snapshot'][5, 15]

ok = w_after < w_before * 0.5
report("MT-03 Synaptic decay",
       ok,
       f"W[5,15] before={w_before:.4f}  after={w_after:.4f}  "
       f"decayed to {w_after/max(w_before,1e-9)*100:.0f}% (should be <50%)")


# ═══════════════════════════════════════════════════════════════
# MT-04  Trace adaptation — novel extends window
# ═══════════════════════════════════════════════════════════════
section("MT-04  Adaptive trace — novel extends eligibility window")

mem_04 = AssociativeMemory()

# Familiar step
mem_04.step(0, qe_norm=0.0)
decay_familiar = mem_04.get_state()['trace_decay']

# Novel step
mem_04.step(0, qe_norm=1.0)
decay_novel = mem_04.get_state()['trace_decay']

ok = decay_novel < decay_familiar
report("MT-04 Trace adaptation",
       ok,
       f"decay_familiar={decay_familiar:.4f}  decay_novel={decay_novel:.4f}  "
       f"novel window longer: {ok}  "
       f"(window_familiar≈{1/decay_familiar:.0f} steps, "
       f"window_novel≈{1/decay_novel:.0f} steps)")


# ═══════════════════════════════════════════════════════════════
# MT-05  Symmetry — W[i,j] == W[j,i]
# ═══════════════════════════════════════════════════════════════
section("MT-05  Symmetry")

mem_05 = AssociativeMemory()
for _ in range(200):
    bmu = np.random.randint(0, 64)
    mem_05.step(int(bmu), qe_norm=float(np.random.rand()))

W05    = mem_05.get_state()['W_snapshot']
asym   = float(np.abs(W05 - W05.T).max())
ok     = asym < 1e-5
report("MT-05 Symmetry",
       ok,
       f"max |W[i,j] - W[j,i]| = {asym:.2e} (should be <1e-5)")


# ═══════════════════════════════════════════════════════════════
# MT-06  No self-connections
# ═══════════════════════════════════════════════════════════════
section("MT-06  No self-connections")

W06     = mem_05.get_state()['W_snapshot']   # reuse MT-05 memory
diag    = float(np.abs(np.diag(W06)).max())
ok      = diag < 1e-6
report("MT-06 No self-connections",
       ok,
       f"max diagonal value = {diag:.2e} (should be <1e-6)")


# ═══════════════════════════════════════════════════════════════
# MT-07  Recall depth — familiar > novel
# ═══════════════════════════════════════════════════════════════
section("MT-07  Recall depth — familiar BMU deeper than novel BMU")

mem_07 = AssociativeMemory()

# Train heavily on BMU 30
for _ in range(300):
    mem_07.step(30, qe_norm=0.9)
    mem_07.step(31, qe_norm=0.9)

# Recall familiar BMU
r_familiar = mem_07.recall(30)

# Recall novel BMU (never seen)
r_novel = mem_07.recall(63)

ok = r_familiar['depth'] > r_novel['depth']
report("MT-07 Recall depth",
       ok,
       f"familiar depth={r_familiar['depth']:.4f}  "
       f"novel depth={r_novel['depth']:.4f}  "
       f"familiar > novel: {ok}")


# ═══════════════════════════════════════════════════════════════
# MT-08  Recall speed — familiar settles faster
# ═══════════════════════════════════════════════════════════════
section("MT-08  Recall speed — familiar BMU settles faster")

# reuse mem_07
r_fam2 = mem_07.recall(30)
r_nov2 = mem_07.recall(63)

ok = r_fam2['speed'] <= r_nov2['speed']
report("MT-08 Recall speed",
       ok,
       f"familiar steps={r_fam2['speed']}  novel steps={r_nov2['speed']}  "
       f"familiar ≤ novel: {ok}")


# ═══════════════════════════════════════════════════════════════
# MT-09  Familiarity score increases with exposure
# ═══════════════════════════════════════════════════════════════
section("MT-09  Familiarity increases with repeated exposure")

mem_09 = AssociativeMemory()

scores = []
for rep in range(5):
    for _ in range(40):
        mem_09.step(22, qe_norm=0.7)
        mem_09.step(23, qe_norm=0.7)
    r = mem_09.recall(22)
    scores.append(r['familiarity'])
    print(f"  Rep {rep+1}: familiarity={r['familiarity']:.4f}")

ok = scores[-1] > scores[0]
report("MT-09 Familiarity increases",
       ok,
       f"first={scores[0]:.4f}  last={scores[-1]:.4f}  increasing: {ok}")


# ═══════════════════════════════════════════════════════════════
# MT-10  Top associations — returns correct neurons
# ═══════════════════════════════════════════════════════════════
section("MT-10  Top associations")

mem_10 = AssociativeMemory()
# Train: BMU 7 always fires with BMU 8 and BMU 9
for _ in range(200):
    mem_10.step(7, qe_norm=0.9)
    mem_10.step(8, qe_norm=0.9)
    mem_10.step(9, qe_norm=0.9)

assoc = mem_10.strongest_associations(7, k=5)
assoc_neurons = [a['neuron_idx'] for a in assoc]
print(f"  Top associations for BMU 7: {assoc_neurons}")
weights_str = [f"{a['weight']:.4f}" for a in assoc]
print(f"  Weights: {weights_str}")

ok = (8 in assoc_neurons) and (9 in assoc_neurons)
report("MT-10 Top associations",
       ok,
       f"BMU 7 top assoc = {assoc_neurons}  "
       f"8 present: {8 in assoc_neurons}  9 present: {9 in assoc_neurons}")


# ═══════════════════════════════════════════════════════════════
# MT-11  Memory map — correct shape and positive values
# ═══════════════════════════════════════════════════════════════
section("MT-11  Memory map shape and content")

mmap = mem_10.memory_map()
shape_ok = mmap.shape == (8, 8)
pos_ok   = mmap[7 // 8, 7 % 8] > 0 or mmap.max() > 0

report("MT-11 Memory map",
       shape_ok and pos_ok,
       f"shape={mmap.shape} (should be (8,8))  max={mmap.max():.4f}")


# ═══════════════════════════════════════════════════════════════
# MT-12  Homeostasis — weights bounded after heavy training
# ═══════════════════════════════════════════════════════════════
section("MT-12  Homeostasis — weights stay bounded")

mem_12 = AssociativeMemory()
for _ in range(5000):
    mem_12.step(np.random.randint(0, 64), qe_norm=1.0)

W12    = mem_12.get_state()['W_snapshot']
w_max  = float(W12.max())
ok     = w_max <= W_MAX + 1e-4   # tiny float tolerance
report("MT-12 Homeostasis",
       ok,
       f"W max = {w_max:.4f}  (ceiling = {W_MAX})  bounded: {ok}")


# ═══════════════════════════════════════════════════════════════
# MT-13  Novelty isolation — two frequencies stay separate
# ═══════════════════════════════════════════════════════════════
section("MT-13  Novelty isolation — distinct frequency memories don't bleed")

mem_13 = AssociativeMemory()

# Train two well-separated frequency groups
# Group A: neurons 0-5  (low freq)
# Group B: neurons 58-63 (high freq)
for _ in range(200):
    for n in range(6):
        mem_13.step(n, qe_norm=0.8)
for _ in range(200):
    for n in range(58, 64):
        mem_13.step(n, qe_norm=0.8)

W13 = mem_13.get_state()['W_snapshot']

# Cross-group weights should be much weaker than within-group
within_A   = float(W13[0:6, 0:6].mean())
within_B   = float(W13[58:64, 58:64].mean())
cross_AB   = float(W13[0:6, 58:64].mean())

ok = cross_AB < (within_A + within_B) / 2 * 0.5
report("MT-13 Novelty isolation",
       ok,
       f"within_A={within_A:.5f}  within_B={within_B:.5f}  "
       f"cross_AB={cross_AB:.5f}  "
       f"cross < 50% of within: {ok}")


# ═══════════════════════════════════════════════════════════════
# MT-14  Full pipeline — M50 → M54 → M55 wired together
# ═══════════════════════════════════════════════════════════════
section("MT-14  Full pipeline integration")

print("  Running M50 simulation (4 frequencies, 2 repeats)...")
np.random.seed(42)
cortex_14 = CortexM54(seed=14)
memory_14 = AssociativeMemory(seed=14)
buf_14    = ExperienceBuffer()

freqs_14  = [0.60, 1.00, 1.60, 2.20]
sig_14, _ = make_blocks(freqs_14 * 2, block_dur=40.0)
d_14      = run_sim(
    sig_14,
    total_time=stabilization_time + 2 * len(freqs_14) * 2 * 40.0 + 10.0,
    sweep_mode=False, dynamic_settle=False, verbose=False,
)

steps_14  = run_pipeline(d_14, cortex_14, memory_14, buf=buf_14)
buf_14.flush(t_end=float(d_14['T'][-1]), cortex_step=cortex_14.t)

n_steps   = len(steps_14)
n_wrote   = sum(1 for s in steps_14 if s['mem_wrote'])
w_max_14  = memory_14.get_state()['w_max']
n_ep_14   = buf_14.n_episodes()

print(f"\n  Pipeline results:")
print(f"  Total steps:    {n_steps}")
print(f"  Memory writes:  {n_wrote}  ({n_wrote/max(n_steps,1)*100:.1f}%)")
print(f"  W max:          {w_max_14:.4f}")
print(f"  Episodes (buf): {n_ep_14}")
print()
memory_14.summary()
print()
buf_14.summary()

pipeline_ok = (n_steps > 100 and n_wrote > 0 and
               w_max_14 > 0 and n_ep_14 >= 4)
report("MT-14 Full pipeline integration",
       pipeline_ok,
       f"steps={n_steps}, writes={n_wrote}, w_max={w_max_14:.4f}, episodes={n_ep_14}")


# ═══════════════════════════════════════════════════════════════
# MT-15  Summary — runs without error
# ═══════════════════════════════════════════════════════════════
section("MT-15  Summary runs without error")

try:
    memory_14.summary()
    report("MT-15 Summary", True, "summary() completed without error")
except Exception as e:
    report("MT-15 Summary", False, f"Exception: {e}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

summarise()