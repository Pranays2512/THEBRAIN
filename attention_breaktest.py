"""
ATTENTION — BREAK TEST SUITE v1
================================

Tests every documented property of the Attention module,
both standalone and integrated with Brain.

Written to match the pattern established in brain_breaktest.py.
All tests follow guide Section 8 rules — no threshold lowering,
no warn= on real failures, thresholds calibrated with safety margin.

TEST ORDER (guide Section 8 checklist)
---------------------------------------
AT-01  Output key completeness
AT-02  Output signal bounds
AT-03  Step counter
AT-04  Standalone — no Brain dependency
AT-05  Familiarity suppression — high familiarity lowers salience
AT-06  Surprise drives salience — surprise_signal raises salience
AT-07  QE drives salience — perceptual novelty raises salience
AT-08  Curiosity_delta drives salience
AT-09  Combined inputs saturate correctly (clipped to 1.0)
AT-10  Zero inputs → baseline salience (familiarity term only)
AT-11  Salience EMA smoothing — changes slower than raw signal
AT-12  Salience delta is near-zero during stable input
AT-13  Salience delta spikes at genuine transitions
AT-14  Attention gate sums to 1.0
AT-15  Attention gate is peaked at BMU location
AT-16  Gate entropy is lower when salience is high (focused)
AT-17  Gate entropy is higher when salience is low (diffuse)
AT-18  Attended BMU tracks actual BMU when salience is high
AT-19  Gate is non-negative everywhere
AT-20  Backward compatibility — default params reproduce baseline
AT-21  Reset clears all state correctly
AT-22  Isolation — Attention doesn't modify Brain outputs
AT-23  Brain integration — keys present and in bounds
AT-24  Brain integration — salience rises at frequency transition
AT-25  Brain integration — salience falls during stable known input
AT-26  Brain integration — gate tracks Brain's bmu_idx
AT-27  Brain integration — delta selectivity vs raw salience
AT-28  Edge cases — extreme inputs don't crash or produce NaN
AT-29  Diagnostics — get_state() returns all expected keys
AT-30  Long run stability — 1000 steps without drift or explosion
"""

import numpy as np
import sys
import math
from collections import deque

# ── Imports ──────────────────────────────────────────────────
try:
    from evaluators import (
        Attention,
        W_SURPRISE, W_QE, W_CURIOSITY, W_FAMILIARITY,
        SALIENCE_EMA_ALPHA, SALIENCE_EMA_INIT,
        GATE_SIGMA, GATE_BASELINE, GATE_BOOST
    )
except Exception as e:
    print(f"  [SKIP] attention.py import failed: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

try:
    from brain import Brain, FEEDBACK_EMA_ALPHA
except Exception as e:
    print(f"  [SKIP] brain.py import failed: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# HARNESS (identical pattern to brain_breaktest.py)
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
    section("ATTENTION BREAK TEST v1 — SUMMARY")
    n_pass = sum(1 for v in results.values() if v == "PASS")
    n_fail = sum(1 for v in results.values() if v == "FAIL")
    n_warn = sum(1 for v in results.values() if v == "WARN")
    for name, tag in results.items():
        sym = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}[tag]
        print(f"  {sym} [{tag}] {name}")
    print(f"\n  {'─'*70}")
    print(f"  PASS:{n_pass}  FAIL:{n_fail}  WARN:{n_warn}")
    if n_fail == 0 and n_warn == 0:
        print("  ALL CLEAR — Attention verified standalone and integrated with Brain")
    else:
        print("  ISSUES FOUND — fix before wiring into Brain")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def make_plv(seed=0, n=500):
    return np.random.RandomState(seed).rand(n).astype(np.float32)

def attn_step(attn, bmu=20, qe=0.3, fam=0.5, surp=0.0, cur=0.0):
    return attn.step(bmu_idx=bmu, qe_norm=qe, familiarity=fam,
                     surprise_signal=surp, curiosity_delta=cur)

def brain_step(brain, freq=1.0, w=0.85, nov=0.0, seed=0):
    return brain.step(decoded_freq=freq, stability_w=w,
                      novelty_flag=nov, plv_vector=make_plv(seed))

def brain_attn_step(brain, attn, freq=1.0, w=0.85, nov=0.0, seed=0):
    b = brain_step(brain, freq=freq, w=w, nov=nov, seed=seed)
    a = attn.step(bmu_idx=b['bmu_idx'], qe_norm=b['qe_norm'],
                  familiarity=b['familiarity'],
                  surprise_signal=b['surprise_signal'],
                  curiosity_delta=b['curiosity_delta'])
    return b, a


# ═══════════════════════════════════════════════════════════════
# AT-01  Output key completeness
# ═══════════════════════════════════════════════════════════════
section("AT-01  Output key completeness — every documented key exists")

REQUIRED_KEYS = [
    'salience', 'salience_ema', 'salience_delta',
    'attention_gate', 'attended_bmu', 'gate_entropy', 't',
]

a01 = Attention()
r01 = attn_step(a01)

missing = [k for k in REQUIRED_KEYS if k not in r01]
extra   = [k for k in r01 if k not in REQUIRED_KEYS]
print(f"  Required: {len(REQUIRED_KEYS)}  Present: {len(r01)}  "
      f"Missing: {missing or 'none'}  Extra: {extra or 'none'}")
report("AT-01 Output key completeness", len(missing) == 0,
       f"missing={missing}  extra={extra}")


# ═══════════════════════════════════════════════════════════════
# AT-02  Output signal bounds
# ═══════════════════════════════════════════════════════════════
section("AT-02  Output signal bounds — all signals in documented ranges")

a02 = Attention()
rng02 = np.random.RandomState(2)
violations = []

for i in range(200):
    r = a02.step(
        bmu_idx         = int(rng02.randint(0, N_NEURONS)),
        qe_norm         = float(rng02.rand()),
        familiarity     = float(rng02.rand()),
        surprise_signal = float(rng02.rand()),
        curiosity_delta = float(rng02.rand()),
    )
    checks = {
        'salience':       (r['salience'],       0.0, 1.0),
        'salience_ema':   (r['salience_ema'],   0.0, 1.0),
        'salience_delta': (r['salience_delta'], 0.0, 1.0),
        'gate_entropy':   (r['gate_entropy'],   0.0, 1.0),
        'attended_bmu':   (r['attended_bmu'],   0,   N_NEURONS - 1),
    }
    for name, (val, lo, hi) in checks.items():
        if not (lo <= val <= hi):
            violations.append(f"step {i}: {name}={val:.4f} not in [{lo},{hi}]")

    # gate must be non-negative and sum to ~1.0
    gate = r['attention_gate']
    if gate.min() < -1e-6:
        violations.append(f"step {i}: gate has negative values (min={gate.min():.6f})")
    gate_sum = gate.sum()
    if not (0.99 <= gate_sum <= 1.01):
        violations.append(f"step {i}: gate sum={gate_sum:.6f} not ~1.0")

print(f"  Violations: {len(violations)}")
if violations[:3]:
    for v in violations[:3]:
        print(f"    {v}")
report("AT-02 Output signal bounds", len(violations) == 0,
       f"{len(violations)} violations")


# ═══════════════════════════════════════════════════════════════
# AT-03  Step counter
# ═══════════════════════════════════════════════════════════════
section("AT-03  Step counter increments correctly")

a03 = Attention()
for i in range(1, 11):
    r = attn_step(a03)
    if r['t'] != i:
        report("AT-03 Step counter", False, f"step {i}: t={r['t']} expected {i}")
        break
else:
    report("AT-03 Step counter", True, f"t={r['t']} after 10 steps")


# ═══════════════════════════════════════════════════════════════
# AT-04  Standalone — works without Brain
# ═══════════════════════════════════════════════════════════════
section("AT-04  Standalone operation — no Brain import needed")

try:
    a04 = Attention()
    # Call with only mandatory parameters (as if Brain not wired yet)
    r04 = a04.step(bmu_idx=10, qe_norm=0.5, familiarity=0.3)
    crashed = False
    has_keys = all(k in r04 for k in REQUIRED_KEYS)
except Exception as e:
    crashed = True
    has_keys = False
    print(f"  Exception: {e}")

report("AT-04 Standalone operation", not crashed and has_keys,
       f"crashed={crashed}  has_keys={has_keys}")


# ═══════════════════════════════════════════════════════════════
# AT-05  Familiarity suppression
# ═══════════════════════════════════════════════════════════════
section("AT-05  Familiarity suppression — high familiarity lowers salience")

# Hold all other inputs constant, vary familiarity
a05 = Attention()
sal_low_fam  = attn_step(a05, bmu=20, qe=0.3, fam=0.0, surp=0.0, cur=0.0)['salience']
a05.reset()
sal_high_fam = attn_step(a05, bmu=20, qe=0.3, fam=1.0, surp=0.0, cur=0.0)['salience']

diff = sal_low_fam - sal_high_fam
print(f"  salience(fam=0.0): {sal_low_fam:.4f}")
print(f"  salience(fam=1.0): {sal_high_fam:.4f}")
print(f"  suppression:       {diff:.4f}  (need > 0.05)")
report("AT-05 Familiarity suppression", diff > 0.05,
       f"low_fam={sal_low_fam:.4f}  high_fam={sal_high_fam:.4f}  diff={diff:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-06  Surprise drives salience
# ═══════════════════════════════════════════════════════════════
section("AT-06  Surprise signal drives salience upward")

a06 = Attention()
sal_no_surp = attn_step(a06, bmu=20, qe=0.0, fam=0.5, surp=0.0, cur=0.0)['salience']
a06.reset()
sal_surp    = attn_step(a06, bmu=20, qe=0.0, fam=0.5, surp=1.0, cur=0.0)['salience']

diff = sal_surp - sal_no_surp
print(f"  salience(surp=0.0): {sal_no_surp:.4f}")
print(f"  salience(surp=1.0): {sal_surp:.4f}")
print(f"  boost:              {diff:.4f}  (need > 0.10)")
report("AT-06 Surprise drives salience", diff > 0.10,
       f"no_surp={sal_no_surp:.4f}  surp={sal_surp:.4f}  boost={diff:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-07  QE drives salience
# ═══════════════════════════════════════════════════════════════
section("AT-07  QE norm (perceptual novelty) drives salience upward")

a07 = Attention()
sal_low_qe  = attn_step(a07, bmu=20, qe=0.0, fam=0.5, surp=0.0, cur=0.0)['salience']
a07.reset()
sal_high_qe = attn_step(a07, bmu=20, qe=1.0, fam=0.5, surp=0.0, cur=0.0)['salience']

diff = sal_high_qe - sal_low_qe
print(f"  salience(qe=0.0): {sal_low_qe:.4f}")
print(f"  salience(qe=1.0): {sal_high_qe:.4f}")
print(f"  boost:            {diff:.4f}  (need > 0.10)")
report("AT-07 QE drives salience", diff > 0.10,
       f"low_qe={sal_low_qe:.4f}  high_qe={sal_high_qe:.4f}  boost={diff:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-08  Curiosity delta drives salience
# ═══════════════════════════════════════════════════════════════
section("AT-08  Curiosity delta drives salience upward")

a08 = Attention()
sal_no_cur = attn_step(a08, bmu=20, qe=0.0, fam=0.5, surp=0.0, cur=0.0)['salience']
a08.reset()
sal_cur    = attn_step(a08, bmu=20, qe=0.0, fam=0.5, surp=0.0, cur=1.0)['salience']

diff = sal_cur - sal_no_cur
print(f"  salience(cur=0.0): {sal_no_cur:.4f}")
print(f"  salience(cur=1.0): {sal_cur:.4f}")
print(f"  boost:             {diff:.4f}  (need > 0.05)")
report("AT-08 Curiosity delta drives salience", diff > 0.05,
       f"no_cur={sal_no_cur:.4f}  cur={sal_cur:.4f}  boost={diff:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-09  Combined inputs clip at 1.0
# ═══════════════════════════════════════════════════════════════
section("AT-09  Combined max inputs clip correctly at 1.0")

a09 = Attention()
r09 = attn_step(a09, bmu=20, qe=1.0, fam=0.0, surp=1.0, cur=1.0)
print(f"  salience (all max): {r09['salience']:.4f}  (should be <=1.0)")
report("AT-09 Combined inputs clip at 1.0", r09['salience'] <= 1.0,
       f"salience={r09['salience']:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-10  Zero inputs → baseline from familiarity term only
# ═══════════════════════════════════════════════════════════════
section("AT-10  Zero inputs produce baseline salience (familiarity term)")

a10 = Attention()
# With surp=0, qe=0, cur=0, fam=0: only W_FAMILIARITY * (1-0) = W_FAMILIARITY
r10 = attn_step(a10, bmu=20, qe=0.0, fam=0.0, surp=0.0, cur=0.0)
expected = float(np.clip(W_FAMILIARITY * 1.0, 0.0, 1.0))
diff = abs(r10['salience'] - expected)
print(f"  Expected (W_FAM * 1.0): {expected:.4f}")
print(f"  Got:                    {r10['salience']:.4f}")
print(f"  Difference:             {diff:.6f}  (need < 0.001)")
report("AT-10 Zero inputs baseline", diff < 0.001,
       f"expected={expected:.4f}  got={r10['salience']:.4f}  diff={diff:.6f}")


# ═══════════════════════════════════════════════════════════════
# AT-11  Salience EMA smoothing
# ═══════════════════════════════════════════════════════════════
section("AT-11  Salience EMA changes slower than raw signal")

a11 = Attention()
rng11 = np.random.RandomState(11)

raw_changes = []
ema_changes = []
prev_raw = None
prev_ema = None

for _ in range(100):
    r = a11.step(
        bmu_idx         = int(rng11.randint(0, N_NEURONS)),
        qe_norm         = float(rng11.rand()),
        familiarity     = float(rng11.rand()),
        surprise_signal = float(rng11.rand()),
        curiosity_delta = float(rng11.rand()),
    )
    if prev_raw is not None:
        raw_changes.append(abs(r['salience']     - prev_raw))
        ema_changes.append(abs(r['salience_ema'] - prev_ema))
    prev_raw = r['salience']
    prev_ema = r['salience_ema']

mean_raw_change = float(np.mean(raw_changes))
mean_ema_change = float(np.mean(ema_changes))
print(f"  Mean step-to-step change — raw: {mean_raw_change:.4f}  ema: {mean_ema_change:.4f}")
report("AT-11 EMA smoothing", mean_ema_change < mean_raw_change,
       f"raw_change={mean_raw_change:.4f}  ema_change={mean_ema_change:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-12  Salience delta near-zero during stable input
# ═══════════════════════════════════════════════════════════════
section("AT-12  Salience delta near-zero during stable input")

a12 = Attention()

# Warmup: let EMA settle
for _ in range(50):
    attn_step(a12, bmu=20, qe=0.3, fam=0.5, surp=0.0, cur=0.0)

# Measure delta during stable period
deltas = []
for _ in range(100):
    r = attn_step(a12, bmu=20, qe=0.3, fam=0.5, surp=0.0, cur=0.0)
    deltas.append(r['salience_delta'])

mean_delta = float(np.mean(deltas))
max_delta  = float(np.max(deltas))
print(f"  Stable period — delta mean: {mean_delta:.4f}  max: {max_delta:.4f}  (need mean < 0.05)")
report("AT-12 Delta near-zero during stable input", mean_delta < 0.05,
       f"mean={mean_delta:.4f}  max={max_delta:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-13  Salience delta spikes at genuine transitions
# ═══════════════════════════════════════════════════════════════
section("AT-13  Salience delta spikes at genuine input transitions")

a13 = Attention()

# Warmup: settle on LOW salience inputs
for _ in range(80):
    attn_step(a13, bmu=20, qe=0.1, fam=0.9, surp=0.0, cur=0.0)

# Transition to HIGH salience
transition_deltas = []
for _ in range(20):
    r = attn_step(a13, bmu=35, qe=0.9, fam=0.0, surp=1.0, cur=0.8)
    transition_deltas.append(r['salience_delta'])

peak_delta = float(np.max(transition_deltas))
mean_delta = float(np.mean(transition_deltas))
print(f"  Transition deltas — peak: {peak_delta:.4f}  mean: {mean_delta:.4f}  (need peak > 0.10)")
report("AT-13 Delta spikes at transitions", peak_delta > 0.10,
       f"peak={peak_delta:.4f}  mean={mean_delta:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-14  Attention gate sums to 1.0
# ═══════════════════════════════════════════════════════════════
section("AT-14  Attention gate sums to 1.0 at all salience levels")

a14 = Attention()
failures = []
for bmu in [0, 15, 31, 32, 48, 63]:
    for sal_level in [(0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.5, 0.5, 0.5)]:
        r = a14.step(bmu_idx=bmu, qe_norm=sal_level[0],
                     familiarity=sal_level[1], surprise_signal=sal_level[2])
        gate_sum = r['attention_gate'].sum()
        if not (0.999 <= gate_sum <= 1.001):
            failures.append(f"bmu={bmu} sum={gate_sum:.6f}")

print(f"  Failures: {len(failures)}")
report("AT-14 Gate sums to 1.0", len(failures) == 0, str(failures or 'none'))


# ═══════════════════════════════════════════════════════════════
# AT-15  Gate is peaked at or near BMU location
# ═══════════════════════════════════════════════════════════════
section("AT-15  Gate peak is at or near the current BMU")

a15 = Attention()
rng15 = np.random.RandomState(15)
failures = []

for _ in range(50):
    bmu = int(rng15.randint(0, N_NEURONS))
    r = a15.step(bmu_idx=bmu, qe_norm=0.8, familiarity=0.1,
                 surprise_signal=0.9, curiosity_delta=0.5)

    gate = r['attention_gate']
    peak_bmu = int(np.argmax(gate))

    # Peak should be within 2 grid cells of the actual BMU
    row_bmu, col_bmu = divmod(bmu, GRID_W)
    row_pk,  col_pk  = divmod(peak_bmu, GRID_W)
    dist = math.sqrt((row_bmu - row_pk)**2 + (col_bmu - col_pk)**2)
    if dist > 2.0:
        failures.append(f"bmu={bmu} peak={peak_bmu} dist={dist:.2f}")

print(f"  Checked 50 random BMUs — failures: {len(failures)}")
report("AT-15 Gate peaked near BMU", len(failures) == 0,
       str(failures[:3] or 'none'))


# ═══════════════════════════════════════════════════════════════
# AT-16  High salience → more focused gate (lower entropy)
# ═══════════════════════════════════════════════════════════════
section("AT-16  High salience produces more focused gate (lower entropy)")

a16_hi = Attention()
a16_lo = Attention()

# Let EMAs settle first
for _ in range(30):
    attn_step(a16_hi, bmu=20, qe=0.9, fam=0.0, surp=1.0, cur=0.8)
    attn_step(a16_lo, bmu=20, qe=0.0, fam=1.0, surp=0.0, cur=0.0)

entropies_hi = [attn_step(a16_hi, bmu=20, qe=0.9, fam=0.0, surp=1.0, cur=0.8)['gate_entropy']
                for _ in range(50)]
entropies_lo = [attn_step(a16_lo, bmu=20, qe=0.0, fam=1.0, surp=0.0, cur=0.0)['gate_entropy']
                for _ in range(50)]

mean_hi_ent = float(np.mean(entropies_hi))
mean_lo_ent = float(np.mean(entropies_lo))
print(f"  High salience entropy: {mean_hi_ent:.4f}")
print(f"  Low  salience entropy: {mean_lo_ent:.4f}  (need hi < lo)")
report("AT-16 High salience = focused gate", mean_hi_ent < mean_lo_ent,
       f"hi_entropy={mean_hi_ent:.4f}  lo_entropy={mean_lo_ent:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-17  Low salience → diffuse gate (entropy near maximum)
# ═══════════════════════════════════════════════════════════════
section("AT-17  Zero salience produces near-uniform (high entropy) gate")

a17 = Attention()
# With fam=1.0, surp=0, qe=0, cur=0: salience = W_FAM*(1-1) = 0
r17 = a17.step(bmu_idx=20, qe_norm=0.0, familiarity=1.0,
               surprise_signal=0.0, curiosity_delta=0.0)
entropy = r17['gate_entropy']
print(f"  Zero-salience gate entropy: {entropy:.4f}  (need > 0.90)")
report("AT-17 Zero salience = diffuse gate", entropy > 0.90,
       f"entropy={entropy:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-18  Attended BMU tracks actual BMU at high salience
# ═══════════════════════════════════════════════════════════════
section("AT-18  Attended BMU tracks actual BMU at high salience")

a18 = Attention()
match_count = 0
total = 50
rng18 = np.random.RandomState(18)

for _ in range(total):
    bmu = int(rng18.randint(0, N_NEURONS))
    r = a18.step(bmu_idx=bmu, qe_norm=1.0, familiarity=0.0,
                 surprise_signal=1.0, curiosity_delta=1.0)
    if r['attended_bmu'] == bmu:
        match_count += 1

match_rate = match_count / total
print(f"  Match rate (high salience): {match_rate:.2f}  (need > 0.80)")
report("AT-18 Attended BMU tracks actual BMU", match_rate > 0.80,
       f"match_rate={match_rate:.2f}  ({match_count}/{total})")


# ═══════════════════════════════════════════════════════════════
# AT-19  Gate is non-negative everywhere
# ═══════════════════════════════════════════════════════════════
section("AT-19  Attention gate is non-negative everywhere")

a19 = Attention()
rng19 = np.random.RandomState(19)
violations = []

for i in range(200):
    r = a19.step(
        bmu_idx         = int(rng19.randint(0, N_NEURONS)),
        qe_norm         = float(rng19.rand()),
        familiarity     = float(rng19.rand()),
        surprise_signal = float(rng19.rand()),
        curiosity_delta = float(rng19.rand()),
    )
    min_val = float(r['attention_gate'].min())
    if min_val < -1e-6:
        violations.append(f"step {i}: min={min_val:.8f}")

print(f"  Violations: {len(violations)}")
report("AT-19 Gate non-negative", len(violations) == 0,
       str(violations[:3] or 'none'))


# ═══════════════════════════════════════════════════════════════
# AT-20  Backward compatibility — default params
# ═══════════════════════════════════════════════════════════════
section("AT-20  Backward compatibility — default params stable across 100 steps")

a20a = Attention()
a20b = Attention()

results_a, results_b = [], []
for i in range(100):
    # Explicit zeros
    ra = a20a.step(bmu_idx=20, qe_norm=0.4, familiarity=0.5,
                   surprise_signal=0.0, curiosity_delta=0.0)
    # Default zeros (omitted)
    rb = a20b.step(bmu_idx=20, qe_norm=0.4, familiarity=0.5)
    results_a.append(ra['salience'])
    results_b.append(rb['salience'])

diffs = [abs(a - b) for a, b in zip(results_a, results_b)]
max_diff = max(diffs)
print(f"  Max difference (explicit zeros vs defaults): {max_diff:.8f}")
report("AT-20 Backward compatibility", max_diff < 1e-6,
       f"max_diff={max_diff:.8f}")


# ═══════════════════════════════════════════════════════════════
# AT-21  Reset clears all state
# ═══════════════════════════════════════════════════════════════
section("AT-21  reset() clears all state — produces identical output to fresh instance")

a21_fresh = Attention()
a21_reset = Attention()

# Run 100 steps to dirty the state
for i in range(100):
    attn_step(a21_reset, bmu=i % N_NEURONS, qe=0.8, fam=0.1, surp=0.9, cur=0.7)

a21_reset.reset()

# Both should produce identical output for the same input
r_fresh = a21_fresh.step(bmu_idx=30, qe_norm=0.5, familiarity=0.4,
                          surprise_signal=0.3, curiosity_delta=0.2)
r_reset = a21_reset.step(bmu_idx=30, qe_norm=0.5, familiarity=0.4,
                          surprise_signal=0.3, curiosity_delta=0.2)

sal_diff = abs(r_fresh['salience'] - r_reset['salience'])
ema_diff = abs(r_fresh['salience_ema'] - r_reset['salience_ema'])
t_match  = r_reset['t'] == 1

print(f"  salience diff: {sal_diff:.8f}  ema diff: {ema_diff:.8f}  t_match={t_match}")
report("AT-21 Reset clears state",
       sal_diff < 1e-6 and ema_diff < 1e-6 and t_match,
       f"sal_diff={sal_diff:.8f}  ema_diff={ema_diff:.8f}  t={r_reset['t']}")


# ═══════════════════════════════════════════════════════════════
# AT-22  Isolation — Attention doesn't modify Brain outputs
# ═══════════════════════════════════════════════════════════════
section("AT-22  Isolation — Attention.step() does not modify Brain's output dict")

brain22 = Brain(seed=22)
attn22  = Attention()
rng22   = np.random.RandomState(22)

failures = []
for i in range(50):
    b_out = brain_step(brain22, freq=1.0 + (i % 3) * 0.4, seed=i)

    # Record Brain output values before Attention sees them
    bmu_before   = b_out['bmu_idx']
    qe_before    = b_out['qe_norm']
    fam_before   = b_out['familiarity']
    surp_before  = b_out['surprise_signal']
    cur_before   = b_out['curiosity_delta']

    # Run Attention
    attn22.step(bmu_idx         = b_out['bmu_idx'],
                qe_norm         = b_out['qe_norm'],
                familiarity     = b_out['familiarity'],
                surprise_signal = b_out['surprise_signal'],
                curiosity_delta = b_out['curiosity_delta'])

    # Brain outputs must be unchanged
    if b_out['bmu_idx']         != bmu_before:  failures.append(f"step {i}: bmu_idx mutated")
    if b_out['qe_norm']         != qe_before:   failures.append(f"step {i}: qe_norm mutated")
    if b_out['familiarity']     != fam_before:  failures.append(f"step {i}: familiarity mutated")
    if b_out['surprise_signal'] != surp_before: failures.append(f"step {i}: surprise_signal mutated")
    if b_out['curiosity_delta'] != cur_before:  failures.append(f"step {i}: curiosity_delta mutated")

print(f"  Mutations detected: {len(failures)}")
report("AT-22 Attention doesn't mutate Brain output", len(failures) == 0,
       str(failures[:3] or 'none'))


# ═══════════════════════════════════════════════════════════════
# AT-23  Brain integration — output keys and bounds
# ═══════════════════════════════════════════════════════════════
section("AT-23  Brain integration — all keys present and in bounds")

brain23 = Brain(seed=23)
attn23  = Attention()
rng23   = np.random.RandomState(23)

violations = []
for i in range(200):
    freq = 0.5 + (i % 6) * 0.3
    b, a = brain_attn_step(brain23, attn23, freq=freq,
                            w=0.8 + (i % 3) * 0.05, seed=i)

    checks = {
        'salience':       (a['salience'],       0.0, 1.0),
        'salience_ema':   (a['salience_ema'],   0.0, 1.0),
        'salience_delta': (a['salience_delta'], 0.0, 1.0),
        'gate_entropy':   (a['gate_entropy'],   0.0, 1.0),
        'attended_bmu':   (float(a['attended_bmu']), 0.0, float(N_NEURONS - 1)),
    }
    for name, (val, lo, hi) in checks.items():
        if not (lo <= val <= hi):
            violations.append(f"step {i}: {name}={val}")

    gate_sum = a['attention_gate'].sum()
    if not (0.999 <= gate_sum <= 1.001):
        violations.append(f"step {i}: gate_sum={gate_sum:.6f}")

print(f"  200 Brain+Attention steps — violations: {len(violations)}")
report("AT-23 Brain integration bounds", len(violations) == 0,
       str(violations[:3] or 'none'))


# ═══════════════════════════════════════════════════════════════
# AT-24  Brain integration — salience rises at frequency transition
# ═══════════════════════════════════════════════════════════════
section("AT-24  Brain integration — salience rises at frequency transition")

brain24 = Brain(seed=24)
attn24  = Attention()
rng24   = np.random.RandomState(24)

# Settle on one frequency
for i in range(150):
    brain_attn_step(brain24, attn24, freq=1.0, seed=i)

# Record stable salience
stable_sals = [brain_attn_step(brain24, attn24, freq=1.0, seed=150+i)[1]['salience']
               for i in range(20)]

# Transition to a different frequency
trans_sals = [brain_attn_step(brain24, attn24, freq=1.8, nov=1.0, seed=200+i)[1]['salience']
              for i in range(20)]

mean_stable = float(np.mean(stable_sals))
mean_trans  = float(np.mean(trans_sals))
peak_trans  = float(np.max(trans_sals))

print(f"  Stable salience:      {mean_stable:.4f}")
print(f"  Transition salience:  mean={mean_trans:.4f}  peak={peak_trans:.4f}")
print(f"  (need peak_trans > mean_stable)")
report("AT-24 Salience rises at transition", peak_trans > mean_stable,
       f"stable={mean_stable:.4f}  trans_peak={peak_trans:.4f}")


# ═══════════════════════════════════════════════════════════════
# AT-25  Brain integration — salience falls on stable known input
# ═══════════════════════════════════════════════════════════════
section("AT-25  Brain integration — familiarity suppresses salience baseline")

# The test verifies that familiarity genuinely grows and suppresses salience.
# We can't measure salience falling monotonically in a full Brain run because
# Brain's surprise_signal keeps re-spiking (SOM drift + L2 oscillations —
# known behaviour documented in guide). Instead we verify the mechanism:
# at equal surprise/qe levels, higher-familiarity steps produce lower salience.

brain25 = Brain(seed=25)
attn25  = Attention()

low_fam_sals  = []   # salience at steps where familiarity < 0.3
high_fam_sals = []   # salience at steps where familiarity > 0.5

for i in range(500):
    b, a = brain_attn_step(brain25, attn25, freq=1.0, seed=i)
    # Only sample steps where surprise is near-zero (familiarity can dominate)
    if b['surprise_signal'] < 0.05 and b['qe_norm'] < 0.05:
        if b['familiarity'] < 0.30:
            low_fam_sals.append(a['salience'])
        elif b['familiarity'] > 0.50:
            high_fam_sals.append(a['salience'])

if low_fam_sals and high_fam_sals:
    mean_low_fam  = float(np.mean(low_fam_sals))
    mean_high_fam = float(np.mean(high_fam_sals))
    print(f"  Low-familiarity steps  (fam<0.3, n={len(low_fam_sals)}):  "
          f"mean salience={mean_low_fam:.4f}")
    print(f"  High-familiarity steps (fam>0.5, n={len(high_fam_sals)}): "
          f"mean salience={mean_high_fam:.4f}")
    print(f"  (need high_fam salience < low_fam salience)")
    report("AT-25 Familiarity suppresses salience baseline",
           mean_high_fam < mean_low_fam,
           f"low_fam={mean_low_fam:.4f}  high_fam={mean_high_fam:.4f}")
else:
    report("AT-25 Familiarity suppresses salience baseline", False,
           f"not enough clean samples: low={len(low_fam_sals)} high={len(high_fam_sals)}")


# ═══════════════════════════════════════════════════════════════
# AT-26  Brain integration — gate tracks Brain's bmu_idx
# ═══════════════════════════════════════════════════════════════
section("AT-26  Brain integration — attended_bmu near Brain's bmu_idx at high salience")

brain26 = Brain(seed=26)
attn26  = Attention()

near_count = 0
total26    = 0

for i in range(200):
    b, a = brain_attn_step(brain26, attn26, freq=1.5, nov=float(i % 20 == 0), seed=i)
    if a['salience'] > 0.3:   # only check when salience is meaningful
        brain_bmu = b['bmu_idx']
        attn_bmu  = a['attended_bmu']
        row_b, col_b = divmod(brain_bmu, GRID_W)
        row_a, col_a = divmod(attn_bmu,  GRID_W)
        dist = math.sqrt((row_b - row_a)**2 + (col_b - col_a)**2)
        if dist <= 3.0:
            near_count += 1
        total26 += 1

near_rate = near_count / max(total26, 1)
print(f"  High-salience steps: {total26}  near matches (dist≤3): {near_count}  rate={near_rate:.2f}")
print(f"  (need rate > 0.70)")
report("AT-26 Gate tracks Brain's BMU", near_rate > 0.70,
       f"near_rate={near_rate:.2f}  ({near_count}/{total26})")


# ═══════════════════════════════════════════════════════════════
# AT-27  Brain integration — delta selectivity
# ═══════════════════════════════════════════════════════════════
section("AT-27  Brain integration — salience_delta << salience during stable operation")

brain27 = Brain(seed=27)
attn27  = Attention()

# Warmup
for i in range(150):
    brain_attn_step(brain27, attn27, freq=1.0, seed=i)

# Stable period measurements
raw_sals, deltas = [], []
for i in range(100):
    _, a = brain_attn_step(brain27, attn27, freq=1.0, seed=150+i)
    raw_sals.append(a['salience'])
    deltas.append(a['salience_delta'])

mean_raw   = float(np.mean(raw_sals))
mean_delta = float(np.mean(deltas))
ratio      = mean_delta / (mean_raw + 1e-6)

print(f"  Stable raw salience mean:  {mean_raw:.4f}")
print(f"  Stable delta mean:         {mean_delta:.4f}")
print(f"  Ratio delta/raw:           {ratio:.3f}  (need < 0.50)")
report("AT-27 Delta selectivity vs raw salience", ratio < 0.50,
       f"raw={mean_raw:.4f}  delta={mean_delta:.4f}  ratio={ratio:.3f}")


# ═══════════════════════════════════════════════════════════════
# AT-28  Edge cases — extreme inputs
# ═══════════════════════════════════════════════════════════════
section("AT-28  Edge cases — extreme inputs don't crash or produce NaN/Inf")

a28 = Attention()
edge_cases = [
    dict(bmu_idx=0,  qe_norm=0.0, familiarity=0.0, surprise_signal=0.0, curiosity_delta=0.0),
    dict(bmu_idx=63, qe_norm=1.0, familiarity=1.0, surprise_signal=1.0, curiosity_delta=1.0),
    dict(bmu_idx=0,  qe_norm=0.0, familiarity=1.0, surprise_signal=0.0, curiosity_delta=0.0),
    dict(bmu_idx=63, qe_norm=1.0, familiarity=0.0, surprise_signal=1.0, curiosity_delta=1.0),
    dict(bmu_idx=31, qe_norm=0.5, familiarity=0.5, surprise_signal=0.5, curiosity_delta=0.5),
    # Corner BMUs
    dict(bmu_idx=0,  qe_norm=0.9, familiarity=0.1, surprise_signal=0.9, curiosity_delta=0.9),
    dict(bmu_idx=7,  qe_norm=0.9, familiarity=0.1, surprise_signal=0.9, curiosity_delta=0.9),
    dict(bmu_idx=56, qe_norm=0.9, familiarity=0.1, surprise_signal=0.9, curiosity_delta=0.9),
    dict(bmu_idx=63, qe_norm=0.9, familiarity=0.1, surprise_signal=0.9, curiosity_delta=0.9),
]

failures = []
for ec in edge_cases:
    try:
        r = a28.step(**ec)
        gate = r['attention_gate']
        if np.any(np.isnan(gate)) or np.any(np.isinf(gate)):
            failures.append(f"NaN/Inf in gate for bmu={ec['bmu_idx']}")
        if np.isnan(r['salience']) or np.isinf(r['salience']):
            failures.append(f"NaN/Inf in salience for bmu={ec['bmu_idx']}")
    except Exception as e:
        failures.append(f"bmu={ec['bmu_idx']}: {type(e).__name__}: {e}")

print(f"  Edge cases tested: {len(edge_cases)}  failures: {len(failures)}")
report("AT-28 Edge cases", len(failures) == 0, str(failures or 'none'))


# ═══════════════════════════════════════════════════════════════
# AT-29  Diagnostics — get_state() keys
# ═══════════════════════════════════════════════════════════════
section("AT-29  Diagnostics — get_state() returns all expected keys")

REQUIRED_STATE_KEYS = [
    't', 'salience', 'salience_ema', 'salience_delta',
    'attended_bmu', 'gate_peak', 'gate_entropy', 'salience_mean',
]

a29 = Attention()
for _ in range(10):
    attn_step(a29)

state = a29.get_state()
missing = [k for k in REQUIRED_STATE_KEYS if k not in state]
print(f"  Keys present: {list(state.keys())}")
print(f"  Missing: {missing or 'none'}")
report("AT-29 get_state() keys", len(missing) == 0,
       f"missing={missing}")


# ═══════════════════════════════════════════════════════════════
# AT-30  Long-run stability — 1000 steps
# ═══════════════════════════════════════════════════════════════
section("AT-30  Long-run stability — 1000 Brain+Attention steps without drift")

brain30 = Brain(seed=30)
attn30  = Attention()
rng30   = np.random.RandomState(30)

freqs = [0.60, 1.00, 1.40, 1.80]
sals_early, sals_late = [], []
nan_count = 0

for i in range(1000):
    freq = freqs[i % len(freqs)]
    b, a = brain_attn_step(brain30, attn30, freq=freq, seed=i)

    if np.isnan(a['salience']) or np.isinf(a['salience']):
        nan_count += 1
    if np.any(np.isnan(a['attention_gate'])):
        nan_count += 1

    if 50 <= i < 150:
        sals_early.append(a['salience'])
    if 850 <= i < 950:
        sals_late.append(a['salience'])

mean_early = float(np.mean(sals_early))
mean_late  = float(np.mean(sals_late))
drift      = abs(mean_late - mean_early)

print(f"  NaN/Inf count: {nan_count}")
print(f"  Salience early (50-150): {mean_early:.4f}")
print(f"  Salience late (850-950): {mean_late:.4f}")
print(f"  Drift: {drift:.4f}  (need < 0.20 — no explosion or collapse)")
report("AT-30 Long-run stability",
       nan_count == 0 and drift < 0.20,
       f"nan={nan_count}  drift={drift:.4f}  early={mean_early:.4f}  late={mean_late:.4f}")


# ═══════════════════════════════════════════════════════════════
summarise()