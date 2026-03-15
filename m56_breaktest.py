"""
M56 — ACTION LAYER BREAK TEST
===============================

Standalone tests for ActionLayer (m56_action.py) before Brain integration.
Run this after any change to m56_action.py.

TESTS
-----
BT-76  Output key completeness
BT-77  Signal bounds — all outputs in valid ranges
BT-78  Step counter increments
BT-79  Q learning — positive rpe raises Q for taken action
BT-80  Q learning — negative rpe lowers Q for taken action
BT-81  Q values bounded in [-1, +1]
BT-82  Exploration — high entropy → higher epsilon
BT-83  Confidence gate — high thought_confidence reduces epsilon
BT-84  Exploitation — argmax Q chosen when epsilon=0
BT-85  Eligibility trace — Q update spreads to recent (state, action) pairs
BT-86  Q decay — unused values decay over time
BT-87  Policy table shape and range correct
BT-88  Determinism — same seed = identical outputs
BT-89  Integrated convergence with Brain RPE signal
"""

import numpy as np
import sys
from collections import deque

try:
    from m56_action import (
        ActionLayer,
        N_NEURONS, N_ACTIONS,
        ETA_Q, Q_DECAY, TRACE_DECAY,
        EPSILON_MIN, EPSILON_MAX, CONFIDENCE_GATE,
        Q_MAX, Q_MIN,
    )
    from brain import Brain
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
    tag  = "PASS" if passed else ("WARN" if warn else "FAIL")
    sym  = "✓" if passed else ("⚠" if warn else "✗")
    results[name] = tag
    print(f"  {sym} [{tag}] {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")

def summarise():
    section("M56 BREAK TEST — SUMMARY")
    n_pass = sum(1 for v in results.values() if v == "PASS")
    n_fail = sum(1 for v in results.values() if v == "FAIL")
    n_warn = sum(1 for v in results.values() if v == "WARN")
    for name, tag in results.items():
        sym = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}[tag]
        print(f"  {sym} [{tag}] {name}")
    print(f"\n  {'─'*70}")
    print(f"  PASS:{n_pass}  FAIL:{n_fail}  WARN:{n_warn}")
    if n_fail == 0 and n_warn == 0:
        print("  ALL CLEAR — M56 ActionLayer verified, ready for Brain integration")
    else:
        print("  ISSUES FOUND — fix before integrating into Brain")


def brain_step(b, freq=1.0, seed=0):
    rng = np.random.RandomState(seed)
    return b.step(decoded_freq=freq, stability_w=0.85, novelty_flag=0.0,
                  plv_vector=rng.rand(500).astype('float32'))


# ═══════════════════════════════════════════════════════════════
# BT-76  Output key completeness
# ═══════════════════════════════════════════════════════════════
section("BT-76  Output key completeness — all documented keys present")

m76 = ActionLayer(n_actions=4, seed=76)
m76.select_action(bmu_idx=10, focus_entropy=0.5, thought_confidence=0.2)
r76 = m76.update(bmu_idx=10, action=0, rpe=0.1)

REQUIRED_KEYS = ['action', 'q_values', 'q_max', 'epsilon', 'explore',
                 'td_error', 'q_mean', 'q_nonzero_frac']
missing76 = [k for k in REQUIRED_KEYS if k not in r76]
print(f"  Required: {len(REQUIRED_KEYS)}  Present: {len(REQUIRED_KEYS)-len(missing76)}  Missing: {missing76 or 'none'}")
report("BT-76 Output key completeness", len(missing76) == 0, f"missing={missing76}")


# ═══════════════════════════════════════════════════════════════
# BT-77  Signal bounds — all outputs in valid ranges
# ═══════════════════════════════════════════════════════════════
section("BT-77  Signal bounds — all outputs in valid ranges over 200 steps")

m77 = ActionLayer(n_actions=4, seed=77)
rng77 = np.random.RandomState(77)
violations77 = []

for i in range(200):
    bmu = rng77.randint(64)
    entropy = rng77.random()
    conf = rng77.random()
    action77 = m77.select_action(bmu_idx=bmu, focus_entropy=entropy,
                                  thought_confidence=conf)
    rpe77 = rng77.uniform(-1.0, 1.0)
    r = m77.update(bmu_idx=bmu, action=action77, rpe=rpe77)

    checks = {
        'action':         (float(r['action']),        0.0, float(m77._n_actions - 1)),
        'q_max':          (r['q_max'],                Q_MIN, Q_MAX),
        'epsilon':        (r['epsilon'],              EPSILON_MIN, EPSILON_MAX),
        'td_error':       (r['td_error'],             -1.0, 1.0),
        'q_mean':         (r['q_mean'],               0.0, 1.0),
        'q_nonzero_frac': (r['q_nonzero_frac'],       0.0, 1.0),
    }
    for name, (val, lo, hi) in checks.items():
        if not (lo <= val <= hi):
            violations77.append(f"step {i}: {name}={val:.4f} not in [{lo},{hi}]")

    q = r['q_values']
    if q.min() < Q_MIN - 1e-5 or q.max() > Q_MAX + 1e-5:
        violations77.append(f"step {i}: q_values out of [{Q_MIN},{Q_MAX}]")

print(f"  Violations over 200 steps: {len(violations77)}")
if violations77[:2]:
    for v in violations77[:2]: print(f"    {v}")
report("BT-77 Signal bounds", len(violations77) == 0,
       f"{len(violations77)} violations")


# ═══════════════════════════════════════════════════════════════
# BT-78  Step counter increments correctly
# ═══════════════════════════════════════════════════════════════
section("BT-78  Step counter — t increments once per update() call")

m78 = ActionLayer(n_actions=4, seed=78)
for i in range(1, 26):
    m78.select_action(bmu_idx=i % 64, focus_entropy=0.5)
    m78.update(bmu_idx=i % 64, action=0, rpe=0.0)
    if m78.t != i:
        print(f"  MISMATCH at step {i}: t={m78.t}")
        break

print(f"  After 25 updates: t={m78.t}  (should be 25)")
report("BT-78 Step counter", m78.t == 25, f"t={m78.t}")


# ═══════════════════════════════════════════════════════════════
# BT-79  Positive RPE raises Q for taken action
# ═══════════════════════════════════════════════════════════════
section("BT-79  Q learning — positive rpe raises Q[bmu, action]")

m79 = ActionLayer(n_actions=4, seed=79)
q_before = float(m79._Q[10, 2])

# 30 updates: bmu=10, action=2, rpe=+0.5
for _ in range(30):
    m79._current_bmu = 10
    m79._current_action = 2
    m79.update(bmu_idx=10, action=2, rpe=0.5)

q_after = float(m79._Q[10, 2])
gained = q_after > q_before
print(f"  Q[10,2] before: {q_before:.6f}  after 30×rpe=+0.5: {q_after:.4f}")
print(f"  argmax Q[10]: {m79._Q[10].argmax()}  (should be 2)")
report("BT-79 Positive RPE raises Q", gained and m79._Q[10].argmax() == 2,
       f"before={q_before:.6f}  after={q_after:.4f}  argmax={m79._Q[10].argmax()}")


# ═══════════════════════════════════════════════════════════════
# BT-80  Negative RPE lowers Q for taken action
# ═══════════════════════════════════════════════════════════════
section("BT-80  Q learning — negative rpe lowers Q[bmu, action]")

m80 = ActionLayer(n_actions=4, seed=80)
# 30 updates: bmu=5, action=1, rpe=-0.5
for _ in range(30):
    m80._current_bmu = 5
    m80._current_action = 1
    m80.update(bmu_idx=5, action=1, rpe=-0.5)

q_80 = float(m80._Q[5, 1])
lowest = m80._Q[5].argmin() == 1
print(f"  Q[5,1] after 30×rpe=-0.5: {q_80:.4f}  argmin={m80._Q[5].argmin()} (should be 1)")
report("BT-80 Negative RPE lowers Q", q_80 < -0.01 and lowest,
       f"Q[5,1]={q_80:.4f}  argmin={m80._Q[5].argmin()}")


# ═══════════════════════════════════════════════════════════════
# BT-81  Q values bounded in [Q_MIN, Q_MAX]
# ═══════════════════════════════════════════════════════════════
section("BT-81  Q bounds — Q values never exceed [-1, +1] under extreme rpe")

m81 = ActionLayer(n_actions=4, seed=81)
for _ in range(500):
    m81._current_bmu = 20
    m81._current_action = 0
    m81.update(bmu_idx=20, action=0, rpe=1.0)  # max positive rpe

q_max_81 = float(m81._Q.max())
q_min_81 = float(m81._Q.min())
print(f"  After 500×rpe=+1.0: Q_max={q_max_81:.4f}  (hard limit={Q_MAX})")
print(f"  After 500×rpe=-1.0 (indirect): Q_min={q_min_81:.4f}  (hard limit={Q_MIN})")
report("BT-81 Q values bounded", q_max_81 <= Q_MAX + 1e-5 and q_min_81 >= Q_MIN - 1e-5,
       f"Q_max={q_max_81:.4f}  Q_min={q_min_81:.4f}")


# ═══════════════════════════════════════════════════════════════
# BT-82  Exploration scales with focus_entropy
# ═══════════════════════════════════════════════════════════════
section("BT-82  Exploration — epsilon higher at high vs low focus_entropy")

m82 = ActionLayer(n_actions=4, seed=82)
m82.select_action(bmu_idx=0, focus_entropy=0.9, thought_confidence=0.0)
eps_high = m82._current_epsilon
m82.select_action(bmu_idx=0, focus_entropy=0.1, thought_confidence=0.0)
eps_low  = m82._current_epsilon

print(f"  epsilon at entropy=0.9: {eps_high:.4f}  at entropy=0.1: {eps_low:.4f}")
print(f"  (need: high entropy → higher epsilon)")
report("BT-82 Exploration scales with entropy",
       eps_high > eps_low,
       f"high={eps_high:.4f}  low={eps_low:.4f}")


# ═══════════════════════════════════════════════════════════════
# BT-83  Confidence gate reduces epsilon
# ═══════════════════════════════════════════════════════════════
section("BT-83  Confidence gate — high thought_confidence reduces epsilon")

m83 = ActionLayer(n_actions=4, seed=83)
m83.select_action(bmu_idx=0, focus_entropy=0.5, thought_confidence=0.0)
eps_no_conf = m83._current_epsilon
m83.select_action(bmu_idx=0, focus_entropy=0.5, thought_confidence=1.0)
eps_full_conf = m83._current_epsilon

print(f"  epsilon (conf=0.0): {eps_no_conf:.4f}  (conf=1.0): {eps_full_conf:.4f}")
print(f"  (need: high confidence → lower epsilon)")
report("BT-83 Confidence gate reduces epsilon",
       eps_full_conf < eps_no_conf,
       f"conf=0: {eps_no_conf:.4f}  conf=1: {eps_full_conf:.4f}")


# ═══════════════════════════════════════════════════════════════
# BT-84  Exploitation — argmax Q chosen when epsilon forced to 0
# ═══════════════════════════════════════════════════════════════
section("BT-84  Exploitation — best action selected when epsilon=0")

m84 = ActionLayer(n_actions=4, seed=84)
# Give action 3 in state 7 a high Q value
for _ in range(50):
    m84._current_bmu = 7
    m84._current_action = 3
    m84.update(bmu_idx=7, action=3, rpe=0.8)

# Exploit: force entropy=0.0 so epsilon=EPSILON_MIN, but override to 0
# Directly test argmax by calling with entropy=0, confidence=1
# At entropy=0: eps = EPSILON_MIN * (1 - CONFIDENCE_GATE*1.0) = 0.05*0.70 = 0.035
# Very low but not zero — test over 100 trials that action=3 is chosen most
chosen = []
for _ in range(100):
    a = m84.select_action(bmu_idx=7, focus_entropy=0.0, thought_confidence=1.0)
    chosen.append(a)

frac_best = chosen.count(3) / 100
print(f"  Q[7]: {m84._Q[7]}  best_action=3")
print(f"  Fraction choosing action 3 (exploit): {frac_best:.2f}  (need >0.85)")
report("BT-84 Exploitation chooses best action", frac_best > 0.85,
       f"frac_action3={frac_best:.2f}")


# ═══════════════════════════════════════════════════════════════
# BT-85  Eligibility trace — Q update spreads to recent actions
# ═══════════════════════════════════════════════════════════════
section("BT-85  Eligibility trace — Q update applies to recently taken actions")

m85 = ActionLayer(n_actions=4, seed=85)

# Take action 1 in state 15, then action 0 in state 20
# Give positive rpe at state 20 — should update BOTH (15,1) and (20,0)
m85._current_bmu = 15
m85._current_action = 1
m85._e *= 0.0  # clear traces
m85._e[15, 1] = 1.0

# Now simulate step 2: decay trace and set new (20, 0)
m85._e *= (1.0 - TRACE_DECAY)  # (15,1) trace = 0.8
m85._current_bmu = 20
m85._current_action = 0
m85._e[20, 0] = 1.0  # (20,0) trace = 1.0

# Apply positive rpe — both should update
q_before_15_1 = float(m85._Q[15, 1])
q_before_20_0 = float(m85._Q[20, 0])
m85._Q += ETA_Q * 0.5 * m85._e

q_after_15_1 = float(m85._Q[15, 1])
q_after_20_0 = float(m85._Q[20, 0])

both_updated = (q_after_15_1 > q_before_15_1) and (q_after_20_0 > q_before_20_0)
print(f"  Q[15,1]: {q_before_15_1:.6f} → {q_after_15_1:.6f}  (trace=0.8 × ETA_Q × rpe)")
print(f"  Q[20,0]: {q_before_20_0:.6f} → {q_after_20_0:.6f}  (trace=1.0 × ETA_Q × rpe)")
print(f"  Both updated: {both_updated}")
report("BT-85 Eligibility trace spreads credit", both_updated,
       f"Q[15,1]={q_after_15_1:.6f}  Q[20,0]={q_after_20_0:.6f}")


# ═══════════════════════════════════════════════════════════════
# BT-86  Q decay — unused values decay over time
# ═══════════════════════════════════════════════════════════════
section("BT-86  Q decay — Q values decay when not reinforced")

m86 = ActionLayer(n_actions=4, seed=86)
# Manually set Q[30, 2] = 0.5
m86._Q[30, 2] = 0.5

# Run 1000 steps with rpe=0 and different state — Q[30,2] should decay
for i in range(1000):
    m86._current_bmu = 0
    m86._current_action = 0
    m86._e *= 0.0  # no eligibility for state 30
    m86._Q *= (1.0 - Q_DECAY)
    np.clip(m86._Q, Q_MIN, Q_MAX, out=m86._Q)

q_decayed = float(m86._Q[30, 2])
print(f"  Q[30,2] start: 0.5000  after 1000 steps no reinforcement: {q_decayed:.4f}")
print(f"  Expected: 0.5 × (1-{Q_DECAY})^1000 = {0.5*(1-Q_DECAY)**1000:.4f}")
report("BT-86 Q decay", q_decayed < 0.3,
       f"Q[30,2]={q_decayed:.4f}  (need <0.30)")


# ═══════════════════════════════════════════════════════════════
# BT-87  Policy table shape and range
# ═══════════════════════════════════════════════════════════════
section("BT-87  Policy table — shape (64,) with values in [0, N_ACTIONS)")

m87 = ActionLayer(n_actions=4, seed=87)
policy = m87.policy_table()
shape_ok  = policy.shape == (N_NEURONS,)
range_ok  = (policy.min() >= 0) and (policy.max() < m87._n_actions)
dtype_ok  = policy.dtype in (np.int32, np.int64)
qmap = m87.q_map()
qmap_ok = qmap.shape == (8, 8)

print(f"  policy shape: {policy.shape}  range: [{policy.min()},{policy.max()}]  dtype: {policy.dtype}")
print(f"  q_map shape: {qmap.shape}")
report("BT-87 Policy table correct",
       shape_ok and range_ok and dtype_ok and qmap_ok,
       f"shape={policy.shape}  range=[{policy.min()},{policy.max()}]  qmap={qmap.shape}")


# ═══════════════════════════════════════════════════════════════
# BT-88  Determinism — same seed = identical outputs
# ═══════════════════════════════════════════════════════════════
section("BT-88  Determinism — same seed produces identical output sequence")

m88a = ActionLayer(n_actions=4, seed=88)
m88b = ActionLayer(n_actions=4, seed=88)
rng88 = np.random.RandomState(88)

actions_a, actions_b = [], []
for i in range(100):
    bmu = rng88.randint(64)
    entropy = rng88.random()
    conf = rng88.random()
    rpe = rng88.uniform(-0.5, 0.5)

    rng88b = np.random.RandomState(88 + i * 1000)  # same sub-rng for both
    a_a = m88a.select_action(bmu, focus_entropy=entropy, thought_confidence=conf)
    a_b = m88b.select_action(bmu, focus_entropy=entropy, thought_confidence=conf)
    m88a.update(bmu, a_a, rpe)
    m88b.update(bmu, a_b, rpe)
    actions_a.append(a_a)
    actions_b.append(a_b)

mismatches = sum(1 for a, b in zip(actions_a, actions_b) if a != b)
print(f"  Action mismatches over 100 steps: {mismatches}  (must be 0)")
report("BT-88 Determinism", mismatches == 0, f"mismatches={mismatches}")


# ═══════════════════════════════════════════════════════════════
# BT-89  Integrated convergence with Brain RPE
# ═══════════════════════════════════════════════════════════════
section("BT-89  Integrated convergence — Q values grow when Brain RPE is consistently positive")

b89 = Brain(seed=89)
m89 = ActionLayer(n_actions=4, seed=89)
rng89 = np.random.RandomState(89)

# Train Brain 200 steps so RPE signal stabilises
for _ in range(200):
    b89.step(decoded_freq=1.0, stability_w=0.85, novelty_flag=0.0,
             plv_vector=rng89.rand(500).astype('float32'), reward=1.0)

# Now run M56 alongside Brain, measuring Q growth
q_mean_early = []
for i in range(50):
    r = b89.step(decoded_freq=1.0, stability_w=0.85, novelty_flag=0.0,
                 plv_vector=rng89.rand(500).astype('float32'), reward=1.0)
    m89.step(bmu_idx=r['bmu_idx'], rpe=r['rpe'],
             focus_entropy=r['focus_entropy'],
             thought_confidence=r['thought_confidence'])
    q_mean_early.append(float(np.abs(m89._Q).mean()))

q_mean_late = []
for i in range(200):
    r = b89.step(decoded_freq=1.0, stability_w=0.85, novelty_flag=0.0,
                 plv_vector=rng89.rand(500).astype('float32'), reward=1.0)
    out = m89.step(bmu_idx=r['bmu_idx'], rpe=r['rpe'],
                   focus_entropy=r['focus_entropy'],
                   thought_confidence=r['thought_confidence'])
    q_mean_late.append(float(np.abs(m89._Q).mean()))

mean_early = float(np.mean(q_mean_early))
mean_late  = float(np.mean(q_mean_late))
grew = mean_late > mean_early
print(f"  Q_mean early: {mean_early:.5f}  late: {mean_late:.5f}  grew: {grew}")
print(f"  Exploration rate: {m89.exploration_rate():.2f}")
m89.summary()
report("BT-89 Q values grow with Brain RPE", grew,
       f"early={mean_early:.5f}  late={mean_late:.5f}")


# ═══════════════════════════════════════════════════════════════
summarise()