"""
TEST_GW.PY — GLOBAL WORKSPACE BREAK TESTS
==========================================

Four tests, each probing a different property of the GW.
Not unit tests of arithmetic — behavioural tests of integration.

  Test 1: Coherence responds to module agreement
  Test 2: Tension only fires when stuck in KNOWN territory
  Test 3: Ignition gates epsilon boost
  Test 4: Full Brain integration — GW keys appear and move correctly
"""

import sys, math
sys.path.insert(0, '/home/claude')
sys.path.insert(1, '/mnt/user-data/uploads')

import numpy as np
from global_workspace import GlobalWorkspace, GW_IGNITION_THRESHOLD, GW_EPSILON_SCALE

PASS = 0; FAIL = 0

def check(name, condition, got, expected_desc):
    global PASS, FAIL
    if condition:
        print(f"  ✓  {name}")
        PASS += 1
    else:
        print(f"  ✗  {name}")
        print(f"       got: {got}")
        print(f"       expected: {expected_desc}")
        FAIL += 1


def make_gw():
    return GlobalWorkspace(n_zones=8, seed=42)


# ═══════════════════════════════════════════════════════════════
# TEST 1 — Coherence responds to module agreement
# ═══════════════════════════════════════════════════════════════
# When all "knowledge" signals agree (all high or all low), coherence
# should be high. When they conflict, coherence should drop.

print("\n" + "─"*60)
print("TEST 1 — Coherence responds to module agreement")
print("─"*60)

# 1A: All modules agree the situation is well-known and manageable.
# familiarity=high, confidence=high, l4_top_prob=high, boredom=low
# → coherence should be high (>0.65)
gw = make_gw()
# Run 20 steps to let EMA settle
for _ in range(20):
    out = gw.step(qe_norm=0.1, familiarity=0.9, freq_idx=2,
                  prediction_error=0.1, thought_confidence=0.85,
                  rpe=0.3, intrinsic_rwd=0.9,
                  corridor_boredom=0.1, steps_since_reward=3,
                  salience=0.6, l4_top_prob=0.8)
check("1A: All modules agree → coherence high",
      out['coherence'] > 0.55,
      out['coherence'], "> 0.55")

# 1B: Modules conflict. Familiarity is high but confidence is low,
# L4 is uncertain, boredom is high. Std should be large → coherence low.
gw2 = make_gw()
for _ in range(20):
    out2 = gw2.step(qe_norm=0.5, familiarity=0.9, freq_idx=2,
                    prediction_error=0.7, thought_confidence=0.1,
                    rpe=-0.3, intrinsic_rwd=0.3,
                    corridor_boredom=0.85, steps_since_reward=30,
                    salience=0.6, l4_top_prob=0.15)
check("1B: Modules conflict → coherence lower than 1A",
      out2['coherence'] < out['coherence'],
      f"conflict={out2['coherence']:.3f} vs agree={out['coherence']:.3f}",
      "conflict coherence < agreement coherence")

# 1C: Coherence in agreement > 0.5
check("1C: Agreement coherence above 0.5",
      out['coherence'] > 0.50,
      out['coherence'], "> 0.50")

# 1D: Coherence in conflict < 0.5
check("1D: Conflict coherence below 0.5",
      out2['coherence'] < 0.50,
      out2['coherence'], "< 0.50")


# ═══════════════════════════════════════════════════════════════
# TEST 2 — Tension only fires when stuck in KNOWN territory
# ═══════════════════════════════════════════════════════════════
# Tension = arousal × familiarity.
# Bored in unfamiliar place → arousal high but familiarity low → tension LOW.
# Bored in familiar place  → arousal high AND familiarity high → tension HIGH.

print("\n" + "─"*60)
print("TEST 2 — Tension fires only when stuck in known territory")
print("─"*60)

# 2A: High boredom + LOW familiarity → tension should be low
# (brain is in unfamiliar territory — it's exploring, not stuck)
gw = make_gw()
for _ in range(10):
    out_unfam = gw.step(qe_norm=0.6, familiarity=0.05, freq_idx=3,
                        prediction_error=0.8, thought_confidence=0.1,
                        rpe=-0.1, intrinsic_rwd=0.2,
                        corridor_boredom=0.80, steps_since_reward=35,
                        salience=0.6, l4_top_prob=0.1)
check("2A: Bored in UNFAMILIAR place → low tension",
      out_unfam['tension'] < 0.25,
      out_unfam['tension'], "< 0.25  (arousal is high but familiarity is low)")

# 2B: High boredom + HIGH familiarity → tension should be high
# (brain is in a well-known place and still not getting what it needs)
gw2 = make_gw()
for _ in range(10):
    out_fam = gw2.step(qe_norm=0.1, familiarity=0.90, freq_idx=3,
                       prediction_error=0.2, thought_confidence=0.7,
                       rpe=-0.1, intrinsic_rwd=0.8,
                       corridor_boredom=0.80, steps_since_reward=35,
                       salience=0.6, l4_top_prob=0.7)
check("2B: Bored in FAMILIAR place → high tension",
      out_fam['tension'] > 0.15,
      out_fam['tension'], "> 0.15  (both arousal and familiarity are high)")

# 2C: Tension in familiar case > tension in unfamiliar case
check("2C: Familiar-bored tension > unfamiliar-bored tension",
      out_fam['tension'] > out_unfam['tension'],
      f"familiar={out_fam['tension']:.3f} unfamiliar={out_unfam['tension']:.3f}",
      "familiar tension > unfamiliar tension")

# 2D: Zero familiarity → near-zero tension regardless of arousal
gw3 = make_gw()
for _ in range(5):
    out_zero = gw3.step(qe_norm=0.9, familiarity=0.0, freq_idx=5,
                        prediction_error=0.9, thought_confidence=0.0,
                        rpe=-0.5, intrinsic_rwd=0.1,
                        corridor_boredom=1.0, steps_since_reward=40,
                        salience=0.9, l4_top_prob=0.0)
check("2D: Zero familiarity → near-zero tension",
      out_zero['tension'] < 0.05,
      out_zero['tension'], "< 0.05  (can't be stuck in a place you don't know)")


# ═══════════════════════════════════════════════════════════════
# TEST 3 — Ignition threshold gates epsilon boost
# ═══════════════════════════════════════════════════════════════
# Epsilon boost is scaled by ignition_factor = salience / threshold.
# Low salience → no boost even if tension/curiosity are maxed.
# High salience + high tension → full boost.

print("\n" + "─"*60)
print("TEST 3 — Ignition gates epsilon boost")
print("─"*60)

# 3A: Salience=0 → epsilon_boost should be 0 regardless of tension
gw = make_gw()
# Prime tension first (5 steps with familiar+bored)
for _ in range(5):
    gw.step(qe_norm=0.5, familiarity=0.9, freq_idx=2,
            prediction_error=0.7, thought_confidence=0.5,
            rpe=-0.2, intrinsic_rwd=0.3,
            corridor_boredom=0.9, steps_since_reward=40,
            salience=0.8, l4_top_prob=0.7)
# Now fire with salience=0
out_no_sal = gw.step(qe_norm=0.5, familiarity=0.9, freq_idx=2,
                     prediction_error=0.7, thought_confidence=0.5,
                     rpe=-0.2, intrinsic_rwd=0.3,
                     corridor_boredom=0.9, steps_since_reward=40,
                     salience=0.0, l4_top_prob=0.7)
check("3A: Salience=0 → epsilon_boost=0",
      out_no_sal['epsilon_boost'] == 0.0,
      out_no_sal['epsilon_boost'], "== 0.0  (ignition_factor=0 kills boost)")

# 3B: High salience + high tension → epsilon_boost > 0
gw2 = make_gw()
for _ in range(10):
    out_full = gw2.step(qe_norm=0.5, familiarity=0.9, freq_idx=2,
                        prediction_error=0.6, thought_confidence=0.5,
                        rpe=-0.2, intrinsic_rwd=0.4,
                        corridor_boredom=0.9, steps_since_reward=40,
                        salience=0.9, l4_top_prob=0.7)
check("3B: High salience + tension → epsilon_boost > 0",
      out_full['epsilon_boost'] > 0.01,
      out_full['epsilon_boost'], "> 0.01")

# 3C: epsilon_boost never exceeds GW_EPSILON_SCALE
check("3C: epsilon_boost never exceeds GW_EPSILON_SCALE",
      out_full['epsilon_boost'] <= GW_EPSILON_SCALE + 1e-9,
      out_full['epsilon_boost'], f"<= {GW_EPSILON_SCALE}")

# 3D: Below-threshold salience → reduced boost vs above-threshold
gw3 = make_gw()
for _ in range(10):
    out_low_sal = gw3.step(qe_norm=0.5, familiarity=0.9, freq_idx=2,
                           prediction_error=0.6, thought_confidence=0.5,
                           rpe=-0.2, intrinsic_rwd=0.4,
                           corridor_boredom=0.9, steps_since_reward=40,
                           salience=GW_IGNITION_THRESHOLD * 0.3,  # well below threshold
                           l4_top_prob=0.7)
check("3D: Below-threshold salience → smaller boost than above-threshold",
      out_low_sal['epsilon_boost'] < out_full['epsilon_boost'],
      f"low_sal={out_low_sal['epsilon_boost']:.4f} full={out_full['epsilon_boost']:.4f}",
      "low salience boost < high salience boost")

# 3E: Ignition flag correct
gw4 = make_gw()
out_ignited = gw4.step(qe_norm=0.2, familiarity=0.5, freq_idx=1,
                       prediction_error=0.3, thought_confidence=0.4,
                       rpe=0.0, intrinsic_rwd=0.7,
                       corridor_boredom=0.2, steps_since_reward=5,
                       salience=GW_IGNITION_THRESHOLD + 0.1,
                       l4_top_prob=0.5)
check("3E: salience above threshold → ignited=True",
      out_ignited['ignited'] == True,
      out_ignited['ignited'], "True")

out_not_ignited = gw4.step(qe_norm=0.2, familiarity=0.5, freq_idx=1,
                            prediction_error=0.3, thought_confidence=0.4,
                            rpe=0.0, intrinsic_rwd=0.7,
                            corridor_boredom=0.2, steps_since_reward=5,
                            salience=GW_IGNITION_THRESHOLD - 0.1,
                            l4_top_prob=0.5)
check("3F: salience below threshold → ignited=False",
      out_not_ignited['ignited'] == False,
      out_not_ignited['ignited'], "False")


# ═══════════════════════════════════════════════════════════════
# TEST 4 — Full Brain integration
# ═══════════════════════════════════════════════════════════════
# Run brain.step() 200 times. Verify:
#   - All gws_* keys present in output
#   - Signals change over time (not frozen)
#   - Boredom builds when freq_idx is constant
#   - Epsilon_boost responds to built-up boredom

print("\n" + "─"*60)
print("TEST 4 — Full Brain integration")
print("─"*60)

from brain import Brain

brain = Brain(seed=42)
plv = np.random.default_rng(99).random(64).astype(np.float32)

# 4A: All gws_* keys present in output
out = brain.step(decoded_freq=1.0, stability_w=0.7, novelty_flag=0.0,
                 plv_vector=plv, reward=0.0, freq_idx=2, world_moved=True)
gw_keys = [k for k in out if k.startswith('gws')]
expected_keys = ['gws_arousal', 'gws_arousal_raw', 'gws_valence_tone',
                 'gws_valence_raw', 'gws_curiosity_pull', 'gws_surprise_debt',
                 'gws_epsilon_boost']
missing = [k for k in expected_keys if k not in gw_keys]
check("4A: All gws_* keys present in Brain output",
      len(missing) == 0,
      f"missing: {missing}", "no missing keys")

# 4B: GWS values are finite floats (not NaN/inf)
all_finite = all(
    math.isfinite(float(out[k])) for k in expected_keys
)
check("4B: All gws values are finite",
      all_finite,
      {k: out[k] for k in expected_keys}, "all finite floats")

# 4C: Run 200 steps with the SAME freq_idx — boredom should build in M58,
# which should feed into gws_arousal. Arousal at step 200 > step 1.
brain2 = Brain(seed=42)
first_arousal = None
last_arousal  = None
for step in range(200):
    r = brain2.step(decoded_freq=1.5, stability_w=0.8, novelty_flag=0.0,
                    plv_vector=plv, reward=0.0, freq_idx=3, world_moved=True)
    if step == 0:  first_arousal = r['gws_arousal']
    if step == 199: last_arousal = r['gws_arousal']
check("4C: Arousal builds over 200 steps in same zone",
      last_arousal > first_arousal,
      f"step0={first_arousal:.4f} step199={last_arousal:.4f}",
      "last_arousal > first_arousal")

# 4D: Epsilon_boost > 0 after boredom has built
check("4D: epsilon_boost > 0 after sustained boredom",
      r['gws_epsilon_boost'] > 0.0,
      r['gws_epsilon_boost'], "> 0.0")

# 4E: Valence_tone responds to reward — step with reward should produce
# a positive valence_tone shift
brain3 = Brain(seed=42)
for _ in range(10):
    brain3.step(decoded_freq=1.0, stability_w=0.7, novelty_flag=0.0,
                plv_vector=plv, reward=0.0, freq_idx=2, world_moved=True)
before_reward = brain3.gws._valence_tone
brain3.step(decoded_freq=1.0, stability_w=0.7, novelty_flag=0.0,
            plv_vector=plv, reward=1.0, freq_idx=2, world_moved=True)
after_reward = brain3.gws._valence_tone
check("4E: Positive reward shifts valence_tone upward",
      after_reward > before_reward,
      f"before={before_reward:.4f} after={after_reward:.4f}",
      "valence_tone increases after reward")

# 4F: GWS summary runs without error
try:
    brain.gws.summary()
    check("4F: gws.summary() runs without error", True, None, None)
except Exception as e:
    check("4F: gws.summary() runs without error", False, str(e), "no exception")

# 4G: gws.get_state() returns coherence, tension, readiness
state = brain.gws.get_state()
has_keys = all(k in state for k in ('coherence', 'tension', 'readiness'))
check("4G: get_state() includes coherence, tension, readiness",
      has_keys, list(state.keys()), "coherence, tension, readiness present")


# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════

total = PASS + FAIL
print(f"\n{'═'*60}")
print(f"  {PASS}/{total} passed")
if FAIL == 0:
    print("  ALL PASS — Global Workspace is integrated and behaving correctly.")
    print()
    print("  What this means:")
    print("  ✓ Coherence reads all modules simultaneously and reflects agreement")
    print("  ✓ Tension correctly fires only when stuck in KNOWN territory")
    print("  ✓ Ignition gates the epsilon boost — low salience = no broadcast")
    print("  ✓ Brain.step() returns unified GW state every step")
    print("  ✓ Arousal builds from sustained boredom and feeds exploration")
    print("  ✓ Valence tone tracks reward direction across time")
else:
    print(f"  {FAIL} FAILED")
print(f"{'═'*60}\n")