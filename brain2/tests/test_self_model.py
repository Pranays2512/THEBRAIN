"""
test_self_model.py — Component 9: Self-Model unit tests

Tests:
  1. observe() accepts InternalState without error
  2. obs_count increments after each observation
  3. identity vector is non-zero after observations
  4. identity drifts toward observed states
  5. drift() near 0 when current state matches identity
  6. drift() > 0 when current state diverges from learned identity
  7. current_concept returns valid neuron index
  8. arousal_trend is positive when arousal increasing
  9. arousal_trend is negative when arousal decreasing
  10. Save/load preserves obs_count and identity
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tempfile
import brain2

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def test(name, condition):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    return condition

def make_state(valence=0., arousal=0., pred_error=0., wm_load=0.5,
               salience=0., attention_focus=0.5, mean_saliency=0.,
               approach=0., avoidance=0., arousal_trend=0.):
    s = brain2.InternalState()
    s.valence         = valence
    s.arousal         = arousal
    s.salience        = salience
    s.pred_error      = pred_error
    s.wm_load         = wm_load
    s.attention_focus = attention_focus
    s.mean_saliency   = mean_saliency
    s.approach        = approach
    s.avoidance       = avoidance
    s.arousal_trend   = arousal_trend
    return s

def run():
    print("\nSelf-Model — Component 9")
    print("=" * 50)
    results = []

    sm = brain2.SelfModel(n_self_neurons=16, seed=42)

    # 1. observe() works
    try:
        sm.observe(make_state(valence=0.5, arousal=0.3))
        results.append(test("observe() accepts InternalState", True))
    except Exception as e:
        results.append(test(f"observe() accepts InternalState (err: {e})", False))

    # 2. obs_count increments
    before = sm.obs_count
    for _ in range(5):
        sm.observe(make_state(valence=0.1, arousal=0.2))
    results.append(test(f"obs_count increments ({before} → {sm.obs_count})",
                        sm.obs_count == before + 5))

    # 3. identity non-zero after observations
    identity = np.array(sm.identity())
    results.append(test(f"identity vector non-zero (sum={float(np.sum(np.abs(identity))):.4f})",
                        float(np.sum(np.abs(identity))) > 1e-6))

    # 4. identity drifts toward observed states
    sm2 = brain2.SelfModel(n_self_neurons=8)
    # Flood with calm states
    for _ in range(100):
        sm2.observe(make_state(valence=0.9, arousal=0.1, pred_error=0.05))
    id2 = np.array(sm2.identity())
    # Identity valence should be close to 0.9
    results.append(test(f"Identity drifts toward observations (valence≈{id2[0]:.3f})",
                        id2[0] > 0.5))

    # 5. drift ≈ 0 when state matches identity
    typical_state = make_state(valence=0.9, arousal=0.1, pred_error=0.05)
    d = sm2.drift(typical_state)
    results.append(test(f"drift small for typical state ({d:.3f})",
                        d < 0.3))

    # 6. drift > 0 when state diverges
    anomalous = make_state(valence=-0.9, arousal=0.95, pred_error=0.9,
                           wm_load=0.9, salience=0.9, mean_saliency=0.9,
                           approach=0., avoidance=1.)
    d_anom = sm2.drift(anomalous)
    results.append(test(f"drift large for anomalous state ({d_anom:.3f} > {d:.3f})",
                        d_anom > d))

    # 7. current_concept returns valid index [0, n_self_neurons)
    sm3 = brain2.SelfModel(n_self_neurons=16)
    for _ in range(10):
        sm3.observe(make_state(valence=0.3, arousal=0.5))
    concept = sm3.current_concept(make_state(valence=0.3, arousal=0.5))
    results.append(test(f"current_concept valid index (got {concept})",
                        0 <= concept < 16))

    # 8. Arousal trend positive when increasing
    sm4 = brain2.SelfModel(n_self_neurons=8)
    for i in range(10):
        sm4.observe(make_state(arousal=float(i) * 0.1))
    trend = sm4.arousal_trend()
    results.append(test(f"arousal_trend positive when increasing ({trend:.3f})",
                        trend > 0.0))

    # 9. Arousal trend negative when decreasing
    sm5 = brain2.SelfModel(n_self_neurons=8)
    for i in range(10):
        sm5.observe(make_state(arousal=1.0 - float(i) * 0.1))
    trend2 = sm5.arousal_trend()
    results.append(test(f"arousal_trend negative when decreasing ({trend2:.3f})",
                        trend2 < 0.0))

    # 10. Save/load
    sm6 = brain2.SelfModel(n_self_neurons=16)
    for _ in range(20):
        sm6.observe(make_state(valence=0.5, arousal=0.4, pred_error=0.3))
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
        path = tf.name
    sm6.save(path)
    sm6_loaded = brain2.SelfModel.load(path)
    os.unlink(path)
    id_orig   = np.array(sm6.identity())
    id_loaded = np.array(sm6_loaded.identity())
    diff = float(np.max(np.abs(id_orig - id_loaded)))
    results.append(test(f"Save/load preserves obs_count+identity (diff={diff:.6f}, n={sm6_loaded.obs_count})",
                        diff < 1e-5 and sm6_loaded.obs_count == sm6.obs_count))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Component 9 (Self-Model): READY\n")
        return True
    else:
        print("Component 9 (Self-Model): NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
