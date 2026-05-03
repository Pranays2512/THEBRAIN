"""
test_emotion.py — Component 6: Emotion unit tests

Tests:
  1. Initial state is neutral (valence=0, arousal=0)
  2. trigger() updates valence and arousal correctly
  3. from_prediction_error() increases arousal proportionally
  4. from_reward(+1) increases valence, from_reward(-1) decreases valence
  5. tick() decays valence and arousal toward neutral
  6. salience = 0 at neutral, > 0 when aroused + valenced
  7. lr_modulator >= 1.0 always (never reduces learning)
  8. approach_mode and avoidance_mode are mutually exclusive
  9. peak_valence/peak_arousal track extremes and decay slowly
  10. Save/load round-trip preserves state
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tempfile
import brain2

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def test(name, condition):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    return condition

def run():
    print("\nEmotion — Component 6")
    print("=" * 50)
    results = []

    # 1. Initial state is neutral
    em = brain2.Emotion(decay_rate=0.05)
    results.append(test(f"Initial state neutral (v={em.valence:.3f}, a={em.arousal:.3f})",
                        em.valence == 0.0 and em.arousal == 0.0))

    # 2. trigger() updates state
    em2 = brain2.Emotion()
    event = brain2.EmotionEvent(valence_delta=0.5, arousal_delta=0.3, intensity=1.0)
    em2.trigger(event)
    results.append(test(f"trigger() updates (v={em2.valence:.3f}, a={em2.arousal:.3f})",
                        em2.valence > 0.0 and em2.arousal > 0.0))

    # 3. from_prediction_error() increases arousal
    em3 = brain2.Emotion()
    em3.from_prediction_error(0.8)
    results.append(test(f"Prediction error → arousal (a={em3.arousal:.3f})",
                        em3.arousal > 0.2))

    # 4. Reward modulates valence
    pos = brain2.Emotion()
    neg = brain2.Emotion()
    pos.from_reward(1.0)
    neg.from_reward(-1.0)
    results.append(test(f"Positive reward → positive valence ({pos.valence:.3f})",
                        pos.valence > 0.0))
    results.append(test(f"Negative reward → negative valence ({neg.valence:.3f})",
                        neg.valence < 0.0))

    # 5. tick() decays toward neutral
    em4 = brain2.Emotion(decay_rate=0.5)
    em4.from_reward(1.0)
    v_before = em4.valence
    a_before = em4.arousal
    for _ in range(10):
        em4.tick()
    results.append(test(f"tick() decays valence ({v_before:.3f} → {em4.valence:.3f})",
                        abs(em4.valence) < abs(v_before)))
    results.append(test(f"tick() decays arousal ({a_before:.3f} → {em4.arousal:.3f})",
                        em4.arousal < a_before))

    # 6. Salience at neutral = 0, > 0 when active
    neutral = brain2.Emotion()
    sal_neutral = neutral.salience
    active = brain2.Emotion()
    active.trigger(brain2.EmotionEvent(0.8, 0.7, 1.0))
    sal_active = active.salience
    results.append(test(f"Salience: neutral={sal_neutral:.3f}, active={sal_active:.3f}",
                        sal_neutral == 0.0 and sal_active > 0.0))

    # 7. lr_modulator always >= 1.0
    em5 = brain2.Emotion()
    results.append(test(f"lr_modulator neutral={em5.lr_modulator:.3f} >= 1.0",
                        em5.lr_modulator >= 1.0))
    em5.from_reward(1.0)
    results.append(test(f"lr_modulator aroused={em5.lr_modulator:.3f} >= 1.0",
                        em5.lr_modulator >= 1.0))

    # 8. approach/avoidance mutually exclusive
    approach = brain2.Emotion()
    approach.trigger(brain2.EmotionEvent(0.5, 0.6, 1.0))
    avoid = brain2.Emotion()
    avoid.trigger(brain2.EmotionEvent(-0.5, 0.6, 1.0))
    results.append(test(f"approach_mode when positive valence+arousal ({approach.valence:.2f},{approach.arousal:.2f})",
                        approach.approach_mode and not approach.avoidance_mode))
    results.append(test(f"avoidance_mode when negative valence+arousal ({avoid.valence:.2f},{avoid.arousal:.2f})",
                        avoid.avoidance_mode and not avoid.approach_mode))

    # 9. Peaks track extremes and decay slower
    em6 = brain2.Emotion(decay_rate=0.5, peak_decay=0.01)
    em6.trigger(brain2.EmotionEvent(0.9, 0.9, 1.0))
    peak_v_after = em6.peak_valence
    peak_a_after = em6.peak_arousal
    # Fast decay current state, peaks should remain higher
    for _ in range(5):
        em6.tick()
    results.append(test(f"Peak decays slower than state (peak_v={em6.peak_valence:.3f} >= v={em6.valence:.3f})",
                        em6.peak_valence >= em6.valence - 0.01))

    # 10. Save/load
    em7 = brain2.Emotion(decay_rate=0.1)
    em7.trigger(brain2.EmotionEvent(0.6, 0.4, 1.0))
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
        path = tf.name
    em7.save(path)
    em8 = brain2.Emotion.load(path)
    os.unlink(path)
    results.append(test(f"Save/load preserves state (v:{em7.valence:.4f}=={em8.valence:.4f})",
                        abs(em7.valence - em8.valence) < 1e-5 and
                        abs(em7.arousal - em8.arousal) < 1e-5))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Component 6 (Emotion): READY\n")
        return True
    else:
        print("Component 6 (Emotion): NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
