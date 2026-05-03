"""
test_predictor.py — Component 4: Prediction Engine unit tests

Tests:
  1. Output dim matches input dim
  2. Error decreases with repeated training on same sequence
  3. Error larger on novel input than trained input (novelty detection)
  4. Offline mode: no weight change
  5. Reset clears state (same input after reset = same output)
  6. Save/load round-trip
  7. Prediction error is non-negative
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

def run():
    print("\nPredictor — Component 4")
    print("=" * 50)
    results = []

    rng = np.random.default_rng(0)
    DIM = 32

    p = brain2.Predictor(input_dim=DIM, hidden_dim=64, lr=0.01)

    # 1. Output dim matches input dim
    v = rng.random(DIM).astype(np.float32)
    pred = p.step(v)
    results.append(test(f"Output dim = input dim ({len(pred)} == {DIM})",
                        len(pred) == DIM))

    # 2. Error decreases with training on same sequence
    p2 = brain2.Predictor(input_dim=DIM, hidden_dim=64, lr=0.01)
    seq = [rng.random(DIM).astype(np.float32) for _ in range(5)]
    errors_early, errors_late = [], []
    for epoch in range(300):
        p2.reset()
        for i in range(len(seq)-1):
            p2.step(seq[i], seq[i+1])
            if epoch < 5:
                errors_early.append(p2.last_error)
            elif epoch >= 295:
                errors_late.append(p2.last_error)
    early_mean = float(np.mean(errors_early))
    late_mean  = float(np.mean(errors_late))
    results.append(test(f"Error decreases with training ({early_mean:.4f} → {late_mean:.4f})",
                        late_mean < early_mean * 0.8))

    # 3. Novel input has higher error than trained input
    p3 = brain2.Predictor(input_dim=DIM, hidden_dim=64, lr=0.01)
    known_seq = [rng.random(DIM).astype(np.float32) for _ in range(4)]
    for _ in range(500):
        p3.reset()
        for i in range(len(known_seq)-1):
            p3.step(known_seq[i], known_seq[i+1])

    p3.reset()
    for i in range(len(known_seq)-1):
        p3.step(known_seq[i], known_seq[i+1])
    known_err = p3.last_error

    p3.reset()
    novel_seq = [rng.random(DIM).astype(np.float32) for _ in range(4)]
    for i in range(len(novel_seq)-1):
        p3.step(novel_seq[i], novel_seq[i+1])
    novel_err = p3.last_error

    results.append(test(f"Novel input > trained input error ({novel_err:.4f} > {known_err:.4f})",
                        novel_err > known_err))

    # 4. Offline mode: no weight change
    p4 = brain2.Predictor(input_dim=DIM, hidden_dim=64, lr=0.05)
    v1 = rng.random(DIM).astype(np.float32)
    v2 = rng.random(DIM).astype(np.float32)
    pred_before = np.array(p4.step(v1))
    p4.reset()
    p4.set_offline(True)
    for _ in range(100):
        p4.step(v1, v2)
    p4.reset()
    pred_after = np.array(p4.step(v1))
    diff = float(np.max(np.abs(pred_before - pred_after)))
    results.append(test(f"Offline mode: no weight change (diff={diff:.6f})",
                        diff < 1e-5))

    # 5. Reset clears temporal state
    p5 = brain2.Predictor(input_dim=DIM, hidden_dim=64, lr=0.0)
    va = rng.random(DIM).astype(np.float32)
    vb = rng.random(DIM).astype(np.float32)
    # First: prime with va then query vb
    p5.step(va)
    pred1 = np.array(p5.step(vb))
    # After reset: no history, query vb
    p5.reset()
    pred2 = np.array(p5.step(vb))
    diff = float(np.max(np.abs(pred1 - pred2)))
    results.append(test(f"Reset clears state (pred differs: diff={diff:.4f})",
                        diff > 1e-4))

    # 6. Save / load round-trip
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
        path = tf.name
    p2.save(path)
    p2_loaded = brain2.Predictor.load(path)
    os.unlink(path)
    p2.reset(); p2_loaded.reset()
    v_test = rng.random(DIM).astype(np.float32)
    pred_orig   = np.array(p2.step(v_test))
    pred_loaded = np.array(p2_loaded.step(v_test))
    diff = float(np.max(np.abs(pred_orig - pred_loaded)))
    results.append(test(f"Save/load round-trip (diff={diff:.6f})",
                        diff < 1e-5))

    # 7. Prediction error non-negative
    p6 = brain2.Predictor(input_dim=DIM, hidden_dim=64, lr=0.01)
    errs = []
    for _ in range(20):
        a = rng.random(DIM).astype(np.float32)
        b = rng.random(DIM).astype(np.float32)
        p6.step(a, b)
        errs.append(p6.last_error)
    results.append(test(f"All errors non-negative (min={min(errs):.4f})",
                        all(e >= 0 for e in errs)))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Component 4 (Predictor): READY\n")
        return True
    else:
        print("Component 4 (Predictor): NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
