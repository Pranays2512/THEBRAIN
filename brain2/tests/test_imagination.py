"""
test_imagination.py — Component 5: Imagination unit tests

Tests:
  1. Simulation produces correct frame count
  2. Offline mode: predictor weights unchanged after simulation
  3. Coherence score is in [0, 1]
  4. Different start states → different simulations
  5. evaluate() returns +1 when simulation reaches goal
  6. dream() produces n_dreams simulations
  7. extract_frames filters low-coherence simulations
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import brain2

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def test(name, condition):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    return condition

def run():
    print("\nImagination — Component 5")
    print("=" * 50)
    results = []

    rng = np.random.default_rng(0)
    DIM = 16

    pred = brain2.Predictor(input_dim=DIM, hidden_dim=32, lr=0.01)
    # Train briefly so predictor has some structure
    seq = [rng.random(DIM).astype(np.float32) for _ in range(6)]
    for _ in range(100):
        pred.reset()
        for i in range(len(seq)-1):
            pred.step(seq[i], seq[i+1])

    imag = brain2.Imagination(pred, max_steps=10)

    # 1. Simulation frame count = steps + 1 (includes start)
    start = rng.random(DIM).astype(np.float32)
    sim = imag.simulate(start, steps=5)
    results.append(test(f"Simulation frame count (got {len(sim.frames)}, expected 6)",
                        len(sim.frames) == 6))

    # 2. Offline mode: predictor unchanged after simulation
    start2 = rng.random(DIM).astype(np.float32)
    pred.reset()
    pred_before = np.array(pred.step(start2))
    imag.simulate(start2, steps=10)  # should not change weights
    pred.reset()
    pred_after = np.array(pred.step(start2))
    diff = float(np.max(np.abs(pred_before - pred_after)))
    results.append(test(f"Offline: predictor unchanged (diff={diff:.6f})", diff < 1e-5))

    # 3. Coherence in [0, 1]
    results.append(test(f"Coherence in [0,1] ({sim.coherence:.3f})",
                        0.0 <= sim.coherence <= 1.0 + 1e-5))

    # 4. Different starts → different simulations
    s1 = imag.simulate(np.zeros(DIM, dtype=np.float32), steps=5)
    s2 = imag.simulate(np.ones(DIM,  dtype=np.float32), steps=5)
    f1_last = np.array(s1.frames[-1])
    f2_last = np.array(s2.frames[-1])
    diff2 = float(np.max(np.abs(f1_last - f2_last)))
    results.append(test(f"Different starts → different outcomes (diff={diff2:.4f})",
                        diff2 > 1e-4))

    # 5. evaluate() returns ~+1 when last frame ≈ goal
    goal = np.array(sim.frames[-1])  # use last frame as goal
    val = imag.evaluate(sim, goal, threshold=0.7)
    results.append(test(f"evaluate() ≈ +1 when goal = last frame ({val:.3f})",
                        val > 0.5))

    # 6. dream() produces correct count
    dreams = imag.dream(n_dreams=5, steps_per_dream=8, seeds=[])
    results.append(test(f"dream() produces {len(dreams)} simulations", len(dreams) == 5))

    # 7. extract_frames filters low coherence
    all_frames = imag.extract_frames(dreams, min_coherence=0.0)
    filtered   = imag.extract_frames(dreams, min_coherence=0.99)
    results.append(test(f"extract_frames filters low coherence ({len(all_frames)} → {len(filtered)})",
                        len(filtered) <= len(all_frames)))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Component 5 (Imagination): READY\n")
        return True
    else:
        print("Component 5 (Imagination): NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
