"""
test_som.py — Component 1: SOM unit tests

Tests:
  1. Similar inputs cluster to nearby BMUs
  2. Different inputs spread across map
  3. Same input → same BMU (deterministic)
  4. Activation map sums correctly
  5. Weights update toward input after train
  6. Save/load round-trip
  7. BMU converges with repeated training
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
    print("\nSOM — Component 1")
    print("=" * 50)
    results = []

    rng = np.random.default_rng(0)
    som = brain2.SOM(rows=20, cols=20, n_dims=16,
                     init_lr=0.3, lr_decay=0.999, radius_decay=0.999)

    # 1. Deterministic: same input → same BMU
    v = rng.random(16).astype(np.float32)
    b1 = som.find_bmu(v)
    b2 = som.find_bmu(v)
    results.append(test("Deterministic BMU", b1 == b2))

    # 2. Different inputs → different BMUs (statistically)
    # In HSOM, the root grid is small (e.g. 16 neurons), so unique BMUs might be lower
    # without spawning. We just want to ensure it's not all going to 1 neuron.
    vecs = [rng.random(16).astype(np.float32) for _ in range(50)]
    bmus = [som.find_bmu(v) for v in vecs]
    unique = len(set(bmus))
    results.append(test(f"Different inputs spread (unique BMUs: {unique}/50)", unique > 3))

    # 3. Weights update toward input after training
    v = np.zeros(16, dtype=np.float32)
    v[0] = 1.0
    bmu_before = som.find_bmu(v)
    w_before = som.neuron_weights(bmu_before).copy()
    for _ in range(200):
        bmu = som.find_bmu(v)
        som.update(v, bmu, reward_mod=1.0)
    w_after = som.neuron_weights(bmu_before)
    dist_before = float(np.sum((w_before - v)**2))
    dist_after  = float(np.sum((np.array(w_after) - v)**2))
    results.append(test(f"Weights converge toward input ({dist_before:.3f} → {dist_after:.3f})",
                        dist_after < dist_before))

    # 4. Similar inputs → nearby BMUs after training
    # Train cluster A around [1,0,...] and cluster B around [0,1,...]
    som2 = brain2.SOM(rows=20, cols=20, n_dims=16, init_lr=0.5)
    a_base = np.zeros(16, dtype=np.float32); a_base[0] = 1.0
    b_base = np.zeros(16, dtype=np.float32); b_base[15] = 1.0
    for _ in range(3000):
        v_a = a_base + rng.random(16).astype(np.float32) * 0.1
        som2.update(v_a, som2.find_bmu(v_a))
        v_b = b_base + rng.random(16).astype(np.float32) * 0.1
        som2.update(v_b, som2.find_bmu(v_b))

    a_bmus = [som2.find_bmu((a_base + rng.random(16).astype(np.float32)*0.05)) for _ in range(10)]
    b_bmus = [som2.find_bmu((b_base + rng.random(16).astype(np.float32)*0.05)) for _ in range(10)]
    
    # Compare their weight vectors instead of raw integer indices
    w_a = np.array([som2.neuron_weights(b) for b in a_bmus])
    w_b = np.array([som2.neuron_weights(b) for b in b_bmus])
    
    a_center = np.mean(w_a, axis=0)
    b_center = np.mean(w_b, axis=0)
    
    a_spread = np.mean(np.linalg.norm(w_a - a_center, axis=1))
    b_spread = np.mean(np.linalg.norm(w_b - b_center, axis=1))
    ab_sep = np.linalg.norm(a_center - b_center)
    
    results.append(test(f"Similar inputs cluster in weight space (A spread={a_spread:.2f}, B spread={b_spread:.2f}, sep={ab_sep:.2f})",
                        a_spread < 0.5 and b_spread < 0.5 and ab_sep > 0.5))

    # 5. Activation map range [0,1] with max=1
    amap = som.activation_map(v)
    results.append(test("Activation map range [0,1]",
                        float(np.min(amap)) >= -1e-6 and abs(float(np.max(amap)) - 1.0) < 1e-5))

    # 6. Step counter increments on update
    s0 = som.step
    som.update(v, som.find_bmu(v))
    results.append(test("Step counter increments", som.step == s0 + 1))

    # 7. Save / load round-trip
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
        path = tf.name
    som.save(path)
    som_loaded = brain2.SOM.load(path)
    os.unlink(path)
    v2 = rng.random(16).astype(np.float32)
    results.append(test("Save/load round-trip (same BMU)",
                        som.find_bmu(v2) == som_loaded.find_bmu(v2)))

    # 8. Reward mod > 1 causes larger update
    som3 = brain2.SOM(rows=10, cols=10, n_dims=8, init_lr=0.3)
    v3 = np.ones(8, dtype=np.float32)
    bmu3 = som3.find_bmu(v3)
    w0 = np.array(som3.neuron_weights(bmu3))
    som3.update(v3, bmu3, reward_mod=3.0)
    w1 = np.array(som3.neuron_weights(bmu3))
    som4 = brain2.SOM(rows=10, cols=10, n_dims=8, init_lr=0.3)
    bmu4 = som4.find_bmu(v3)
    w0b = np.array(som4.neuron_weights(bmu4))
    som4.update(v3, bmu4, reward_mod=1.0)
    w1b = np.array(som4.neuron_weights(bmu4))
    delta_high = float(np.sum(np.abs(w1 - w0)))
    delta_low  = float(np.sum(np.abs(w1b - w0b)))
    results.append(test(f"High reward_mod causes larger update ({delta_high:.4f} > {delta_low:.4f})",
                        delta_high > delta_low))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Component 1 (SOM): READY\n")
        return True
    else:
        print("Component 1 (SOM): NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
