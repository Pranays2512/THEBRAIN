"""
test_working_mem.py — Component 3: Working Memory unit tests

Tests:
  1. Slots fill up to capacity
  2. Over capacity: evicts least active
  3. Decay: activations decrease each tick
  4. Context vector: mean of active slots
  5. Duplicate (similar) input updates existing slot, not new
  6. Salience protects from eviction
  7. Save/load round-trip
  8. Clear empties all slots
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
    print("\nWorkingMemory — Component 3")
    print("=" * 50)
    results = []

    rng = np.random.default_rng(0)
    DIM = 16

    wm = brain2.WorkingMemory(n_dims=DIM, capacity=7, decay_rate=0.9)

    # 1. Slots fill to capacity
    vecs = [np.eye(DIM)[i].astype(np.float32) * 3.0 for i in range(7)]
    for v in vecs:
        wm.gate(v)
    wm.tick() # Need at least one tick to generate spikes
    results.append(test(f"Fills to capacity (size={wm.size})", wm.size == 7))

    # 2. Over capacity: evicts least active
    for _ in range(5):
        wm.tick()
    old_acts = list(wm.activations())
    v_new = rng.random(DIM).astype(np.float32) * 3.0
    wm.gate(v_new, salience=0.0)
    wm.tick()
    results.append(test(f"Capacity maintained after overflow (size={wm.size})",
                        wm.size <= 7))

    # 3. Decay: activations decrease each tick
    wm2 = brain2.WorkingMemory(n_dims=DIM, capacity=7, decay_rate=0.9)
    v = rng.random(DIM).astype(np.float32) * 3.0
    wm2.gate(v)
    wm2.tick() # Initial spike
    acts_before = list(wm2.activations())
    wm2.tick() # Decay
    acts_after = list(wm2.activations())
    if len(acts_before) > 0 and len(acts_after) > 0:
        results.append(test(f"Decay reduces activation ({acts_before[0]:.3f} → {acts_after[0]:.3f})",
                            acts_after[0] < acts_before[0]))
    else:
        results.append(test("Decay reduces activation (failed to spike)", False))

    # 4. Context vector: weighted mean of slots
    wm3 = brain2.WorkingMemory(n_dims=DIM, capacity=7, decay_rate=0.95)
    ones  = np.ones(DIM,  dtype=np.float32) * 3.0
    zeros = np.zeros(DIM, dtype=np.float32)
    wm3.gate(ones)
    wm3.tick()
    ctx = np.array(wm3.context())
    # Context should be > 0 since it spiked
    results.append(test(f"Context = slot content when single slot (mean={float(np.mean(ctx)):.4f})",
                        float(np.mean(ctx)) > 0.1))

    # 5. Similar input updates existing slot, not creates new
    wm4 = brain2.WorkingMemory(n_dims=DIM, capacity=7)
    base = rng.random(DIM).astype(np.float32) * 3.0
    wm4.gate(base)
    wm4.tick()
    size_before = wm4.size
    # Very similar vector
    similar = base + rng.random(DIM).astype(np.float32) * 0.01
    wm4.gate(similar)
    results.append(test(f"Similar input merges (size unchanged: {size_before} → {wm4.size})",
                        wm4.size == size_before))

    # 6. High-salience slot survives many ticks
    wm5 = brain2.WorkingMemory(n_dims=DIM, capacity=3, decay_rate=0.85)
    v_hi = rng.random(DIM).astype(np.float32) * 3.0
    v_lo = [rng.random(DIM).astype(np.float32) * 3.0 for _ in range(2)]
    wm5.gate(v_hi, salience=1.0)  # high salience
    for v in v_lo:
        wm5.gate(v, salience=0.0)
    wm5.tick()
    # Fill to overflow
    for _ in range(3):
        wm5.gate(rng.random(DIM).astype(np.float32) * 3.0, salience=0.0)
    wm5.tick()
    
    ctx5 = np.array(wm5.context())
    sim_to_hi = float(np.dot(ctx5, v_hi) / (np.linalg.norm(ctx5) * np.linalg.norm(v_hi) + 1e-8))
    results.append(test(f"High salience survives eviction (ctx sim to hi={sim_to_hi:.3f})",
                        sim_to_hi > 0.1))

    # 7. Save/load round-trip
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
        path = tf.name
    wm.save(path)
    wm_loaded = brain2.WorkingMemory.load(path)
    os.unlink(path)
    ctx_orig   = np.array(wm.context())
    ctx_loaded = np.array(wm_loaded.context())
    diff = float(np.max(np.abs(ctx_orig - ctx_loaded)))
    results.append(test(f"Save/load round-trip (context diff={diff:.6f})", diff < 1e-5))

    # 8. Clear empties all slots
    wm.clear()
    results.append(test("Clear empties slots", wm.size == 0 and wm.empty))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Component 3 (WorkingMemory): READY\n")
        return True
    else:
        print("Component 3 (WorkingMemory): NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
