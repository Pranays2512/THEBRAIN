"""
test_episodic.py — Component 2: Episodic Memory unit tests

Tests:
  1. Low surprise: episode not stored
  2. High surprise: episode stored
  3. Retrieve returns most similar episode
  4. Dissimilar query returns different episode than similar query
  5. Max capacity: old episodes evicted
  6. Consolidation: reduces episode count, creates prototypes
  7. Save/load round-trip
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
    print("\nEpisodicMemory — Component 2")
    print("=" * 50)
    results = []

    rng = np.random.default_rng(0)
    DIM = 32

    # 1. Low surprise: not stored
    em = brain2.EpisodicMemory(n_dims=DIM, surprise_threshold=0.5)
    for _ in range(5):
        em.observe(rng.random(DIM).astype(np.float32))
    stored = em.commit(prediction_error=0.1)
    results.append(test("Low surprise: not stored", not stored and em.episode_count == 0))

    # 2. High surprise: stored
    for _ in range(5):
        em.observe(rng.random(DIM).astype(np.float32))
    stored = em.commit(prediction_error=0.9)
    results.append(test("High surprise: stored", stored and em.episode_count == 1))

    # 3. Retrieve returns most similar
    em2 = brain2.EpisodicMemory(n_dims=DIM, surprise_threshold=0.1)
    # Store episode A (ones in first half)
    v_a = np.zeros(DIM, dtype=np.float32); v_a[:DIM//2] = 1.0
    for _ in range(3):
        em2.observe(v_a + rng.random(DIM).astype(np.float32) * 0.05)
    em2.commit(0.9)

    # Store episode B (ones in second half)
    v_b = np.zeros(DIM, dtype=np.float32); v_b[DIM//2:] = 1.0
    for _ in range(3):
        em2.observe(v_b + rng.random(DIM).astype(np.float32) * 0.05)
    em2.commit(0.9)

    # Query with v_a — should get episode A (first stored)
    top = em2.retrieve_topk(v_a.astype(np.float32), k=1)
    a_idx = top[0][1] if top else -1
    # Query with v_b — should get episode B (second stored)
    top = em2.retrieve_topk(v_b.astype(np.float32), k=1)
    b_idx = top[0][1] if top else -1
    results.append(test(f"Retrieve: different episodes for different queries (a={a_idx}, b={b_idx})",
                        a_idx != b_idx))

    # 4. Retrieve returns non-None result
    ep = em2.retrieve(v_a.astype(np.float32))
    results.append(test("Retrieve returns episode (list of frames)",
                        ep is not None and len(ep) > 0))

    # 5. Max capacity evicts old
    em3 = brain2.EpisodicMemory(n_dims=DIM, max_episodes=5, surprise_threshold=0.1)
    for i in range(10):
        for _ in range(2):
            em3.observe(rng.random(DIM).astype(np.float32))
        em3.commit(1.0)
    results.append(test(f"Max capacity eviction (count={em3.episode_count} <= 5)",
                        em3.episode_count <= 5))

    # 6. Consolidation
    em4 = brain2.EpisodicMemory(n_dims=DIM, max_episodes=200, surprise_threshold=0.1)
    base = rng.random(DIM).astype(np.float32)
    # Store 30 very similar episodes (should consolidate)
    for _ in range(30):
        v = base + rng.random(DIM).astype(np.float32) * 0.02
        for _ in range(2):
            em4.observe(v)
        em4.commit(1.0)
    count_before = em4.episode_count
    protos = em4.consolidate(similarity_threshold=0.90)
    count_after = em4.episode_count
    results.append(test(f"Consolidation reduces episodes ({count_before} → {count_after}, protos={em4.prototype_count})",
                        count_after < count_before and em4.prototype_count > 0))

    # 7. Save/load round-trip
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
        path = tf.name
    em2.save(path)
    em2_loaded = brain2.EpisodicMemory.load(path)
    os.unlink(path)
    results.append(test("Save/load round-trip (same episode count)",
                        em2.episode_count == em2_loaded.episode_count))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Component 2 (EpisodicMemory): READY\n")
        return True
    else:
        print("Component 2 (EpisodicMemory): NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
