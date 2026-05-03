"""
test_language.py — Component 8: Language unit tests

Tests:
  1. Encode known word returns n_dims vector
  2. Encode unknown word returns zero vector
  3. Decode concept vec returns closest word
  4. hear() moves word embedding toward activation (Hebbian)
  5. Words heard in same context become closer in embedding space
  6. best_word returns single string
  7. familiarity increases with repeated hearing
  8. Save/load round-trip preserves embeddings
  9. Inner speech loop: encode → decode → re-encode produces coherent cycle
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
    print("\nLanguage — Component 8")
    print("=" * 50)
    results = []

    rng = np.random.default_rng(0)
    DIM = 32

    lang = brain2.Language(n_dims=DIM, lr=0.1)

    # 1. Encode known word
    v_fire = rng.random(DIM).astype(np.float32)
    lang.register_word("fire", v_fire)
    enc = np.array(lang.encode("fire"))
    results.append(test(f"Encode known word (len={len(enc)})", len(enc) == DIM))

    # 2. Encode unknown word returns zeros
    enc_unk = np.array(lang.encode("xyzzy_unknown"))
    results.append(test("Encode unknown = zero vector",
                        float(np.sum(np.abs(enc_unk))) < 1e-6))

    # 3. Decode concept vec returns closest word
    lang.register_word("water", rng.random(DIM).astype(np.float32))
    lang.register_word("food",  rng.random(DIM).astype(np.float32))
    top = lang.decode(v_fire, k=1)
    results.append(test(f"Decode returns fire for fire vec (got '{top[0][0]}')",
                        top[0][0] == "fire"))

    # 4. hear() moves embedding toward activation
    lang2 = brain2.Language(n_dims=DIM, lr=0.5)
    target = np.ones(DIM, dtype=np.float32)
    lang2.register_word("hot", rng.random(DIM).astype(np.float32))
    vec_before = np.array(lang2.encode("hot"))
    for _ in range(20):
        lang2.hear("hot", target)
    vec_after = np.array(lang2.encode("hot"))
    dist_before = float(np.linalg.norm(vec_before - target))
    dist_after  = float(np.linalg.norm(vec_after  - target))
    results.append(test(f"hear() moves embedding toward activation ({dist_before:.3f} → {dist_after:.3f})",
                        dist_after < dist_before))

    # 5. Words heard in same context become closer
    lang3 = brain2.Language(n_dims=DIM, lr=0.3)
    context = rng.random(DIM).astype(np.float32)
    lang3.register_word("warm")
    lang3.register_word("safe")
    lang3.register_word("cold")
    cold_ctx = rng.random(DIM).astype(np.float32)  # different context
    for _ in range(50):
        lang3.hear("warm", context + rng.random(DIM).astype(np.float32) * 0.05)
        lang3.hear("safe", context + rng.random(DIM).astype(np.float32) * 0.05)
        lang3.hear("cold", cold_ctx + rng.random(DIM).astype(np.float32) * 0.05)
    warm = np.array(lang3.encode("warm"))
    safe = np.array(lang3.encode("safe"))
    cold = np.array(lang3.encode("cold"))
    sim_ws = float(np.dot(warm, safe) / (np.linalg.norm(warm) * np.linalg.norm(safe) + 1e-8))
    sim_wc = float(np.dot(warm, cold) / (np.linalg.norm(warm) * np.linalg.norm(cold) + 1e-8))
    results.append(test(f"Same-context words closer (warm-safe={sim_ws:.3f} > warm-cold={sim_wc:.3f})",
                        sim_ws > sim_wc))

    # 6. best_word returns string
    bw = lang.best_word(v_fire)
    results.append(test(f"best_word returns string ('{bw}')", isinstance(bw, str) and len(bw) > 0))

    # 7. Familiarity increases with repeated hearing
    lang4 = brain2.Language(n_dims=DIM)
    lang4.register_word("test")
    ctx = rng.random(DIM).astype(np.float32)
    fam0 = lang4.familiarity("test")
    for _ in range(10): lang4.hear("test", ctx)
    fam10 = lang4.familiarity("test")
    results.append(test(f"Familiarity increases ({fam0:.3f} → {fam10:.3f})",
                        fam10 > fam0))

    # 8. Save/load round-trip
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
        path = tf.name
    lang3.save(path)
    lang3_loaded = brain2.Language.load(path)
    os.unlink(path)
    w1 = np.array(lang3.encode("warm"))
    w2 = np.array(lang3_loaded.encode("warm"))
    diff = float(np.max(np.abs(w1 - w2)))
    results.append(test(f"Save/load preserves embeddings (diff={diff:.6f})", diff < 1e-5))

    # 9. Inner speech coherence: encode → decode → re-encode stable
    lang5 = brain2.Language(n_dims=DIM, lr=0.1)
    concept = rng.random(DIM).astype(np.float32)
    for word in ["think", "feel", "know", "dream"]:
        # Place words in different regions
        w_vec = rng.random(DIM).astype(np.float32)
        lang5.register_word(word, w_vec)
    # Hear each word 20x in its region
    for word in ["think", "feel", "know", "dream"]:
        w_vec = np.array(lang5.encode(word))
        for _ in range(20):
            lang5.hear(word, w_vec + rng.random(DIM).astype(np.float32) * 0.05)
    # Inner speech loop: concept → best_word → encode → should be similar to concept
    w1_vec = np.array(lang5.encode("think"))
    bw = lang5.best_word(w1_vec)
    bw_vec = np.array(lang5.encode(bw))
    sim = float(np.dot(w1_vec, bw_vec) / (np.linalg.norm(w1_vec)*np.linalg.norm(bw_vec)+1e-8))
    results.append(test(f"Inner speech loop coherent: think→'{bw}'→sim={sim:.3f}",
                        bw == "think" and sim > 0.9))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Component 8 (Language): READY\n")
        return True
    else:
        print("Component 8 (Language): NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
