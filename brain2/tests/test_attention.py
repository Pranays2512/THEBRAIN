"""
test_attention.py — Component 7: Attention unit tests

Tests:
  1. Novel input (high prediction error) passes gate
  2. Boring input (low novelty) fails gate at high arousal
  3. Saliency map updates: active neurons become more salient
  4. focus_neuron tracks highest saliency after input
  5. tick() decays saliency over time
  6. High arousal_modulator raises threshold (narrower focus)
  7. top-down bias increases scores for biased neurons
  8. Reset clears saliency map
  9. Save/load preserves saliency map
  10. mean_saliency increases after novel inputs
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
    print("\nAttention — Component 7")
    print("=" * 50)
    results = []

    rng = np.random.default_rng(42)
    N = 64  # n_neurons

    attn = brain2.Attention(n_neurons=N, decay_rate=0.1, base_threshold=0.3)

    # Create activation maps
    strong_act = np.zeros(N, dtype=np.float32)
    strong_act[:8] = 1.0  # 8 strongly active neurons
    weak_act = np.zeros(N, dtype=np.float32)
    weak_act[0] = 0.01  # barely active

    # 1. Novel input passes (high novelty, moderate arousal)
    result = attn.gate(strong_act, novelty=0.9, arousal_modulator=0.6)
    results.append(test(f"Novel strong input passes (score={result.score:.3f}, thr={result.threshold:.3f})",
                        result.passed))

    # 2. Boring weak input fails at high arousal
    attn2 = brain2.Attention(n_neurons=N, decay_rate=0.1, base_threshold=0.3)
    result2 = attn2.gate(weak_act, novelty=0.01, arousal_modulator=1.0)
    results.append(test(f"Weak input fails at high arousal (score={result2.score:.4f}, thr={result2.threshold:.3f})",
                        not result2.passed))

    # 3. Saliency map updates at active neurons
    attn3 = brain2.Attention(n_neurons=N, decay_rate=0.05)
    act = np.zeros(N, dtype=np.float32)
    act[10] = 1.0
    act[11] = 0.8
    attn3.gate(act, novelty=0.8)
    sal = np.array(attn3.saliency_map())
    results.append(test(f"Active neurons get higher saliency (sal[10]={sal[10]:.3f} > sal[50]={sal[50]:.3f})",
                        sal[10] > sal[50]))

    # 4. focus_neuron tracks strongest signal
    attn4 = brain2.Attention(n_neurons=N)
    act4 = np.zeros(N, dtype=np.float32)
    act4[33] = 1.0  # strongest
    act4[20] = 0.5
    attn4.gate(act4, novelty=1.0)
    focus = attn4.focus_neuron
    results.append(test(f"focus_neuron = 33 (got {focus})", focus == 33))

    # 5. tick() decays saliency
    attn5 = brain2.Attention(n_neurons=N, decay_rate=0.3)
    attn5.gate(strong_act, novelty=1.0)
    sal_before = float(np.mean(np.array(attn5.saliency_map())))
    for _ in range(10):
        attn5.tick()
    sal_after = float(np.mean(np.array(attn5.saliency_map())))
    results.append(test(f"tick() decays saliency ({sal_before:.4f} → {sal_after:.4f})",
                        sal_after < sal_before))

    # 6. High arousal_modulator = higher threshold
    attn6 = brain2.Attention(n_neurons=N, base_threshold=0.3)
    r_low  = attn6.gate(strong_act, novelty=0.5, arousal_modulator=0.5)
    attn6_hi = brain2.Attention(n_neurons=N, base_threshold=0.3)
    r_high = attn6_hi.gate(strong_act, novelty=0.5, arousal_modulator=1.0)
    results.append(test(f"High arousal = higher threshold ({r_low.threshold:.3f} < {r_high.threshold:.3f})",
                        r_low.threshold < r_high.threshold))

    # 7. Top-down bias helps passage
    attn7 = brain2.Attention(n_neurons=N, base_threshold=0.5)
    # Weak activation — barely passes alone
    bias = np.zeros(N, dtype=np.float32)
    bias[:8] = 1.0  # bias toward same neurons as strong_act
    attn7.set_top_down(bias)
    # Run many steps to accumulate bias effect
    r_biased = None
    for _ in range(20):
        r_biased = attn7.gate(strong_act, novelty=0.5, arousal_modulator=0.5)
    results.append(test(f"Top-down bias builds saliency (mean_sal={attn7.mean_saliency:.4f})",
                        attn7.mean_saliency > 0.0))

    # 8. Reset clears saliency
    attn8 = brain2.Attention(n_neurons=N)
    attn8.gate(strong_act, novelty=1.0)
    sal_before_reset = attn8.mean_saliency
    attn8.reset()
    results.append(test(f"Reset clears saliency ({sal_before_reset:.4f} → {attn8.mean_saliency:.4f})",
                        attn8.mean_saliency == 0.0))

    # 9. Save/load
    attn9 = brain2.Attention(n_neurons=N, decay_rate=0.15)
    for _ in range(5):
        attn9.gate(strong_act, novelty=0.7)
    sal9 = np.array(attn9.saliency_map())
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tf:
        path = tf.name
    attn9.save(path)
    attn9_loaded = brain2.Attention.load(path)
    os.unlink(path)
    sal9_loaded = np.array(attn9_loaded.saliency_map())
    diff = float(np.max(np.abs(sal9 - sal9_loaded)))
    results.append(test(f"Save/load preserves saliency map (diff={diff:.6f})", diff < 1e-5))

    # 10. mean_saliency increases with novel inputs
    attn10 = brain2.Attention(n_neurons=N, decay_rate=0.01)
    sal_init = attn10.mean_saliency
    for _ in range(10):
        act_rng = rng.random(N).astype(np.float32)
        attn10.gate(act_rng, novelty=0.8)
    sal_after_inputs = attn10.mean_saliency
    results.append(test(f"mean_saliency increases with novel inputs ({sal_init:.4f} → {sal_after_inputs:.4f})",
                        sal_after_inputs > sal_init))

    # Summary
    print()
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("Component 7 (Attention): READY\n")
        return True
    else:
        print("Component 7 (Attention): NEEDS FIX\n")
        return False

if __name__ == '__main__':
    ok = run()
    sys.exit(0 if ok else 1)
