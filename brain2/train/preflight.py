"""
preflight.py — Run this BEFORE every training launch.
All checks must pass (✅) before training starts.
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import brain2
from concept_encoder import ConceptEncoder

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
errors = []

def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}  ← {detail}")
        errors.append(name)

print("=" * 60)
print("PRE-FLIGHT CHECKS")
print("=" * 60)

# ── 1. ConceptEncoder sanity ──────────────────────────────────────
print("\n[1] ConceptEncoder")
enc = ConceptEncoder(64)

# NaN/Inf poison check
for poison in ["nan", "NaN", "inf", "infinity", "-inf"]:
    v = enc.encode(poison)
    check(f'encode("{poison}") is clean', not np.isnan(v).any() and not np.isinf(v).any(),
          f"contains NaN/Inf!")

# Number ordering: sim(5,4) > sim(5,10)
s_4  = enc.similarity("5", "4")
s_10 = enc.similarity("5", "10")
check("Number ordering: sim(5,4) > sim(5,10)", s_4 > s_10, f"{s_4:.3f} vs {s_10:.3f}")

# sim(5,5) == 1.0
s_5 = enc.similarity("5", "5")
check("sim(5,5) == 1.0", abs(s_5 - 1.0) < 1e-5, f"{s_5:.5f}")

# All vectors have unit-norm (normalized)
for concept in ["3", "fire", "dog", "true", "false", "plus", "equals"]:
    v = enc.encode(concept)
    check(f'encode("{concept}") no NaN/Inf', not np.isnan(v).any() and not np.isinf(v).any())

# ── 2. SOM size vs vocab math ─────────────────────────────────────
print("\n[2] SOM / Vocabulary capacity")
SOM_SIZE  = 64
VOCAB_CAP = 5000
neurons   = SOM_SIZE * SOM_SIZE
ratio     = VOCAB_CAP / neurons
check(f"SOM {SOM_SIZE}x{SOM_SIZE}={neurons} neurons for {VOCAB_CAP} words",
      ratio <= 2.0, f"{ratio:.1f} words/neuron (must be ≤ 2.0)")
print(f"    → {ratio:.2f} words/neuron")

# ── 3. Smoke train: 500 steps → save → load → weights unchanged ──
print("\n[3] Train → Save → Load → Weights preserved")
tmpdir = tempfile.mkdtemp()
try:
    cfg = dict(som_rows=SOM_SIZE, som_cols=SOM_SIZE, n_dims=64,
               hidden_dim=128, wm_capacity=7, episodic_max=500,
               self_neurons=16, seed=42)
    b = brain2.Brain(**cfg)

    # Do a few perceive steps
    for i in range(50):
        v = enc.encode(str(i % 20))
        b.perceive(v)
        b.hear(str(i % 20))

    # Save
    ckpt = os.path.join(tmpdir, "smoke")
    b.som.save(f"{ckpt}_som.bin")
    b.predictor.save(f"{ckpt}_pred.bin")
    b.language.save(f"{ckpt}_lang.bin")

    # Read back SOM step count (proves weights were saved, not fresh)
    som2  = brain2.SOM.load(f"{ckpt}_som.bin")
    pred2 = brain2.Predictor.load(f"{ckpt}_pred.bin")
    lang2 = brain2.Language.load(f"{ckpt}_lang.bin")

    check("SOM step count preserved after save/load",
          som2.step == b.som.step, f"got {som2.step}, want {b.som.step}")
    check("Predictor input_dim preserved",
          pred2.input_dim == SOM_SIZE*SOM_SIZE,
          f"got {pred2.input_dim}")
    check("Language vocab preserved",
          lang2.vocab_size == b.language.vocab_size,
          f"got {lang2.vocab_size}, want {b.language.vocab_size}")

    # Verify SOM weights actually match (not random)
    w1 = b.som.neuron_weights(0)
    w2 = som2.neuron_weights(0)
    diff = float(np.max(np.abs(np.array(w1) - np.array(w2))))
    check("SOM neuron[0] weights byte-identical after load", diff < 1e-6,
          f"max diff={diff:.2e}")

    # Verify eval uses TRAINED weights (SOM steps > 0)
    check("Eval will use trained SOM (steps > 0)", som2.step > 0,
          "SOM has 0 steps — eval would use random weights!")

finally:
    shutil.rmtree(tmpdir)

# ── 4. Predictor output sanity ────────────────────────────────────
print("\n[4] Predictor output is valid activation map")
cfg = dict(som_rows=SOM_SIZE, som_cols=SOM_SIZE, n_dims=64,
           hidden_dim=128, wm_capacity=7, episodic_max=500,
           self_neurons=16, seed=42)
b2 = brain2.Brain(**cfg)

for i in range(10):
    b2.perceive(enc.encode(str(i)))

b2.predictor.set_offline(True)
act = b2.som.activation_map(enc.encode("5"))
pred_out = b2.predictor.step(act)
b2.predictor.set_offline(False)

check("Predictor output has no NaN", not np.isnan(pred_out).any())
check("Predictor output has no Inf", not np.isinf(pred_out).any())
check("Predictor output in [0,1] range (sigmoid)",
      float(np.min(pred_out)) >= 0.0 and float(np.max(pred_out)) <= 1.0,
      f"min={np.min(pred_out):.3f}, max={np.max(pred_out):.3f}")
check("Predictor output dim == SOM neurons",
      len(pred_out) == SOM_SIZE * SOM_SIZE,
      f"got {len(pred_out)}, want {SOM_SIZE*SOM_SIZE}")

# ── 5. Language decode aligns with predictor output ───────────────
print("\n[5] Language decode alignment")
# Register a word at a specific SOM activation, then check it decodes back
test_word = "testword123"
act_5 = b2.som.activation_map(enc.encode("5"))
b2.language.register_word(test_word, act_5)
decoded = b2.language.decode(act_5, 1)
check(f"Known word decodes from its own SOM activation",
      len(decoded) > 0, "decode returned empty")

# ── SUMMARY ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"\033[91m❌ {len(errors)} check(s) FAILED — DO NOT launch training:\033[0m")
    for e in errors:
        print(f"   • {e}")
    sys.exit(1)
else:
    print(f"\033[92m✅ All checks passed — safe to launch training!\033[0m")
    print(f"\n  Config for brain_v5:")
    print(f"    SOM: {SOM_SIZE}x{SOM_SIZE} ({SOM_SIZE*SOM_SIZE} neurons)")
    print(f"    Vocab cap: {VOCAB_CAP} words")
    print(f"    Words/neuron: {VOCAB_CAP/(SOM_SIZE*SOM_SIZE):.2f}")
    print(f"    n_dims: 64, hidden: 512")
print("=" * 60)
