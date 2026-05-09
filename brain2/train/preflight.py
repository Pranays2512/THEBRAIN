"""
preflight.py — cheap checks to run before an expensive training launch.
All checks must pass before committing to a long run.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import brain2
from concept_encoder import ConceptEncoder
from conceptnet_loader import ConceptNetLoader
from train import HealthError, checkpoint_health

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []


def check(name, cond, detail=""):
    if cond:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name} <- {detail}")
        errors.append(name)


print("=" * 60)
print("PRE-FLIGHT CHECKS")
print("=" * 60)

SOM_SIZE = 64
N_DIMS = 64
VOCAB_CAP = 5000
enc = ConceptEncoder(N_DIMS)

print("\n[1] ConceptEncoder")
for poison in ["nan", "NaN", "inf", "infinity", "-inf"]:
    v = enc.encode(poison)
    check(f'encode("{poison}") is clean',
          not np.isnan(v).any() and not np.isinf(v).any(),
          "contains NaN/Inf")

s_4 = enc.similarity("5", "4")
s_10 = enc.similarity("5", "10")
check("Number ordering: sim(5,4) > sim(5,10)", s_4 > s_10, f"{s_4:.3f} vs {s_10:.3f}")
check("sim(5,5) == 1.0", abs(enc.similarity("5", "5") - 1.0) < 1e-5)

for concept in ["3", "fire", "dog", "true", "false", "plus", "equals"]:
    v = enc.encode(concept)
    check(f'encode("{concept}") no NaN/Inf', not np.isnan(v).any() and not np.isinf(v).any())

print("\n[2] SOM / Vocabulary capacity")
neurons = SOM_SIZE * SOM_SIZE
ratio = VOCAB_CAP / neurons
check(f"SOM {SOM_SIZE}x{SOM_SIZE}={neurons} neurons for {VOCAB_CAP} words",
      ratio <= 2.0, f"{ratio:.1f} words/neuron")
print(f"    -> {ratio:.2f} words/neuron")

print("\n[3] Train -> Save -> Full Resume")
tmpdir = tempfile.mkdtemp()
try:
    cfg = dict(som_rows=SOM_SIZE, som_cols=SOM_SIZE, n_dims=N_DIMS,
               hidden_dim=128, wm_capacity=7, episodic_max=500,
               self_neurons=16, seed=42)
    b = brain2.Brain(**cfg)

    for i in range(50):
        b.perceive(enc.encode(str(i % 20)))
        b.hear(str(i % 20))

    ckpt = os.path.join(tmpdir, "smoke")
    b.predictor.save(f"{ckpt}_predictor.bin")
    b.language.save(f"{ckpt}_language.bin")
    b.som.save(f"{ckpt}_som.bin")
    b.episodic.save(f"{ckpt}_episodic.bin")
    b.emotion.save(f"{ckpt}_emotion.bin")
    b.self_model.save(f"{ckpt}_self.bin")
    b.symbolic_table.save(f"{ckpt}_symbolic.bin")

    som2 = brain2.SOM.load(f"{ckpt}_som.bin")
    pred2 = brain2.Predictor.load(f"{ckpt}_predictor.bin")
    lang2 = brain2.Language.load(f"{ckpt}_language.bin")

    check("SOM step count preserved after save/load", som2.step == b.som.step,
          f"got {som2.step}, want {b.som.step}")
    check("Predictor input_dim preserved", pred2.input_dim == neurons, f"got {pred2.input_dim}")
    check("Language vocab preserved", lang2.vocab_size == b.language.vocab_size,
          f"got {lang2.vocab_size}, want {b.language.vocab_size}")

    diff = float(np.max(np.abs(np.array(b.som.neuron_weights(0)) - np.array(som2.neuron_weights(0)))))
    check("SOM neuron[0] weights byte-identical after load", diff < 1e-6, f"max diff={diff:.2e}")

    b3 = brain2.Brain(**cfg)
    b3.load_components(f"{ckpt}_predictor.bin", f"{ckpt}_language.bin", f"{ckpt}_som.bin",
                       f"{ckpt}_episodic.bin", f"{ckpt}_emotion.bin",
                       f"{ckpt}_self.bin", f"{ckpt}_symbolic.bin")
    check("Brain resume restores trained SOM steps", b3.som.step == b.som.step,
          f"got {b3.som.step}, want {b.som.step}")
    check("Brain resume restores predictor dimensions", b3.predictor.input_dim == b.som.n_neurons,
          f"got {b3.predictor.input_dim}, want {b.som.n_neurons}")
finally:
    shutil.rmtree(tmpdir)

print("\n[4] Predictor output")
b2 = brain2.Brain(SOM_SIZE, SOM_SIZE, N_DIMS, 128, 7, 500, 16, 42)
for i in range(10):
    b2.perceive(enc.encode(str(i)))

b2.predictor.set_offline(True)
act = b2.som.activation_map(enc.encode("5"))
pred_out = b2.predictor.step(act)
b2.predictor.set_offline(False)

check("Predictor output has no NaN", not np.isnan(pred_out).any())
check("Predictor output has no Inf", not np.isinf(pred_out).any())
check("Predictor output in [0,1]", float(np.min(pred_out)) >= 0.0 and float(np.max(pred_out)) <= 1.0,
      f"min={np.min(pred_out):.3f}, max={np.max(pred_out):.3f}")
check("Predictor output dim == SOM neurons", len(pred_out) == neurons, f"got {len(pred_out)}")

print("\n[5] Language decode alignment")
test_word = "testword123"
act_5 = b2.som.activation_map(enc.encode("5"))
b2.language.register_word(test_word, act_5)
decoded = b2.language.decode(act_5, 1)
check("Known word decodes from its own SOM activation",
      len(decoded) > 0 and decoded[0][0] == test_word,
      f"decoded={decoded[:1]}")

print("\n[6] SOM sharpness + trainer health")
act_test = np.asarray(b2.som.activation_map(enc.encode("test_sharpness")), dtype=np.float32)
check("SOM map is sharp enough (no fog)", float(np.mean(act_test)) < 0.2,
      f"mean activation is {np.mean(act_test):.3f}")
check("SOM map has strong peak", float(np.max(act_test)) > 0.8,
      f"max activation is {np.max(act_test):.3f}")
try:
    checkpoint_health(b2, enc, "preflight", 0, fatal=True)
    check("Trainer health tripwire passes on clean brain", True)
except HealthError as exc:
    check("Trainer health tripwire passes on clean brain", False, str(exc))

print("\n[7] Vocab cap enforcement")
tiny_path = tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv")
try:
    for i in range(20):
        tiny_path.write(
            f"/a/[/r/IsA/,/c/en/word{i}/,/c/en/thing{i}/]\t"
            f"/r/IsA\t/c/en/word{i}\t/c/en/thing{i}\t{{\"weight\": 2.0}}\n"
        )
    tiny_path.close()

    tiny_loader = ConceptNetLoader(n_dims=N_DIMS, vocab_cap=10)
    seqs = list(tiny_loader.sequences(tiny_path.name, max_seqs=1000))
    check("Vocab cap built successfully", tiny_loader._allowed_words is not None,
          "allowed words is None")
    if tiny_loader._allowed_words is not None:
        check("Vocab cap strictly enforced", len(tiny_loader._allowed_words) == 10,
              f"got {len(tiny_loader._allowed_words)}")
    check("Vocab cap filters emitted sequences", len(seqs) <= 5,
          f"got {len(seqs)} sequences from cap=10")
finally:
    try:
        os.unlink(tiny_path.name)
    except OSError:
        pass

print("\n" + "=" * 60)
if errors:
    print(f"\033[91m{len(errors)} check(s) FAILED - DO NOT launch training:\033[0m")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)

print("\033[92mAll checks passed - safe to launch training.\033[0m")
print(f"\n  Config for next run:")
print(f"    SOM: {SOM_SIZE}x{SOM_SIZE} ({neurons} neurons)")
print(f"    Vocab cap: {VOCAB_CAP} words")
print(f"    Words/neuron: {ratio:.2f}")
print(f"    n_dims: {N_DIMS}, hidden: 512")
print("=" * 60)