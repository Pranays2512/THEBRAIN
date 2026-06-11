import brain2
import json

print("Init Brain...")
b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
b.load_components(
        predictor_path="checkpoints/math_brain/predictor.bin",
        language_path="checkpoints/math_brain/language.bin",
        som_path="checkpoints/math_brain/som.bin",
        episodic_path="checkpoints/math_brain/episodic.bin",
        emotion_path="checkpoints/math_brain/emotion.bin",
        self_path="checkpoints/math_brain/self.bin",
        symbolic_path="checkpoints/math_brain/symbolic.bin",
        binding_path="checkpoints/math_brain/binding.bin",
        bg_path="checkpoints/math_brain/bg.bin",
        procedures_path="checkpoints/math_brain/procedures.bin",
        hpred_path="checkpoints/math_brain/hpred.bin"
    )

with open("data/math_corpus.json") as f:
    corpus = json.load(f)

print("Finding roots...")
roots_items = [p for p in corpus if p["input"].startswith("roots")]
pair = roots_items[0]
inp = pair["input"]
print("Input:", inp)
tokens = inp.split()
print("Tokens:", tokens, len(tokens))

b_val = tokens[5] + tokens[6]
c_val = tokens[8] + tokens[9]

print("Writing...")
b.scratchpad.write("object", b.language.encode(b_val), "math_arg")
b.scratchpad.write("a_operator", b.language.encode(c_val), "math_arg")

print("Forcing...")
b.force_reason_step(30, "3_and_1")
print("Done!")
