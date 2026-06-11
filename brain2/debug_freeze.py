import brain2
import json

rl_corpus = []
with open("data/math_corpus.json") as f:
    rl_corpus = json.load(f)

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

sample_corpus = rl_corpus[:100]
print("Running loop...")
for idx, pair in enumerate(sample_corpus):
    inp = pair["input"]
    tokens = inp.split()
    op_to_force = 20
    b.reset_sequence()
    if len(tokens) >= 3 and b.language.knows(tokens[0]) and b.language.knows(tokens[2]):
        b.scratchpad.write("subject", b.language.encode(tokens[0]), "math_arg")
        b.scratchpad.write("object", b.language.encode(tokens[2]), "math_arg")
    if len(tokens) >= 2 and b.language.knows(tokens[1]):
        b.scratchpad.write("a_operator", b.language.encode(tokens[1]), "math_arg")
    
    print(f"Step {idx} - start_reasoning")
    b.start_reasoning()
    print(f"Step {idx} - force_reason_step")
    b.force_reason_step(op_to_force, "reply")
    print(f"Step {idx} - reinforce_bg")
    b.reinforce_bg(1.0)
    print(f"Step {idx} - DONE")
