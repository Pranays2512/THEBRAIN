import brain2
import json

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

b.reset_sequence()
b.scratchpad.write("subject", b.language.encode("29"), "math_arg")
b.scratchpad.write("object", b.language.encode("20"), "math_arg")
b.scratchpad.write("a_operator", b.language.encode("+"), "math_arg")
b.start_reasoning()

sol = b.direct_reason_step("reply")
print(f"Chosen Op: {sol}")
