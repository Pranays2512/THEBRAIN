import brain2
import json
import time

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

t0 = time.time()
b.reset_sequence()
t1 = time.time()
print(f"reset_sequence: {t1-t0:.6f}")

v = b.language.encode("20")
t2 = time.time()
print(f"encode 1: {t2-t1:.6f}")

b.scratchpad.write("subject", v, "math_arg")
t3 = time.time()
print(f"write 1: {t3-t2:.6f}")

b.start_reasoning()
t4 = time.time()
print(f"start_reasoning: {t4-t3:.6f}")

b.force_reason_step(20, "reply")
t5 = time.time()
print(f"force_reason_step: {t5-t4:.6f}")

b.reinforce_bg(1.0)
t6 = time.time()
print(f"reinforce_bg: {t6-t5:.6f}")
