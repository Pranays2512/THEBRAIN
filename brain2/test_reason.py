import os, sys, brain2
import numpy as np

b = brain2.Brain(8, 8, 16)
checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "stage3_conversation")
b.load_components(
    predictor_path=os.path.join(checkpoint_dir, "predictor.bin"),
    language_path=os.path.join(checkpoint_dir, "language.bin"),
    som_path=os.path.join(checkpoint_dir, "som.bin"),
    episodic_path=os.path.join(checkpoint_dir, "episodic.bin"),
    emotion_path=os.path.join(checkpoint_dir, "emotion.bin"),
    self_path=os.path.join(checkpoint_dir, "self.bin"),
    symbolic_path=os.path.join(checkpoint_dir, "symbolic.bin"),
    binding_path=os.path.join(checkpoint_dir, "binding.bin"),
    bg_path=os.path.join(checkpoint_dir, "bg.bin"),
    procedures_path=os.path.join(checkpoint_dir, "procedures.bin"),
    hpred_path=os.path.join(checkpoint_dir, "hpred.bin")
)

subj_vec = np.array(b.language.encode("dog"), dtype=np.float32)
rel_vec = np.array(b.language.encode("isa"), dtype=np.float32)
b.scratchpad.clear()
b.scratchpad.write("subject", subj_vec, "context")
b.scratchpad.write("relation", rel_vec, "context")
b.scratchpad.write("comparison", np.zeros(16, dtype=np.float32), "eval")
goal_vec = np.array(b.language.encode("reply"), dtype=np.float32)
b.scratchpad.write("goal", goal_vec, "goal")

print("Starting reason_step()...")
b.start_reasoning()
op1 = b.reason_step("reply", 0.0)
print(f"Reasoning ops: [{op1}]")
res_vec = b.scratchpad.read("result")
print(f"Decoded: {b.language.best_word(res_vec) if len(res_vec) > 0 else '...'}")

b.scratchpad.clear()
b.scratchpad.write("subject", subj_vec, "context")
b.scratchpad.write("relation", rel_vec, "context")
b.scratchpad.write("comparison", np.zeros(16, dtype=np.float32), "eval")
b.scratchpad.write("goal", goal_vec, "goal")

print("Starting reason()...")
ops = b.reason("reply", 5, 0.0)
print(f"Reasoning ops: {ops}")
res_vec = b.scratchpad.read("result")
print(f"Decoded: {b.language.best_word(res_vec) if len(res_vec) > 0 else '...'}")
