import brain2
import time
import os

b = brain2.Brain(8, 8, 16)
checkpoint_dir = "checkpoints/stage5_math"
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

start = time.time()
for _ in range(100):
    b.reset_sequence()
    b.scratchpad.write("subject",  b.language.encode("97"), "context")
    b.scratchpad.write("relation", b.language.encode("9"), "context")
    b.scratchpad.write("object",   b.language.encode("24"), "context")
    b.force_reason_step(2,  "solve")
    b.force_reason_step(3,  "solve")
    b.force_reason_step(15, "solve")
end = time.time()
print(f"Math (with Memoization): {end-start:.4f}s")
