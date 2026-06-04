# Check: 17/40 = 0.425. Python round to 0.42 (banker's rounding) but f"{17/40:.2f}" = "0.42"
# But maybe the engine computes 0.43?
print(f"{17/40:.2f}")  # expected
import brain2
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
b.symbolic_table.seed_math_symbols()
for i in range(1000): b.symbolic_table.bind(str(i))

cat = "probability"
q = "probability of 17 out of 40"
for w in q.split(): b.language.register_word(w)

b.scratchpad.write("subject", b.language.encode("17"), "context")
b.scratchpad.write("object",  b.language.encode("40"),  "context")
b.scratchpad.write("goal", b.language.encode(cat), "goal")
seq = b.procedures.retrieve(b.language.encode(cat))
print("seq:", seq)
for op in seq: b.force_reason_step(op, "reply")
spoken = b.get_spoken_words()
print("spoken:", spoken)
