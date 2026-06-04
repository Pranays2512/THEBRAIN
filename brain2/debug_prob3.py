import brain2
import os, re
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
for i in range(1000):
    b.symbolic_table.bind(str(i))

cat = "probability"
q = "probability of 19 out of 83"
exp = "0.23"

for w in q.split(): b.language.register_word(w)
prob_match = re.match(r"probability\s*of\s*(\d+)\s*(?:in|out of)\s*(\d+)", q)
subj_val, obj_val = prob_match.groups()
print(f"subj_val={subj_val}, obj_val={obj_val}")

b.scratchpad.write("subject", b.language.encode(subj_val), "context")
b.scratchpad.write("object",  b.language.encode(obj_val),  "context")
b.scratchpad.write("goal", b.language.encode(cat), "goal")

# Use saved sequence
seq = b.procedures.retrieve(b.language.encode(cat))
print(f"seq: {seq}")
for op in seq:
    b.force_reason_step(op, "reply")
    spoken = b.get_spoken_words()
    if spoken:
        print(f"Op {op} -> spoken: {spoken}")

spoken = b.get_spoken_words()
print("Final spoken:", spoken)
b.clear_spoken_words()
ans = spoken[-1] if spoken else ""
print(f"ans={ans}, expected={exp}")
