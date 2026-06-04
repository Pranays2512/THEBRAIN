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

subj, rel, obj = "pranay0", "is", "creator0"
for w in [subj, rel, obj]:
    b.language.register_word(w)
    b.symbolic_table.bind(w)
b.binding.bind(b.language.encode(subj), b.language.encode(rel), b.language.encode(obj))

ans_vec, conf = b.binding.query(b.language.encode(subj), b.language.encode(rel), True, 0.3, 4)
ans = b.language.best_word(ans_vec)
print(f"Query (pranay0, is) -> {ans} (conf: {conf})")
