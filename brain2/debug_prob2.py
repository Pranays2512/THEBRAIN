import brain2
b = brain2.Brain(8, 8, 16)
checkpoint_dir = "checkpoints/stage5_math"
import os
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

# Test procedure retrieval
for cat in ["probability", "permute", "area", "power"]:
    b.language.register_word(cat)
    seq = b.procedures.retrieve(b.language.encode(cat))
    print(f"Procedure '{cat}': {seq}")
