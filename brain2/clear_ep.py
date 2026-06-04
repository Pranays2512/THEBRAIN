import brain2
b = brain2.Brain(8, 8, 16)
ckpt = "checkpoints/stage5_math"
b.load_components(
    predictor_path=f"{ckpt}/predictor.bin",
    language_path=f"{ckpt}/language.bin",
    som_path=f"{ckpt}/som.bin",
    emotion_path=f"{ckpt}/emotion.bin",
    self_path=f"{ckpt}/self.bin",
    symbolic_path=f"{ckpt}/symbolic.bin",
    binding_path=f"{ckpt}/binding.bin",
    bg_path=f"{ckpt}/bg.bin",
    procedures_path=f"{ckpt}/procedures.bin",
    hpred_path=f"{ckpt}/hpred.bin"
)
# Save it back (episodic memory is fresh, 0 episodes)
b.save_components(ckpt)
print("Cleared episodic memory!")
