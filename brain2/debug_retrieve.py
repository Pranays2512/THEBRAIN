import brain2
import os
b = brain2.Brain(8, 8, 16)
ckpt = "checkpoints/stage5_math"
b.load_components(
    predictor_path=f"{ckpt}/predictor.bin",
    language_path=f"{ckpt}/language.bin",
    som_path=f"{ckpt}/som.bin",
    episodic_path=f"{ckpt}/episodic.bin",
    emotion_path=f"{ckpt}/emotion.bin",
    self_path=f"{ckpt}/self.bin",
    symbolic_path=f"{ckpt}/symbolic.bin",
    binding_path=f"{ckpt}/binding.bin",
    bg_path=f"{ckpt}/bg.bin",
    procedures_path=f"{ckpt}/procedures.bin",
    hpred_path=f"{ckpt}/hpred.bin"
)
v_pranay = b.language.encode("pranay")
print("Best word for pranay:", b.language.best_word(v_pranay))
print("Best word for 941:", b.language.best_word(b.language.encode("941")))
