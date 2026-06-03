import brain2
import os

print("Initializing Brain v3...")
b = brain2.Brain(som_rows=8, som_cols=8, n_dims=16)
ckpt_dir = "checkpoints/stage4_parsing"
b.load_components(
    predictor_path=f"{ckpt_dir}/predictor.bin",
    language_path=f"{ckpt_dir}/language.bin",
    som_path=f"{ckpt_dir}/som.bin",
    episodic_path=f"{ckpt_dir}/episodic.bin",
    emotion_path=f"{ckpt_dir}/emotion.bin",
    self_path=f"{ckpt_dir}/self.bin",
    symbolic_path=f"{ckpt_dir}/symbolic.bin",
    binding_path=f"{ckpt_dir}/binding.bin",
    bg_path=f"{ckpt_dir}/bg.bin",
    procedures_path=f"{ckpt_dir}/procedures.bin",
    hpred_path=f"{ckpt_dir}/hpred.bin"
)

# Teach it numbers 0 to 100
for i in range(101):
    word = str(i)
    if not b.symbolic_table.knows(word):
        b.learn_word(word)

b.save_components(ckpt_dir)
print("Brain has learned numbers 0 to 100!")
