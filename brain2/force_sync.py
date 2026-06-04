import brain2
b = brain2.Brain(som_rows=10, som_cols=10, n_dims=32)
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
for i in range(1001):
    word = str(i)
    vec = b.symbolic_table.lookup(word)
    # Force language to use the EXACT SAME vector as symbolic
    b.language.register_word(word, vec)

b.save_components(ckpt_dir)
print("Forced synced Language and Symbolic for numbers 0-1000!")
