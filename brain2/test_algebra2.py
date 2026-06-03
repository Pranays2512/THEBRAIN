import brain2
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
b.scratchpad.clear()
b.scratchpad.write("subject", b.language.encode("10"), "context")
b.scratchpad.write("relation", b.language.encode("2"), "context")
b.scratchpad.write("object", b.language.encode("4"), "context")

b.force_reason_step(2, "solve") # MATH_SUB
sub_res = b.scratchpad.read("result")
print("After SUB:", b.language.best_word(sub_res))

b.force_reason_step(3, "solve") # MATH_DIV
div_res = b.scratchpad.read("result")
print("After DIV:", b.language.best_word(div_res))
