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
v6_lang = b.language.encode("6")
v6_sym = b.symbolic_table.lookup("6")
print("v6 lang == v6 sym?", (v6_lang == v6_sym).all())

v492_lang = b.language.encode("492")
print("v492 lang == v6 sym?", (v492_lang == v6_sym).all())

res = b.language.best_word(v6_sym)
print("Language best word for v6_sym:", res)

# Why? Let's check dot product manually!
def cos_sim(a, b):
    return sum(x*y for x,y in zip(a,b)) / (sum(x*x for x in a)**0.5 * sum(x*x for x in b)**0.5)

print("v6_sym vs v6_lang cos_sim:", cos_sim(v6_sym, v6_lang))
print("v6_sym vs v492_lang cos_sim:", cos_sim(v6_sym, v492_lang))
