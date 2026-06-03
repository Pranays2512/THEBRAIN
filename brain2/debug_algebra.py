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

# Test 10 - 4
s = b.symbolic_table.lookup("10")
o = b.symbolic_table.lookup("4")
res_sub = b.symbolic_table.apply("-", s, o)

# In OP_MATH_SUB, it does:
# s_sym = symbolic.nearest_symbol(subj)
# o_sym = symbolic.nearest_symbol(obj)
# res = stoi(s_sym) - stoi(o_sym)
# res_sym = to_string(res)
# pad.write("result", symbolic.lookup(res_sym))

s_sym = b.symbolic_table.nearest_symbol(s)
o_sym = b.symbolic_table.nearest_symbol(o)
print(f"s_sym: {s_sym}, o_sym: {o_sym}")

try:
    res = int(s_sym) - int(o_sym)
    res_sym = str(res)
    print(f"res_sym: {res_sym}")
    vec_res = b.symbolic_table.lookup(res_sym)
    print("Language best word for res:", b.language.best_word(vec_res))
except Exception as e:
    print("Error:", e)

