import brain2
import os

print("Initializing Brain v3 for Self Ontology...")
b = brain2.Brain(som_rows=8, som_cols=8, n_dims=16)

ckpt_dir = "checkpoints/stage5_math"
try:
    b.load_components(
        predictor_path=f"{ckpt_dir}/predictor.bin",
        language_path=f"{ckpt_dir}/language.bin",
        som_path=f"{ckpt_dir}/som.bin",
        episodic_path=f"{ckpt_dir}/episodic.bin",
        emotion_path=f"{ckpt_dir}/emotion.bin",
        self_path=f"{ckpt_dir}/self.bin",
        symbolic_path=f"{ckpt_dir}/symbolic.bin",
        binding_path=f"{ckpt_dir}/binding.bin",
        bg_path=f"{ckpt_dir}/bg.bin" if os.path.exists(f"{ckpt_dir}/bg.bin") else "",
        procedures_path=f"{ckpt_dir}/procedures.bin",
        hpred_path=f"{ckpt_dir}/hpred.bin"
    )
except Exception as e:
    print("Error loading:", e)
    exit(1)

self_facts = [
    ("you", "are", "brain"),
    ("you", "are", "ai"),
    ("you", "can", "think"),
    ("you", "have", "memory"),
    ("pranay", "is", "creator")
]

count = 0
for s, r, o in self_facts:
    # 1. Teach words
    for w in [s, r, o]:
        if not b.symbolic_table.knows(w):
            b.learn_word(w)
            
    # 2. Encode and Bind
    s_vec = b.language.encode(s)
    r_vec = b.language.encode(r)
    o_vec = b.language.encode(o)
    b.binding.bind(s_vec, r_vec, o_vec)
    count += 1

print(f"Successfully injected {count} Self facts into Binding Memory!")
b.save_components(ckpt_dir)
print("Self-Ontology saved to stage4_parsing.")
