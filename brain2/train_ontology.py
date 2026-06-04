import brain2
import os

print("Initializing Brain v3...")
b = brain2.Brain(som_rows=10, som_cols=10, n_dims=32)

ckpt_dir = "checkpoints/stage4_parsing"
try:
    # Load existing components so we don't overwrite everything
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

count = 0
with open("ontology_dataset.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            s, r, o = parts
            
            # 1. Teach the words to the Language and Symbolic Memory
            for w in [s, r, o]:
                if not b.symbolic_table.knows(w):
                    b.learn_word(w)
                    
            # 2. Encode to vectors
            s_vec = b.language.encode(s)
            r_vec = b.language.encode(r)
            o_vec = b.language.encode(o)
            
            # 3. Bind fact in memory
            b.binding.bind(s_vec, r_vec, o_vec)
            count += 1

print(f"Successfully injected {count} facts into Binding Memory!")
b.save_components(ckpt_dir)
print("Ontology saved to stage4_parsing.")
