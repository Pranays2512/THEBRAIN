import brain2
import os

b = brain2.Brain(som_rows=8, som_cols=8, n_dims=16)
checkpoint_dir = "checkpoints/stage4_parsing"
if os.path.exists(checkpoint_dir):
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
for i in range(1000):
    b.learn_word(str(i))
b.symbolic_table.seed_math_symbols()

for w in ["permute", "probability", "area", "power"]:
    if not b.symbolic_table.knows(w):
        b.learn_word(w)

print("Training Procedural Memory for Math Algorithms...")

# 1. Permutations: nPr = n! / (n-r)!
# Sequence:
# 2  (MATH_SUB): n - r -> result
# 25 (STORE_TMP): result -> relation
# 22 (MATH_FACT): n! -> result
# 23 (MATH_FACT_REL): (n-r)! -> relation
# 3  (MATH_DIV): result / relation -> result
# 15 (SPEAK): speaks result
# 8  (HALT)
seq_perm = [2, 25, 22, 23, 3, 15, 8]
b.reset_sequence()
bmu = b.som.activation_map(b.language.encode("permute"))
b.working_mem.gate(bmu * 10.0, 1.0)
b.working_mem.tick()
b.consolidate_procedure(seq_perm, "permute")

# 2. Probability: target / total
seq_prob = [3, 15, 8]
b.reset_sequence()
bmu = b.som.activation_map(b.language.encode("probability"))
b.working_mem.gate(bmu * 10.0, 1.0)
b.working_mem.tick()
b.consolidate_procedure(seq_prob, "probability")

# 3. Area of Rectangle: l * b
seq_area = [21, 15, 8]
b.reset_sequence()
bmu = b.som.activation_map(b.language.encode("area"))
b.working_mem.gate(bmu * 10.0, 1.0)
b.working_mem.tick()
b.consolidate_procedure(seq_area, "area")

# 4. Exponents: subject ^ object
seq_pow = [24, 15, 8]
b.reset_sequence()
bmu = b.som.activation_map(b.language.encode("power"))
b.working_mem.gate(bmu * 10.0, 1.0)
b.working_mem.tick()
b.consolidate_procedure(seq_pow, "power")


out_dir = "checkpoints/stage5_math"
os.makedirs(out_dir, exist_ok=True)
b.save_components(out_dir)
print(f"Saved Procedural Memory to {out_dir}!")
