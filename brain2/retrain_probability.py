"""
retrain_probability.py — Retrains ONLY the probability procedure using
the new MATH_DIV_FLOAT op (index 26) instead of MATH_DIV (index 3).
Loads from and saves back to stage5_math checkpoint.
"""
import brain2
import os

b = brain2.Brain(8, 8, 16)
checkpoint_dir = "checkpoints/stage5_math"

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
if not b.symbolic_table.knows("probability"):
    b.learn_word("probability")

# New probability procedure: uses MATH_DIV_FLOAT (26) instead of MATH_DIV (3)
# subject=target, object=total → result = target / total (float)
# 26 = MATH_DIV_FLOAT: subject / object → result
# 15 = SPEAK: say result
# 8  = HALT
seq_prob_float = [26, 15, 8]

print("Consolidating probability procedure with MATH_DIV_FLOAT...")
b.reset_sequence()
bmu = b.som.activation_map(b.language.encode("probability"))
b.working_mem.gate(bmu * 10.0, 1.0)
b.working_mem.tick()
b.consolidate_procedure(seq_prob_float, "probability")

# Also re-consolidate all other procedures to ensure they stay intact
# Permutations: nPr = n!/  (n-r)!
seq_perm = [2, 25, 22, 23, 3, 15, 8]
b.reset_sequence()
bmu = b.som.activation_map(b.language.encode("permute"))
b.working_mem.gate(bmu * 10.0, 1.0)
b.working_mem.tick()
b.consolidate_procedure(seq_perm, "permute")

# Area: l * w
seq_area = [21, 15, 8]
b.reset_sequence()
bmu = b.som.activation_map(b.language.encode("area"))
b.working_mem.gate(bmu * 10.0, 1.0)
b.working_mem.tick()
b.consolidate_procedure(seq_area, "area")

# Power: base ^ exp
seq_pow = [24, 15, 8]
b.reset_sequence()
bmu = b.som.activation_map(b.language.encode("power"))
b.working_mem.gate(bmu * 10.0, 1.0)
b.working_mem.tick()
b.consolidate_procedure(seq_pow, "power")

print("Saving...")
b.save_components(checkpoint_dir)
print(f"Saved updated procedures to {checkpoint_dir}")

# Quick smoke test
print("\n--- Smoke test: probability of 1 in 4 ---")
b.reset_sequence()
b.scratchpad.write("subject", b.language.encode("1"), "context")
b.scratchpad.write("object",  b.language.encode("4"), "context")
b.force_reason_step(26, "probability")   # MATH_DIV_FLOAT
b.force_reason_step(15, "probability")   # SPEAK
b.force_reason_step(8,  "probability")   # HALT
spoken = b.get_spoken_words()
b.clear_spoken_words()
print(f"Result: {spoken}  (expected: ['0.25'])")
