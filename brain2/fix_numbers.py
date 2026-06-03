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

# First, let's fix any ZERO vectors in Language or Symbolic
for i in range(1001):
    word = str(i)
    # Force it to learn (if it already knows it but it's a zero vector, this won't help, but let's see)
    if not b.symbolic_table.knows(word):
        b.learn_word(word)
    else:
        # It knows it. But is the vector zero?
        vec = b.symbolic_table.lookup(word)
        if sum(abs(x) for x in vec) < 1e-5:
            # It's a zero vector! We need to overwrite it!
            # We can't overwrite in C++ easily without a function, but we can call b.learn_word... wait, learn_word skips if knows() is true.
            pass

    # Actually, we can just ensure that Language has the non-zero vector from Symbolic!
    # Wait, if Symbolic is zero, we are doomed. Let's just check if 3 has a zero vector.
    vec3 = b.symbolic_table.lookup("3")
    # print(f"Vec 3: {vec3}")

b.save_components(ckpt_dir)
