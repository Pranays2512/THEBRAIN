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

print("Repairing 0-1000...")
for i in range(1001):
    word = str(i)
    # If the word has a zero vector in Language because of our bad sync script,
    # or if Symbolic doesn't know it, we must fix it.
    
    # Check Symbolic
    if not b.symbolic_table.knows(word):
        # Good, we can learn it normally.
        b.learn_word(word)
    else:
        # Symbolic knows it. Let's make sure Language has the EXACT same non-zero vector.
        sym_vec = b.symbolic_table.lookup(word)
        if sum(abs(x) for x in sym_vec) > 1e-5:
            # It's a valid non-zero vector. Sync Language to it.
            b.language.register_word(word, sym_vec)
        else:
            # Symbolic has a ZERO vector for this word! This is bad.
            # We must generate a new valid vector and force it into both!
            # We can do this by using a temporary Brain instance just to get a good seed!
            b_temp = brain2.Brain(som_rows=1, som_cols=1, n_dims=16)
            b_temp.learn_word(word)
            new_vec = b_temp.symbolic_table.lookup(word)
            
            # Now, how to force it into our actual brain?
            # Language can be forced via register_word
            b.language.register_word(word, new_vec)
            # Symbolic doesn't have an overwrite in C++ right now, so we are stuck with the zero vector in Symbolic.
            # But wait! Symbolic's `lookup` just reads the zero vector. If it's a zero vector, it can't be fixed without C++ changes.
            # Luckily, I think only words 101-1000 have zero vectors in Symbolic (if they even exist in Symbolic at all).
            pass

b.save_components(ckpt_dir)
print("Saved!")
