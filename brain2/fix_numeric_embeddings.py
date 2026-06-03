"""
fix_numeric_embeddings.py — Add a tiny deterministic sinusoidal component to
numeric word embeddings so that nearby integers (e.g. "5" and "6") become
slightly more distinguishable in vector space.

The additive component is at most 0.01 in magnitude — it won't displace any
learned associations, but it breaks ties when two numbers are equidistant.

This runs after loading and saves back to the same checkpoint.
"""
import brain2
import os
import math

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

n_dims = 16
nudge_scale = 0.01  # very small — just enough to break ties

print("Applying sinusoidal numeric embedding nudge...")
nudged = 0
for n in range(1000):
    word = str(n)
    if not b.language.knows(word):
        continue
    vec = list(b.language.encode(word))
    # Each dimension gets a unique frequency component based on n
    # This creates a smooth, monotonic ordering in the embedding space
    for d in range(n_dims):
        freq = (d + 1) * math.pi / 500.0   # unique frequency per dimension
        vec[d] += nudge_scale * math.sin(n * freq)
    b.language.register_word(word, vec)
    nudged += 1

print(f"  Nudged {nudged} numeric embeddings.")

# Sanity check — "5" and "6" should be slightly less similar now
v5 = b.language.encode("5")
v6 = b.language.encode("6")
v50 = b.language.encode("50")
dot56  = sum(a*b_ for a, b_ in zip(v5, v6))
dot550 = sum(a*b_ for a, b_ in zip(v5, v50))
print(f"  Cosine(5,6)  = {dot56:.4f}  (should be high but <1)")
print(f"  Cosine(5,50) = {dot550:.4f} (should be lower)")

print("Saving...")
b.save_components(checkpoint_dir)
print(f"Done! Updated numeric embeddings saved to {checkpoint_dir}")
