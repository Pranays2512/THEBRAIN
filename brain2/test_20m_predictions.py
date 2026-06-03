import brain2
import sys

print("Loading massive 20M parameter Brain...")
b = brain2.Brain(som_rows=64, som_cols=64, n_dims=256)
ckpt_dir = "checkpoints/stage5_20m"
try:
    b.load_components(
        predictor_path=f"{ckpt_dir}/predictor.bin",
        language_path=f"{ckpt_dir}/language.bin",
        som_path=f"{ckpt_dir}/som.bin",
        episodic_path=f"{ckpt_dir}/episodic.bin",
        emotion_path=f"{ckpt_dir}/emotion.bin",
        self_path=f"{ckpt_dir}/self.bin",
        symbolic_path=f"{ckpt_dir}/symbolic.bin",
        binding_path="",
        bg_path="",
        procedures_path="",
        hpred_path=""
    )
except Exception as e:
    print("Error loading checkpoints:", e)
    sys.exit(1)

print("Brain successfully loaded from checkpoints/stage5_20m!")

# Test 1: Vocabulary
words_to_test = ["romeo", "juliet", "citizen", "king", "queen", "love"]
print("\n--- Vocabulary Knowledge ---")
for w in words_to_test:
    knows = b.symbolic_table.knows(w)
    print(f"Knows '{w}'? {'Yes' if knows else 'No'}")

# Test 2: Predictive Grammar (Unsupervised Learning Results)
print("\n--- Grammar / Prediction Test ---")
prompt_word = "first"
print(f"Feeding prompt word: '{prompt_word}'")
if not b.symbolic_table.knows(prompt_word):
    print("Warning: prompt word not in vocabulary.")
else:
    vec = b.language.encode(prompt_word)
    # Perceive it (updates SOM and steps predictor forward)
    b.perceive(vec)
    
    # Think 5 steps ahead using the Predictor
    print("Predicting next 5 concepts...")
    res = b.think(5)
    print("Words predicted:", res.words)
    print(f"Coherence score: {res.coherence:.2f}")

print("\n--- Let's try another prompt ---")
prompt_word2 = "good"
print(f"Feeding prompt word: '{prompt_word2}'")
if b.symbolic_table.knows(prompt_word2):
    vec2 = b.language.encode(prompt_word2)
    b.perceive(vec2)
    res2 = b.think(5)
    print("Words predicted:", res2.words)
    print(f"Coherence score: {res2.coherence:.2f}")

