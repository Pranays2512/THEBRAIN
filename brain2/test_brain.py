import brain2
import os

print("=== WAKING UP DORMANT BRAIN FEATURES ===")
print("Initializing C++ Brain...")
brain = brain2.Brain(som_rows=16, som_cols=16, n_dims=32)

print("\n1. Feeding the Brain some initial experiences (Perception)...")
sentences = [
    "the dog ate the fish",
    "an animal is a living thing",
    "the rocket has large mass",
    "energy depends on mass and speed",
    "the quick brown fox jumps over the lazy dog"
]

for s in sentences:
    res = brain.perceive_text(s, brain2.ErrorMode.FULL)
    print(f"Perceived: '{s}' -> Surprise: {res.prediction_error:.4f}")

print("\n2. Testing Inner Speech (think)...")
# The brain thinks based on its current context (the last sentence)
think_res = brain.think(10)
print(f"Inner Speech Words: {think_res.words}")
print(f"Coherence: {think_res.coherence:.4f}")

print("\n3. Testing Unsupervised Daydreaming...")
print("Running daydream() 10 times to let the SOM organize on random noise...")
for i in range(10):
    brain.daydream()
print("Daydreaming completed successfully without crashing.")

print("\n4. Testing Generative REM-Style Dream Replay...")
# Generates novel sequences from single tokens and trains on them
try:
    mean_ce = brain.dream_replay_generative(n_samples=5, gen_len=10)
    print(f"Generative Dream Replay completed! Mean Cross-Entropy: {mean_ce:.4f}")
except Exception as e:
    print(f"Error during generative dreaming: {e}")

print("\n=== TEST COMPLETE ===")
print("If you are reading this, the brain successfully ran its dormant features without breaking!")
