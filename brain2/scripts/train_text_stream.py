"""
train_text_stream.py - Script to train Brain2 on sequential text data.
Demonstrates the capability of the Hierarchical Architecture (HSOM, Tiered WM, Tree-Episodic).
"""

import sys, os
import numpy as np

# Add the brain2 directory to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import brain2

def main():
    print("==================================================")
    print("       Brain2 Sequence Learning Environment       ")
    print("==================================================\n")

    # 1. Define the Structured Narrative Corpus
    corpus = [
        "alice went to the store",
        "alice bought an apple",
        "the apple was red",
        "bob went to the park",
        "bob saw a dog",
        "the dog was brown"
    ]

    print("Corpus to learn:")
    for c in corpus:
        print(f"  > {c}")
    print()

    # 2. Create pseudo-random "sensory" embeddings for each word
    vocab = set()
    for sentence in corpus:
        for word in sentence.split():
            vocab.add(word)

    rng = np.random.default_rng(42)
    n_dims = 16
    word_vecs = {word: rng.random(n_dims).astype(np.float32) for word in vocab}

    # 3. Initialize the Hierarchical Brain
    print("Initializing Brain...")
    # 20x20 SOM is the max allowed, we start with 4x4 and spawn up to 400 total nodes
    brain = brain2.Brain(som_rows=20, som_cols=20, n_dims=n_dims, wm_capacity=7)
    
    import random
    
    # 4. Training Loop (Online Sequence Learning)
    print("\nTraining Phase (50 epochs)...")
    for epoch in range(50):
        # Shuffle corpus to prevent recency bias (catastrophic overwriting) at the end of each epoch
        random.shuffle(corpus)
        for sentence in corpus:
            for word in sentence.split():
                vec = word_vecs[word]
                brain.perceive(vec)
                brain.hear(word)
            brain.reset_sequence()

    print("\nTraining Complete.")
    print("--------------------------------------------------")
    print(f"Words in Vocabulary: {brain.language.vocab_size}")
    print(f"Episodic Memories Stored (Tree Roots): {brain.episodic.episode_count}")
    
    # 5. Testing & Verification
    print("\n==================================================")
    print("              Testing & Verification              ")
    print("==================================================\n")

    def test_sequence(seq):
        brain.reset_sequence()
        print(f"Input Sequence: '{seq}'")
        for word in seq.split():
            brain.perceive(word_vecs[word])
            brain.hear(word)
            # Force the prompt into working memory
            act = brain.som.activation_map(word_vecs[word])
            brain.working_mem.gate(act * 1.5, 1.0)
            brain.working_mem.tick()
        
        ctx = brain.working_mem.context()
        
        think_res = brain.think(steps=2)
        predicted = think_res.words[-1] if len(think_res.words) > 1 else (think_res.words[0] if think_res.words else "<silence>")
        
        print(f"  -> Brain Predicts: '{predicted}'")
        print(f"  -> Coherence: {think_res.coherence:.3f}\n")

    # Test predictive recall
    test_sequence("alice went to the")
    test_sequence("bob went to the")
    
    # Test longer context recall
    test_sequence("alice bought an")
    test_sequence("bob saw a")

    # 6. Rest Phase (Dreaming & Consolidation)
    print("Initiating Rest Phase (Dreaming/Consolidation)...")
    brain.reset_sequence()
    frames = brain.dream(n_dreams=5, steps_per_dream=5)
    print(f"Extracted {len(frames)} highly-coherent dream frames.")
    
    # Check episodic consolidation
    protos = brain.episodic.prototype_count
    print(f"Episodic Prototypes Formed: {protos}")
    print("\nBrain2 successfully processed the sequence hierarchy!")

if __name__ == "__main__":
    main()
