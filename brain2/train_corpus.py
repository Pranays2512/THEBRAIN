import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def train_corpus():
    b = brain2.Brain(som_rows=4, som_cols=4, n_dims=16, episodic_max=2000)
    
    # Load corpus
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "simple_stories.txt")
    if not os.path.exists(corpus_path):
        print(f"Corpus not found at {corpus_path}")
        return
        
    with open(corpus_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    epochs = 20
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1} ---")
        total_error = 0.0
        word_count = 0
        
        # Read story
        for line in lines:
            # We treat punctuation simply by adding space around it
            sentence = line.replace(".", " .").lower()
            words = sentence.split()
            
            b.reset_sequence()
            for w in words:
                word_vec = b.language.encode(w)
                res = b.perceive(word_vec)
                
                total_error += res.prediction_error
                word_count += 1
                
                # We could reinforce BG here based on valence/arousal, but we just want to observe prediction
                b.reinforce_bg(0.0)
                
        # End of day "Dream" phase
        print("Dreaming and consolidating...")
        b.dream(n_dreams=5, steps_per_dream=10)
        
        avg_error = total_error / max(1, word_count)
        episodes = b.episodic.episode_count
        prototypes = b.episodic.prototype_count
        
        print(f"Avg Prediction Error: {avg_error:.4f}")
        print(f"Episodic Memories: {episodes} | Prototypes: {prototypes}")
        if episodes > 0:
            print(f"Compression Ratio: {(prototypes / episodes):.2%}")
            
    print("\nTraining Complete.")
    print("Testing a familiar sentence:")
    b.reset_sequence()
    for w in "alice went to the store .".split():
        res = b.perceive(b.language.encode(w))
        print(f"'{w}': error={res.prediction_error:.4f}, attention_passed={res.attention_passed}")

if __name__ == '__main__':
    train_corpus()
