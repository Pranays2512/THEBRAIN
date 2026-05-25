import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def word_generator(filepath, chunk_size=1024*1024):
    """Generator to read a massive file chunk-by-chunk and yield words."""
    with open(filepath, 'r', encoding='utf-8') as f:
        tail = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if tail:
                    yield tail
                break
            
            # Combine previous tail with new chunk
            text = tail + chunk
            words = text.split()
            
            # If the chunk didn't end with whitespace, the last word might be cut off
            if not chunk[-1].isspace():
                tail = words.pop() if words else ""
            else:
                tail = ""
                
            for w in words:
                yield w

def train_massive_corpus():
    # Massive Brain Initialization:
    # 32x32 SOM = 1024 distinct concept clusters for the vast vocabulary
    print("Initializing Brain V4 (Massive Configuration)...")
    b = brain2.Brain(som_rows=32, som_cols=32, n_dims=16, episodic_max=50000)
    
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "text8")
    if not os.path.exists(corpus_path):
        print(f"Corpus not found at {corpus_path}. Please download it first.")
        return
        
    print(f"Starting stream from {corpus_path}...")
    
    DREAM_INTERVAL = 10000  # Dream every 10k words
    PRINT_INTERVAL = 10000  # Print progress every 10k words
    
    start_time = time.time()
    total_error = 0.0
    interval_error = 0.0
    word_count = 0
    
    for word in word_generator(corpus_path):
        word_count += 1
        
        # We don't have punctuation in text8 (it's strictly lowercase letters and spaces),
        # so we just feed a continuous stream of text.
        
        if not b.language.knows(word):
            b.language.register_word(word)
            
        word_vec = b.language.encode(word)
        res = b.perceive(word_vec)
        b.hear(word)
        
        b.reinforce_bg(0.0)
        
        error = res.prediction_error
        total_error += error
        interval_error += error
        
        # Periodic Dreaming for episodic consolidation
        if word_count % DREAM_INTERVAL == 0:
            b.dream(n_dreams=5, steps_per_dream=10)
            
        # Periodic Progress Output
        if word_count % PRINT_INTERVAL == 0:
            avg_interval_error = interval_error / PRINT_INTERVAL
            elapsed = time.time() - start_time
            speed = word_count / elapsed
            vocab_size = b.language.vocab_size
            
            print(f"Words: {word_count:,} | "
                  f"Speed: {speed:,.1f} w/s | "
                  f"Avg Err: {avg_interval_error:.4f} | "
                  f"Vocab: {vocab_size:,} | "
                  f"Eps: {b.episodic.episode_count:,} / Protos: {b.episodic.prototype_count:,}", flush=True)
                  
            interval_error = 0.0
            
            # Since this takes hours, we can gracefully exit or just let it run.
            # We'll just let it run. User can kill it whenever.

    print("\nTraining Complete!")
    total_time = time.time() - start_time
    print(f"Processed {word_count:,} words in {total_time:,.2f} seconds ({word_count/total_time:,.1f} words/sec).")
    print(f"Final Global Avg Error: {total_error / max(1, word_count):.4f}")

if __name__ == '__main__':
    train_massive_corpus()
