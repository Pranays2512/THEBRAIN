import os
import sys
import re
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def clean_text(text):
    # Remove Gutenberg header and footer
    start_match = re.search(r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*', text)
    end_match = re.search(r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*', text)
    
    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]
        
    # Lowercase
    text = text.lower()
    
    # Pad punctuation with spaces so they become separate tokens
    text = re.sub(r'([.,!?\'"()\[\]{};:])', r' \1 ', text)
    
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def train_large_corpus():
    # Initialize a slightly larger brain for the large vocabulary
    # 8x8 SOM gives 64 clusters, which is fine for abstract grouping
    b = brain2.Brain(som_rows=10, som_cols=10, n_dims=32, episodic_max=10000)
    
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "alice.txt")
    if not os.path.exists(corpus_path):
        print(f"Corpus not found at {corpus_path}. Please download it first.")
        return
        
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    print("Cleaning and tokenizing corpus...")
    clean_corpus = clean_text(raw_text)
    words = clean_corpus.split()
    total_words = len(words)
    print(f"Total words to process: {total_words}")
    
    DREAM_INTERVAL = 500
    PRINT_INTERVAL = 1000
    
    start_time = time.time()
    total_error = 0.0
    interval_error = 0.0
    
    print("\nStarting Training...")
    
    for i, word in enumerate(words):
        # We simulate sentence boundaries resetting the sequence context
        if word in ['.', '!', '?']:
            b.reset_sequence()
            
        word_vec = b.language.encode(word)
        res = b.perceive(word_vec)
        
        # We add some small random reinforcement just to keep TD(lambda) active, 
        # normally this would come from an internal drive or external task
        b.reinforce_bg(0.0)
        
        error = res.prediction_error
        total_error += error
        interval_error += error
        
        # Dream (consolidate) periodically
        if (i + 1) % DREAM_INTERVAL == 0:
            b.dream(n_dreams=5, steps_per_dream=10)
            
        # Print progress
        if (i + 1) % PRINT_INTERVAL == 0:
            avg_interval_error = interval_error / PRINT_INTERVAL
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            progress = ((i + 1) / total_words) * 100
            
            print(f"[{progress:.1f}%] Words: {i + 1}/{total_words} | "
                  f"Speed: {speed:.1f} w/s | "
                  f"Avg Err: {avg_interval_error:.4f} | "
                  f"Eps: {b.episodic.episode_count} / Protos: {b.episodic.prototype_count}")
                  
            interval_error = 0.0

    print("\nTraining Complete!")
    total_time = time.time() - start_time
    print(f"Processed {total_words} words in {total_time:.2f} seconds ({total_words/total_time:.1f} words/sec).")
    print(f"Final Global Avg Error: {total_error / total_words:.4f}")
    
    print("\nTesting semantic familiarity (Alice fell down the rabbit hole):")
    b.reset_sequence()
    for w in "alice fell down the rabbit hole .".split():
        res = b.perceive(b.language.encode(w))
        print(f"'{w}': error={res.prediction_error:.4f}, attention_passed={res.attention_passed}")

if __name__ == '__main__':
    train_large_corpus()
