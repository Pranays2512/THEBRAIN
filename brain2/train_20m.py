import brain2
import os
import sys
import time
import urllib.request
import re

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "tiny_shakespeare.txt"
CKPT_DIR = "checkpoints/stage5_20m"

def calculate_parameters(rows, cols, dims, bg_hidden):
    n_nodes = rows * cols
    
    # SOM
    som_params = n_nodes * dims
    # Predictor (n_nodes x n_nodes)
    pred_params = n_nodes * n_nodes
    # Basal Ganglia
    bg_params = n_nodes * bg_hidden + bg_hidden * 17
    # Decoder RNN
    dec_params = n_nodes * 64 + 64 * 64
    
    total = som_params + pred_params + bg_params + dec_params
    return total, {
        "SOM": som_params,
        "Predictor": pred_params,
        "BasalGanglia": bg_params,
        "Decoder": dec_params
    }

def main():
    print("=" * 60)
    print(" 20 MILLION PARAMETER BRAIN UNSUPERVISED TRAINING ")
    print("=" * 60)
    
    # 1. Download Corpus
    if not os.path.exists(DATA_FILE):
        print(f"Downloading TinyShakespeare ({DATA_FILE})...")
        urllib.request.urlretrieve(URL, DATA_FILE)
    else:
        print(f"Using existing {DATA_FILE}")
        
    print("Parsing text...", end="", flush=True)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read().lower()
    
    # Simple word tokenization
    words = re.findall(r'\b[a-z]+\b', text)
    total_words = len(words)
    print(f" Found {total_words} words.")
    
    # 2. Initialize Brain
    ROWS, COLS, DIMS = 64, 64, 256
    print("\nInitializing Brain v3 Engine in C++...")
    b = brain2.Brain(som_rows=ROWS, som_cols=COLS, n_dims=DIMS)
    
    total_params, breakdown = calculate_parameters(ROWS, COLS, DIMS, 128)
    print(f"\n[ ARCHITECTURE SCALING SUCCESS ]")
    print(f"Total Parameters: {total_params:,d}")
    print(f"  - SOM Grid:       {breakdown['SOM']:,d}")
    print(f"  - Predictor:      {breakdown['Predictor']:,d}")
    print(f"  - Logic Engine:   {breakdown['BasalGanglia']:,d}")
    print(f"  - Vocab Decoder:  {breakdown['Decoder']:,d}")
    
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    print("\nStarting Unsupervised Pre-training...")
    print("The Brain is now reading the corpus, grounding words into concepts, and learning grammar.")
    
    start_time = time.time()
    last_print_time = start_time
    
    # 3. Training Loop
    for i, w in enumerate(words):
        # Unsupervised learning loop
        if not b.symbolic_table.knows(w):
            b.learn_word(w)
            
        # The Language module encodes the word to 256D, then the Brain perceives it,
        # updating the SOM clusters and the Predictor matrix dynamically.
        b.perceive(b.language.encode(w))
        
        # Progress Bar Logic
        if i % 1000 == 0 and i > 0:
            now = time.time()
            elapsed = now - start_time
            speed = i / elapsed
            rem_words = total_words - i
            eta_sec = rem_words / speed
            
            # Format ETA
            m, s = divmod(int(eta_sec), 60)
            h, m = divmod(m, 60)
            eta_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            
            sys.stdout.write(f"\rProgress: [{i:7d} / {total_words:7d}] "
                             f"| Speed: {speed:5.0f} w/s | ETA: {eta_str}")
            sys.stdout.flush()
            
            # Periodically save
            if now - last_print_time > 60:  # save every 60 seconds
                b.save_components(CKPT_DIR)
                last_print_time = now

    print("\n\nTraining Complete! Saving final checkpoint...")
    b.save_components(CKPT_DIR)
    print(f"Successfully saved 20M parameter brain to {CKPT_DIR}")

if __name__ == "__main__":
    main()
