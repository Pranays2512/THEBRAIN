#!/usr/bin/env python3
"""
train_conversational.py — Train the Brain exclusively on massive conversational data to achieve true language mastery.
"""

import os, sys, json, time, random
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def initialize_vocab(b, corpus):
    """Seed the symbolic and language tables with all words in the corpus."""
    words = set()
    for pair in corpus:
        words.update(pair["input"].split())
        words.update(pair["target"].split())
    
    for word in sorted(words):
        b.language.register_word(word)
        b.symbolic_table.bind(word)

def train():
    N_DIMS = 128
    SOM_ROWS = 256
    SOM_COLS = 256
    HIDDEN_DIM = 256
    EPOCHS = 1

    print(f"Initializing Brain (Dims: {N_DIMS}, SOM: {SOM_ROWS}x{SOM_COLS}, Hidden: {HIDDEN_DIM})...")
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS, hidden_dim=HIDDEN_DIM)
    
    # Load massive squad weights as the base
    base_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "massive_squad")
    if os.path.exists(base_ckpt):
        print(f"Loading base checkpoints from {base_ckpt}...")
        try:
            b.load_components(
                predictor_path=os.path.join(base_ckpt, "predictor.bin"),
                language_path=os.path.join(base_ckpt, "language.bin"),
                som_path=os.path.join(base_ckpt, "som.bin"),
                episodic_path=os.path.join(base_ckpt, "episodic.bin"),
                emotion_path=os.path.join(base_ckpt, "emotion.bin"),
                self_path=os.path.join(base_ckpt, "self.bin"),
                symbolic_path=os.path.join(base_ckpt, "symbolic.bin"),
                binding_path=os.path.join(base_ckpt, "binding.bin"),
                bg_path=os.path.join(base_ckpt, "bg.bin"),
                procedures_path=os.path.join(base_ckpt, "procedures.bin"),
                hpred_path=os.path.join(base_ckpt, "hpred.bin")
            )
        except Exception as e:
            print(f"Failed to load checkpoints: {e}")
            return
    else:
        print("Base checkpoints not found! Cannot train.")
        return

    # Load datasets
    print("Loading Conversational Massive dataset...")
    with open(os.path.join(os.path.dirname(__file__), "data", "conversational_massive.json"), "r") as f:
        conv_corpus = json.load(f)
        
    print(f"Conversational corpus size: {len(conv_corpus)}")
    
    initialize_vocab(b, conv_corpus)
    
    print("\nStarting Fluent Conversational Training...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- EPOCH {epoch}/{EPOCHS} ---")
        random.shuffle(conv_corpus)
        
        total_processed = 0
        for pair in conv_corpus:
            b.reset_sequence()
            seq = pair["input"].split() + pair["target"].split()
            for word in seq:
                vec = b.language.encode(word)
                b.perceive(vec)
            
            total_processed += 1
            if total_processed % 2000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {total_processed} pairs | Elapsed: {elapsed:.1f}s | Dict Size: {b.language.vocab_size}")
                print(f"Predictor Spatial Error (L2): {b.predictor.last_error:.6f}")
                
            if total_processed > 0 and total_processed % 10000 == 0:
                ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "fluent_brain")
                os.makedirs(ckpt_dir, exist_ok=True)
                print(f"Auto-saving checkpoint to {ckpt_dir}...", flush=True)
                b.save_components(ckpt_dir)
                import gc
                gc.collect()

        # Run quick generation test
        for test_prompt in ["what is your name", "how are you doing", "what do you think of music"]:
            b.reset_sequence()
            for tw in test_prompt.split():
                if b.language.knows(tw):
                    b.perceive(b.language.encode(tw))
            res = b.think(6)
            print(f"  Test [{test_prompt}] -> {' '.join([w for w in res.words if w])}")

    # Checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "fluent_brain")
    os.makedirs(ckpt_dir, exist_ok=True)
    b.save_components(ckpt_dir)
    print(f"\nTraining Complete. Saved to {ckpt_dir}")

if __name__ == "__main__":
    train()
