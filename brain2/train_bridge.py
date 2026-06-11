#!/usr/bin/env python3
"""
train_bridge.py — Train the Brain on an interleaved dataset of conversational fluency and SQuAD facts to prevent catastrophic forgetting.
"""

import os, sys, json, time, random, gc
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
    EPOCHS = 20

    print(f"Initializing Brain (Dims: {N_DIMS}, SOM: {SOM_ROWS}x{SOM_COLS}, Hidden: {HIDDEN_DIM})...")
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS, hidden_dim=HIDDEN_DIM)
    
    # Load massive squad weights
    massive_ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "massive_squad")
    if os.path.exists(massive_ckpt):
        print(f"Loading Massive SQuAD checkpoints from {massive_ckpt}...")
        try:
            b.load_components(
                predictor_path=os.path.join(massive_ckpt, "predictor.bin"),
                language_path=os.path.join(massive_ckpt, "language.bin"),
                som_path=os.path.join(massive_ckpt, "som.bin"),
                episodic_path=os.path.join(massive_ckpt, "episodic.bin"),
                emotion_path=os.path.join(massive_ckpt, "emotion.bin"),
                self_path=os.path.join(massive_ckpt, "self.bin"),
                symbolic_path=os.path.join(massive_ckpt, "symbolic.bin"),
                binding_path=os.path.join(massive_ckpt, "binding.bin"),
                bg_path=os.path.join(massive_ckpt, "bg.bin"),
                procedures_path=os.path.join(massive_ckpt, "procedures.bin"),
                hpred_path=os.path.join(massive_ckpt, "hpred.bin")
            )
        except Exception as e:
            print(f"Failed to load checkpoints: {e}")
            return
    else:
        print("Massive SQuAD checkpoints not found! Cannot train bridge.")
        return

    # Load datasets
    print("Loading Bridge and SQuAD datasets...")
    with open(os.path.join(os.path.dirname(__file__), "data", "bridge_corpus.json"), "r") as f:
        bridge_corpus = json.load(f)
        
    with open(os.path.join(os.path.dirname(__file__), "data", "squad_qa.json"), "r") as f:
        squad_corpus = json.load(f)
        
    # Sample a subset of SQuAD to prevent catastrophic forgetting
    squad_sample = random.sample(squad_corpus, min(2000, len(squad_corpus)))
    
    # Combine and initialize vocab
    mixed_corpus = bridge_corpus + squad_sample
    print(f"Mixed corpus size: {len(mixed_corpus)} ({len(bridge_corpus)} bridge, {len(squad_sample)} SQuAD)")
    
    initialize_vocab(b, mixed_corpus)
    
    print("\nStarting Interleaved Bridge Training...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- EPOCH {epoch}/{EPOCHS} ---")
        random.shuffle(mixed_corpus)
        
        total_processed = 0
        for pair in mixed_corpus:
            b.reset_sequence()
            seq = pair["input"].split() + pair["target"].split()
            for word in seq:
                vec = b.language.encode(word)
                b.perceive(vec)
            
            total_processed += 1
            if total_processed % 500 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {total_processed} pairs | Elapsed: {elapsed:.1f}s | Dict Size: {b.language.vocab_size}")
                print(f"Predictor Spatial Error (L2): {b.predictor.last_error:.6f}")

        # Run quick generation test
        for test_prompt in ["what is your name", "what is the capital of france", "what is a computer"]:
            b.reset_sequence()
            for tw in test_prompt.split():
                if b.language.knows(tw):
                    b.perceive(b.language.encode(tw))
            res = b.think(6)
            print(f"  Test [{test_prompt}] -> {' '.join([w for w in res.words if w])}")

    # Checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "bridge_squad")
    os.makedirs(ckpt_dir, exist_ok=True)
    b.save_components(ckpt_dir)
    print(f"\nTraining Complete. Saved to {ckpt_dir}")

if __name__ == "__main__":
    train()
