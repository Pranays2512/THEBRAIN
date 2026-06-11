#!/usr/bin/env python3
"""
train_fluency.py — Autoregressive Language Fluency Training

This script trains the Brain's LSTMs and SOM to achieve natural conversational fluency.
By passing sequential word vectors through `brain.perceive()`, the Predictor naturally 
learns to forecast the next concept state. When later prompted via `brain.think()`, 
it will autocomplete the sequence, forming a conversational reply.
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

# Training Hyperparameters for Scaled Architecture
N_DIMS = 128
SOM_ROWS = 256
SOM_COLS = 256
HIDDEN_DIM = 256
EPOCHS = 1000
PRINT_INTERVAL = 100

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
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "conversational_corpus.json")
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus not found at {corpus_path}")
        return
        
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
        
    print(f"Initializing Brain (Dims: {N_DIMS}, SOM: {SOM_ROWS}x{SOM_COLS}, Hidden: {HIDDEN_DIM})...")
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS, hidden_dim=HIDDEN_DIM)
    
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
            
    initialize_vocab(b, corpus)
    print(f"Vocabulary loaded: {len(corpus)} QA pairs.")
    
    print("\nStarting Neural Conversational Training...")
    start_time = time.time()
    
    # Train Predictor via perceive()
    for epoch in range(1, EPOCHS + 1):
        for pair in corpus:
            b.reset_sequence()
            
            # Combine input and target into one sequence
            seq = pair["input"].split() + pair["target"].split()
            
            for word in seq:
                vec = b.language.encode(word)
                b.perceive(vec)
                
        if epoch % PRINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{EPOCHS} | Elapsed: {elapsed:.2f}s")
            
            # Quick evaluation on a sample
            b.reset_sequence()
            test_prompt = "hello"
            b.perceive(b.language.encode(test_prompt))
            res = b.think(2)
            print(f"  Test [hello] -> {' '.join(res.words)}")
            
            b.reset_sequence()
            test_prompt2 = "what is your"
            for w in test_prompt2.split():
                b.perceive(b.language.encode(w))
            res2 = b.think(3)
            print(f"  Test [{test_prompt2}] -> {' '.join(res2.words)}")
    
    # Checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "fluent_squad")
    os.makedirs(ckpt_dir, exist_ok=True)
    b.save_components(ckpt_dir)
    print(f"\nTraining Complete. Saved to {ckpt_dir}")
    
    # Interactive Chat
    print("\n=== INTERACTIVE CHAT MODE ===")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_in = input("\nYou: ").strip().lower()
            if user_in in ["exit", "quit", "q"]:
                break
            if not user_in:
                continue
                
            b.reset_sequence()
            words = user_in.split()
            for w in words:
                if b.language.knows(w):
                    b.perceive(b.language.encode(w))
                else:
                    b.language.register_word(w)
                    b.symbolic_table.bind(w)
                    b.perceive(b.language.encode(w))
                    
            # Let it think
            reply = b.think(4) # Generate up to 4 words
            # Filter out empty or repeated words
            clean_reply = []
            for w in reply.words:
                if w and (not clean_reply or clean_reply[-1] != w):
                    clean_reply.append(w)
            
            print("Brain:", " ".join(clean_reply))
            
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    train()
