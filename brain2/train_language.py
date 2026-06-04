#!/usr/bin/env python3
"""
train_language.py — Stage 1: Linguistic and Conceptual Bootstrapping

This script feeds a simple corpus of linguistic constructs and logical statements
into the Brain's cognitive loop. This forces the Self-Organizing Map (SOM) to
cluster concepts, the Predictor to learn sequence transitions, and the Episodic
Memory to store the experiences.

Once complete, the entire brain state is saved so that Stage 2 (Math Training)
can load it and build on top of these grounded concepts.
"""

import os, sys, time, random
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

# Ensure same configuration as train_algebra.py
SOM_ROWS = 8
SOM_COLS = 8
N_DIMS = 16

def initialize_embeddings(brain):
    """Ensure determinism across training scripts by pre-registering vocabulary."""
    brain.symbolic_table.seed_math_symbols()
    
    # Register numbers
    for i in range(1000):
        brain.symbolic_table.bind(str(i))
        
    # Register core reasoning words
    words = sorted(["x", "goal", "relation", "math", "op_symbol", "object", "comparison", "eval", "subject", "result"])
    for word in words:
        brain.language.register_word(word)
        brain.symbolic_table.bind(word)
        
    # Additional linguistic corpus words for this stage
    corpus_words = ["apple", "fruit", "isa", "red", "color", "dog", "animal", "barks", "cat", "meows"]
    for word in sorted(corpus_words):
        brain.language.register_word(word)
        brain.symbolic_table.bind(word)

def main():
    print("Initializing Brain for Stage 1: Language Training...", flush=True)
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS)
    initialize_embeddings(b)
    
    # ── Corpus Definition ──────────────────────────────────────────────────
    # Simple sentences to learn relationships and sequences
    corpus = [
        "apple isa fruit",
        "apple color red",
        "dog isa animal",
        "dog barks",
        "cat isa animal",
        "cat meows"
    ]
    
    print("Feeding corpus to the Cognitive Loop...")
    start_time = time.time()
    
    epochs = 100
    for epoch in range(epochs):
        for sentence in corpus:
            words = sentence.split()
            # Push each word through perceive to train Predictor and SOM
            for word in words:
                vec_np = b.language.encode(word)
                if vec_np.shape[0] != N_DIMS:
                    print(f"Dim mismatch: {vec_np.shape} vs {N_DIMS}", flush=True)
                if np.sum(np.abs(vec_np)) > 0:
                    b.perceive(vec_np)
            
            # Explicitly bind the triples to train the BindingMemory
            if len(words) == 3:
                subj = np.array(b.symbolic_table.lookup(words[0]), dtype=np.float32)
                rel = np.array(b.symbolic_table.lookup(words[1]), dtype=np.float32)
                obj = np.array(b.symbolic_table.lookup(words[2]), dtype=np.float32)
                b.bind_triple(subj, rel, obj)
                
    elapsed = time.time() - start_time
    print(f"Language training completed in {elapsed:.2f}s.")
    
    # ── Procedural Reasoning Bootstrap ────────────────────────────────────────
    # Teach the brain the goal→opseq associations so downstream math training
    # doesn't start from scratch. This seeds BG memory with the correct procedures.
    print("Bootstrapping procedural reasoning pathways...", flush=True)
    
    # Op codes (must match basal_ganglia.hpp enum values)
    OP_MATH_SUB = 2;  OP_MATH_DIV = 3;  OP_HALT = 8
    OP_SPEAK = 15;    OP_MUL = 21;      OP_PERM_N = 22; OP_PERM_K = 23
    OP_POWER = 24;    OP_DIV_FLOAT = 26
    
    for goal_word in ["solve", "permute", "probability", "area", "power", "remember"]:
        brain.language.register_word(goal_word)
        brain.symbolic_table.bind(goal_word)
    
    # Goal → procedure map
    PROCEDURES = {
        "solve":       [OP_MATH_SUB, OP_MATH_DIV, OP_SPEAK, OP_HALT],
        "permute":     [OP_MATH_SUB, OP_PERM_N,   OP_PERM_K, OP_MATH_DIV, OP_SPEAK, OP_HALT],
        "probability": [OP_DIV_FLOAT, OP_SPEAK,   OP_HALT],
        "area":        [OP_MUL, OP_SPEAK, OP_HALT],
        "power":       [OP_POWER, OP_SPEAK, OP_HALT],
    }
    
    for goal_word, op_seq in PROCEDURES.items():
        goal_vec = np.array(brain.language.encode(goal_word), dtype=np.float32)
        brain.scratchpad.write("goal", goal_vec, "goal")
        brain.start_reasoning()
        for op in op_seq:
            brain.force_reason_step(op, goal_word)
            brain.reinforce_bg(1.0)
    
    print("Procedural pathway bootstrap complete.")
    
    # ── Save Checkpoint ────────────────────────────────────────────────────
    stage1_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "stage1_language")
    os.makedirs(stage1_dir, exist_ok=True)
    print(f"Saving Brain state to {stage1_dir}...")
    b.save_components(stage1_dir)
    print("Stage 1 complete! You can now run Stage 2 (train_algebra.py).")

if __name__ == "__main__":
    main()
