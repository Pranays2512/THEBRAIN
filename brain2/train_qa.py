#!/usr/bin/env python3
"""
train_qa.py — Teach the BG Controller to perform autonomous Question Answering.

Problem type: 
Given an interrogative concept ("?") in one of the slots (e.g., subject="?", relation="isa", object="apple"),
the BG must learn to shift attention to the missing piece, query memory, and speak the result.

Target Op Sequence:
ATTEND -> BIND_QUERY -> SPEAK -> HALT
"""

import os, sys, random, time, json
import numpy as np
import brain2

# Configuration
N_DIMS         = 128
SOM_ROWS       = 256
SOM_COLS       = 256
HIDDEN_DIM     = 256
N_EPISODES     = 10000
MAX_STEPS      = 6
SAVE_INTERVAL  = 500
CHECKPOINT_DIR = "checkpoints/massive_squad"

OP_READ = 0; OP_WRITE = 1; OP_MATH_SUB = 2; OP_MATH_DIV = 3
OP_COMPARE = 4; OP_BIND_QUERY = 5; OP_RETRIEVE = 6; OP_ANALOGY = 7
OP_HALT = 8; OP_STORE_SUBJ = 9; OP_STORE_REL = 10; OP_STORE_OBJ = 11
OP_NOT = 12; OP_BIND_ISA = 13; OP_ASK_USER = 14; OP_SPEAK = 15; OP_ATTEND = 16

print(f"Initializing Brain (Dims: {N_DIMS}, SOM: {SOM_ROWS}x{SOM_COLS}, Hidden: {HIDDEN_DIM})...")
b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS, hidden_dim=HIDDEN_DIM)

def load_all():
    if os.path.exists(f"{CHECKPOINT_DIR}/predictor.bin"):
        print(f"Loading Brain from {CHECKPOINT_DIR}...")
        b.load_components(
            predictor_path=f"{CHECKPOINT_DIR}/predictor.bin",
            language_path=f"{CHECKPOINT_DIR}/language.bin",
            som_path=f"{CHECKPOINT_DIR}/som.bin",
            episodic_path=f"{CHECKPOINT_DIR}/episodic.bin",
            emotion_path=f"{CHECKPOINT_DIR}/emotion.bin",
            self_path=f"{CHECKPOINT_DIR}/self.bin",
            symbolic_path=f"{CHECKPOINT_DIR}/symbolic.bin",
            binding_path=f"{CHECKPOINT_DIR}/binding.bin",
            procedures_path=f"{CHECKPOINT_DIR}/procedures.bin",
            hpred_path=f"{CHECKPOINT_DIR}/hpred.bin"
        )
        if os.path.exists(f"{CHECKPOINT_DIR}/bg.bin"):
            b.load_bg(f"{CHECKPOINT_DIR}/bg.bin")
    else:
        print("Checkpoint not found! Run train_massive_corpus.py first.")
        sys.exit(1)

def run_episode(pair):
    b.reset_sequence()
    b.clear_spoken_words()

    q_text = pair["input"]
    a_text = pair["target"]
    
    # Preload the working memory by perceiving the question
    b.perceive_text(q_text)
    
    # The actor must learn to extract context and trigger inner speech (SPEAK)
    b.scratchpad.write("goal", b.language.encode("reply"), "goal")
    
    b.start_reasoning()
    
    # 1. Attend to context
    b.force_reason_step(OP_ATTEND, "reply")
    b.reinforce_bg(1.0)
    
    # 2. Retrieve semantic memories if necessary
    b.force_reason_step(OP_RETRIEVE, "reply")
    b.reinforce_bg(1.0)
    
    # 3. Trigger SPEAK to hand off to the Predictor (Inner Speech)
    b.force_reason_step(OP_SPEAK, "reply")
    b.reinforce_bg(1.0)
    
    # 4. Halt
    b.force_reason_step(OP_HALT, "reply")
    b.reinforce_bg(1.0)
    
    return 1.0, [OP_ATTEND, OP_RETRIEVE, OP_SPEAK, OP_HALT]

if __name__ == "__main__":
    print("Starting Phase 3 Reinforcement Learning...", flush=True)
    load_all()
    
    corpus_path = "data/squad_qa.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} SQuAD QA pairs.")
    
    print("Training Basal Ganglia Controller with Teacher Forcing...", flush=True)
    
    wins = 0
    start_time = time.time()
    
    # Shuffle for RL
    random.shuffle(corpus)
    
    # Train on a subset for RL Phase (Teacher forcing is highly sample efficient)
    subset = corpus[:N_EPISODES]
    
    for ep, pair in enumerate(subset, 1):
        reward, ops = run_episode(pair)
        if reward >= 1.0:
            wins += 1
            
        if ep % 50 == 0:
            win_rate = wins / 50
            elapsed = time.time() - start_time
            print(f"Ep {ep:4d}/{N_EPISODES} | WinRate: {win_rate*100:5.1f}% | Last Ops: {ops} | Time: {elapsed:.1f}s", flush=True)
            wins = 0
            start_time = time.time()
            
        if ep % SAVE_INTERVAL == 0:
            b.save_bg(f"{CHECKPOINT_DIR}/bg.bin")
            
    b.save_bg(f"{CHECKPOINT_DIR}/bg.bin")
    print("Finished Phase 3 QA Reasoning Training. BG routing policy saved.")
