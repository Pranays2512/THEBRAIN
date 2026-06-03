#!/usr/bin/env python3
"""
train_qa.py — Teach the BG Controller to perform autonomous Question Answering.

Problem type: 
Given an interrogative concept ("?") in one of the slots (e.g., subject="?", relation="isa", object="apple"),
the BG must learn to shift attention to the missing piece, query memory, and speak the result.

Target Op Sequence:
ATTEND -> BIND_QUERY -> SPEAK -> HALT
"""

import os, sys, random, time
import numpy as np
import brain2

# Configuration
N_EPISODES     = 50000
MAX_STEPS      = 6
SAVE_INTERVAL  = 5000
PRINT_INTERVAL = 1000
CHECKPOINT_DIR = "checkpoints/stage4_parsing"

OP_READ = 0; OP_WRITE = 1; OP_MATH_SUB = 2; OP_MATH_DIV = 3
OP_COMPARE = 4; OP_BIND_QUERY = 5; OP_RETRIEVE = 6; OP_ANALOGY = 7
OP_HALT = 8; OP_STORE_SUBJ = 9; OP_STORE_REL = 10; OP_STORE_OBJ = 11
OP_NOT = 12; OP_BIND_ISA = 13; OP_ASK_USER = 14; OP_SPEAK = 15; OP_ATTEND = 16

b = brain2.Brain(som_rows=8, som_cols=8, n_dims=16)

def load_all():
    if os.path.exists(CHECKPOINT_DIR):
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
    else:
        print("Checkpoint not found!")
        sys.exit(1)

def run_episode(is_query=True):
    b.scratchpad.clear()
    b.clear_spoken_words()

    if is_query:
        q_word = "?"
        rel_word = "isa"
        obj_word = "apple"
        
        b.scratchpad.write("subject", b.language.encode(obj_word), "context")
        b.scratchpad.write("relation", b.language.encode(rel_word), "context")
        b.scratchpad.write("object", b.language.encode(q_word), "context")
        b.scratchpad.write("goal", b.language.encode("reply"), "goal")
        
        b.start_reasoning()
        b.force_reason_step(OP_BIND_QUERY, "reply")
        b.reinforce_bg(1.0)
        b.force_reason_step(OP_SPEAK, "reply")
        b.reinforce_bg(1.0)
        b.force_reason_step(OP_HALT, "reply")
        b.reinforce_bg(1.0)
    else:
        # Fact: apple is fruit
        subj_word = "apple"
        rel_word = "isa"
        obj_word = "fruit"
        
        b.scratchpad.write("subject", b.language.encode(subj_word), "context")
        b.scratchpad.write("relation", b.language.encode(rel_word), "context")
        b.scratchpad.write("object", b.language.encode(obj_word), "context")
        b.scratchpad.write("goal", b.language.encode("reply"), "goal")
        
        b.start_reasoning()
        b.force_reason_step(13, "reply") # OP_BIND_ISA
        b.reinforce_bg(1.0)
        
        # When learning a fact, it might be polite to say "Got it." or just halt.
        # Let's just HALT.
        b.force_reason_step(OP_HALT, "reply")
        b.reinforce_bg(1.0)
    
    # Test Policy
    b.scratchpad.clear()
    if is_query:
        b.scratchpad.write("subject", b.language.encode("apple"), "context")
        b.scratchpad.write("relation", b.language.encode("isa"), "context")
        b.scratchpad.write("object", b.language.encode("?"), "context")
    else:
        b.scratchpad.write("subject", b.language.encode("apple"), "context")
        b.scratchpad.write("relation", b.language.encode("isa"), "context")
        b.scratchpad.write("object", b.language.encode("fruit"), "context")
        
    b.scratchpad.write("goal", b.language.encode("reply"), "goal")
    
    b.start_reasoning()
    ops_taken = []
    for step in range(MAX_STEPS):
        op = b.reason_step("reply", 0.0)
        ops_taken.append(op)
        if op == OP_HALT:
            break
            
    expected = [OP_BIND_QUERY, OP_SPEAK, OP_HALT] if is_query else [13, OP_HALT]
    return 1.0 if ops_taken == expected else 0.0, ops_taken

if __name__ == "__main__":
    print("Starting script...", flush=True)
    load_all()
    print("Training BG Controller with Teacher Forcing...", flush=True)
    
    wins = 0
    start_time = time.time()
    
    for ep in range(1, 1000 + 1):  # Teacher forcing is very fast!
        is_q = (ep % 2 == 0)
        reward, ops = run_episode(is_q)
        if reward >= 1.0:
            wins += 1
            
        if ep % 100 == 0:
            win_rate = wins / 100
            elapsed = time.time() - start_time
            print(f"Ep {ep:4d}/1000 | WinRate: {win_rate*100:5.1f}% | Last Ops: {ops} | Time: {elapsed:.1f}s", flush=True)
            wins = 0
            start_time = time.time()
            
    b.save_bg(f"{CHECKPOINT_DIR}/bg.bin")
    print("Finished QA Training.")
