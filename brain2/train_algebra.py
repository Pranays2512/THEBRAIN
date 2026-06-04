#!/usr/bin/env python3
"""
train_algebra.py — Teach the BG Controller to perform autonomous Equation Solving.

Problem type: 
Given an equation `ax + b = c`, where:
subject = c
relation = a
object = b
goal = solve

Target Op Sequence:
OP_MATH_SUB (c - b -> result)
OP_MATH_DIV (result / a -> result)
OP_SPEAK (say the result)
OP_HALT
"""

import os, sys, random, time
import numpy as np
import brain2

# Configuration
N_EPISODES     = 5000
MAX_STEPS      = 6
CHECKPOINT_DIR = "checkpoints/stage4_parsing"

OP_READ = 0; OP_WRITE = 1; OP_MATH_SUB = 2; OP_MATH_DIV = 3
OP_COMPARE = 4; OP_BIND_QUERY = 5; OP_RETRIEVE = 6; OP_ANALOGY = 7
OP_HALT = 8; OP_STORE_SUBJ = 9; OP_STORE_REL = 10; OP_STORE_OBJ = 11
OP_NOT = 12; OP_BIND_ISA = 13; OP_ASK_USER = 14; OP_SPEAK = 15; OP_ATTEND = 16

b = brain2.Brain(som_rows=10, som_cols=10, n_dims=32)

def load_all():
    if os.path.exists(CHECKPOINT_DIR):
        print(f"Loading Brain from {CHECKPOINT_DIR}...")
        bg_path_file = f"{CHECKPOINT_DIR}/bg.bin"
        b.load_components(
            predictor_path=f"{CHECKPOINT_DIR}/predictor.bin",
            language_path=f"{CHECKPOINT_DIR}/language.bin",
            som_path=f"{CHECKPOINT_DIR}/som.bin",
            episodic_path=f"{CHECKPOINT_DIR}/episodic.bin",
            emotion_path=f"{CHECKPOINT_DIR}/emotion.bin",
            self_path=f"{CHECKPOINT_DIR}/self.bin",
            symbolic_path=f"{CHECKPOINT_DIR}/symbolic.bin",
            binding_path=f"{CHECKPOINT_DIR}/binding.bin",
            bg_path=bg_path_file if os.path.exists(bg_path_file) else "",
            procedures_path=f"{CHECKPOINT_DIR}/procedures.bin",
            hpred_path=f"{CHECKPOINT_DIR}/hpred.bin"
        )
    else:
        print("Checkpoint not found!")
        sys.exit(1)

def run_episode(is_solve=True):
    b.scratchpad.clear()
    b.clear_spoken_words()

    if is_solve:
        # Example: 2x + 4 = 10 -> subject=10, relation=2, object=4
        subj_word = "10"
        rel_word = "2"
        obj_word = "4"
        
        # Ensure vocabulary knows the numbers
        for w in [subj_word, rel_word, obj_word]:
            if not b.symbolic_table.knows(w):
                b.learn_word(w)
                
        b.scratchpad.write("subject", b.language.encode(subj_word), "context")
        b.scratchpad.write("relation", b.language.encode(rel_word), "context")
        b.scratchpad.write("object", b.language.encode(obj_word), "context")
        b.scratchpad.write("goal", b.language.encode("solve"), "goal")
        
        b.start_reasoning()
        
        # Step 1: Subtraction (c - b)
        b.force_reason_step(OP_MATH_SUB, "solve")
        b.reinforce_bg(1.0)
        
        # Step 2: Division (res / a)
        b.force_reason_step(OP_MATH_DIV, "solve")
        b.reinforce_bg(1.0)
        
        # Step 3: Speak
        b.force_reason_step(OP_SPEAK, "solve")
        b.reinforce_bg(1.0)
        
        # Step 4: Halt
        b.force_reason_step(OP_HALT, "solve")
        b.reinforce_bg(1.0)
    else:
        # Fallback to QA so it doesn't forget how to answer queries!
        q_word = "?"
        rel_word = "color"
        obj_word = "apple"
        
        b.scratchpad.write("subject", b.language.encode(obj_word), "context")
        b.scratchpad.write("relation", b.language.encode(rel_word), "context")
        b.scratchpad.write("object", b.language.encode(q_word), "context")
        b.scratchpad.write("goal", b.language.encode("reply"), "goal")
        
        b.start_reasoning()
        # [BIND_QUERY, SPEAK_SUBJ, SPEAK_REL, SPEAK_OBJ, HALT]
        b.force_reason_step(OP_BIND_QUERY, "reply")
        b.reinforce_bg(1.0)
        b.force_reason_step(17, "reply") # SPEAK_SUBJ
        b.reinforce_bg(1.0)
        b.force_reason_step(18, "reply") # SPEAK_REL
        b.reinforce_bg(1.0)
        b.force_reason_step(OP_SPEAK, "reply") # SPEAK (result)
        b.reinforce_bg(1.0)
        b.force_reason_step(OP_HALT, "reply")
        b.reinforce_bg(1.0)
    
    # Test Policy
    b.scratchpad.clear()
    if is_solve:
        b.scratchpad.write("subject", b.language.encode("10"), "context")
        b.scratchpad.write("relation", b.language.encode("2"), "context")
        b.scratchpad.write("object", b.language.encode("4"), "context")
        b.scratchpad.write("goal", b.language.encode("solve"), "goal")
    else:
        b.scratchpad.write("subject", b.language.encode("apple"), "context")
        b.scratchpad.write("relation", b.language.encode("color"), "context")
        b.scratchpad.write("object", b.language.encode("?"), "context")
        b.scratchpad.write("goal", b.language.encode("reply"), "goal")
    
    b.start_reasoning()
    ops_taken = []
    for step in range(MAX_STEPS):
        op = b.reason_step("solve" if is_solve else "reply", 0.0)
        ops_taken.append(op)
        if op == OP_HALT:
            break
            
    expected = [OP_MATH_SUB, OP_MATH_DIV, OP_SPEAK, OP_HALT] if is_solve else [OP_BIND_QUERY, 17, 18, OP_SPEAK, OP_HALT]
    return 1.0 if ops_taken == expected else 0.0, ops_taken

if __name__ == "__main__":
    print("Starting Algebra Curriculum...", flush=True)
    load_all()
    print("Training BG Controller with Teacher Forcing...", flush=True)
    
    wins = 0
    start_time = time.time()
    
    for ep in range(1, N_EPISODES + 1):
        is_solve = (ep % 2 == 0)
        reward, ops = run_episode(is_solve)
        if reward >= 1.0:
            wins += 1
            
        if ep % 100 == 0:
            win_rate = wins / 100
            elapsed = time.time() - start_time
            print(f"Ep {ep:4d}/{N_EPISODES} | WinRate: {win_rate*100:5.1f}% | Last Ops: {ops} | Time: {elapsed:.1f}s", flush=True)
            wins = 0
            start_time = time.time()
            
    b.save_bg(f"{CHECKPOINT_DIR}/bg.bin")
    b.save_components(CHECKPOINT_DIR)  # Save symbolic and language since we learned numbers!
    print("Finished Algebra Training.")
