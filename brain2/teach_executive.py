#!/usr/bin/env python3
"""
teach_executive.py — Reinforcement Learning for Executive Function

This script trains the Brain's Basal Ganglia (Actor-Critic) to consciously select 
logical operations (Op::MATH_ADD, Op::ANALOGY, etc.) instead of relying on the 
fuzzy predictive neural network.
"""

import os, sys, time, random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

# Action indices mapped from basal_ganglia.hpp Op enum
OP_READ       = 0
OP_WRITE      = 1
OP_MATH_SUB   = 2
OP_MATH_DIV   = 3
OP_ANALOGY    = 7
OP_HALT       = 8
OP_MATH_ADD   = 20
OP_MATH_MUL   = 21

def train_executive():
    print("Initializing Brain for Executive Training...")
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "math_brain")
    if os.path.exists(os.path.join(ckpt_dir, "bg.bin")):
        print(f"Loading checkpoints from {ckpt_dir}...")
        try:
            b.load_components(
                predictor_path=os.path.join(ckpt_dir, "predictor.bin"),
                language_path=os.path.join(ckpt_dir, "language.bin"),
                som_path=os.path.join(ckpt_dir, "som.bin"),
                episodic_path=os.path.join(ckpt_dir, "episodic.bin"),
                emotion_path=os.path.join(ckpt_dir, "emotion.bin"),
                self_path=os.path.join(ckpt_dir, "self.bin"),
                symbolic_path=os.path.join(ckpt_dir, "symbolic.bin"),
                binding_path=os.path.join(ckpt_dir, "binding.bin"),
                bg_path=os.path.join(ckpt_dir, "bg.bin"),
                procedures_path=os.path.join(ckpt_dir, "procedures.bin"),
                hpred_path=os.path.join(ckpt_dir, "hpred.bin")
            )
        except Exception as e:
            print(f"Error loading: {e}")
            return
            
    print("Generating Executive Curriculum...")
    
    # Simple curriculum of addition and subtraction
    tasks = []
    for _ in range(5000):
        if random.random() > 0.5:
            # Addition Task
            v1 = random.randint(1, 50)
            v2 = random.randint(1, 50)
            target_op = OP_MATH_ADD
            goal_word = "+"
        else:
            # Subtraction Task
            v1 = random.randint(50, 100)
            v2 = random.randint(1, 50)
            target_op = OP_MATH_SUB
            goal_word = "-"
            
        tasks.append({
            "sub": str(v1),
            "obj": str(v2),
            "target_op": target_op,
            "goal": goal_word
        })

    # Analogy Tasks
    tasks.append({"sub": "dog", "rel": "has", "ctx": "bird", "target_op": OP_ANALOGY, "goal": "analogy"})
    tasks.append({"sub": "car", "rel": "needs", "ctx": "human", "target_op": OP_ANALOGY, "goal": "analogy"})
    
    # Multiply analogy tasks to balance the dataset
    analogy_tasks = [t for t in tasks if t["target_op"] == OP_ANALOGY] * 500
    tasks.extend(analogy_tasks)
    
    random.shuffle(tasks)
    
    print(f"Generated {len(tasks)} tasks.")
    print("Starting Actor-Critic Reinforcement Learning...")
    
    epochs = 3
    correct = 0
    total = 0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- EPOCH {epoch}/{epochs} ---")
        correct = 0
        total = 0
        
        for i, task in enumerate(tasks):
            b.reset_sequence()
            
            # Make sure vocab knows the numbers/words
            for w in [task["sub"], task.get("obj", ""), task.get("rel", ""), task.get("ctx", ""), task["goal"]]:
                if w and not b.language.knows(w):
                    b.language.register_word(w)
            
            # Stage the Working Memory / Scratchpad
            b.scratchpad.write("subject", b.language.encode(task["sub"]), "ctx")
            if "obj" in task:
                b.scratchpad.write("object", b.language.encode(task["obj"]), "ctx")
            if "rel" in task:
                b.scratchpad.write("relation", b.language.encode(task["rel"]), "ctx")
            if "ctx" in task:
                b.scratchpad.write("context_map", b.language.encode(task["ctx"]), "ctx")
                
            # Force Basal Ganglia to make a decision
            chosen_op = b.direct_reason_step(task["goal"])
            
            # Calculate Dopamine Reward
            if chosen_op == task["target_op"]:
                reward = 1.0
                correct += 1
            elif chosen_op in [OP_MATH_ADD, OP_MATH_SUB, OP_ANALOGY, OP_MATH_MUL, OP_MATH_DIV]:
                # Right category, wrong specific operation
                reward = -0.5
            else:
                # Completely wrong operation (like READ or WRITE)
                reward = -1.0
                
            total += 1
            
            # Apply TD(lambda) reinforcement to Basal Ganglia Actor-Critic weights
            b.reinforce_bg(reward)
            
            if total % 1000 == 0:
                acc = (correct / total) * 100
                print(f"Processed {total} tasks | Accuracy: {acc:.1f}%")
                # Reset rolling stats
                correct = 0
                total = 0
                
    # Save the trained executive
    out_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "executive_brain")
    os.makedirs(out_dir, exist_ok=True)
    b.save_components(out_dir)
    print(f"\nTraining Complete! Brain executive functions saved to {out_dir}")

if __name__ == "__main__":
    train_executive()
