#!/usr/bin/env python3
import os, sys, random
import numpy as np
import brain2

SOM_ROWS = 8
SOM_COLS = 8
N_DIMS = 16
N_EPISODES = 5000
MAX_STEPS = 5

OP_RETRIEVE = 6
OP_HALT = 8

def train():
    print("Initializing Brain for Episodic Training...")
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS)
    
    stage_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "stage5_math")
    if os.path.exists(stage_dir):
        print(f"Loading Stage 5 Math components from {stage_dir}...")
        b.load_components(
            predictor_path=os.path.join(stage_dir, "predictor.bin"),
            language_path=os.path.join(stage_dir, "language.bin"),
            som_path=os.path.join(stage_dir, "som.bin"),
            episodic_path=os.path.join(stage_dir, "episodic.bin"),
            emotion_path=os.path.join(stage_dir, "emotion.bin"),
            self_path=os.path.join(stage_dir, "self.bin"),
            symbolic_path=os.path.join(stage_dir, "symbolic.bin"),
            binding_path=os.path.join(stage_dir, "binding.bin"),
            bg_path=os.path.join(stage_dir, "bg.bin"),
            procedures_path=os.path.join(stage_dir, "procedures.bin"),
            hpred_path=os.path.join(stage_dir, "hpred.bin")
        )
        
    for w in ["i", "say", "remember", "focus"]:
        if not b.symbolic_table.knows(w):
            b.symbolic_table.bind(w)
            b.language.register_word(w)

    print("Training 'remember' procedure...")
    remember_ops = [6, 15, 8]  # RETRIEVE, SPEAK, HALT
    
    # We consolidate the procedure so it is locked in neural memory
    b.reset_sequence()
    goal_vec = b.language.encode("remember")
    bmu = b.som.activation_map(goal_vec)
    b.working_mem.gate(bmu * 10.0, 1.0)
    b.working_mem.tick()
    
    b.consolidate_procedure(remember_ops, "remember")
    print("Consolidated 'remember' procedure.")
    
    b.save_components(stage_dir)
    print("Done. Saved to stage5_math.")

if __name__ == "__main__":
    train()
