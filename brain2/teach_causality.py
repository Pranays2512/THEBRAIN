#!/usr/bin/env python3
"""
teach_causality.py — Phase 2: Causal Simulation Training

Feeds the Brain strict logical cause-and-effect sequences.
When the Brain mispredicts a causal outcome, its PredictiveCodingLayer 
will trigger a violent Anti-Hebbian unlearning of its WorkingMemory, 
forcing the neural weights to align with causal reality.
"""

import os, sys, random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def teach_causality():
    print("Loading Phase 1 Brain for Phase 2 Causal Training...")
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "massive_squad")
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

    causal_rules = [
        "fire burns wood to ash",
        "water extinguishes fire",
        "sun heats water to steam",
        "gravity pulls objects down",
        "ice melts into water",
        "rain makes ground wet",
        "seed grows into tree",
        "wind blows leaves away",
        "food gives energy to animals",
        "birds fly in sky"
    ]
    
    # Generate 10000 causal streams
    print("Generating Causal Curriculum...")
    curriculum = []
    for _ in range(10000):
        curriculum.append(random.choice(causal_rules))
        
    print(f"Starting Causal Simulation on {len(curriculum)} streams...")
    
    total_l2 = 0.0
    for i, stream in enumerate(curriculum):
        b.reset_sequence()
        b.perceive_text(stream)
        
        # After perceiving the sequence, evaluate error
        error = b.predictor.last_error
        total_l2 += error
        
        if i % 1000 == 0:
            avg_err = total_l2 / 1000.0 if i > 0 else error
            print(f"Stream {i:05d} | Avg Causal Surprise (Error): {avg_err:.5f}")
            total_l2 = 0.0
            
    out_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "causal_brain")
    os.makedirs(out_dir, exist_ok=True)
    b.save_components(out_dir)
    print(f"\nPhase 2 Complete! Causal Brain saved to {out_dir}")

if __name__ == "__main__":
    teach_causality()
