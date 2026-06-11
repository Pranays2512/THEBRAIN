#!/usr/bin/env python3
"""
test_phase2.py — 100-Case Causal Reasoning Validation Suite

Evaluates if the Brain's Predictive Coding successfully unlearned hallucinations 
and properly anticipates logical cause-and-effect sequences.
"""

import os, sys, random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def test_causality():
    print("Loading Phase 2 Brain for Causal Validation...")
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "causal_brain")
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
        print(f"Error loading checkpoints: {e}")
        return

    causal_tests = [
        {"prompt": "fire burns wood to", "target": "ash"},
        {"prompt": "water extinguishes", "target": "fire"},
        {"prompt": "sun heats water to", "target": "steam"},
        {"prompt": "gravity pulls objects", "target": "down"},
        {"prompt": "ice melts into", "target": "water"},
        {"prompt": "rain makes ground", "target": "wet"},
        {"prompt": "seed grows into", "target": "tree"},
        {"prompt": "wind blows leaves", "target": "away"},
        {"prompt": "food gives energy to", "target": "animals"},
        {"prompt": "birds fly in", "target": "sky"}
    ]
    
    # We duplicate to get 100 tests to evaluate consistency and jitter
    test_cases = causal_tests * 10
    random.shuffle(test_cases)
    
    passed = 0
    
    print("\n--- Running 100-Case Causal Reasoning Test ---\n")
    
    for i, test in enumerate(test_cases):
        b.reset_sequence()
        prompt = test["prompt"]
        target = test["target"]
        
        b.perceive_text(prompt)
        
        # Test 1-word generation coherence
        res = b.think(1)
        generated = " ".join([w for w in res.words if w]).strip()
        
        if generated == target:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        print(f"Test {i+1:03d} | Prompt: [{prompt}] -> Predicted: [{generated}] (Target: {target}) | {status}")
        
    pass_rate = passed / 100.0
    
    print("\n--- Phase 2 Validation Complete ---")
    print(f"Pass Rate: {pass_rate * 100:.1f}%")
    
    if pass_rate >= 0.90:
        print("\nRESULT: Phase 2 VERIFIED. The Brain has internalized causal structures.")
    else:
        print("\nRESULT: Phase 2 FAILED. Predictive Coding did not eliminate hallucinations.")

if __name__ == "__main__":
    test_causality()
