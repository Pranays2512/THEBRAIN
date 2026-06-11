#!/usr/bin/env python3
"""
test_phase1.py — 100-Case Fluency Validation Suite

Evaluates the Brain's raw topological prediction accuracy on 100 novel and familiar conversational prompts.
"""

import os, sys, json, random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def test_fluency():
    print("Loading Phase 1 Brain for 100-Case Validation...")
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
        print(f"Error loading checkpoints: {e}")
        return

    corpus_path = os.path.join(os.path.dirname(__file__), "data", "squad_qa.json")
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
        
    conv_path = os.path.join(os.path.dirname(__file__), "data", "conversational_corpus.json")
    with open(conv_path, "r") as f:
        conv = json.load(f)
        
    corpus.extend(conv)
    random.shuffle(corpus)
    
    test_cases = corpus[:100]
    
    passed = 0
    total_l2 = 0.0
    
    print("\n--- Running 100-Case Fluency Test ---\n")
    
    for i, test in enumerate(test_cases):
        b.reset_sequence()
        prompt = test["input"]
        
        # Feed the prompt
        b.perceive_text(prompt)
        
        # Test generation coherence and error
        res = b.think(4)
        error = b.predictor.last_error
        
        total_l2 += error
        
        # Criteria for passing: L2 error must be low, meaning the brain wasn't totally confused
        if error < 0.25:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
            
        print(f"Test {i+1:03d} | L2 Error: {error:.4f} | {status} | Prompt: {prompt[:30]}... -> {' '.join([w for w in res.words if w])}")
        
    avg_l2 = total_l2 / 100.0
    pass_rate = passed / 100.0
    
    print("\n--- Phase 1 Validation Complete ---")
    print(f"Pass Rate: {pass_rate * 100:.1f}%")
    print(f"Average L2 Prediction Error: {avg_l2:.4f}")
    
    if pass_rate >= 0.85:
        print("\nRESULT: Phase 1 VERIFIED. The Brain is topologically fluent.")
    else:
        print("\nRESULT: Phase 1 FAILED. More unsupervised training required.")

if __name__ == "__main__":
    test_fluency()
