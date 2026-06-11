#!/usr/bin/env python3
"""
test_phase3.py — 100-Case Executive Logic Validation Suite

Evaluates if the Brain's Actor-Critic Basal Ganglia successfully learned to 
consciously trigger exact C++ Mathematical and Analogical operations instead of guessing.
"""

import os, sys, random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

OP_MATH_ADD = 20
OP_MATH_SUB = 2
OP_ANALOGY  = 7

def test_executive():
    print("Loading Phase 3 Brain for Executive Logic Validation...")
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "executive_brain")
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

    # Generate 100 Test Cases
    tests = []
    
    for _ in range(33): # Math Add
        v1 = random.randint(1, 100)
        v2 = random.randint(1, 100)
        tests.append({"type": "ADD", "sub": str(v1), "obj": str(v2), "goal": "+", "target_op": OP_MATH_ADD, "ans": str(v1+v2)})
        
    for _ in range(33): # Math Sub
        v1 = random.randint(100, 200)
        v2 = random.randint(1, 100)
        tests.append({"type": "SUB", "sub": str(v1), "obj": str(v2), "goal": "-", "target_op": OP_MATH_SUB, "ans": str(v1-v2)})
        
    for _ in range(34): # Analogy
        tests.append({"type": "ANALOGY", "sub": "dog", "rel": "has", "ctx": "bird", "goal": "analogy", "target_op": OP_ANALOGY, "ans": "feathers"})

    random.shuffle(tests)
    
    passed = 0
    print("\n--- Running 100-Case Executive Logic Test ---\n")
    
    for i, test in enumerate(tests):
        b.reset_sequence()
        
        # Register words so Brain doesn't panic
        for w in [test["sub"], test.get("obj", ""), test.get("rel", ""), test.get("ctx", ""), test["goal"], test["ans"]]:
            if w and not b.language.knows(w):
                b.language.register_word(w)
                
        # Load Scratchpad Context
        b.scratchpad.write("subject", b.language.encode(test["sub"]), "ctx")
        if "obj" in test: b.scratchpad.write("object", b.language.encode(test["obj"]), "ctx")
        if "rel" in test: b.scratchpad.write("relation", b.language.encode(test["rel"]), "ctx")
        if "ctx" in test: b.scratchpad.write("context_map", b.language.encode(test["ctx"]), "ctx")
        
        # Trigger Actor-Critic to pick an Operation
        chosen_op = b.direct_reason_step(test["goal"])
        
        # Read the result from the scratchpad
        res_vec = b.scratchpad.read("result")
        
        if len(res_vec) == 0:
            ans = "<BLANK>"
        else:
            ans = b.language.best_word(res_vec)
            
        status = "FAIL"
        if chosen_op == test["target_op"] and ans == test["ans"]:
            status = "PASS"
            passed += 1
            
        # Optional: partial credit for correct op but wrong symbol math
        if chosen_op == test["target_op"] and ans != test["ans"]:
            status = "PARTIAL (Op correct, Math failed)"
            
        print(f"Test {i+1:03d} [{test['type']}] | Selected Op: {chosen_op} | Expected Op: {test['target_op']} | Output: {ans} (Expected: {test['ans']}) | {status}")
        
    pass_rate = passed / 100.0
    
    print("\n--- Phase 3 Validation Complete ---")
    print(f"Pass Rate: {pass_rate * 100:.1f}%")
    
    if pass_rate == 1.0:
        print("\nRESULT: Phase 3 VERIFIED. Executive Function is flawless. The Neuro-Symbolic integration is a complete success.")
    else:
        print("\nRESULT: Phase 3 FAILED. Basal Ganglia routing errors detected.")

if __name__ == "__main__":
    test_executive()
