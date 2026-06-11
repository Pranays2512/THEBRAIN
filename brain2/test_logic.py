#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

def test_logic():
    print("Loading Brain...")
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "math_brain")
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
    b.symbolic_table.seed_math_symbols()
    b.symbolic_table.bind("x", None, brain2.SymbolOp.NONE, "variable")
    b.symbolic_table.bind("eval", None, brain2.SymbolOp.SEQUENCE, "instruction")
    b.symbolic_table.bind("roots", None, brain2.SymbolOp.SEQUENCE, "instruction")

    tests = [
        "2 + 2",
        "5 - 3",
        "10 + 15",
        "eval 2 x ^ 2 + 3 x + 1 for x = 2",
        "roots of x ^ 2 - 4 = 0",
        "if 5 > 3 then"
    ]
    
    print("\n--- Testing Logic Engine (Zero-Hallucination) ---")
    for t in tests:
        res = b.cognitive_step(t)
        print(f"[{t}] => {res.strip()}")
        
if __name__ == "__main__":
    test_logic()
