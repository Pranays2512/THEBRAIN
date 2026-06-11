#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

def test_sym():
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
    
    v1 = b.language.encode("2")
    v2 = b.language.encode("3")
    res_vec = b.symbolic_op("+", v1, v2)
    word = b.language.best_word(res_vec)
    print(f"2 + 3 = {word}")

if __name__ == "__main__":
    test_sym()
