#!/usr/bin/env python3
"""
chat_fluency.py — Interactive Chat with the Fluency-Trained 128D Brain
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def main():
    print("Initializing Brain (Dims: 128, SOM: 256x256, Hidden: 256)...")
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "fluent_squad")
    if os.path.exists(ckpt_dir):
        print(f"Loading trained neural fluency weights from {ckpt_dir}...")
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
            print("Could not load all components, starting fresh. Error:", e)
    else:
        print(f"Checkpoint directory {ckpt_dir} not found. Brain is untrained.")
        
    print("\n=== INTERACTIVE CHAT MODE ===")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            user_in = input("\nYou: ").strip().lower()
            if user_in in ["exit", "quit", "q"]:
                break
            if not user_in:
                continue
                
            b.reset_sequence()
            words = user_in.split()
            for w in words:
                if b.language.knows(w):
                    b.perceive(b.language.encode(w))
                else:
                    # On-the-fly zero-shot learning of new words
                    b.language.register_word(w)
                    b.symbolic_table.bind(w)
                    b.perceive(b.language.encode(w))
                    
            # Let it think and predict the semantic continuation
            reply = b.think(4) 
            
            clean_reply = []
            for w in reply.words:
                if w and (not clean_reply or clean_reply[-1] != w):
                    clean_reply.append(w)
            
            print("Brain:", " ".join(clean_reply))
            
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
