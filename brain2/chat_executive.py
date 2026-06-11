#!/usr/bin/env python3
"""
chat_executive.py — Interactive Chat with the Massive 512D Executive Brain
"""

import os, sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def main():
    # Hyperparameters (MUST MATCH MASSIVE TRAINING)
    N_DIMS = 512
    SOM_ROWS = 512
    SOM_COLS = 512
    HIDDEN_DIM = 512
    
    print(f"Initializing Brain (Dims: {N_DIMS}, SOM: {SOM_ROWS}x{SOM_COLS}, Hidden: {HIDDEN_DIM})...")
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS, hidden_dim=HIDDEN_DIM)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "executive_brain")
    if os.path.exists(ckpt_dir):
        print(f"Loading trained neural weights from {ckpt_dir}...")
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
            print("Checkpoints loaded successfully.")
        except Exception as e:
            print("Could not load all components. Error:", e)
            sys.exit(1)
    else:
        print(f"Checkpoint directory {ckpt_dir} not found. Exiting.")
        sys.exit(1)
        
    print("\n=== INTERACTIVE MASSIVE BRAIN CHAT MODE ===")
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
                    # Zero-shot registration of OOV words
                    b.language.register_word(w)
                    b.symbolic_table.bind(w)
                    b.perceive(b.language.encode(w))
                    
            reply = b.think(6) 
            
            clean_reply = []
            for w in reply.words:
                if w and (not clean_reply or clean_reply[-1] != w):
                    clean_reply.append(w)
            
            print("Brain:", " ".join(clean_reply))
            
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
