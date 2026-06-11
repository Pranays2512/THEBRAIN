#!/usr/bin/env python3
"""
chat_math.py — Interactive mathematical reasoning with the Brain.
Loads the mathematical checkpoints and evaluates deterministic logic.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def chat():
    N_DIMS = 128
    SOM_ROWS = 256
    SOM_COLS = 256
    HIDDEN_DIM = 256

    print(f"Initializing Math Brain (Dims: {N_DIMS}, SOM: {SOM_ROWS}x{SOM_COLS}, Hidden: {HIDDEN_DIM})...")
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS, hidden_dim=HIDDEN_DIM)
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "math_brain")
    if os.path.exists(ckpt_dir):
        print(f"Loading trained mathematical weights from {ckpt_dir}...")
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
            # Ensure symbolic math operators are seeded properly
            b.symbolic_table.seed_math_symbols()
            b.symbolic_table.bind("x", None, brain2.SymbolOp.NONE, "variable")
            b.symbolic_table.bind("eval", None, brain2.SymbolOp.SEQUENCE, "instruction")
            b.symbolic_table.bind("roots", None, brain2.SymbolOp.SEQUENCE, "instruction")
            
        except Exception as e:
            print(f"Failed to load checkpoints: {e}")
            return
    else:
        print("Checkpoints not found. Cannot start.")
        return

    print("\n=== INTERACTIVE MATHEMATICS MODE ===")
    print("Type equations like '1 + 1' or 'roots of x ^ 2 - 4 = 0'")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            user_in = input("\nYou: ").strip()
            if not user_in:
                continue
            if user_in.lower() in ["exit", "quit"]:
                break
                
            words = user_in.split()
            known = [w for w in words if b.language.knows(w) or b.symbolic_table.knows(w)]
            if not known:
                print("Brain: [I don't recognize any of those symbols.]")
                continue
            
            # 1. Perceive
            b.reset_sequence()
            for w in words:
                if b.language.knows(w):
                    b.perceive(b.language.encode(w))
            
            # 2. Load the operands into the C++ LogicEngine's memory
            if len(words) >= 3 and b.language.knows(words[0]) and b.language.knows(words[2]):
                b.scratchpad.write("subject", b.language.encode(words[0]), "math_arg")
                b.scratchpad.write("object", b.language.encode(words[2]), "math_arg")
                
            # Also write the full context (including the operator) to the scratchpad
            # so the Basal Ganglia can differentiate between + and -
            # We can just encode the operator itself and put it in 'relation'
            if len(words) >= 3 and b.language.knows(words[1]):
                b.scratchpad.write("relation", b.language.encode(words[1]), "math_arg")
            
            # 3. Activate the Zero-Hallucination Cognitive Loop
            b.start_reasoning()
            b.reason("reply", 5, 0.0)
            
            # 4. Speak
            result_vec = b.scratchpad.read("result")
            
            if len(result_vec) > 0:
                res_str = b.language.best_word(result_vec)
                print(f"Brain: {res_str}")
            else:
                # Fallback to probabilistic thinking if it couldn't map the logic
                res = b.think(6)
                out_words = [w for w in res.words if w]
                print(f"Brain (MDN Fallback): {' '.join(out_words)}")
            
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    chat()
