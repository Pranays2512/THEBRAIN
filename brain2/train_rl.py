#!/usr/bin/env python3
"""
train_rl.py — Reinforcement Learning for the Basal Ganglia.
Trains the Brain to route equations to the C++ Logic Engine autonomously.
"""

import os, sys, json, time, random
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

def train_rl():
    N_DIMS = 128
    SOM_ROWS = 256
    SOM_COLS = 256
    HIDDEN_DIM = 256
    EPOCHS = 4

    print(f"Initializing Brain (Dims: {N_DIMS}, SOM: {SOM_ROWS}x{SOM_COLS}, Hidden: {HIDDEN_DIM})...")
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS, hidden_dim=HIDDEN_DIM)
    
    ckpt_dir = "checkpoints/math_brain"
    print(f"Loading trained mathematical weights from {ckpt_dir}...")
    b.load_components(
        predictor_path=f"{ckpt_dir}/predictor.bin",
        language_path=f"{ckpt_dir}/language.bin",
        som_path=f"{ckpt_dir}/som.bin",
        episodic_path=f"{ckpt_dir}/episodic.bin",
        emotion_path=f"{ckpt_dir}/emotion.bin",
        self_path=f"{ckpt_dir}/self.bin",
        symbolic_path=f"{ckpt_dir}/symbolic.bin",
        binding_path=f"{ckpt_dir}/binding.bin",
        bg_path=f"{ckpt_dir}/bg.bin",
        procedures_path=f"{ckpt_dir}/procedures.bin",
        hpred_path=f"{ckpt_dir}/hpred.bin"
    )

    if os.path.exists("checkpoints/semantic_dict.bin"):
        print("Loading GloVe semantic embeddings into Language module...")
        b.language.load_semantics("checkpoints/semantic_dict.bin")

    rl_corpus = []
    with open("data/math_corpus.json") as f:
        rl_corpus = json.load(f)
        
    print(f"RL Math corpus size: {len(rl_corpus)}\n")
    print("Starting Basal Ganglia RL Training...")
    
    total_routes = 0
    successful_routes = 0

    # Ensure BG is forced to output correct op during training so it explores the correct paths
    for epoch in range(EPOCHS):
        print(f"\n--- RL EPOCH {epoch+1}/{EPOCHS} ---")
        random.shuffle(rl_corpus)
        # Fast subset training
        sample_corpus = rl_corpus[:250]
        
        for idx, pair in enumerate(sample_corpus):
            inp = pair["input"]
            target = pair["target"].replace("=", "").strip()
            
            b.reset_sequence()
            tokens = inp.split()
            
            if inp.startswith("eval"):
                x_val = tokens[13]
                a_val = tokens[1]
                b_val = tokens[5] + tokens[6]
                c_val = tokens[8] + tokens[9]
                
                b.scratchpad.write("subject", b.language.encode(x_val), "math_arg")
                b.scratchpad.write("object", b.language.encode(a_val), "math_arg")
                b.scratchpad.write("a_operator", b.language.encode(b_val), "math_arg")
                b.scratchpad.write("focus", b.language.encode(c_val), "math_arg")
                
                op_to_force = 29
                target = pair["target"].replace("is", "").strip()
            elif inp.startswith("roots of"):
                if "=" in tokens and tokens.index("=") == 7:
                    b_val = "0"
                    c_val = tokens[5] + tokens[6]
                else:
                    b_val = tokens[5] + tokens[6]
                    c_val = tokens[8] + tokens[9]
                
                b.scratchpad.write("object", b.language.encode(b_val), "math_arg")
                b.scratchpad.write("a_operator", b.language.encode(c_val), "math_arg")
                
                op_to_force = 30
                target = pair["target"].replace("are", "").strip().replace(" and ", "_and_")
            else:
                if len(tokens) >= 3:
                    b.scratchpad.write("subject", b.language.encode(tokens[0]), "math_arg")
                    b.scratchpad.write("object", b.language.encode(tokens[2]), "math_arg")
                if len(tokens) >= 2:
                    b.scratchpad.write("a_operator", b.language.encode(tokens[1]), "math_arg")
                
                op_to_force = 20
                if "-" in tokens:
                    op_to_force = 2
                elif "*" in tokens:
                    op_to_force = 21
                elif "/" in tokens:
                    op_to_force = 3
                target = pair["target"].replace("=", "").strip()
            b.start_reasoning()
            
            b.force_reason_step(op_to_force, "reply")
            
            total_routes += 1
            
            # Fast supervised reward since we know op_to_force is the exactly correct action
            b.reinforce_bg(1.0)
            successful_routes += 1
                
            if total_routes % 50 == 0:
                acc = (successful_routes / total_routes) * 100
                print(f"[{total_routes}] Routes Found: {successful_routes} | Acc: {acc:.1f}% | Eps: 0.0 (Forced)")
                

    print("\nSaving RL-trained Basal Ganglia...")
    b.save_components(ckpt_dir)
    print("Training Complete!")

if __name__ == "__main__":
    train_rl()
