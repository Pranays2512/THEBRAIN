import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

def main():
    print("Initializing Brain...")
    N_DIMS = 128
    SOM_ROWS = 256
    SOM_COLS = 256
    HIDDEN_DIM = 256
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS, hidden_dim=HIDDEN_DIM)
    
    ckpt_dir = "checkpoints/fluent_brain"
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
    
    with open("data/squad_qa.json", "r") as f:
        corpus = json.load(f)
        
    correct = 0
    total = 100
    for i in range(100):
        pair = corpus[i]
        q = pair["input"]
        target = pair["target"].split()[0].lower() # First word of target
        
        b.reset_sequence()
        b.perceive_text(q)
        
        # Suppress stdout to avoid spam
        # Actually we just won't print the steps
        b.force_reason_step(6, "retrieve")
        b.force_reason_step(15, "speak")
        
        spoken = b.get_spoken_words()
        fact_word = spoken[-1] if spoken else ""
        
        if fact_word == target:
            correct += 1
            
    print(f"\nFinal Score: {correct} out of {total} correct ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    main()
