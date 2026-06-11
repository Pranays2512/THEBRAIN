#!/usr/bin/env python3
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
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "fluent_brain")
    print(f"Loading fluent_brain checkpoints from {ckpt_dir}...")
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

    print("\n" + "="*40)
    print("TEST 1: CONVERSATIONAL FLUENCY")
    print("="*40)
    fluency_prompts = [
        "hello there",
        "how are you",
        "who are you",
        "what do you think of space",
        "do you sleep"
    ]
    for p in fluency_prompts:
        b.reset_sequence()
        b.perceive_text(p)
        res = b.think(6)
        reply = " ".join([w for w in res.words if w])
        print(f"User:  {p}\nBrain: {reply}\n")

    print("="*40)
    print("TEST 2: EPISODIC FACTS & LOGIC BLEND")
    print("="*40)
    
    corpus_path = "data/squad_qa.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
        
    for i in range(10): # Test 10 random facts
        pair = corpus[i]
        q = pair["input"]
        target = pair["target"]
        
        b.reset_sequence()
        b.perceive_text(q)
        
        # Episodic Retrieval
        b.force_reason_step(6, "retrieve")
        b.force_reason_step(15, "speak")
        
        spoken = b.get_spoken_words()
        fact_word = spoken[-1] if spoken else ""
        
        # Logic Blend: Perceive the fact, then think
        b.perceive_text(fact_word)
        res = b.think(4)
        rest = " ".join([w for w in res.words if w])
        
        print(f"Q:      {q}")
        print(f"Target: {target}")
        print(f"Brain:  {fact_word} {rest}\n")

if __name__ == "__main__":
    main()
