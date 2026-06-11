import os, sys, json, time
import brain2

def main():
    print("Initializing Brain (Dims: 128, SOM: 256x256, Hidden: 256)...")
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    ckpt_dir = "checkpoints/massive_squad"
    
    print("Loading all Phase 1-3 checkpoints...")
    b.load_components(
        predictor_path=f"{ckpt_dir}/predictor.bin",
        language_path=f"{ckpt_dir}/language.bin",
        som_path=f"{ckpt_dir}/som.bin",
        episodic_path="", # Start fresh so we don't crash loading the old payload-less format
        emotion_path=f"{ckpt_dir}/emotion.bin",
        self_path=f"{ckpt_dir}/self.bin",
        symbolic_path=f"{ckpt_dir}/symbolic.bin",
        binding_path=f"{ckpt_dir}/binding.bin",
        procedures_path=f"{ckpt_dir}/procedures.bin",
        hpred_path=f"{ckpt_dir}/hpred.bin"
    )
    b.load_bg(f"{ckpt_dir}/bg.bin")

    corpus_path = "data/squad_qa.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    # We only need to store the episodes we're going to test, plus some extra.
    # Storing 1000 facts to prove the architecture works
    subset = corpus[:1000]
    print(f"Storing {len(subset)} SQuAD facts into Episodic Memory payloads...")
    
    start = time.time()
    for i, pair in enumerate(subset):
        b.reset_sequence()
        q = pair["input"]
        target = pair["target"]
        
        # 1. Perceive the question (creates a unique episodic trajectory in the SOM)
        b.perceive_text(q)
        
        # 2. Encode the target answer (if multi-word, just take the first important word for now to prove retrieval)
        target_word = target.split()[0]
        target_vec = b.language.encode(target_word)
        
        # 3. Commit the episode with a high error (1.0 forces storage) and the answer as the payload
        res = b.commit_episode(1.0, target_vec)
        if not res:
            print(f"WARNING: Failed to commit episode for question {i}")
        
        if (i+1) % 100 == 0:
            print(f"Stored {i+1} facts...")
            
    print(f"Completed in {time.time() - start:.2f}s")
    b.save_components(ckpt_dir)
    print("Episodic Memory updated and saved!")

if __name__ == "__main__":
    main()
