import os, sys, json
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
        episodic_path=f"{ckpt_dir}/episodic.bin",
        emotion_path=f"{ckpt_dir}/emotion.bin",
        self_path=f"{ckpt_dir}/self.bin",
        symbolic_path=f"{ckpt_dir}/symbolic.bin",
        binding_path=f"{ckpt_dir}/binding.bin",
        procedures_path=f"{ckpt_dir}/procedures.bin",
        hpred_path=f"{ckpt_dir}/hpred.bin"
    )
    b.load_bg(f"{ckpt_dir}/bg.bin")

    corpus_path = "data/squad_qa.json"
    print(f"Loading {corpus_path}...")
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    print("\nStarting evaluation of 100 questions...\n")
    correct = 0
    with open("100_answers_squad_final.log", "w") as out:
        for i in range(100):
            pair = corpus[i]
            q = pair["input"]
            target = pair["target"]
            
            b.reset_sequence()
            
            # 1. Perceive the question context
            b.perceive_text(q)
            
            # 2. Force the Reasoning Engine to execute OP_RETRIEVE (index 6)
            # This directly triggers Episodic Memory search using the working memory context
            b.force_reason_step(6, "retrieve")
            
            # 3. Force OP_SPEAK (index 15) to decode the retrieved payload into a word
            b.force_reason_step(15, "speak")
            
            spoken = b.get_spoken_words()
            first_word = spoken[-1] if spoken else ""
            
            # If Episodic Memory failed, first_word will be empty or "...". 
            # In that case, fallback to Imagination (LSTM Predictor)
            if first_word in ["", "..."]:
                res_fallback = b.think(1)
                first_word = res_fallback.words[0] if res_fallback.words else "..."
            
            # Predict the rest of the sequence up to the length of the target answer
            target_len = len(target.split())
            if target_len > 1:
                res2 = b.think(target_len)
                full_reply = first_word + " " + " ".join([w for w in res2.words if w and w != first_word])
            else:
                full_reply = first_word
                
            out.write(f"Q: {q}\nTarget: {target}\nBrain: {full_reply}\n{'-'*40}\n")
            print(f"[{i+1}/100] Q: {q} | Brain: {full_reply}")

if __name__ == "__main__":
    main()
