#!/usr/bin/env python3
"""
train_massive_corpus.py — Unsupervised Predictor Training on Massive SQuAD corpus
"""

import os, sys, time, json, gc
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

# Training Hyperparameters
# N_DIMS: embedding dim. GloVe vectors are 50-d; 64 leaves a little headroom
# and keeps SIMD-friendly alignment. The old value (512) zero-padded 462 dims,
# making every SOM scan / LM-head sgemm ~8x more expensive for no information.
N_DIMS = 64
SOM_ROWS = 128
SOM_COLS = 128
HIDDEN_DIM = 512
EPOCHS = 1
SAVE_INTERVAL = 10000 # Save every 10,000 QA pairs

def safe_register_word(b, word):
    """Register word if out-of-vocabulary."""
    if not b.language.knows(word):
        b.language.register_word(word)
        b.symbolic_table.bind(word)

def train():
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "squad_train_80k.json")
    if not os.path.exists(corpus_path):
        print(f"Error: {corpus_path} not found.")
        return
        
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} Q&A pairs.")
    
    print(f"Initializing Brain (Dims: {N_DIMS}, SOM: {SOM_ROWS}x{SOM_COLS}, Hidden: {HIDDEN_DIM})...")
    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS, n_dims=N_DIMS, hidden_dim=HIDDEN_DIM)
    
    glove_txt = os.path.join(os.path.dirname(__file__), "glove.6B.50d.txt")
    if os.path.exists(glove_txt):
        print("Loading GloVe Semantic Embeddings...")
        import numpy as np
        with open(glove_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                if not word.isalpha(): continue
                vec50 = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                vec = np.zeros(N_DIMS, dtype=np.float32)
                vec[:min(50, N_DIMS)] = vec50[:min(50, N_DIMS)]
                b.language.register_word(word, vec)
    
    # Freeze vocabulary to prevent semantic drift/catastrophic forgetting
    b.language.freeze_vocabulary()
    print("Vocabulary frozen.")
    
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "executive_brain")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    start_time = time.time()
    total_processed = 0
    start_epoch = 1
    
    state_path = os.path.join(ckpt_dir, "training_state.json")
    if os.path.exists(os.path.join(ckpt_dir, "predictor.bin")):
        print(f"Loading existing checkpoints from {ckpt_dir}...")
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
            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    state = json.load(f)
                    total_processed = state.get("total_processed", 0)
                    start_epoch = state.get("epoch", 1)
                print(f"Resuming from Epoch {start_epoch}, Total Processed: {total_processed}")
        except Exception as e:
            print(f"Failed to load checkpoints: {e}")

    print("Extracting active corpus vocabulary for LM Softmax head...")
    corpus_vocab = set()
    for pair in corpus:
        for w in pair["input"].split():
            corpus_vocab.add(w)
        for w in pair["target"].split():
            corpus_vocab.add(w)
    
    active_indices = []
    for w in corpus_vocab:
        idx = b.language.word_id(w)
        if idx >= 0:
            active_indices.append(idx)
            
    print(f"Corpus vocab size: {len(corpus_vocab)}. Active indices found: {len(active_indices)}")
    b.set_active_vocab(active_indices)

    ewma_ce = -1.0   # initialized on first valid CE sample
    start_pair = total_processed  # for ms/pair calculation

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n--- EPOCH {epoch}/{EPOCHS} ---")
        
        for i, pair in enumerate(corpus):
            b.reset_sequence()
            
            # Fused pass: LM training + cognitive pipeline in one LSTM forward per segment
            # Real per-token CE drives sigma-gated WM and episodic memory
            ce_in  = b.train_lm_sequence_fused(pair["input"])
            ce_tgt = b.train_lm_sequence_fused(pair["target"])
            
            total_processed += 1
            
            # EWMA of CE (alpha=0.002, ~500-pair smoothing window)
            pair_ce = (ce_in + ce_tgt) / 2.0 if (ce_in > 0 or ce_tgt > 0) else None
            if pair_ce is not None:
                if ewma_ce < 0:
                    ewma_ce = pair_ce  # initialize
                else:
                    ewma_ce = 0.998 * ewma_ce + 0.002 * pair_ce
            
            if total_processed % 50 == 0:
                import math
                elapsed = time.time() - start_time
                avg_ce = (ce_in + ce_tgt) / 2.0
                window_ppl = math.exp(min(avg_ce, 20))
                ewma_ppl = math.exp(min(ewma_ce, 20)) if ewma_ce >= 0 else float('nan')
                print(f"Processed {total_processed} pairs | Elapsed: {elapsed:.1f}s | {elapsed/max(total_processed-start_pair,1)*1000:.0f}ms/pair", flush=True)
                print(f"  Window CE: {avg_ce:.4f} | Window PPL: {window_ppl:.1f} | EWMA CE: {ewma_ce:.4f} | EWMA PPL: {ewma_ppl:.1f}", flush=True)
                print(b.get_profiling_report(), flush=True)
            if total_processed % 1000 == 0:
                b.som.prune_dead_branches(10000)
                
            if total_processed > 0 and total_processed % SAVE_INTERVAL == 0:
                print(f"Saving checkpoint to {ckpt_dir}...", flush=True)
                b.save_components(ckpt_dir)
                with open(state_path, "w") as f:
                    json.dump({"epoch": epoch, "total_processed": total_processed}, f)
                gc.collect() # Force garbage collection to prevent memory ballooning
                
                # Run a quick generation test
                b.reset_sequence()
                test_prompt = "what is the capital of"
                for tw in test_prompt.split():
                    if b.language.knows(tw):
                        b.perceive(b.language.encode(tw))
                res = b.think(4)
                print(f"  Test [{test_prompt}] -> {' '.join([w for w in res.words if w])}")

    b.save_components(ckpt_dir)
    print("Massive Training Complete.")

if __name__ == "__main__":
    train()
