#!/usr/bin/env python3
"""
run_scorecard.py — brain2 baseline scorecard.

The single source of truth for "did this change help?". Run before and after
any architecture change; nothing merges if these numbers regress.

Metrics:
  1. lm_eval_ce / lm_eval_ppl  — cross-entropy on a held-out 10% split after
                                 one fused-training pass over the other 90%
  2. train_words_per_sec       — fused-path training throughput
  3. fact_retrieval_acc        — one-shot (subject, relation) -> object recall
                                 through BindingMemory, decoded by nearest
                                 embedding over the full vocabulary
  4. peak_rss_mb               — peak resident memory of the run

Writes scorecard.json. If scorecard_baseline.json exists, prints the delta;
otherwise saves this run as the baseline.
"""

import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

SEED = 42
N_DIMS = 64
SOM_ROWS = 128
SOM_COLS = 128
HIDDEN_DIM = 512
N_WORDS = 100_000         # corpus slice size (30k overfit within one epoch)
CHUNK_WORDS = 200         # words per fused-training segment
N_FACTS = 50              # one-shot triples for binding retrieval
HERE = os.path.dirname(os.path.abspath(__file__))


def load_corpus():
    path = os.path.join(HERE, "tiny_shakespeare.txt")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    words = [w.lower().strip(".,;:!?'\"()-") for w in text.split()]
    words = [w for w in words if w]
    return words[:N_WORDS]


def load_embeddings(vocab, rng):
    """GloVe-50 where available (padded to N_DIMS), seeded random otherwise."""
    emb = {w: rng.standard_normal(N_DIMS).astype(np.float32) * 0.3 for w in vocab}
    glove = os.path.join(HERE, "glove.6B.50d.txt")
    n_glove = 0
    if os.path.exists(glove):
        vocab_set = set(vocab)
        with open(glove, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts[0] in vocab_set:
                    v = np.zeros(N_DIMS, dtype=np.float32)
                    v[:50] = [float(x) for x in parts[1 : 51]]
                    emb[parts[0]] = v
                    n_glove += 1
    return emb, n_glove


def main():
    rng = np.random.default_rng(SEED)
    words = load_corpus()
    vocab = sorted(set(words))
    print(f"Corpus: {len(words)} words, vocab {len(vocab)}")

    emb, n_glove = load_embeddings(vocab, rng)
    print(f"Embeddings: {n_glove}/{len(vocab)} from GloVe, rest random (seeded)")

    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS,
                     n_dims=N_DIMS, hidden_dim=HIDDEN_DIM, seed=SEED)
    for w in vocab:
        b.language.register_word(w, emb[w])
    b.language.freeze_vocabulary()
    b.set_active_vocab([b.language.word_id(w) for w in vocab])

    # ── 1+2. LM training pass and held-out eval ──────────────────────────
    chunks = [" ".join(words[i : i + CHUNK_WORDS])
              for i in range(0, len(words), CHUNK_WORDS)]
    split = int(len(chunks) * 0.9)
    train_chunks, eval_chunks = chunks[:split], chunks[split:]

    n_train_words = sum(len(c.split()) for c in train_chunks)
    t0 = time.time()
    for c in train_chunks:
        b.reset_sequence()
        b.train_lm_sequence_fused(c)
    train_time = time.time() - t0
    words_per_sec = n_train_words / train_time

    b.predictor.set_offline(True)
    eval_ces = []
    for c in eval_chunks:
        b.reset_sequence()
        ce = b.train_lm_sequence(c)  # offline: forward + CE only, no updates
        if ce > 0:
            eval_ces.append(ce)
    b.predictor.set_offline(False)
    eval_ce = float(np.mean(eval_ces))
    eval_ppl = float(np.exp(min(eval_ce, 20)))

    # ── 3. One-shot fact retrieval through BindingMemory ─────────────────
    cand = rng.choice(vocab, size=(N_FACTS, 3))
    all_emb = np.stack([emb[w] for w in vocab])            # [V, D]
    all_norm = all_emb / (np.linalg.norm(all_emb, axis=1, keepdims=True) + 1e-8)
    correct = 0
    for s, r, o in cand:
        b.bind_triple(emb[s], emb[r], emb[o])
    for s, r, o in cand:
        vec, conf = b.binding_query(emb[s], emb[r], True, 0.0)
        vec = np.asarray(vec, dtype=np.float32)
        if vec.size == 0 or np.linalg.norm(vec) < 1e-8:
            continue
        sims = all_norm @ (vec / np.linalg.norm(vec))
        if vocab[int(np.argmax(sims))] == o:
            correct += 1
    fact_acc = correct / N_FACTS

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

    result = {
        "lm_eval_ce": round(eval_ce, 4),
        "lm_eval_ppl": round(eval_ppl, 1),
        "train_words_per_sec": round(words_per_sec, 0),
        "fact_retrieval_acc": round(fact_acc, 3),
        "peak_rss_mb": round(peak_rss_mb, 0),
        "config": {"n_dims": N_DIMS, "som": f"{SOM_ROWS}x{SOM_COLS}",
                   "hidden_dim": HIDDEN_DIM, "n_words": len(words),
                   "vocab": len(vocab), "seed": SEED},
    }

    print("\n──── SCORECARD ────")
    for k in ("lm_eval_ce", "lm_eval_ppl", "train_words_per_sec",
              "fact_retrieval_acc", "peak_rss_mb"):
        print(f"  {k:22s} {result[k]}")

    out = os.path.join(HERE, "scorecard.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {out}")

    baseline_path = os.path.join(HERE, "scorecard_baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            base = json.load(f)
        print("\n──── vs baseline ────")
        for k in ("lm_eval_ce", "lm_eval_ppl", "train_words_per_sec",
                  "fact_retrieval_acc", "peak_rss_mb"):
            if k in base:
                delta = result[k] - base[k]
                print(f"  {k:22s} {base[k]} → {result[k]}  ({delta:+.3f})")
    else:
        with open(baseline_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"No baseline found — saved this run as {baseline_path}")


if __name__ == "__main__":
    main()
