#!/usr/bin/env python3
"""
run_scorecard.py — brain2 baseline scorecard.

The single source of truth for "did this change help?". Run before and after
any architecture change; nothing merges if these numbers regress.

  TOKENIZER=word   (default) whitespace word-level
  TOKENIZER=bpe              hybrid: known words whole, OOV -> BPE pieces

Metrics (tokenizer-INVARIANT ones first, so word vs bpe are comparable):
  1. bits_per_char       — total held-out NLL / ln(2) / chars. The honest
                           cross-tokenizer language-modeling number.
  2. nll_per_word        — total held-out NLL / held-out words
  3. lm_eval_ppl_tok     — exp(NLL/token); tokenizer-DEPENDENT, kept for continuity
  4. oov_gen_bits        — bits/char on a held-out set of words never seen in
                           training (BPE should beat word-level: shared pieces
                           vs a single unseen random row)
  5. train_words_per_sec — fused-path throughput (source words/sec, comparable)
  6. fact_retrieval_acc  — one-shot (subject,relation)->object via BindingMemory
  7. peak_rss_mb

Writes scorecard.json; prints delta vs scorecard_baseline.json if present.
"""

import json
import math
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2
from bpe import BPE

SEED = 42
N_DIMS = 64
SOM_ROWS = 128
SOM_COLS = 128
HIDDEN_DIM = 512
N_WORDS = 100_000
CHUNK_WORDS = 200
N_FACTS = 50
N_BPE_MERGES = 2000
TOKENIZER = os.environ.get("TOKENIZER", "word")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_corpus_words():
    with open(os.path.join(HERE, "tiny_shakespeare.txt"), encoding="utf-8") as f:
        text = f.read()
    words = [w.lower().strip(".,;:!?'\"()-") for w in text.split()]
    return [w for w in words if w][:N_WORDS]


def load_glove(vocab_set):
    """Return {word: 50-d-padded-to-N_DIMS vector} for GloVe words in vocab."""
    out = {}
    path = os.path.join(HERE, "glove.6B.50d.txt")
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts[0] in vocab_set:
                v = np.zeros(N_DIMS, dtype=np.float32)
                v[:50] = [float(x) for x in parts[1:51]]
                out[parts[0]] = v
    return out


def seeded_vec(token, rng_scale=0.3):
    """Deterministic frozen code for a token (word piece or OOV word)."""
    h = abs(hash(token)) % (2**32)
    return (np.random.default_rng(h).standard_normal(N_DIMS) * rng_scale).astype(np.float32)


def register_token_vocab(b, tokens, glove):
    for t in tokens:
        if b.language.knows(t):
            continue
        b.language.register_word(t, glove.get(t, seeded_vec(t)))


def main():
    rng = np.random.default_rng(SEED)
    words = load_corpus_words()
    n_chars = sum(len(w) + 1 for w in words)  # +1 approximates whitespace
    word_vocab = sorted(set(words))
    glove = load_glove(set(word_vocab))
    print(f"Corpus: {len(words)} words, {n_chars} chars, vocab {len(word_vocab)}")
    print(f"GloVe coverage: {len(glove)}/{len(word_vocab)}")
    print(f"Tokenizer: {TOKENIZER}")

    # keep_whole = words we trust to whole-token (GloVe-grounded). For the OOV
    # probe we also hold out a slice of GloVe words entirely from training.
    held_out_words = set(rng.choice(sorted(glove), size=min(300, len(glove)),
                                    replace=False)) if glove else set()
    keep_whole = set(glove) - held_out_words

    bpe = None
    if TOKENIZER == "bpe":
        counts = {}
        for w in words:
            if w not in keep_whole:
                counts[w] = counts.get(w, 0) + 1
        bpe = BPE()
        bpe.train(counts, N_BPE_MERGES)

    def tokenize(text):
        if bpe is None:
            return text.split()
        return bpe.tokenize(text, keep_whole)

    # Build the full token vocabulary the model will see. held_out_words are
    # deliberately NOT registered — they model open-vocabulary input the frozen
    # brain has never been told about. In word mode they are unknown and get
    # silently skipped (zero coverage); in BPE mode they decompose into
    # registered pieces and stay scorable. This is the structural difference.
    tok_words = tokenize(" ".join(words))
    tok_vocab = sorted(t for t in set(tok_words) if t not in held_out_words)

    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS,
                     n_dims=N_DIMS, hidden_dim=HIDDEN_DIM, seed=SEED)
    register_token_vocab(b, tok_vocab, glove)
    b.language.freeze_vocabulary()
    b.set_active_vocab([b.language.word_id(t) for t in tok_vocab])
    print(f"Token vocab: {len(tok_vocab)}")

    # ── train / eval split on the source word stream ─────────────────────
    chunks = [words[i:i + CHUNK_WORDS] for i in range(0, len(words), CHUNK_WORDS)]
    split = int(len(chunks) * 0.9)
    train_chunks, eval_chunks = chunks[:split], chunks[split:]
    n_train_words = sum(len(c) for c in train_chunks)

    t0 = time.time()
    for c in train_chunks:
        b.reset_sequence()
        b.train_lm_sequence_fused(" ".join(tokenize(" ".join(c))))
    train_time = time.time() - t0
    words_per_sec = n_train_words / train_time

    def held_out_nll(chunk_list):
        total_nll, total_tok, total_chars, total_words = 0.0, 0, 0, 0
        for c in chunk_list:
            b.reset_sequence()
            nll, ntok = b.eval_text_nll(" ".join(tokenize(" ".join(c))))
            total_nll += nll
            total_tok += ntok
            total_chars += sum(len(w) + 1 for w in c)
            total_words += len(c)
        return total_nll, total_tok, total_chars, total_words

    nll, ntok, nchar, nword = held_out_nll(eval_chunks)
    bits_per_char = nll / math.log(2) / max(nchar, 1)
    nll_per_word = nll / max(nword, 1)
    ppl_tok = math.exp(min(nll / max(ntok, 1), 20))

    # ── open-vocab coverage probe ────────────────────────────────────────
    # Sentences seeded with held_out_words (never registered). Coverage =
    # scorable input tokens / total input tokens. Word mode skips unknowns
    # (low coverage = silently dropped text); BPE decomposes them (~full).
    oov_coverage = float("nan")
    if held_out_words:
        hw = sorted(held_out_words)
        scorable, total = 0, 0
        for _ in range(50):
            sent = list(rng.choice(hw, size=12))
            toks = tokenize(" ".join(sent))
            b.reset_sequence()
            _, ntok = b.eval_text_nll(" ".join(toks))
            scorable += ntok + 1            # +1: last token has no target to score
            total += len(toks)
        oov_coverage = scorable / max(total, 1)

    # ── one-shot fact retrieval (whole-word GloVe vectors) ───────────────
    fact_vocab = sorted(glove) if glove else word_vocab
    emb = {w: (glove.get(w) if w in glove else seeded_vec(w)) for w in fact_vocab}
    all_emb = np.stack([emb[w] for w in fact_vocab])
    all_norm = all_emb / (np.linalg.norm(all_emb, axis=1, keepdims=True) + 1e-8)
    cand = rng.choice(fact_vocab, size=(N_FACTS, 3))
    for s, r, o in cand:
        b.bind_triple(emb[s], emb[r], emb[o])
    correct = 0
    for s, r, o in cand:
        vec, _ = b.binding_query(emb[s], emb[r], True, 0.0)
        vec = np.asarray(vec, dtype=np.float32)
        if vec.size and np.linalg.norm(vec) > 1e-8:
            sims = all_norm @ (vec / np.linalg.norm(vec))
            if fact_vocab[int(np.argmax(sims))] == o:
                correct += 1
    fact_acc = correct / N_FACTS

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

    result = {
        "tokenizer": TOKENIZER,
        "bits_per_char": round(bits_per_char, 4),
        "nll_per_word": round(nll_per_word, 4),
        "lm_eval_ppl_tok": round(ppl_tok, 1),
        "oov_coverage": round(oov_coverage, 4) if oov_coverage == oov_coverage else None,
        "train_words_per_sec": round(words_per_sec, 0),
        "fact_retrieval_acc": round(fact_acc, 3),
        "peak_rss_mb": round(peak_rss_mb, 0),
        "config": {"n_dims": N_DIMS, "hidden_dim": HIDDEN_DIM,
                   "n_words": len(words), "token_vocab": len(tok_vocab),
                   "bpe_merges": N_BPE_MERGES if bpe else 0, "seed": SEED},
    }

    print("\n──── SCORECARD ────")
    keys = ["bits_per_char", "nll_per_word", "lm_eval_ppl_tok", "oov_coverage",
            "train_words_per_sec", "fact_retrieval_acc", "peak_rss_mb"]
    for k in keys:
        print(f"  {k:22s} {result[k]}")

    with open(os.path.join(HERE, "scorecard.json"), "w") as f:
        json.dump(result, f, indent=2)

    base_path = os.path.join(HERE, "scorecard_baseline.json")
    if os.path.exists(base_path):
        with open(base_path) as f:
            base = json.load(f)
        print(f"\n──── vs baseline (tokenizer={base.get('tokenizer','?')}) ────")
        for k in keys:
            if base.get(k) is not None and result[k] is not None:
                print(f"  {k:22s} {base[k]} → {result[k]}  ({result[k]-base[k]:+.4f})")
    else:
        with open(base_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nNo baseline — saved this run as {base_path}")


if __name__ == "__main__":
    main()
