#!/usr/bin/env python3
"""
train_and_eval_squad.py — real multi-epoch SQuAD run with the fixed pipeline.

Trains on data/squad_train_80k.json, evaluates held-out bits/char and
per-token CE on data/squad_test_6k.json after every epoch. Demonstrates the
data-scarcity hypothesis: with ~30x more data than the scorecard slice, the
train/eval gap should shrink versus the 30k-word case.

Env knobs (all optional):
  EPOCHS=3  TRAIN_PAIRS=0(all)  INPUT_DROPOUT=0  LSTM_WD=0  TOKENIZER=word|bpe

Writes checkpoints to checkpoints/squad_run/ and a JSON log of per-epoch
metrics to squad_run_log.json.
"""

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2
from bpe import BPE

HERE = os.path.dirname(os.path.abspath(__file__))
N_DIMS = 64
SOM_ROWS = SOM_COLS = 128
HIDDEN_DIM = 512
SEED = 42
EPOCHS = int(os.environ.get("EPOCHS", "3"))
TRAIN_PAIRS = int(os.environ.get("TRAIN_PAIRS", "0"))      # 0 = all
INPUT_DROPOUT = float(os.environ.get("INPUT_DROPOUT", "0"))
LSTM_WD = float(os.environ.get("LSTM_WD", "0"))
TOKENIZER = os.environ.get("TOKENIZER", "word")
N_BPE_MERGES = 4000


def load(path):
    with open(os.path.join(HERE, "data", path)) as f:
        return json.load(f)


def pair_words(p):
    return p.get("input", "").split() + p.get("target", "").split()


def main():
    train = load("squad_train_80k.json")
    test = load("squad_test_6k.json")
    if TRAIN_PAIRS:
        train = train[:TRAIN_PAIRS]
    print(f"train pairs {len(train)}, test pairs {len(test)}, "
          f"epochs {EPOCHS}, tokenizer {TOKENIZER}, "
          f"dropout {INPUT_DROPOUT}, lstm_wd {LSTM_WD}", flush=True)

    # Vocabulary over all source words.
    word_vocab = set()
    for p in train + test:
        word_vocab.update(pair_words(p))

    # GloVe where available.
    glove = {}
    gpath = os.path.join(HERE, "glove.6B.50d.txt")
    if os.path.exists(gpath):
        with open(gpath, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts[0] in word_vocab:
                    v = np.zeros(N_DIMS, dtype=np.float32)
                    v[:50] = [float(x) for x in parts[1:51]]
                    glove[parts[0]] = v
    keep_whole = set(glove)
    print(f"word vocab {len(word_vocab)}, GloVe {len(glove)}", flush=True)

    bpe = None
    if TOKENIZER == "bpe":
        counts = {}
        for p in train:
            for w in pair_words(p):
                if w not in keep_whole:
                    counts[w] = counts.get(w, 0) + 1
        bpe = BPE()
        bpe.train(counts, N_BPE_MERGES)

    def toks(text):
        return text.split() if bpe is None else bpe.tokenize(text, keep_whole)

    def seeded_vec(t):
        h = abs(hash(t)) % (2**32)
        return (np.random.default_rng(h).standard_normal(N_DIMS) * 0.3).astype(np.float32)

    b = brain2.Brain(som_rows=SOM_ROWS, som_cols=SOM_COLS,
                     n_dims=N_DIMS, hidden_dim=HIDDEN_DIM, seed=SEED)
    b.predictor.input_dropout = INPUT_DROPOUT
    b.predictor.lstm_weight_decay = LSTM_WD

    # Build + register the token vocabulary.
    tok_vocab = set()
    for p in train + test:
        for field in ("input", "target"):
            tok_vocab.update(toks(p.get(field, "")))
    for t in sorted(tok_vocab):
        b.language.register_word(t, glove.get(t, seeded_vec(t)))
    b.language.freeze_vocabulary()
    b.set_active_vocab([b.language.word_id(t) for t in sorted(tok_vocab)])
    print(f"token vocab {len(tok_vocab)}", flush=True)

    ckpt = os.path.join(HERE, "checkpoints", "squad_run")
    os.makedirs(ckpt, exist_ok=True)

    def evaluate():
        b.predictor.set_offline(True)
        nll, chars, ntok = 0.0, 0, 0
        for p in test:
            for field in ("input", "target"):
                txt = p.get(field, "")
                if not txt:
                    continue
                b.reset_sequence()
                n, k = b.eval_text_nll(" ".join(toks(txt)))
                nll += n
                ntok += k
                chars += len(txt) + 1
        b.predictor.set_offline(False)
        return (nll / math.log(2) / max(chars, 1),     # bits/char
                math.exp(min(nll / max(ntok, 1), 20)))  # ppl/token

    log = []
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_ce, n = 0.0, 0
        for i, p in enumerate(train):
            b.reset_sequence()
            for field in ("input", "target"):
                txt = p.get(field, "")
                if txt:
                    ce = b.train_lm_sequence_fused(" ".join(toks(txt)))
                    if ce > 0:
                        tr_ce += ce
                        n += 1
            if (i + 1) % 5000 == 0:
                print(f"  epoch {epoch} {i+1}/{len(train)} "
                      f"train_ce {tr_ce/max(n,1):.3f} "
                      f"({(i+1)/(time.time()-t0):.0f} pairs/s)", flush=True)
        bpc, ppl = evaluate()
        rec = {"epoch": epoch, "train_ce": round(tr_ce / max(n, 1), 4),
               "eval_bits_per_char": round(bpc, 4), "eval_ppl_tok": round(ppl, 1),
               "epoch_sec": round(time.time() - t0, 0)}
        log.append(rec)
        print(f"EPOCH {epoch}: train_ce {rec['train_ce']} | "
              f"eval bits/char {rec['eval_bits_per_char']} | "
              f"eval ppl/tok {rec['eval_ppl_tok']} | {rec['epoch_sec']}s", flush=True)
        b.save_components(ckpt)
        with open(os.path.join(HERE, "squad_run_log.json"), "w") as f:
            json.dump(log, f, indent=2)

    print(f"DONE in {time.time()-t_start:.0f}s. Log -> squad_run_log.json", flush=True)


if __name__ == "__main__":
    main()
