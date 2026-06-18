#!/usr/bin/env python3
"""
test_consolidation.py — hardening test for Consolidation/dreaming (#7).

The consolidation lives in the training loop (auto-replay: the brain rehearses
recently-learned sequences while learning new ones). This pins the production
guarantee: with auto-replay ON, catastrophic forgetting of an earlier task is
substantially reduced when a new task interferes — and the new task is still
learned. Protocol: learn A -> learn B (interference) -> retest A.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import brain2

N_DIMS = 64
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def markov(words, rng, fan=3):
    return {w: list(rng.choice(words, size=fan, replace=False)) for w in words}


def corpus(words, m, n, length, rng):
    out = []
    for _ in range(n):
        w = rng.choice(words)
        seq = [w]
        for _ in range(length - 1):
            w = rng.choice(m[w]); seq.append(w)
        out.append(" ".join(seq))
    return out


def build(vocab, emb, auto):
    b = brain2.Brain(som_rows=64, som_cols=64, n_dims=N_DIMS, hidden_dim=256, seed=11)
    for w in vocab:
        b.language.register_word(w, emb[w])
    b.language.freeze_vocabulary()
    b.set_active_vocab([b.language.word_id(w) for w in vocab])
    b.auto_replay = auto
    return b


def nll(b, chunks):
    b.predictor.set_offline(True)
    tot, n = 0.0, 0
    for c in chunks:
        b.reset_sequence()
        x, k = b.eval_text_nll(c)
        tot += x; n += k
    b.predictor.set_offline(False)
    return tot / max(n, 1)


def run():
    print("\nConsolidation (dreaming) — hardening test")
    rng = np.random.default_rng(11)
    poolA = [f"a{i}" for i in range(60)]
    poolB = [f"b{i}" for i in range(60)]
    vocab = sorted(poolA + poolB)
    emb = {w: (rng.standard_normal(N_DIMS) * 0.3).astype(np.float32) for w in vocab}
    mkA, mkB = markov(poolA, rng), markov(poolB, rng)
    A_train = corpus(poolA, mkA, 80, 40, rng)
    A_eval = corpus(poolA, mkA, 10, 40, rng)
    B_train = corpus(poolB, mkB, 80, 40, rng)
    B_eval = corpus(poolB, mkB, 10, 40, rng)

    forget, b_loss = {}, {}
    for auto in (False, True):
        b = build(vocab, emb, auto)
        for _ in range(3):                       # learn A
            for c in A_train:
                b.reset_sequence(); b.train_lm_sequence_fused(c)
        a0 = nll(b, A_eval)
        for c in B_train:                        # interfering learning
            b.reset_sequence(); b.train_lm_sequence_fused(c)
        forget[auto] = nll(b, A_eval) - a0
        b_loss[auto] = nll(b, B_eval)

    print(f"  A forgetting:  no-replay {forget[False]:+.3f}   auto-replay {forget[True]:+.3f}")
    print(f"  B learned (loss, lower=better):  no-replay {b_loss[False]:.2f}  auto-replay {b_loss[True]:.2f}")

    check("forgetting happens without replay", forget[False] > 0.8)
    check("auto-replay cuts forgetting >=50%", forget[True] < 0.5 * forget[False])
    # the new task is not sacrificed: B loss with replay ~ B loss without it
    check("replay does not sacrifice the new task", b_loss[True] < b_loss[False] + 0.4)

    # knobs are honored and don't crash
    b = build(vocab, emb, True)
    b.replay_interval = 2; b.replay_samples = 6
    b.train_lm_sequence_fused(A_train[0])
    check("replay knobs settable and run", b.replay_interval == 2 and b.replay_samples == 6)

    print(f"\nConsolidation: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
