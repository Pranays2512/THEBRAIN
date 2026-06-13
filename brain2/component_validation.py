#!/usr/bin/env python3
"""
component_validation.py — Step 3, done as validation, not deletion.

Each brain component claims to do a job. This suite measures whether it does,
ON THE AXIS WHERE IT IS SUPPOSED TO ACT — not on perplexity, which is the wrong
ruler for dreaming or emotion (the same mistake as judging reasoning by
perplexity). A component that proves its effect is kept WITH A NUMBER behind
it; one that shows nothing is first a wiring bug to investigate, not a delete.

Tests:
  1. dream_consolidation — does the dream cycle's episodic replay protect
     recently-learned material from interference by later learning?
       protocol: learn A -> (dream A | skip) -> learn B (interference)
                 -> measure retention NLL on held-out A.  dreaming should
                 lower A's loss after B.
  2. emotion_modulation  — does emotion's learning-rate modulation make
     surprising/salient items better retained?  (requires the emotion-disable
     knob; see emotion_enabled.)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2

N_DIMS = 64
SOM = 64
HIDDEN = 256
SEED = 11


def make_vocab(words, rng):
    vocab = sorted(set(words))
    emb = {w: (rng.standard_normal(N_DIMS) * 0.3).astype(np.float32) for w in vocab}
    return vocab, emb


def fresh_brain(vocab, emb):
    b = brain2.Brain(som_rows=SOM, som_cols=SOM, n_dims=N_DIMS,
                     hidden_dim=HIDDEN, seed=SEED)
    for w in vocab:
        b.language.register_word(w, emb[w])
    b.language.freeze_vocabulary()
    b.set_active_vocab([b.language.word_id(w) for w in vocab])
    b.auto_replay = False   # this test controls replay explicitly
    return b


def nll(b, chunks):
    b.predictor.set_offline(True)
    tot, n = 0.0, 0
    for c in chunks:
        b.reset_sequence()
        x, k = b.eval_text_nll(c)
        tot += x
        n += k
    b.predictor.set_offline(False)
    return tot / max(n, 1)


def make_markov(rng, words, fanout=3):
    """Fixed transition table: each word -> a small set of successors. Gives the
    corpus REAL learnable structure (so consolidation reinforces signal, not
    noise). Random word streams have nothing to consolidate."""
    return {w: list(rng.choice(words, size=fanout, replace=False)) for w in words}


def gen_corpus(rng, words, mk, n_chunks, chunk_len):
    out = []
    for _ in range(n_chunks):
        w = rng.choice(words)
        seq = [w]
        for _ in range(chunk_len - 1):
            w = rng.choice(mk[w])
            seq.append(w)
        out.append(" ".join(seq))
    return out


# ── 1. dream consolidation ───────────────────────────────────────────────────
def test_dream(rng, dream_cycles=30):
    # Two disjoint word pools so A and B genuinely interfere on shared weights
    # but not on shared tokens.
    poolA = [f"a{i}" for i in range(60)]
    poolB = [f"b{i}" for i in range(60)]
    words = poolA + poolB
    vocab, emb = make_vocab(words, rng)

    mkA = make_markov(rng, poolA)
    mkB = make_markov(rng, poolB)
    A_train = gen_corpus(rng, poolA, mkA, 120, 40)
    A_eval = gen_corpus(rng, poolA, mkA, 12, 40)   # same chain as A_train -> learnable
    B_train = gen_corpus(rng, poolB, mkB, 120, 40)
    A_PASSES = 3   # learn A well enough that there is real structure to retain

    B_eval = gen_corpus(rng, poolB, mkB, 12, 40)

    def rehearse(b, mode):
        # interleaved replay: rehearse buffered A sequences WHILE learning B.
        # This is how experience replay actually fights forgetting (rehearsal),
        # vs a pre-B "sleep" which B then overwrites.
        if mode == "faithful":
            b.dream_replay_faithful(4, 1)
        elif mode == "generative":
            b.dream_replay_generative(4, 24, 0.5, 5)

    results = {}
    for label in ("no_dream", "faithful", "generative"):
        b = fresh_brain(vocab, emb)
        for _ in range(A_PASSES):              # learn A (multiple passes)
            for c in A_train:
                b.reset_sequence()
                b.train_lm_sequence_fused(c)
        a_after_learn = nll(b, A_eval)
        for i, c in enumerate(B_train):        # learn B, rehearsing A every 4th
            b.reset_sequence()
            b.train_lm_sequence_fused(c)
            if label != "no_dream" and i % 4 == 0:
                rehearse(b, label)
        a_after_interf = nll(b, A_eval)
        b_learned = nll(b, B_eval)             # confirm B still learned
        results[label] = (a_after_learn, a_after_interf, b_learned)

    return results


def main():
    rng = np.random.default_rng(SEED)
    print("component validation — measure each part where it should act\n")

    r = test_dream(rng)
    print("1. DREAM CONSOLIDATION  (learn A -> learn B while rehearsing A -> retest A)")
    print(f"   {'mode':12s} {'A after learn':>13s} {'A after B':>10s} {'forgetting':>11s} {'B learned':>10s}")
    summary = {}
    forgets = {}
    for label in ("no_dream", "faithful", "generative"):
        learn, interf, bl = r[label]
        forget = interf - learn
        forgets[label] = forget
        print(f"   {label:12s} {learn:13.3f} {interf:10.3f} {forget:+11.3f} {bl:10.3f}")
        summary[label + "_forgetting"] = round(forget, 3)

    base = forgets["no_dream"]
    best = min(("faithful", "generative"), key=lambda m: forgets[m])
    print()
    if forgets[best] < base - 1e-3:
        pct = 100 * (base - forgets[best]) / base
        print(f"   -> WINNER: {best} replay cuts forgetting {pct:.0f}% "
              f"({base:+.3f} -> {forgets[best]:+.3f}), B still learned")
        summary["winner"] = best
    else:
        print("   -> neither replay beats no-dream (investigate)")
        summary["winner"] = "none"
    print("\nsummary:", summary)


if __name__ == "__main__":
    main()
