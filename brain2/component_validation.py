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


def gen_corpus(rng, vocab_words, n_chunks, chunk_len):
    return [" ".join(rng.choice(vocab_words, size=chunk_len)) for _ in range(n_chunks)]


# ── 1. dream consolidation ───────────────────────────────────────────────────
def test_dream(rng, dream_cycles=30):
    # Two disjoint word pools so A and B genuinely interfere on shared weights
    # but not on shared tokens.
    poolA = [f"a{i}" for i in range(60)]
    poolB = [f"b{i}" for i in range(60)]
    words = poolA + poolB
    vocab, emb = make_vocab(words, rng)

    A_train = gen_corpus(rng, poolA, 40, 40)
    A_eval = gen_corpus(rng, poolA, 10, 40)
    B_train = gen_corpus(rng, poolB, 40, 40)

    results = {}
    diag = {}
    for label, do_dream in (("no_dream", False), ("dream", True)):
        b = fresh_brain(vocab, emb)
        for c in A_train:                      # learn A
            b.reset_sequence()
            b.train_lm_sequence_fused(c)
        a_after_learn = nll(b, A_eval)
        if do_dream:                           # sleep on A
            ep_before = b.episodic.episode_count
            nll_before = a_after_learn
            for _ in range(dream_cycles):
                b.dream(20, 15)
            # diagnostics: did dreaming change the LM or the episodic store?
            diag["episodes_before_dream"] = ep_before
            diag["episodes_after_dream"] = b.episodic.episode_count
            diag["A_nll_lm_change_from_dream"] = round(nll(b, A_eval) - nll_before, 5)
        for c in B_train:                      # interfering learning
            b.reset_sequence()
            b.train_lm_sequence_fused(c)
        a_after_interf = nll(b, A_eval)
        results[label] = (a_after_learn, a_after_interf)

    return results, diag


def main():
    rng = np.random.default_rng(SEED)
    print("component validation — measure each part where it should act\n")

    r, diag = test_dream(rng)
    nl, ni = r["no_dream"]
    dl, di = r["dream"]
    print("1. DREAM CONSOLIDATION (learn A -> sleep? -> learn B -> retest A)")
    print(f"   A loss after learning A:        no_dream {nl:.3f} | dream {dl:.3f}")
    print(f"   A loss after B interference:    no_dream {ni:.3f} | dream {di:.3f}")
    forget_no = ni - nl
    forget_dr = di - dl
    print(f"   forgetting (rise in A loss):    no_dream {forget_no:+.3f} | dream {forget_dr:+.3f}")
    print("   wiring diagnostics:")
    print(f"     LM change from dreaming:      {diag.get('A_nll_lm_change_from_dream')}  (≈0 => replay not training the predictor)")
    print(f"     episodes before/after dream:  {diag.get('episodes_before_dream')} -> {diag.get('episodes_after_dream')}  (drop => consolidate() erasing, not strengthening)")
    verdict = ("dreaming REDUCES forgetting" if forget_dr < forget_no - 1e-3
               else "no measurable dream benefit — WIRING BUG, not a reason to cut")
    print(f"   -> {verdict}\n")

    print("summary:", {
        "dream_forgetting_no": round(forget_no, 3),
        "dream_forgetting_yes": round(forget_dr, 3),
        "dream_helps": bool(forget_dr < forget_no - 1e-3),
        "lm_change_from_dream": diag.get("A_nll_lm_change_from_dream"),
        "episodes_after_dream": diag.get("episodes_after_dream"),
    })


if __name__ == "__main__":
    main()
