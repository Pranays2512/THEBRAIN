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


def fresh_brain(vocab, emb, som_size=SOM):
    b = brain2.Brain(som_rows=som_size, som_cols=som_size, n_dims=N_DIMS,
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


# ── 2. emotion: salience-weighted learning ───────────────────────────────────
def som_qe(b, words, emb):
    """Mean SOM quantization error: distance from each word's vector to its BMU
    neuron. Lower = the SOM has imprinted a neuron onto that word."""
    qe = []
    for w in words:
        v = emb[w]
        bmu = b.som.find_bmu(v)
        nw = np.asarray(b.som.neuron_weights(bmu), dtype=np.float32)
        qe.append(float(np.linalg.norm(v - nw)))
    return float(np.mean(qe))


def test_emotion(n_seeds=5):
    """Emotion scales the SOM learning rate by arousal (= surprise). Claim:
    surprising/salient items get imprinted more strongly. Two EQUAL-frequency
    word pools — 'pred' in a fixed predictable pattern (low surprise once
    learned), 'rand' at random positions (always high surprise). Only the
    surprise differs, so any QE gap that emotion opens is the salience effect.
    SOM resource pressure (12x12=144 neurons, 300 words) makes imprinting
    contested. The effect is weak/noisy, so average the differential over
    seeds rather than trust one lucky run."""
    rows = {False: [], True: []}
    diffs = []
    for seed in range(1, n_seeds + 1):
        rng = np.random.default_rng(100 + seed)
        pred = [f"p{i}" for i in range(150)]
        rand = [f"r{i}" for i in range(150)]
        words = pred + rand
        vocab, emb = make_vocab(words, rng)
        fixed = " ".join(rng.choice(pred, size=40))
        corpus = [fixed if rng.random() < 0.5 else " ".join(rng.choice(rand, size=40))
                  for _ in range(300)]
        rng.shuffle(corpus)
        qe = {}
        for emo in (False, True):
            b = fresh_brain(vocab, emb, som_size=12)
            b.emotion.modulation_enabled = emo
            for c in corpus:
                b.reset_sequence()
                b.train_lm_sequence_fused(c)
            qe[emo] = (som_qe(b, pred, emb), som_qe(b, rand, emb))
            rows[emo].append(qe[emo])
        gain_pred = qe[False][0] - qe[True][0]
        gain_rand = qe[False][1] - qe[True][1]
        diffs.append(gain_rand - gain_pred)   # >0 => emotion favors surprising
    mean = {e: (float(np.mean([r[0] for r in rows[e]])),
                float(np.mean([r[1] for r in rows[e]]))) for e in (False, True)}
    return mean, diffs


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
    print()

    mean, diffs = test_emotion()
    qp_off, qr_off = mean[False]
    qp_on, qr_on = mean[True]
    mean_diff = sum(diffs) / len(diffs)
    favoring = sum(1 for d in diffs if d > 0)
    print("2. EMOTION = salience-weighted learning (SOM imprinting, lower QE better)")
    print(f"   {'':14s} {'predictable':>12s} {'surprising':>12s}   (mean over {len(diffs)} seeds)")
    print(f"   emotion OFF    {qp_off:12.3f} {qr_off:12.3f}")
    print(f"   emotion ON     {qp_on:12.3f} {qr_on:12.3f}")
    print(f"   surprising-favoring differential per seed: "
          f"{[round(d, 3) for d in diffs]}")
    print(f"   mean differential {mean_diff:+.4f}  ({favoring}/{len(diffs)} seeds favor surprising)")
    if mean_diff > 0 and favoring > len(diffs) / 2:
        print("   -> emotion DOES bias imprinting toward surprising items, but the")
        print("      effect is WEAK (modulation is gentle: lr 1.5x->2.0x). Wired and")
        print("      directionally correct; strengthen modulation if it should matter more.")
        emo_verdict = "weak-directional"
    else:
        print("   -> no reliable emotion effect (investigate)")
        emo_verdict = "none"
    summary["emotion_mean_differential"] = round(mean_diff, 4)
    summary["emotion_seeds_favoring"] = f"{favoring}/{len(diffs)}"
    summary["emotion"] = emo_verdict

    print("\nsummary:", summary)


if __name__ == "__main__":
    main()
