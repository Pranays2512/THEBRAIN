#!/usr/bin/env python3
"""
retrain_language.py — Re-train language on top of existing brain weights.

Adds the expanded corpus (527 pairs, ~2991 TP transitions) without
touching the body-grounded SOM weights.

Usage:
    python retrain_language.py          # 25 epochs, ~3-4 min
    python retrain_language.py --full   # 50 epochs, ~7 min
"""

import sys, os, time, argparse
import numpy as np

sys.dont_write_bytecode = True

from brain_fast import FastBrain, SILENCE, N_MFCC
from vocab_extra import VOCABULARY
from dialogue_corpus import DIALOGUES

BRAIN_FILE = 'brain_fast.pkl'

rng = np.random.default_rng(42)

REWARD_RESPONSE = 0.8
REWARD_INPUT    = 0.1
NOISE_INPUT     = 0.10
NOISE_SELF      = 0.05

def noisy_mfcc(word, noise_std):
    if word not in VOCABULARY:
        return []
    mean_vec, n_frames = VOCABULARY[word]
    return [(mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
            for _ in range(n_frames)]

def feed_words(brain, words, reward, noise_std):
    bmus = []
    for word in words:
        frames = noisy_mfcc(word, noise_std)
        if not frames:
            continue
        for i, frame in enumerate(frames):
            brain.hear(frame)
            r = reward if i == len(frames) - 1 else 0.0
            out = brain.step(reward=r)
        bmus.append(out['bmu'])
        mean_vec, _ = VOCABULARY[word]
        clean_bmu = brain.som.find_bmu(mean_vec.astype(np.float32))
        if word not in brain.word_to_bmu:
            brain.word_to_bmu[word] = clean_bmu
            brain.bmu_to_word[clean_bmu] = word
        brain.hear(SILENCE)
        brain.step()
    return bmus

def tokenize(text):
    return [w for w in text.lower().split() if w in VOCABULARY]

def train_epoch(brain, dialogues, epoch):
    n_exchanges = 0
    words_learned = 0
    total_reward = 0.0
    order = rng.permutation(len(dialogues))
    for idx in order:
        you_text, brain_text = dialogues[idx]
        you_words   = tokenize(you_text)
        brain_words = tokenize(brain_text)
        if not you_words or not brain_words:
            continue
        for w in you_words:
            brain.word_tp.observe(w)
        brain.hear_word_separator()
        for w in brain_words:
            brain.word_tp.observe(w)
        brain.hear_word_end()
        feed_words(brain, you_words, REWARD_INPUT, NOISE_INPUT)
        for _ in range(6):
            brain.hear(SILENCE)
            brain.step()
        bmus = feed_words(brain, brain_words, REWARD_RESPONSE, NOISE_SELF)
        words_learned  += len(brain_words)
        total_reward   += REWARD_RESPONSE * len(brain_words)
        if bmus:
            brain.record_turn(bmus, REWARD_RESPONSE)
        brain.hear(SILENCE)
        brain.step()
        n_exchanges += 1
    brain.dream(n_sequences=min(30, len(brain._dream_buffer)))
    return {'exchanges': n_exchanges, 'words_learned': words_learned,
            'mean_reward': total_reward / max(words_learned, 1)}

def evaluate(brain, dialogues, n_sample=20):
    sample = rng.choice(len(dialogues), size=min(n_sample, len(dialogues)), replace=False)
    overlaps = []
    for idx in sample:
        you_text, brain_text = dialogues[idx]
        you_words   = tokenize(you_text)
        brain_words = set(tokenize(brain_text))
        if not you_words or not brain_words:
            continue
        generated    = brain.word_tp_generate(you_words, max_len=8)
        gen_set      = set(generated)
        overlaps.append(len(gen_set & brain_words) / len(brain_words))
    return float(np.mean(overlaps)) if overlaps else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='50 epochs')
    args = parser.parse_args()

    N_EPOCHS = 50 if args.full else 25

    if not os.path.exists(BRAIN_FILE):
        print(f"No {BRAIN_FILE} found. Run train_fast.py first.")
        sys.exit(1)

    print(f"Loading {BRAIN_FILE}...")
    brain = FastBrain.load(BRAIN_FILE)
    print(brain.status())
    print(f"Corpus: {len(DIALOGUES)} pairs | Epochs: {N_EPOCHS}")
    print()

    best_score = -1.0
    patience   = 0
    t0 = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        stats = train_epoch(brain, DIALOGUES, epoch)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
              f"exchanges={stats['exchanges']}  "
              f"learned={stats['words_learned']}  "
              f"word_tp={brain.word_tp.n_transitions()} transitions  "
              f"[{elapsed:.0f}s]")

        if epoch % 5 == 0:
            score = evaluate(brain, DIALOGUES, n_sample=25)
            print(f"    → Word TP overlap: {score:.3f}")
            if score > best_score:
                best_score = score
                patience   = 0
                brain.save(BRAIN_FILE)
                print(f"    → Saved (best so far)")
            else:
                patience += 1
                if patience >= 3:
                    print(f"    → Early stopping")
                    break

    brain.save(BRAIN_FILE)
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(brain.status())
    print("\nRun: python live_fast.py")

if __name__ == '__main__':
    main()