"""
train_fast.py — Train the 5000-neuron FastBrain on the dialogue corpus.

Usage:
    python train_fast.py          # trains from scratch or resumes
    python train_fast.py --reset  # ignore existing checkpoint, start fresh

After training, run:
    python live_fast.py           # talk to the trained brain

Training procedure (same philosophy as train_dialogue.py):
  For each dialogue ("what you say", "brain should respond"):
    1. Feed input words at low reward (0.1) — context, not the target.
    2. Feed correct response words at high reward (0.8) — learn this.
    3. Word TP records input → SEP → response sequence.
  Repeat for N_EPOCHS, with early stopping.
  After each epoch, run dream() for offline TP consolidation.

Why this is genuine learning (not hardcoding):
  The brain never sees an IF/ELSE rule for any dialogue pair.
  After thousands of repetitions with reward, the TP matrix naturally
  produces "i feel good" after "how are you" — because those BMU
  transitions were reinforced more than any others.
"""

import sys
import os
import time
import argparse
import numpy as np
from collections import defaultdict

sys.dont_write_bytecode = True

from brain_fast import FastBrain, SILENCE, N_MFCC
from vocab_extra import VOCABULARY

BRAIN_FILE = 'brain_fast.pkl'
N_EPOCHS   = 25
REWARD_RESPONSE = 0.8    # brain hears the correct response words
REWARD_INPUT    = 0.1    # brain hears what "you" said (context signal)
NOISE_INPUT     = 0.10   # noise on external voice
NOISE_SELF      = 0.05   # brain's own voice is clearer
SAVE_EVERY      = 5
EVAL_EVERY      = 5
PATIENCE        = 3

rng = np.random.default_rng(42)


def noisy_mfcc(word: str, noise_std: float) -> list[np.ndarray]:
    """Return noisy MFCC frames for word, or [] if unknown."""
    if word not in VOCABULARY:
        return []
    mean_vec, n_frames = VOCABULARY[word]
    return [
        (mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
        for _ in range(n_frames)
    ]


def feed_words(brain: FastBrain, words: list[str],
               reward: float, noise_std: float) -> list[int]:
    """
    Feed a word sequence through the brain.
    Returns list of BMUs (one per word, at the last MFCC frame).
    """
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

        # Update BMU→word map from clean MFCC
        mean_vec, _ = VOCABULARY[word]
        clean_bmu = brain.som.find_bmu(mean_vec.astype(np.float32))
        if word not in brain.word_to_bmu:
            brain.word_to_bmu[word] = clean_bmu
            brain.bmu_to_word[clean_bmu] = word

        # Silence gap between words
        brain.hear(SILENCE)
        brain.step()

    return bmus


def tokenize(text: str) -> list[str]:
    """Return only words that exist in the vocabulary."""
    return [w for w in text.lower().split() if w in VOCABULARY]


def train_epoch(brain: FastBrain, dialogues: list, epoch: int) -> dict:
    """Run one pass through all dialogues, shuffled."""
    n_exchanges   = 0
    words_learned = 0
    total_reward  = 0.0

    order = rng.permutation(len(dialogues))

    for idx in order:
        you_text, brain_text = dialogues[idx]
        you_words   = tokenize(you_text)
        brain_words = tokenize(brain_text)

        if not you_words or not brain_words:
            continue

        # ── Word TP: record the full exchange ─────────────────────────
        for w in you_words:
            brain.word_tp.observe(w)
        brain.hear_word_separator()
        for w in brain_words:
            brain.word_tp.observe(w)
        brain.hear_word_end()

        # ── Phase 1: Brain hears YOU speak ────────────────────────────
        feed_words(brain, you_words, reward=REWARD_INPUT, noise_std=NOISE_INPUT)

        # Settling pause — integrate before responding
        for _ in range(6):
            brain.hear(SILENCE)
            brain.step()

        # ── Phase 2: Brain hears correct response ─────────────────────
        bmus = feed_words(brain, brain_words,
                          reward=REWARD_RESPONSE, noise_std=NOISE_SELF)
        words_learned  += len(brain_words)
        total_reward   += REWARD_RESPONSE * len(brain_words)

        # Buffer for dream consolidation
        if bmus:
            brain.record_turn(bmus, REWARD_RESPONSE)

        # Short silence between exchanges
        brain.hear(SILENCE)
        brain.step()
        n_exchanges += 1

    # Offline consolidation after each epoch
    brain.dream(n_sequences=min(30, len(brain._dream_buffer)))

    return {
        'epoch':         epoch,
        'exchanges':     n_exchanges,
        'words_learned': words_learned,
        'mean_reward':   total_reward / max(words_learned, 1),
    }


def evaluate(brain: FastBrain, dialogues: list, n_sample: int = 15) -> float:
    """
    Word TP accuracy: for N random dialogues, generate a response
    and measure word overlap with the target.
    Returns mean overlap ratio (0–1).
    """
    sample_idx = rng.choice(len(dialogues),
                            size=min(n_sample, len(dialogues)),
                            replace=False)
    overlaps = []
    for idx in sample_idx:
        you_text, brain_text = dialogues[idx]
        you_words   = tokenize(you_text)
        brain_words = set(tokenize(brain_text))

        if not you_words or not brain_words:
            continue

        generated    = brain.word_tp_generate(you_words, max_len=8)
        generated_set = set(generated)

        overlap = len(generated_set & brain_words) / len(brain_words)
        overlaps.append(overlap)

    return float(np.mean(overlaps)) if overlaps else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true',
                        help='Start fresh, ignore existing checkpoint')
    args = parser.parse_args()

    from dialogue_corpus import DIALOGUES

    # ── Load or create brain ──────────────────────────────────────────
    if not args.reset and os.path.exists(BRAIN_FILE):
        print(f"Resuming from {BRAIN_FILE}...")
        brain = FastBrain.load(BRAIN_FILE)
        print(brain.status())
    else:
        print("Creating new 5000-neuron FastBrain...")
        brain = FastBrain()

    print(f"\n  Vocabulary:  {len(VOCABULARY)} words")
    print(f"  Corpus:      {len(DIALOGUES)} dialogues")
    print(f"  Epochs:      {N_EPOCHS}  (patience={PATIENCE})")
    print(f"  Reward:      input={REWARD_INPUT}, response={REWARD_RESPONSE}")
    print()

    # ── Training loop ─────────────────────────────────────────────────
    t_start       = time.time()
    best_score    = -1.0
    patience_ctr  = 0

    for epoch in range(1, N_EPOCHS + 1):
        stats = train_epoch(brain, DIALOGUES, epoch)
        elapsed = time.time() - t_start

        print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
              f"exchanges={stats['exchanges']}  "
              f"learned={stats['words_learned']} words  "
              f"reward={stats['mean_reward']:.3f}  "
              f"word_tp={brain.word_tp.n_transitions()} transitions  "
              f"[{elapsed:.0f}s]")

        if epoch % SAVE_EVERY == 0:
            brain.save(BRAIN_FILE)
            print(f"    → Saved to {BRAIN_FILE}")

        if epoch % EVAL_EVERY == 0:
            score = evaluate(brain, DIALOGUES, n_sample=15)
            print(f"    → Word TP overlap: {score:.3f}")

            if score > best_score:
                best_score    = score
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"    → Early stopping (no improvement for {PATIENCE} evals)")
                    break

    # ── Final save ────────────────────────────────────────────────────
    brain.save(BRAIN_FILE)
    total = time.time() - t_start
    print(f"\n  Done in {total:.1f}s")
    print(f"  {brain.status()}")
    print(f"  Brain saved → {BRAIN_FILE}")
    print(f"  Run: python live_fast.py")


if __name__ == '__main__':
    main()
