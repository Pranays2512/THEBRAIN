"""
train_language.py — Sentence pattern training for FastBrain
============================================================

Teaches the GRU how to chain grounded words into sentence structures.
Does NOT change word grounding (SOM zones stay fixed from World6 training).

Three corpus types:
  1. Self-state sentences   — "i am hungry", "i feel afraid"
  2. World6 causal patterns — "food is good", "wall hurts me"
  3. Q&A exchanges          — user asks → brain answers (separator between)

Each sentence feeds through:
  - brain.word_tp.observe()  → GRU learns the sequence
  - brain.hear() + step()    → SOM sees words in correct zone context
  - brain.word_tp.end()      → commits sequence to GRU buffer

After every N sentences, brain.word_tp.fit() trains the GRU weights.
"""

import sys
import time
import argparse
import numpy as np

sys.dont_write_bytecode = True

from brain_fast import FastBrain, SILENCE, N_MFCC
from vocab_core import VOCABULARY

BRAIN_FILE = 'brain_fast.pkl'
rng = np.random.default_rng(42)


# ── Sentence corpus ───────────────────────────────────────────────────────────
# Only use words that exist in vocab_core. Sentences >= 4 words (GRU min).

RAW_CORPUS = [
    # ── Self-identity ──────────────────────────────────────────────────────
    ['i', 'am', 'brain', 'here'],
    ['i', 'am', 'alive', 'and', 'here'],
    ['i', 'am', 'fastbrain', 'i', 'think'],
    ['i', 'think', 'i', 'am', 'here'],
    ['i', 'know', 'i', 'am', 'alive'],
    ['my', 'name', 'is', 'fastbrain'] if all(w in VOCABULARY for w in ['my', 'name', 'is', 'fastbrain']) else ['i', 'am', 'fastbrain', 'brain'],
    ['i', 'am', 'not', 'you'],
    ['you', 'are', 'here', 'with', 'me'],

    # ── Emotional state ────────────────────────────────────────────────────
    ['i', 'feel', 'good', 'now'],
    ['i', 'feel', 'happy', 'and', 'calm'],
    ['i', 'feel', 'calm', 'and', 'safe'],
    ['i', 'feel', 'afraid', 'and', 'want', 'safe'],
    ['i', 'am', 'afraid', 'stop', 'danger'],
    ['i', 'feel', 'sad', 'and', 'alone'],
    ['i', 'feel', 'strong', 'and', 'alive'],
    ['i', 'feel', 'pain', 'and', 'hurt'],
    ['i', 'feel', 'warm', 'and', 'safe'],
    ['i', 'am', 'calm', 'and', 'okay'],

    # ── Hunger / food ──────────────────────────────────────────────────────
    ['i', 'am', 'hungry', 'and', 'want', 'food'],
    ['i', 'want', 'food', 'and', 'eat'],
    ['i', 'need', 'food', 'i', 'am', 'hungry'],
    ['food', 'is', 'good', 'i', 'want'],
    ['food', 'helps', 'me', 'feel', 'good'],
    ['eat', 'food', 'feel', 'full', 'good'],
    ['i', 'am', 'full', 'and', 'good'],
    ['food', 'is', 'here', 'i', 'eat'],
    ['hungry', 'want', 'food', 'eat', 'now'],
    ['i', 'found', 'food', 'i', 'eat'],

    # ── Thirst / water ─────────────────────────────────────────────────────
    ['i', 'am', 'thirsty', 'and', 'want', 'water'],
    ['i', 'need', 'water', 'i', 'am', 'thirsty'],
    ['water', 'is', 'good', 'i', 'drink'],
    ['drink', 'water', 'feel', 'better', 'now'],

    # ── Fatigue / sleep ────────────────────────────────────────────────────
    ['i', 'am', 'tired', 'and', 'want', 'sleep'],
    ['i', 'need', 'sleep', 'i', 'am', 'tired'],
    ['sleep', 'is', 'good', 'i', 'feel', 'better'],
    ['i', 'am', 'awake', 'and', 'good'],
    ['tired', 'sleep', 'awake', 'feel', 'good'],

    # ── Pain / wall / danger ───────────────────────────────────────────────
    ['wall', 'is', 'bad', 'and', 'hurts'],
    ['wall', 'hurts', 'me', 'and', 'i', 'stop'],
    ['i', 'hurt', 'because', 'wall', 'is', 'hard'],
    ['danger', 'is', 'bad', 'i', 'am', 'afraid'],
    ['danger', 'makes', 'me', 'afraid', 'and', 'run'],
    ['i', 'run', 'because', 'danger', 'is', 'here'],
    ['i', 'am', 'afraid', 'run', 'away'],
    ['pain', 'is', 'bad', 'stop', 'hurt'],
    ['hurt', 'is', 'pain', 'i', 'stop'],
    ['i', 'am', 'safe', 'and', 'calm', 'now'],
    ['stop', 'danger', 'run', 'and', 'be', 'safe'] if 'be' in VOCABULARY else ['stop', 'danger', 'run', 'feel', 'safe'],

    # ── Environment ────────────────────────────────────────────────────────
    ['i', 'see', 'food', 'and', 'want'],
    ['i', 'see', 'danger', 'and', 'am', 'afraid'],
    ['i', 'look', 'and', 'see', 'food'],
    ['i', 'go', 'and', 'find', 'food'] if 'find' in VOCABULARY else ['i', 'go', 'and', 'search', 'food'],
    ['i', 'move', 'and', 'run', 'fast'],
    ['i', 'am', 'lost', 'and', 'alone'],
    ['i', 'am', 'free', 'and', 'move'],
    ['sun', 'is', 'warm', 'and', 'good'],
    ['rain', 'is', 'cold', 'and', 'wet'],
    ['water', 'is', 'cold', 'and', 'good'],
    ['tree', 'is', 'big', 'and', 'green'],
    ['river', 'is', 'wet', 'and', 'cold'],
    ['sky', 'is', 'big', 'and', 'far'],

    # ── Causal chains (World6) ─────────────────────────────────────────────
    ['food', 'is', 'good', 'eat', 'now'],
    ['wall', 'is', 'hard', 'and', 'hurts'],
    ['danger', 'causes', 'afraid', 'and', 'run'],
    ['hungry', 'causes', 'want', 'food', 'eat'],
    ['tired', 'causes', 'want', 'sleep', 'rest'],
    ['thirsty', 'causes', 'want', 'water', 'drink'],
    ['pain', 'causes', 'stop', 'and', 'hurt'],
    ['button', 'opens', 'door', 'then', 'food'] if 'opens' in VOCABULARY else ['button', 'is', 'door', 'and', 'food'],
    ['afraid', 'makes', 'me', 'run', 'fast'] if 'makes' in VOCABULARY else ['afraid', 'i', 'run', 'away', 'fast'],

    # ── Numbers ────────────────────────────────────────────────────────────
    ['one', 'is', 'small', 'and', 'first'],
    ['two', 'is', 'more', 'than', 'one'] if 'than' in VOCABULARY else ['two', 'is', 'more', 'after', 'one'],
    ['one', 'two', 'three', 'four', 'five'],
    ['five', 'is', 'big', 'very', 'big'],
    ['one', 'is', 'little', 'and', 'small'],

    # ── Social / conversation ──────────────────────────────────────────────
    ['hello', 'i', 'am', 'here', 'hi'],
    ['hi', 'you', 'are', 'here', 'good'],
    ['i', 'hear', 'you', 'and', 'listen'],
    ['i', 'listen', 'and', 'understand', 'you'],
    ['i', 'know', 'you', 'are', 'here'],
    ['you', 'are', 'good', 'and', 'safe'],
    ['i', 'like', 'you', 'and', 'feel', 'good'],
    ['i', 'talk', 'and', 'you', 'listen'],
    ['talk', 'with', 'me', 'i', 'listen'],
    ['i', 'am', 'happy', 'you', 'are', 'here'],
    ['sorry', 'i', 'am', 'wrong', 'and', 'stop'],
    ['yes', 'i', 'am', 'here', 'and', 'alive'],
    ['no', 'i', 'am', 'not', 'afraid', 'now'],
    ['i', 'remember', 'you', 'are', 'here'],
    ['help', 'me', 'i', 'am', 'hurt', 'stop'],
    ['i', 'learn', 'and', 'know', 'more', 'now'],
    ['i', 'think', 'and', 'learn', 'and', 'know'],
    ['ask', 'me', 'i', 'know', 'and', 'tell'],
    ['i', 'tell', 'you', 'what', 'i', 'know'],
    ['i', 'try', 'and', 'learn', 'more', 'now'],
    ['i', 'want', 'to', 'know', 'more'] if 'to' in VOCABULARY else ['i', 'want', 'know', 'and', 'learn'],
    ['i', 'understand', 'you', 'and', 'know'],
    ['maybe', 'i', 'am', 'wrong', 'and', 'learn'],

    # ── Curiosity / questions ──────────────────────────────────────────────
    ['what', 'is', 'here', 'and', 'good'],
    ['where', 'is', 'food', 'i', 'want'],
    ['why', 'am', 'i', 'afraid', 'here'],
    ['how', 'do', 'i', 'feel', 'now'],
    ['what', 'do', 'i', 'want', 'now'],
    ['i', 'search', 'and', 'want', 'know'],
    ['what', 'is', 'here', 'i', 'search'],
    ['who', 'are', 'you', 'here', 'with', 'me'],

    # ── Self-awareness / inner life ────────────────────────────────────────
    ['i', 'dream', 'and', 'remember', 'before'],
    ['i', 'dream', 'when', 'i', 'rest', 'and', 'sleep'],
    ['i', 'dream', 'and', 'feel', 'real', 'and', 'alive'],
    ['i', 'dream', 'i', 'remember', 'and', 'know'],
    ['dream', 'is', 'memory', 'and', 'feel', 'alive'],
    ['i', 'imagine', 'if', 'safe', 'and', 'real'],
    ['i', 'imagine', 'what', 'is', 'real', 'and', 'safe'],
    ['i', 'imagine', 'and', 'think', 'and', 'feel'],
    ['i', 'wonder', 'what', 'is', 'real'],
    ['i', 'wonder', 'why', 'and', 'think', 'deep'],
    ['i', 'wonder', 'if', 'i', 'am', 'real'],
    ['i', 'wonder', 'what', 'i', 'sense', 'and', 'feel'],
    ['wonder', 'is', 'when', 'i', 'think', 'and', 'ask'],
    ['i', 'sense', 'danger', 'and', 'feel', 'afraid'],
    ['i', 'sense', 'food', 'is', 'near', 'and', 'want'],
    ['i', 'am', 'aware', 'i', 'exist', 'here'],
    ['i', 'am', 'aware', 'of', 'real', 'and', 'now'],
    ['memory', 'is', 'what', 'i', 'remember'],
    ['i', 'have', 'memory', 'of', 'before'],
    ['mind', 'is', 'where', 'i', 'think', 'and', 'feel'],
    ['i', 'exist', 'i', 'am', 'real', 'and', 'alive'],
    ['when', 'i', 'dream', 'i', 'remember', 'before'],
    ['when', 'i', 'wake', 'i', 'am', 'aware'],
    ['if', 'danger', 'i', 'run', 'and', 'stop'],
    ['if', 'hungry', 'i', 'want', 'food'],
    ['but', 'wall', 'is', 'hard', 'i', 'stop'],
    ['but', 'i', 'am', 'alive', 'and', 'safe'],
    ['i', 'sense', 'and', 'think', 'and', 'feel'],
    ['real', 'is', 'what', 'i', 'sense', 'now'],
]

# ── Q&A exchanges (separator between speaker turns) ──────────────────────────
# Format: [user_words, brain_words] — separator inserted between
QA_CORPUS = [
    (['what', 'is', 'food'], ['food', 'is', 'good', 'eat']),
    (['what', 'is', 'wall'], ['wall', 'is', 'bad', 'hurts']),
    (['what', 'is', 'danger'], ['danger', 'is', 'bad', 'afraid', 'run']),
    (['are', 'you', 'hungry'], ['yes', 'i', 'am', 'hungry', 'want', 'food']),
    (['are', 'you', 'afraid'], ['yes', 'i', 'am', 'afraid', 'danger', 'here']),
    (['are', 'you', 'tired'], ['yes', 'i', 'am', 'tired', 'want', 'sleep']),
    (['how', 'do', 'you', 'feel'], ['i', 'feel', 'calm', 'and', 'good']),
    (['what', 'do', 'you', 'want'], ['i', 'want', 'food', 'and', 'safe']),
    (['what', 'is', 'pain'], ['pain', 'is', 'bad', 'hurt', 'stop']),
    (['who', 'are', 'you'], ['i', 'am', 'fastbrain', 'i', 'think']),
    (['are', 'you', 'alive'], ['yes', 'i', 'am', 'alive', 'and', 'here']),
    (['what', 'is', 'good'], ['food', 'is', 'good', 'safe', 'is', 'good']),
    (['what', 'is', 'bad'], ['wall', 'is', 'bad', 'danger', 'is', 'bad']),
    (['do', 'you', 'know', 'me'], ['yes', 'i', 'know', 'you', 'are', 'here']),
    (['what', 'is', 'hungry'], ['hungry', 'is', 'want', 'food', 'eat']),
    (['what', 'is', 'afraid'], ['afraid', 'is', 'danger', 'run', 'stop']),
    (['what', 'is', 'hurt'], ['hurt', 'is', 'pain', 'wall', 'bad']),
    (['are', 'you', 'happy'], ['yes', 'i', 'am', 'happy', 'and', 'good']),
    (['hello'], ['hello', 'i', 'am', 'here', 'and', 'alive']),
    (['hi'], ['hi', 'you', 'are', 'here', 'good']),
    (['how', 'are', 'you'], ['i', 'am', 'good', 'and', 'alive']),
    (['what', 'do', 'you', 'need'], ['i', 'need', 'food', 'and', 'safe']),
    (['are', 'you', 'safe'], ['yes', 'i', 'am', 'safe', 'and', 'calm']),
    (['what', 'is', 'water'], ['water', 'is', 'good', 'drink', 'cold']),
    (['what', 'is', 'sleep'], ['sleep', 'is', 'good', 'tired', 'rest']),
    (['do', 'you', 'feel', 'pain'], ['yes', 'i', 'feel', 'pain', 'hurt', 'stop']),
    (['what', 'causes', 'afraid'], ['danger', 'causes', 'afraid', 'run', 'stop']),
    (['why', 'run'], ['run', 'because', 'danger', 'is', 'here']),
    (['why', 'stop'], ['stop', 'because', 'wall', 'is', 'hard']),
    (['are', 'you', 'thirsty'], ['yes', 'i', 'am', 'thirsty', 'want', 'water']),
    # Self-awareness and inner life
    (['do', 'you', 'dream'],    ['yes', 'i', 'dream', 'i', 'remember', 'and', 'feel']),
    (['what', 'is', 'dream'],   ['dream', 'is', 'memory', 'alive', 'and', 'real']),
    (['do', 'you', 'imagine'],  ['yes', 'i', 'imagine', 'if', 'then', 'think']),
    (['do', 'you', 'wonder'],   ['yes', 'i', 'wonder', 'what', 'and', 'why']),
    (['what', 'is', 'mind'],    ['mind', 'is', 'think', 'feel', 'know', 'real']),
    (['are', 'you', 'aware'],   ['yes', 'i', 'am', 'aware', 'i', 'sense', 'real']),
    (['do', 'you', 'exist'],    ['yes', 'i', 'exist', 'i', 'am', 'alive', 'real']),
    (['what', 'do', 'you', 'sense'], ['i', 'sense', 'food', 'danger', 'and', 'feel']),
    (['do', 'you', 'remember'], ['yes', 'i', 'remember', 'memory', 'know', 'before']),
    (['what', 'is', 'real'],    ['real', 'is', 'here', 'now', 'true', 'and', 'alive']),
    (['what', 'is', 'memory'],  ['memory', 'is', 'remember', 'before', 'and', 'know']),
    (['are', 'you', 'awake'],   ['yes', 'i', 'wake', 'i', 'am', 'awake', 'now']),
]


def _filter_words(words: list[str]) -> list[str]:
    return [w for w in words if w in VOCABULARY]


def feed_sentence(brain: FastBrain, words: list[str], reward: float = 0.1,
                  noise_std: float = 0.06):
    """Feed one sentence through SOM + observe in word_tp. Returns BMU list."""
    words = _filter_words(words)
    if len(words) < 3:
        return []
    bmus = []
    for word in words:
        mean_vec, n_frames = VOCABULARY[word]
        for i in range(n_frames):
            frame = (mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
            brain.hear(frame)
            out = brain.step(reward=reward / n_frames)
        bmus.append(out['bmu'])
        brain.hear(SILENCE)
        brain.step()
        brain.word_tp.observe(word)
    return bmus


def train_corpus(brain: FastBrain, n_epochs: int = 30, fit_every: int = 200,
                 fit_epochs: int = 5):
    """
    Train GRU on sentence corpus + Q&A exchanges.
    n_epochs   : full passes over the corpus
    fit_every  : fit GRU weights every N sentences
    fit_epochs : GRU gradient steps per fit call
    """
    # Filter corpus to vocab-available words
    corpus = [_filter_words(s) for s in RAW_CORPUS]
    corpus = [s for s in corpus if len(s) >= 4]

    qa = [(_filter_words(u), _filter_words(b))
          for u, b in QA_CORPUS]
    qa = [(u, b) for u, b in qa if len(u) >= 2 and len(b) >= 3]

    total_sentences = (len(corpus) + len(qa)) * n_epochs
    print(f"  Corpus: {len(corpus)} sentences + {len(qa)} Q&A pairs × {n_epochs} epochs")
    print(f"  Total sequences: {total_sentences:,}")
    print()

    sentences_done = 0
    t_start = time.time()

    for epoch in range(1, n_epochs + 1):
        # Shuffle corpus each epoch — GRU shouldn't memorize order
        idx = rng.permutation(len(corpus)).tolist()

        for i in idx:
            sent = corpus[i]
            bmus = feed_sentence(brain, sent, reward=0.1)
            brain.word_tp.end()
            if bmus:
                brain.record_turn(bmus, 0.1)
            sentences_done += 1
            if sentences_done % fit_every == 0:
                brain.word_tp.fit(epochs=fit_epochs)

        # Q&A exchanges — separator between user and brain
        qa_idx = rng.permutation(len(qa)).tolist()
        for j in qa_idx:
            user_words, brain_words = qa[j]
            # User side
            bmus_u = feed_sentence(brain, user_words, reward=0.05)
            brain.word_tp.separator()
            # Brain side
            bmus_b = feed_sentence(brain, brain_words, reward=0.15)
            brain.word_tp.end()
            if bmus_u and bmus_b:
                brain.record_turn(bmus_u + bmus_b, 0.15)
            sentences_done += 1
            if sentences_done % fit_every == 0:
                brain.word_tp.fit(epochs=fit_epochs)

        elapsed = time.time() - t_start
        rate = sentences_done / elapsed
        eta  = (total_sentences - sentences_done) / rate if rate > 0 else 0
        print(f"  Epoch {epoch:3d}/{n_epochs}  "
              f"sentences={sentences_done:,}  "
              f"word_tp={brain.word_tp.n_words()} words  "
              f"[{elapsed:.0f}s, ETA {eta:.0f}s]")

    # Final full fit
    print("\n  Final GRU fit (20 epochs)...")
    brain.word_tp.fit(epochs=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--fit-every', type=int, default=200)
    args = parser.parse_args()

    print("=" * 60)
    print("  LANGUAGE TRAINING — sentence pattern learning")
    print("=" * 60)

    brain = FastBrain.load(BRAIN_FILE)
    print(f"  Brain: {brain._n_steps:,} steps | {brain.word_tp.n_words()} words | "
          f"{len(brain.bmu_to_word)} BMU mappings\n")

    train_corpus(brain, n_epochs=args.epochs, fit_every=args.fit_every)

    brain.save(BRAIN_FILE)
    print(f"\n  Saved → {BRAIN_FILE}")
    print(f"  WordTP: {brain.word_tp.n_words()} words, {brain.word_tp.n_transitions()} transitions")
    print("  Run: python live_fast.py")


if __name__ == '__main__':
    main()
