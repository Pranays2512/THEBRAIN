"""
train_dialogue.py — Train brain on conversational patterns.

This is genuine learning, not hardcoding.

The brain hears each dialogue exchange many times.
When it hears you say something, the TP matrix learns
what word sequence comes next (the brain's response).
The dopamine system reinforces the correct response words
with high reward, so those transitions become strong.

After training, the brain's TP matrix naturally continues
"how are you" → "i feel good" because it was reinforced
thousands of times — not because any rule says so.

This is how children learn language: repetition + reward,
not rules.

Usage:
    python train_dialogue.py
    # Saves to brain_trained.pkl when done
    # Run live.py after to talk to the trained brain
"""

import sys, time, os
import numpy as np
from collections import Counter

sys.dont_write_bytecode = True
from brain import Brain
from vocab_extra import VOCABULARY, SILENCE, N_MFCC

BRAIN_FILE   = 'brain_trained.pkl'
N_EPOCHS     = 20       # 359 dialogues × 20 epochs = 7180 reinforced exchanges
REWARD_SELF  = 0.8      # reward when brain hears its OWN correct response
REWARD_INPUT = 0.1      # small reward just for hearing human input
NOISE_INPUT  = 0.10     # noise for "you" words (external voice)
NOISE_SELF   = 0.05     # noise for brain's own words (self-hearing is clearer)
SAVE_EVERY   = 5        # save checkpoint every N epochs
EVAL_EVERY   = 5        # evaluate every N epochs
PATIENCE     = 3        # stop if eval declines for this many consecutive evals

rng = np.random.default_rng(42)


def say(word: str, noise_std: float = 0.10) -> list:
    """Return noisy MFCC frames for a word."""
    if word not in VOCABULARY:
        return []
    mean_vec, n_frames = VOCABULARY[word]
    return [(mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
            for _ in range(n_frames)]


def feed_words(brain: Brain, words: list[str],
               reward: float, noise_std: float):
    """Feed a word sequence through the brain with given reward."""
    bmus = []
    for word in words:
        frames = say(word, noise_std)
        if not frames:
            continue
        for i, frame in enumerate(frames):
            brain.hear(frame)
            r = reward if i == len(frames) - 1 else 0.0
            out = brain.step(reward=r)
            bmus.append(out['m71_phoneme_bmu'])
        brain.hear(SILENCE)
        brain.step()
    return bmus


def tokenize(text: str) -> list[str]:
    """Split text into known vocabulary words only."""
    words = text.lower().split()
    known = [w for w in words if w in VOCABULARY]
    return known


def train_epoch(brain: Brain, dialogues: list, epoch: int) -> dict:
    """Run one pass through all dialogues."""
    n_exchanges = 0
    n_words_heard = 0
    n_words_learned = 0
    total_reward = 0.0

    # Shuffle order each epoch so brain doesn't overfit to sequence
    order = rng.permutation(len(dialogues))

    for idx in order:
        you_text, brain_text = dialogues[idx]

        you_words   = tokenize(you_text)
        brain_words = tokenize(brain_text)

        if not you_words or not brain_words:
            continue

        # ── Word-level TP: record the full exchange ──────────────
        # Feed input words → separator → response words → end
        for w in you_words:
            brain.hear_word(w)
        brain.hear_word_separator()
        for w in brain_words:
            brain.hear_word(w)
        brain.hear_word_end()

        # ── Phase 1: Brain hears YOU speak ──────────────────────────
        # Small positive reward — this is just context, not the target
        feed_words(brain, you_words, reward=REWARD_INPUT, noise_std=NOISE_INPUT)
        n_words_heard += len(you_words)

        # Settling — integrate what was heard before responding
        for _ in range(8):
            brain.hear(SILENCE)
            brain.step()

        # ── Phase 2: Brain hears the CORRECT response ────────────────
        # High reward — this is what it should learn to say
        # This is like a parent saying the right word after a child's attempt
        bmus = feed_words(brain, brain_words,
                          reward=REWARD_SELF, noise_std=NOISE_SELF)
        n_words_learned += len(brain_words)
        total_reward += REWARD_SELF * len(brain_words)

        # Record this turn for dreaming
        if bmus:
            brain.record_turn(bmus, REWARD_SELF,
                              brain.selfmodel._last_label)

        # Short silence between exchanges
        brain.hear(SILENCE)
        brain.step()
        n_exchanges += 1

    # ── Dream after each epoch to consolidate ────────────────────────
    if hasattr(brain, '_word_trajectories') and brain._word_trajectories:
        brain.dream_language(n_sequences=20)

    return {
        'epoch': epoch,
        'exchanges': n_exchanges,
        'words_heard': n_words_heard,
        'words_learned': n_words_learned,
        'mean_reward': total_reward / max(n_words_learned, 1),
    }


def evaluate(brain: Brain, dialogues: list, n_sample: int = 10) -> float:
    """
    Quick evaluation: for N sample inputs, generate a TP response
    and measure how many words overlap with the target.
    Returns mean word overlap ratio (0–1).
    """
    from collections import Counter

    N = brain.phoneme_seq._P.shape[0]

    # Build bmu→word map quickly
    bmu_to_word: dict[int, str] = {}
    for word in list(VOCABULARY.keys())[:100]:   # quick subset
        mean_vec, _ = VOCABULARY[word]
        brain.hear(mean_vec.astype(np.float32))
        out = brain.step()
        bmu = out['m71_phoneme_bmu']
        if bmu not in bmu_to_word:
            bmu_to_word[bmu] = word

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

        # Feed input
        heard_bmus = []
        for word in you_words:
            mean_vec, _ = VOCABULARY[word]
            brain.hear(mean_vec.astype(np.float32))
            out = brain.step()
            heard_bmus.append(out['m71_phoneme_bmu'])

        # TP walk
        if not heard_bmus:
            continue
        tp = sum(brain.phoneme_seq._P[b] for b in heard_bmus if 0 <= b < N)
        total = float(tp.sum())
        if total < 1e-9:
            continue
        tp /= total

        # Sample top 5 BMUs
        top_bmus = np.argsort(tp)[-5:]
        generated = set()
        for b in top_bmus:
            w = bmu_to_word.get(int(b))
            if w:
                generated.add(w)

        if brain_words:
            overlap = len(generated & brain_words) / len(brain_words)
            overlaps.append(overlap)

    return float(np.mean(overlaps)) if overlaps else 0.0


def main():
    from dialogue_corpus import DIALOGUES

    # ── Load brain ────────────────────────────────────────────────────
    if os.path.exists(BRAIN_FILE):
        print(f"Loading brain from {BRAIN_FILE}...")
        brain = Brain.load(BRAIN_FILE)
    else:
        print("No brain found. Run teach_english.py first.")
        sys.exit(1)

    # Ensure word_tp exists (backwards compat with old pickles)
    if not hasattr(brain, 'word_tp'):
        from brain import WordTransitionPredictor
        brain.word_tp = WordTransitionPredictor()

    print(f"  Vocabulary: {len(VOCABULARY)} words")
    print(f"  Corpus:     {len(DIALOGUES)} dialogues")
    print(f"  Epochs:     {N_EPOCHS}  (early stopping patience={PATIENCE})")
    print(f"  Reward:     input={REWARD_INPUT}, self={REWARD_SELF}")
    print()

    # ── Training loop with early stopping ──────────────────────────────
    t_start = time.time()
    best_score = -1.0
    patience_counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        stats = train_epoch(brain, DIALOGUES, epoch)

        elapsed = time.time() - t_start
        print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
              f"exchanges={stats['exchanges']}  "
              f"learned={stats['words_learned']} words  "
              f"reward={stats['mean_reward']:.3f}  "
              f"word_tp={brain.word_tp.n_transitions()} transitions  "
              f"[{elapsed:.0f}s]")

        # Checkpoint save
        if epoch % SAVE_EVERY == 0:
            brain.save(BRAIN_FILE)
            print(f"    → Saved checkpoint to {BRAIN_FILE}")

        # Evaluation with early stopping
        if epoch % EVAL_EVERY == 0:
            score = evaluate(brain, DIALOGUES, n_sample=15)
            print(f"    → Eval word overlap: {score:.3f}")

            # Also test word-level TP quality
            tp_score = evaluate_word_tp(brain, DIALOGUES, n_sample=15)
            print(f"    → Word TP accuracy: {tp_score:.3f}")

            combined = 0.3 * score + 0.7 * tp_score  # word TP matters more
            if combined > best_score:
                best_score = combined
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"    → Early stopping: score declined for {PATIENCE} evals")
                    break

    # ── Final save ────────────────────────────────────────────────────
    brain.save(BRAIN_FILE)
    total = time.time() - t_start
    print(f"\n  Training complete in {total:.1f}s")
    print(f"  Word TP: {brain.word_tp.n_words()} words, "
          f"{brain.word_tp.n_transitions()} transitions")
    print(f"  Brain saved → {BRAIN_FILE}")
    print(f"  Run: python live.py")


def evaluate_word_tp(brain: Brain, dialogues: list, n_sample: int = 10) -> float:
    """
    Evaluate word-level TP accuracy:
    For each sample dialogue, generate a response from the input words
    and measure word overlap with the expected response.
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

        # Generate using word TP
        generated = brain.word_tp_generate(you_words, max_len=8)
        generated_set = set(generated)

        if brain_words:
            overlap = len(generated_set & brain_words) / len(brain_words)
            overlaps.append(overlap)

    return float(np.mean(overlaps)) if overlaps else 0.0


if __name__ == '__main__':
    main()
