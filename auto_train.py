#!/usr/bin/env python3
"""
auto_train.py — Automated zone-response reinforcement training.

Runs thousands of exchanges without human input:
  1. Pick a random zone and input word from that zone
  2. Feed through acoustic SOM to get BMUs
  3. Generate a response via the intent bridge
  4. Score it: does the response contain zone-expected words?
  5. Hear own response back with appropriate reward
  6. Dream every N turns to consolidate

This builds up correct zone→expression associations much faster than
manual conversation. After running, use live_fast.py for fine-tuning
via real interaction.

Usage:
    python auto_train.py              # 2000 turns
    python auto_train.py --turns 5000
    python auto_train.py --turns 500 --quick
"""

import sys
import os
import time
import argparse
import numpy as np
from collections import defaultdict

sys.dont_write_bytecode = True

from brain_fast import N_MFCC
from vocab_core import VOCABULARY

BRAIN_FILE = 'brain_fast.pkl'
rng = np.random.default_rng(int(time.time()) % (2**31))

if not os.path.exists(BRAIN_FILE):
    print(f"No {BRAIN_FILE} found. Run train_world6_fast.py first.")
    sys.exit(1)

# ── Import live_fast (loads the brain once, updates BMU map, prints status) ───
import live_fast as lf

# Use live_fast's already-loaded, already-updated brain — no double load.
brain = lf.brain

ZONE_ANCHORS    = lf.ZONE_ANCHORS
ZONE_EXPRESSION = lf.ZONE_EXPRESSION
_WORD_TO_ZONE   = lf._WORD_TO_ZONE

# ── Acoustic helpers ──────────────────────────────────────────────────────────

def say_frames(word: str, noise_std: float = 0.08) -> list[np.ndarray]:
    if word not in VOCABULARY:
        return []
    mean_vec, n_frames = VOCABULARY[word]
    return [
        (mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
        for _ in range(n_frames)
    ]


def feed_word_acoustic(word: str, reward: float = 0.0) -> int | None:
    frames = say_frames(word)
    if not frames:
        return None
    for i, frame in enumerate(frames):
        brain.hear(frame)
        out = brain.step(reward=reward if i == len(frames) - 1 else 0.0)
    brain.hear(np.zeros(N_MFCC, dtype=np.float32))
    brain.step()
    return out['bmu']


def feed_input_words(words: list[str]) -> list[int]:
    bmus = []
    for word in words:
        bmu = feed_word_acoustic(word, reward=0.05)
        if bmu is not None:
            bmus.append(bmu)
    for _ in range(4):
        brain.hear(np.zeros(N_MFCC, dtype=np.float32))
        brain.step()
    return bmus


def hear_response(words: list[str], reward: float) -> list[int]:
    bmus = []
    for word in words:
        bmu = feed_word_acoustic(word, reward=reward)
        if bmu is not None:
            bmus.append(bmu)
    if bmus:
        brain.record_turn(bmus, max(reward, 0.0))
    return bmus


# ── Response scorer ───────────────────────────────────────────────────────────

def score_response(response_words: list[str], expected_zone: str) -> float:
    """
    Score 0..1: how zone-appropriate is the response?

    A response scores 1.0 if all its words are in the expected zone's
    expression list. Scores 0 if no words match.
    """
    if not response_words:
        return 0.0
    expected = set(ZONE_EXPRESSION[expected_zone])
    # Also accept zone anchor words as valid (they ground the zone)
    valid = expected | set(ZONE_ANCHORS[expected_zone])
    hits = sum(1 for w in response_words if w in valid)
    return hits / len(response_words)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_auto_training(n_turns: int = 2000, dream_every: int = 20,
                      verbose: bool = True):
    zones     = list(ZONE_ANCHORS.keys())
    stats     = defaultdict(lambda: {'attempts': 0, 'rewarded': 0, 'score_sum': 0.0})
    rewarded  = 0
    total_score = 0.0

    print(f"\n{'='*60}")
    print(f"AUTO TRAINING: {n_turns} turns, dream every {dream_every}")
    print(f"{'='*60}\n")

    t0 = time.time()

    for turn in range(1, n_turns + 1):
        # Pick a zone and 1-2 anchor words from it
        zone        = rng.choice(zones)
        anchors     = [w for w in ZONE_ANCHORS[zone] if w in VOCABULARY]
        if not anchors:
            continue
        n_words     = rng.integers(1, min(3, len(anchors) + 1))
        input_words = list(rng.choice(anchors, size=int(n_words), replace=False))

        # Feed input through acoustic pipeline
        heard_bmus = feed_input_words(input_words)

        # Generate response via intent bridge
        response   = lf.generate_response(heard_bmus, input_words)
        resp_words = response.split() if response else []

        # Score and reward
        score  = score_response(resp_words, zone)
        reward = 1.0 if score >= 0.5 else 0.0

        if resp_words:
            hear_response(resp_words, reward=reward)

        # Teach WordTP the correct pattern (regardless of what was generated)
        # This reinforces the zone→expression associations directly
        correct_seeds = [w for w in ZONE_EXPRESSION[zone][:3] if w in VOCABULARY]
        if correct_seeds and reward < 1.0:
            for w in input_words:
                brain.word_tp.observe(w)
            brain.hear_word_separator()
            for w in correct_seeds:
                brain.word_tp.observe(w)
            brain.hear_word_end()

        # Stats
        stats[zone]['attempts']  += 1
        stats[zone]['rewarded']  += int(reward > 0)
        stats[zone]['score_sum'] += score
        if reward > 0:
            rewarded += 1
        total_score += score

        # Dream
        if turn % dream_every == 0:
            brain.dream(n_sequences=15)

        # Progress report
        if verbose and turn % 200 == 0:
            elapsed  = time.time() - t0
            accuracy = rewarded / turn
            avg_score = total_score / turn
            print(f"  Turn {turn:5d}/{n_turns}  "
                  f"reward_rate={accuracy:.2f}  "
                  f"avg_score={avg_score:.2f}  "
                  f"[{elapsed:.0f}s]")
            for z in zones:
                s = stats[z]
                if s['attempts'] > 0:
                    rate = s['rewarded'] / s['attempts']
                    avg  = s['score_sum'] / s['attempts']
                    print(f"    {z:8s}: {s['attempts']:4d} tries, "
                          f"reward={rate:.2f}, avg={avg:.2f}")
            print()

    # Final report
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"  Total turns:  {n_turns}")
    print(f"  Rewarded:     {rewarded} ({rewarded/n_turns:.1%})")
    print(f"  Avg score:    {total_score/n_turns:.3f}")
    print(f"  Word TP:      {brain.word_tp.n_words()} words, "
          f"{brain.word_tp.n_transitions()} transitions")

    # Quick generation check
    print(f"\n{'─'*40}")
    print("Response check after training:")
    test_cases = [
        (['hungry'],           'food'),
        (['hurt', 'wall'],     'pain'),
        (['tired', 'sleep'],   'rest'),
        (['go', 'open'],       'action'),
        (['hi', 'hello'],      'social'),
        (['afraid', 'danger'], 'danger'),
    ]
    for words, expected in test_cases:
        bmus  = feed_input_words(words)
        resp  = lf.generate_response(bmus, words)
        score = score_response(resp.split(), expected) if resp else 0.0
        ok    = '✓' if score >= 0.5 else '~'
        print(f"  {ok} {str(words):25s} [{expected:6s}]  → {resp!r}  (score={score:.2f})")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--turns', type=int, default=2000)
    parser.add_argument('--dream-every', type=int, default=20)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    run_auto_training(
        n_turns    = args.turns,
        dream_every= args.dream_every,
        verbose    = not args.quiet,
    )

    brain.save(BRAIN_FILE)
    print(f"\nSaved to {BRAIN_FILE}")
    print("Run: python live_fast.py")


if __name__ == '__main__':
    main()
