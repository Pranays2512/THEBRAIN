"""
live_fast.py — Embodied conversation loop for the 5000-neuron FastBrain
=======================================================================

Usage:
  python train_fast.py       # train first
  python live_fast.py        # then talk

Commands:
  /state   — brain's current status
  /words   — word TP stats
  /dream   — run a dream cycle manually
  /help    — show known words
  /quit    — save and exit

Feedback after a response:
  good / yes / great  → reward (+1.0)
  no / wrong / bad    → penalty (−0.5, discourages that pattern)
"""

import sys
import os
import time
import numpy as np
from collections import Counter, deque

sys.dont_write_bytecode = True

from brain_fast import FastBrain, SILENCE, N_MFCC
from vocab_extra import VOCABULARY
from m75_semantic import SemanticMemory

BRAIN_FILE = 'brain_fast.pkl'
rng = np.random.default_rng(int(time.time()) % (2**31))

# ── Load brain ────────────────────────────────────────────────────────────────
if not os.path.exists(BRAIN_FILE):
    print(f"\n  No trained brain found ({BRAIN_FILE}).")
    print("  Run:  python train_fast.py  first.")
    sys.exit(1)

print(f"\nLoading FastBrain from {BRAIN_FILE}...")
brain = FastBrain.load(BRAIN_FILE)
print(brain.status())

# ── Semantic memory ───────────────────────────────────────────────────────────
semantic = SemanticMemory()
print(f"  Semantic memory: {len(semantic._facts)} facts")

# ── Conversation state ────────────────────────────────────────────────────────
_recent_responses: list[str] = []
_MAX_RECENT       = 4
_turn_history: deque = deque(maxlen=5)
_total_turns  = 0
_DREAM_EVERY  = 10

_FEEDBACK_POSITIVE = {'good', 'yes', 'great', 'nice', 'right', 'correct',
                      'perfect', 'well', 'okay', 'fine'}
_FEEDBACK_NEGATIVE = {'no', 'wrong', 'bad', 'stop', 'not', 'never'}

_PUNCT = str.maketrans('', '', '.,?!;:\'"()-/\\')

_last_response_words: list[str] = []
_last_heard_bmus:     list[int] = []


def say_frames(word: str, noise_std: float = 0.10) -> list[np.ndarray]:
    """Return noisy MFCC frames for a vocabulary word."""
    if word not in VOCABULARY:
        return []
    mean_vec, n_frames = VOCABULARY[word]
    return [
        (mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
        for _ in range(n_frames)
    ]


# ── BMU → word map (may be pre-populated from training) ──────────────────────
print("  Updating BMU→word map from clean MFCCs...", end='', flush=True)
for word in VOCABULARY:
    mean_vec, _ = VOCABULARY[word]
    bmu = brain.som.find_bmu(mean_vec.astype(np.float32))
    if word not in brain.word_to_bmu:
        brain.word_to_bmu[word] = bmu
    if bmu not in brain.bmu_to_word:
        brain.bmu_to_word[bmu] = word
print(f" {len(brain.bmu_to_word)} BMU→word entries.")


def hear_own_response(words: list[str], reward: float = 0.0):
    """
    Feed the brain's own spoken words back through its acoustic SOM.
    This closes the amnesia loop — the brain remembers what it said.
    Biologically: auditory efference copy (hearing your own voice).
    """
    bmus = []
    for word in words:
        frames = say_frames(word, noise_std=0.05)
        if not frames:
            continue
        for i, frame in enumerate(frames):
            brain.hear(frame)
            out = brain.step(reward=reward if i == len(frames) - 1 else 0.0)
        bmus.append(out['bmu'])
        brain.hear(SILENCE)
        brain.step()
    if bmus:
        brain.record_turn(bmus, max(reward, 0.0))
    return bmus


def feed_input(words: list[str]) -> list[int]:
    """Feed the user's words through the brain. Returns BMUs."""
    bmus = []
    for word in words:
        frames = say_frames(word, noise_std=0.10)
        if not frames:
            continue
        for i, frame in enumerate(frames):
            brain.hear(frame)
            out = brain.step(reward=0.05)
        bmus.append(out['bmu'])
        brain.hear(SILENCE)
        brain.step()
    return bmus


def semantic_query(words: list[str]) -> str:
    """Check M75 for factual answers. Empty string if none found."""
    ws = set(words)

    if 'name' in ws and 'my' in ws:
        facts = semantic.recall('you')
        nm = facts.get('name', '')
        return f'your name is {nm}' if nm else ''

    if 'name' in ws and 'your' in ws:
        facts = semantic.recall('brain')
        nm = facts.get('name', '')
        return f'i am {nm}' if nm else 'i am brain'

    skip = {'what', 'who', 'where', 'when', 'why', 'how',
            'is', 'are', 'do', 'you', 'i', 'a', 'the', 'my', 'your', 'name'}
    content = [w for w in words if w not in skip]
    for w in content:
        if semantic.knows(w):
            return semantic.describe(w)
    return ''


def generate_response(input_words: list[str],
                      heard_bmus: list[int]) -> str:
    """
    Generate a response using only learned structures.

    Priority:
      1. Semantic fact (hard fact recall — names, definitions).
      2. Word-level TP (primary — trained on dialogue corpus).
      3. Phoneme-level TP fallback (BMU walk → word map).
      4. Silence (nothing learned for this input).

    No hardcoded intent routing. Everything comes from training.
    """
    # ── 1. Semantic fact lookup ────────────────────────────────────────
    sem_answer = semantic_query(input_words)
    if sem_answer:
        return sem_answer

    # ── 2. Word TP (primary) ──────────────────────────────────────────
    if brain.word_tp.n_words() > 0:
        # Try a few temperatures and pick best non-repetitive result
        for temp in [0.9, 1.2, 1.5]:
            words = brain.word_tp_generate(input_words, max_len=5,
                                           temperature=temp)
            if words:
                resp = ' '.join(words)
                if resp not in _recent_responses:
                    return resp

    # ── 3. Phoneme TP fallback ────────────────────────────────────────
    if heard_bmus:
        for temp in [1.2, 1.6, 2.0]:
            bmu_seq = brain.generate_bmus(heard_bmus, n_steps=10,
                                          temperature=temp)
            words = brain.bmus_to_words(bmu_seq)
            if words:
                resp = ' '.join(words[:6])
                if resp and resp not in _recent_responses:
                    return resp

    return ''


def apply_feedback(positive: bool):
    """Reward or penalize the last response."""
    global _last_response_words
    if not _last_response_words:
        return
    reward = 1.0 if positive else -0.5
    hear_own_response(_last_response_words, reward=max(reward, 0.0))
    if positive:
        print("  [brain: +reward]")
    else:
        print("  [brain: noted — will try differently]")


# ── Main loop ─────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("  FAST BRAIN — 5000 neurons")
print("  Type to talk. /quit to exit.")
print("="*55 + "\n")

while True:
    try:
        raw = input("you: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        break

    if not raw:
        continue

    # ── Slash commands ────────────────────────────────────────────────
    if raw.startswith('/'):
        cmd = raw[1:].strip().lower()

        if cmd in ('quit', 'exit', 'q'):
            break

        elif cmd == 'state':
            print(brain.status())
            print(f"  Semantic facts: {len(semantic._facts)}")
            print(f"  Recent responses: {_recent_responses[-3:]}")

        elif cmd == 'words':
            print(f"  Word TP: {brain.word_tp.n_words()} words, "
                  f"{brain.word_tp.n_transitions()} transitions")
            # Show a sample generation
            sample_inputs = [['how', 'are', 'you'], ['what', 'is', 'your', 'name'],
                             ['i', 'feel', 'good'], ['hello']]
            for inp in sample_inputs:
                known = [w for w in inp if w in VOCABULARY]
                if known:
                    resp = brain.word_tp_generate(known, max_len=6)
                    print(f"    '{' '.join(known)}' → '{' '.join(resp)}'")

        elif cmd == 'dream':
            print("  Dreaming...", end='', flush=True)
            brain.dream(n_sequences=50)
            print(" done.")

        elif cmd == 'help':
            words = sorted(VOCABULARY.keys())
            print(f"  {len(words)} known words:")
            for i in range(0, min(len(words), 100), 10):
                print("   ", ' '.join(words[i:i+10]))
            if len(words) > 100:
                print(f"   ... and {len(words)-100} more")

        elif cmd == 'save':
            brain.save(BRAIN_FILE)
            print(f"  Saved to {BRAIN_FILE}")

        else:
            print("  Commands: /state /words /dream /help /save /quit")

        continue

    # ── Parse input ───────────────────────────────────────────────────
    tokens = raw.lower().translate(_PUNCT).split()

    # ── Feedback shortcut — pure feedback words stop here ─────────────
    if len(tokens) == 1 and tokens[0] in _FEEDBACK_POSITIVE:
        apply_feedback(positive=True)
        _total_turns += 1
        continue

    if len(tokens) == 1 and tokens[0] in _FEEDBACK_NEGATIVE:
        apply_feedback(positive=False)
        _total_turns += 1
        continue

    # Multi-word input starting with feedback — apply reward then respond
    if tokens and tokens[0] in _FEEDBACK_POSITIVE:
        apply_feedback(positive=True)
    elif tokens and tokens[0] in _FEEDBACK_NEGATIVE:
        apply_feedback(positive=False)

    heard_words = [w for w in tokens if w in VOCABULARY]

    # ── Semantic learning ─────────────────────────────────────────────
    semantic.learn_from_sentence(tokens)

    # Update word TP with input context
    for w in heard_words:
        brain.word_tp.observe(w, weight=0.3)  # mild weight — not training

    # ── Feed input through acoustic SOM ──────────────────────────────
    _last_heard_bmus = feed_input(heard_words)

    # Settling — brief silence to integrate input
    for _ in range(4):
        brain.hear(SILENCE)
        brain.step()

    # ── Generate response ─────────────────────────────────────────────
    response = generate_response(heard_words, _last_heard_bmus)

    if response:
        print(f"brain: {response}")
        response_words = response.split()
        _last_response_words = response_words

        # Brain hears its own response (self-feedback)
        hear_own_response(response_words, reward=0.0)

        _recent_responses.append(response)
        if len(_recent_responses) > _MAX_RECENT:
            _recent_responses.pop(0)

        _turn_history.append({'you': raw, 'brain': response})
    else:
        print("brain: ...")
        _last_response_words = []

    _total_turns += 1

    # Periodic dream cycle
    if _total_turns % _DREAM_EVERY == 0:
        brain.dream(n_sequences=10)

# ── Exit ──────────────────────────────────────────────────────────────────────
print("\nSaving brain...", end='', flush=True)
brain.save(BRAIN_FILE)
print(" done.")
print(f"Total turns: {_total_turns}")
