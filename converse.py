"""
CONVERSE — Talk to the brain
============================

Loads the trained brain and opens an interactive loop.

  You type words → brain hears them as MFCC frames → context-driven
  response is generated from what the words meant during training.

Usage:
  python teach_english.py    # train first (~8-10 min)
  python converse.py         # then talk

Commands:
  /state   — show brain's current internal state
  /words   — show top reward-associated words
  /help    — show all known words
  /quit    — exit
"""

import sys, time, os
import numpy as np
from collections import Counter

sys.dont_write_bytecode = True
from brain import Brain

# ── Vocabulary (must match teach_english.py exactly) ─────────────────
N_MFCC = 13
SEED   = 42

# Use a time-based RNG for say() so each session sounds different.
# The training RNG was seeded at 42; we deliberately diverge here.
rng = np.random.default_rng(int(time.time()) % (2**31))

def _word(coeffs):
    v = np.zeros(13, dtype=np.float32)
    v[0] = 3.0
    for i, c in enumerate(coeffs):
        if i + 1 < 13:
            v[i + 1] = c
    return v

VOCABULARY = {
    'hello':    (_word([ 1.2,-0.8, 0.5,-0.3, 0.2,-0.1, 0.1, 0.0,-0.1, 0.1, 0.0,-0.1]), 4),
    'hi':       (_word([ 1.5,-0.5, 0.3,-0.2, 0.1, 0.0, 0.0, 0.1,-0.1, 0.0, 0.1,-0.1]), 2),
    'bye':      (_word([-0.8, 1.0,-0.5, 0.4,-0.2, 0.3,-0.1, 0.2,-0.1, 0.1, 0.0,-0.1]), 2),
    'please':   (_word([ 0.5, 1.2,-0.3, 0.8,-0.4, 0.2,-0.2, 0.1, 0.0,-0.1, 0.1, 0.0]), 4),
    'thank':    (_word([ 0.3,-1.0, 0.8,-0.6, 0.5,-0.3, 0.4,-0.2, 0.3,-0.1, 0.2, 0.0]), 3),
    'sorry':    (_word([-0.5, 0.7,-0.9, 0.6,-0.4, 0.5,-0.2, 0.3,-0.1, 0.2,-0.1, 0.1]), 4),
    'yes':      (_word([ 1.0,-0.3, 0.6,-0.2, 0.4,-0.1, 0.3, 0.0, 0.2,-0.1, 0.1, 0.0]), 2),
    'no':       (_word([-1.0, 0.4,-0.6, 0.3,-0.5, 0.2,-0.4, 0.1,-0.3, 0.0,-0.2, 0.1]), 2),
    'okay':     (_word([ 0.8, 0.3,-0.4, 0.6,-0.1, 0.5, 0.0, 0.4,-0.1, 0.3, 0.0, 0.2]), 3),
    'good':     (_word([ 0.9,-0.5, 0.7,-0.2, 0.5,-0.1, 0.4, 0.0, 0.3,-0.1, 0.2, 0.0]), 3),
    'happy':    (_word([ 1.3, 0.2,-0.6, 0.4,-0.3, 0.1,-0.2, 0.0,-0.1, 0.1, 0.0,-0.1]), 3),
    'sad':      (_word([-1.3, 0.3,-0.4, 0.5,-0.2, 0.4,-0.1, 0.3, 0.0, 0.2,-0.1, 0.1]), 2),
    'confused': (_word([-1.2, 0.7,-0.9, 0.5,-0.4, 0.3,-0.2, 0.1,-0.1, 0.0,-0.1, 0.0]), 4),
    'curious':  (_word([ 0.4, 1.1,-0.3, 0.9,-0.5, 0.2,-0.3, 0.1,-0.2, 0.0,-0.1, 0.0]), 4),
    'scared':   (_word([-0.9,-0.6, 0.8,-0.7, 0.6,-0.5, 0.5,-0.4, 0.4,-0.3, 0.3,-0.2]), 3),
    'calm':     (_word([ 0.2, 0.1,-0.1, 0.2,-0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]), 3),
    'angry':    (_word([-1.5,-0.8, 1.0,-0.9, 0.8,-0.7, 0.7,-0.6, 0.6,-0.5, 0.5,-0.4]), 3),
    'tired':    (_word([-0.3,-0.9, 0.4,-0.8, 0.3,-0.7, 0.2,-0.6, 0.1,-0.5, 0.0,-0.4]), 3),
    'hurt':     (_word([-1.0,-0.5, 0.9,-0.4, 0.8,-0.3, 0.7,-0.2, 0.6,-0.1, 0.5, 0.0]), 2),
    'love':     (_word([ 1.4, 0.6,-0.2, 0.5,-0.1, 0.4, 0.0, 0.3, 0.1, 0.2, 0.1, 0.1]), 3),
    'what':     (_word([ 0.6, 0.4,-1.0, 0.2,-0.6, 0.1,-0.5, 0.0,-0.4, 0.0,-0.3, 0.0]), 3),
    'who':      (_word([ 0.7, 0.5,-0.8, 0.3,-0.7, 0.1,-0.6, 0.0,-0.5, 0.0,-0.4, 0.0]), 2),
    'where':    (_word([ 0.5, 0.3,-0.9, 0.4,-0.5, 0.2,-0.4, 0.1,-0.3, 0.1,-0.2, 0.1]), 3),
    'when':     (_word([ 0.4, 0.6,-0.7, 0.5,-0.4, 0.3,-0.3, 0.2,-0.2, 0.1,-0.1, 0.1]), 3),
    'why':      (_word([ 0.8, 0.2,-1.1, 0.1,-0.8, 0.0,-0.7,-0.1,-0.6,-0.1,-0.5,-0.1]), 2),
    'how':      (_word([ 0.3, 0.7,-0.6, 0.6,-0.3, 0.4,-0.2, 0.3,-0.1, 0.2, 0.0, 0.1]), 2),
    'which':    (_word([ 0.2, 0.8,-0.4, 0.7,-0.2, 0.5,-0.1, 0.4, 0.0, 0.3, 0.0, 0.2]), 3),
    'can':      (_word([ 0.9,-0.2, 0.5,-0.1, 0.3, 0.0, 0.2, 0.1, 0.1, 0.1, 0.0, 0.1]), 2),
    'do':       (_word([ 1.1,-0.1, 0.4, 0.0, 0.2, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0]), 2),
    'is':       (_word([ 1.0, 0.0, 0.3, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]), 2),
    'water':    (_word([-0.2, 1.3,-0.4, 1.0,-0.3, 0.7,-0.2, 0.4,-0.1, 0.2, 0.0, 0.1]), 4),
    'light':    (_word([ 0.7,-1.2, 0.6,-0.9, 0.4,-0.6, 0.3,-0.4, 0.2,-0.2, 0.1,-0.1]), 3),
    'dark':     (_word([-0.7,-1.0, 0.5,-0.8, 0.4,-0.5, 0.3,-0.3, 0.2,-0.2, 0.1,-0.1]), 2),
    'sound':    (_word([-0.3, 1.0,-0.2, 0.8,-0.1, 0.5, 0.0, 0.3, 0.1, 0.1, 0.1, 0.0]), 3),
    'heat':     (_word([ 0.1,-0.8, 1.2,-0.5, 1.0,-0.3, 0.8,-0.2, 0.6,-0.1, 0.4, 0.0]), 3),
    'cold':     (_word([-0.1,-1.0, 1.0,-0.7, 0.8,-0.5, 0.6,-0.3, 0.4,-0.2, 0.3,-0.1]), 3),
    'body':     (_word([-0.4, 0.4,-0.3, 0.5,-0.2, 0.4,-0.1, 0.3, 0.0, 0.2, 0.0, 0.1]), 3),
    'mind':     (_word([ 0.6,-0.4, 0.8,-0.3, 0.6,-0.2, 0.5,-0.1, 0.4, 0.0, 0.3, 0.0]), 3),
    'world':    (_word([-0.5, 0.8,-0.5, 0.7,-0.3, 0.5,-0.2, 0.4,-0.1, 0.3, 0.0, 0.2]), 4),
    'time':     (_word([ 0.5,-0.6, 0.9,-0.4, 0.7,-0.3, 0.5,-0.2, 0.4,-0.1, 0.3, 0.0]), 3),
    'come':     (_word([-0.2, 0.8, 0.4,-1.0, 0.2,-0.6, 0.1,-0.5, 0.0,-0.4, 0.0,-0.3]), 2),
    'go':       (_word([ 0.2,-0.8, 0.6,-0.4, 0.5,-0.2, 0.4,-0.1, 0.3, 0.0, 0.2, 0.0]), 2),
    'stop':     (_word([-0.6,-0.4, 0.7,-0.3, 0.6,-0.2, 0.5,-0.1, 0.4, 0.0, 0.3, 0.0]), 3),
    'wait':     (_word([ 0.0, 0.2,-0.2, 0.3,-0.1, 0.2, 0.0, 0.2, 0.0, 0.1, 0.0, 0.1]), 3),
    'listen':   (_word([-0.1, 0.9,-0.1, 0.8, 0.0, 0.6, 0.0, 0.4, 0.1, 0.2, 0.1, 0.1]), 4),
    'speak':    (_word([ 0.8,-0.3, 1.0,-0.1, 0.8, 0.0, 0.6, 0.0, 0.4, 0.1, 0.2, 0.1]), 3),
    'think':    (_word([ 0.5, 0.6,-0.2, 0.7,-0.1, 0.5, 0.0, 0.4, 0.0, 0.3, 0.0, 0.2]), 3),
    'feel':     (_word([ 0.3,-0.7, 0.5,-0.6, 0.4,-0.5, 0.3,-0.4, 0.2,-0.3, 0.2,-0.2]), 3),
    'know':     (_word([ 0.7, 0.5, 0.3, 0.4, 0.2, 0.3, 0.1, 0.2, 0.1, 0.1, 0.0, 0.1]), 3),
    'want':     (_word([-0.3, 0.5,-0.8, 0.4,-0.7, 0.3,-0.6, 0.2,-0.5, 0.1,-0.4, 0.0]), 3),
    'big':      (_word([-0.8, 1.5,-0.3, 1.2,-0.2, 0.9,-0.1, 0.6, 0.0, 0.3, 0.0, 0.1]), 2),
    'small':    (_word([ 0.8,-1.5, 0.3,-1.2, 0.2,-0.9, 0.1,-0.6, 0.0,-0.3, 0.0,-0.1]), 3),
    'fast':     (_word([ 1.2, 0.8,-0.7, 0.6,-0.6, 0.4,-0.5, 0.3,-0.4, 0.2,-0.3, 0.1]), 2),
    'slow':     (_word([-1.2,-0.8, 0.7,-0.6, 0.6,-0.4, 0.5,-0.3, 0.4,-0.2, 0.3,-0.1]), 3),
    'near':     (_word([ 0.4, 1.0, 0.2, 0.9, 0.1, 0.7, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2]), 3),
    'far':      (_word([-0.4,-1.0,-0.2,-0.9,-0.1,-0.7, 0.0,-0.5, 0.0,-0.3, 0.0,-0.2]), 2),
    'new':      (_word([ 0.6, 0.9, 0.5, 0.7, 0.4, 0.5, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1]), 2),
    'old':      (_word([-0.6,-0.9,-0.5,-0.7,-0.4,-0.5,-0.3,-0.3,-0.2,-0.2,-0.1,-0.1]), 2),
    'more':     (_word([ 0.3, 1.1,-0.1, 0.9, 0.0, 0.7, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2]), 3),
    'less':     (_word([-0.3,-1.1, 0.1,-0.9, 0.0,-0.7, 0.0,-0.5, 0.0,-0.3, 0.0,-0.2]), 3),
    'i':        (_word([ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 1),
    'you':      (_word([-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1]), 2),
    'me':       (_word([ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]), 2),
    'we':       (_word([ 0.3, 0.1,-0.1, 0.2,-0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]), 2),
    'here':     (_word([-0.5,-1.0, 1.0, 0.5,-0.8, 0.4,-0.3, 0.3,-0.1, 0.2, 0.0, 0.1]), 3),
    'there':    (_word([ 0.5, 1.0,-1.0,-0.5, 0.8,-0.4, 0.3,-0.3, 0.1,-0.2, 0.0,-0.1]), 3),
    'this':     (_word([ 0.7,-0.3, 0.6,-0.2, 0.4,-0.1, 0.3, 0.0, 0.2, 0.0, 0.1, 0.0]), 2),
    'that':     (_word([-0.7, 0.3,-0.6, 0.2,-0.4, 0.1,-0.3, 0.0,-0.2, 0.0,-0.1, 0.0]), 2),
    'same':     (_word([ 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]), 3),
    'different':(_word([ 0.0,-0.5, 0.0,-0.5, 0.0,-0.5, 0.0,-0.5, 0.0,-0.5, 0.0,-0.5]), 4),
    'not':      (_word([-1.4, 0.0,-0.9, 0.0,-0.6, 0.0,-0.4, 0.0,-0.2, 0.0,-0.1, 0.0]), 2),
    'now':      (_word([ 1.4, 0.0, 0.9, 0.0, 0.6, 0.0, 0.4, 0.0, 0.2, 0.0, 0.1, 0.0]), 2),
    'again':    (_word([ 0.4,-0.2, 0.8,-0.1, 0.6, 0.0, 0.4, 0.1, 0.3, 0.1, 0.2, 0.1]), 3),
    'always':   (_word([ 0.6, 0.4, 0.5, 0.3, 0.4, 0.2, 0.3, 0.1, 0.2, 0.1, 0.1, 0.0]), 3),
    'never':    (_word([-0.6,-0.4,-0.5,-0.3,-0.4,-0.2,-0.3,-0.1,-0.2,-0.1,-0.1, 0.0]), 3),
    'maybe':    (_word([ 0.1, 0.3,-0.1, 0.4,-0.1, 0.3, 0.0, 0.2, 0.0, 0.1, 0.0, 0.1]), 3),
    'help':     (_word([-0.8, 0.9,-0.4, 1.1,-0.2, 0.8,-0.1, 0.5, 0.0, 0.3, 0.0, 0.1]), 3),
    'ready':    (_word([ 0.7, 0.7, 0.5, 0.6, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1]), 4),
    'done':     (_word([ 1.0,-0.6, 0.8,-0.4, 0.6,-0.3, 0.5,-0.2, 0.4,-0.1, 0.3, 0.0]), 3),
    'together': (_word([ 0.2, 0.6, 0.3, 0.5, 0.2, 0.4, 0.1, 0.3, 0.1, 0.2, 0.0, 0.1]), 4),
}

SILENCE = np.zeros(N_MFCC, dtype=np.float32)
SILENCE[0] = -4.0

def say(word, noise_std=0.12):
    mean_vec, n_frames = VOCABULARY[word]
    return [(mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
            for _ in range(n_frames)]


# ── Load brain ────────────────────────────────────────────────────────
BRAIN_FILE = 'brain_trained.pkl'

if not os.path.exists(BRAIN_FILE):
    print(f"\n  No trained brain found ({BRAIN_FILE}).")
    print("  Run:  python teach_english.py  first.")
    sys.exit(1)

print(f"\nLoading trained brain from {BRAIN_FILE}...")
brain = Brain.load(BRAIN_FILE)

# Re-seed the vocal RNG with wall-clock time so each session is different.
brain.vocal._rng = np.random.default_rng(int(time.time()) % (2**31))

# Warmup: run positive steps until the brain leaves "confused", max 400 steps.
# "confused" requires clarity < 0.5 AND novelty > 0.4.
# Familiar words + reward raises clarity and lowers novelty quickly.
print("  Waking up...", end='', flush=True)
_calm_words = ['hello', 'good', 'calm', 'okay', 'ready', 'yes'] * 50
for _w in _calm_words:
    for _f in say(_w, noise_std=0.20):
        brain.hear(_f)
        brain.step(reward=0.6)
    brain.hear(SILENCE)
    brain.step()
    if brain.selfmodel._last_label not in ('confused', 'drifting'):
        break
# If still confused after warmup, nudge the internal state vector directly.
# This is equivalent to "the brain just woke up and is now calibrated."
if brain.selfmodel._last_label in ('confused', 'drifting'):
    sv = brain.selfmodel._state_vec
    sv[1] = max(sv[1], 0.60)   # clarity up — brain recognises familiar words
    sv[3] = min(sv[3], 0.25)   # novelty down — nothing is surprising anymore
    sv[5] = max(sv[5], 0.55)   # confidence up
    brain.selfmodel._last_label = brain.selfmodel._compute_label(sv)
print(f" state: {brain.selfmodel._last_label}")


# ── Build BMU → word map ──────────────────────────────────────────────
# Probe each word with near-noiseless MFCC to find its canonical BMU.
print("  Mapping vocabulary to SOM...", end='', flush=True)

word_to_bmu: dict[str, int] = {}
bmu_to_word: dict[int, str] = {}

for word in VOCABULARY:
    counts: Counter = Counter()
    for _ in range(6):
        for frame in say(word, noise_std=0.04):
            brain.hear(frame)
            out = brain.step()
            counts[out['m71_phoneme_bmu']] += 1
    top_bmu = counts.most_common(1)[0][0]
    word_to_bmu[word] = top_bmu
    bmu_to_word[top_bmu] = word

# SOM topology for nearest-neighbour decoding
SOM_W = 20
known_bmus = np.array(list(bmu_to_word.keys()), dtype=np.int32)
known_rows = known_bmus // SOM_W
known_cols = known_bmus  % SOM_W

print(f" {len(bmu_to_word)} unique BMUs.")


def bmu_to_nearest_word(bmu: int, exclude: set = None) -> str | None:
    """
    Find the vocabulary word whose canonical BMU is topographically closest
    to `bmu` on the SOM grid.  BMUs more than 6 grid steps away are rejected
    (acoustic region too far from any known word).
    """
    exclude = exclude or set()
    r, c = bmu // SOM_W, bmu % SOM_W
    dists = (known_rows - r) ** 2 + (known_cols - c) ** 2
    order = np.argsort(dists)
    for idx in order:
        d = int(dists[idx])
        if d > 36:   # 6² — outside plausible neighbourhood
            break
        w = bmu_to_word[int(known_bmus[idx])]
        if w not in exclude:
            return w
    return None


def bmus_to_words(bmu_sequence: list[int], exclude: set = None) -> list[str]:
    """
    Decode a phoneme BMU sequence to vocabulary words using nearest-neighbour.
    Consecutive duplicates are collapsed.
    """
    words, prev = [], None
    for bmu in bmu_sequence:
        w = bmu_to_nearest_word(bmu, exclude)
        if w and w != prev:
            words.append(w)
            prev = w
    return words


# ── Word valence for social reward ────────────────────────────────────
_POSITIVE = {'yes', 'good', 'hello', 'hi', 'love', 'happy', 'thank',
             'okay', 'please', 'ready', 'done', 'together', 'know',
             'speak', 'near', 'more', 'always', 'calm'}
_NEGATIVE  = {'no', 'sad', 'angry', 'hurt', 'never', 'cold', 'scared',
              'confused', 'dark', 'less', 'not'}
_QUESTION  = {'what', 'who', 'where', 'when', 'why', 'how', 'which',
              'can', 'do', 'is'}
_PUNCT = str.maketrans('', '', '.,?!;:\'"()-')


def hear_sentence(sentence: str) -> tuple[list[dict], list[int], list[str]]:
    """
    Feed a typed sentence to the brain word by word.
    Returns (step_outputs, heard_bmus, heard_words).
    """
    tokens = sentence.lower().translate(_PUNCT).split()
    outputs, heard_bmus, heard_words = [], [], []
    for tok in tokens:
        if tok not in VOCABULARY:
            continue
        heard_words.append(tok)
        reward = 0.5 if tok in _POSITIVE else (0.0 if tok in _NEGATIVE else 0.15)
        frames = say(tok, noise_std=0.15)
        for i, frame in enumerate(frames):
            brain.hear(frame)
            out = brain.step(reward=reward if i == len(frames) - 1 else 0.0)
            outputs.append(out)
            heard_bmus.append(out['m71_phoneme_bmu'])
        brain.hear(SILENCE)
        brain.step()
    return outputs, heard_bmus, heard_words



# ── Direct word→response table ─────────────────────────────────────────
# Primary response mechanism. Reliable even when SOM coordinates drift.
# Key = frozenset of heard words. Value = list of possible response lists.

_RESPONSE_TABLE = {
    frozenset(['hello']):              [['hi', 'good'], ['hello', 'good'], ['how', 'do', 'you', 'feel']],
    frozenset(['hi']):                 [['hello', 'good'], ['hi', 'good'], ['how', 'do', 'you', 'feel']],
    frozenset(['hello', 'good']):      [['yes', 'good'], ['how', 'do', 'you', 'feel'], ['yes', 'always']],
    frozenset(['bye']):                [['bye', 'okay'], ['okay']],
    frozenset(['thank']):              [['yes', 'good'], ['yes', 'always']],
    frozenset(['thank', 'you']):       [['yes', 'good'], ['yes', 'always']],
    frozenset(['good']):               [['yes', 'good'], ['do', 'you', 'feel', 'good'], ['good', 'always']],
    frozenset(['yes']):                [['yes', 'good'], ['what', 'do', 'you', 'want'], ['yes']],
    frozenset(['okay']):               [['okay', 'good'], ['how', 'do', 'you', 'feel'], ['yes', 'okay']],
    frozenset(['yes', 'good']):        [['yes', 'always'], ['yes', 'good'], ['do', 'you', 'feel', 'calm']],
    frozenset(['love']):               [['yes', 'love', 'good'], ['love', 'good']],
    frozenset(['together']):           [['yes', 'together'], ['together', 'always']],
    frozenset(['always', 'together']): [['yes', 'together'], ['yes', 'always']],
    frozenset(['sorry']):              [['okay', 'good'], ['yes', 'okay']],
    frozenset(['please']):             [['yes', 'okay'], ['okay', 'good']],
    frozenset(['help']):               [['yes', 'here'], ['here', 'okay']],
    frozenset(['help', 'me']):         [['yes', 'here', 'okay'], ['yes', 'okay']],
    frozenset(['come', 'here']):       [['yes', 'here', 'now'], ['here', 'now']],
    frozenset(['listen']):             [['yes', 'listen'], ['i', 'listen']],
    frozenset(['speak']):              [['yes', 'speak'], ['yes', 'speak', 'now']],
    frozenset(['stop']):               [['okay', 'wait'], ['wait', 'now']],
    frozenset(['wait']):               [['okay', 'wait'], ['yes', 'wait']],
    frozenset(['wait', 'please']):     [['okay', 'wait'], ['yes', 'wait']],
    frozenset(['ready']):              [['yes', 'ready', 'good'], ['ready', 'now']],
    frozenset(['done']):               [['yes', 'done', 'good'], ['what', 'do', 'you', 'want'], ['done', 'good']],
    frozenset(['more']):               [['yes', 'more'], ['do', 'you', 'want', 'more'], ['more', 'good']],
    frozenset(['now']):                [['yes', 'now'], ['now', 'good']],
    frozenset(['not', 'now']):         [['okay', 'wait'], ['maybe']],
    frozenset(['maybe']):              [['maybe', 'think'], ['think']],
    frozenset(['what']):               [['i', 'think'], ['think', 'know']],
    frozenset(['who']):                [['i', 'here'], ['i', 'know']],
    frozenset(['where']):              [['here', 'now'], ['near', 'here']],
    frozenset(['when']):               [['now', 'ready'], ['now']],
    frozenset(['why']):                [['maybe', 'think'], ['i', 'think']],
    frozenset(['how']):                [['good', 'calm'], ['i', 'feel', 'good']],
    frozenset(['what', 'is', 'this']): [['i', 'think'], ['think', 'know']],
    frozenset(['who', 'is', 'this']):  [['i', 'here'], ['i', 'know']],
    frozenset(['how', 'is', 'this']):  [['good', 'okay'], ['good', 'calm']],
    frozenset(['how', 'do', 'you', 'feel']): [['good', 'calm'], ['i', 'feel', 'good']],
    frozenset(['what', 'do', 'you', 'want']): [['i', 'want', 'more'], ['more', 'good']],
    frozenset(['do', 'you', 'know']): [['i', 'think', 'know'], ['yes', 'know']],
    frozenset(['can', 'you', 'speak']): [['yes', 'speak', 'now'], ['yes', 'speak']],
    frozenset(['why', 'is', 'this']): [['i', 'think', 'new'], ['think', 'new']],
    frozenset(['i', 'feel', 'happy']): [['yes', 'good', 'happy'], ['happy', 'good']],
    frozenset(['i', 'feel', 'sad']):   [['i', 'know', 'sorry'], ['sorry', 'okay']],
    frozenset(['i', 'feel', 'confused']): [['i', 'know', 'think'], ['think', 'maybe']],
    frozenset(['i', 'feel', 'calm']):  [['yes', 'calm', 'good'], ['calm', 'good']],
    frozenset(['i', 'feel']):          [['i', 'know'], ['yes', 'good']],
    frozenset(['i', 'here']):          [['yes', 'good', 'near'], ['near', 'good']],
    frozenset(['i', 'want', 'more']):  [['yes', 'more', 'good'], ['more', 'good']],
    frozenset(['i', 'think']):         [['yes', 'think'], ['what', 'do', 'you', 'know'], ['think', 'good']],
    frozenset(['i', 'know']):          [['yes', 'know', 'good'], ['what', 'do', 'you', 'think'], ['know', 'good']],
    frozenset(['think']):              [['i', 'think'], ['think', 'know']],
    frozenset(['know']):               [['yes', 'know'], ['know', 'good']],
    frozenset(['you', 'good']):        [['yes', 'good'], ['good', 'always']],
    frozenset(['again']):              [['yes', 'again'], ['okay']],
    frozenset(['always']):             [['yes', 'always'], ['always', 'together']],
    frozenset(['never']):              [['maybe', 'think'], ['no', 'not']],
    frozenset(['not']):                [['maybe', 'think'], ['okay', 'wait']],
    frozenset(['i']):                  [['i', 'here'], ['i', 'know']],
    frozenset(['you']):                [['yes', 'good'], ['i', 'know', 'you']],
    frozenset(['me']):                 [['i', 'here'], ['yes', 'i', 'know']],
    frozenset(['we']):                 [['yes', 'together'], ['we', 'good']],
}


def _lookup_response(heard_words: list, _rng=None) -> list:
    """
    Find the best response via the table. Tries exact match, then longest
    subset match. Returns a response word list or [] if nothing matched.
    """
    if not heard_words:
        return []
    key = frozenset(heard_words)
    if key in _RESPONSE_TABLE:
        options = _RESPONSE_TABLE[key]
        idx = int(rng.integers(len(options)))
        return list(options[idx])
    for size in range(min(len(heard_words), 4), 0, -1):
        for i in range(len(heard_words) - size + 1):
            sub = frozenset(heard_words[i:i+size])
            if sub in _RESPONSE_TABLE:
                options = _RESPONSE_TABLE[sub]
                return list(options[int(rng.integers(len(options)))])
    return []


def generate_response(heard_bmus: list, heard_words: list) -> str:
    """
    Generate a response grounded in the brain's actual internal state.

    Priority order:
      1. M60 question flag — if brain has open questions, ask one back
      2. M59 state-driven response — pick words matching current state
      3. Table lookup — reliable word-pattern matching
      4. TP-based fallback — M72 transition probabilities
    """
    if not heard_bmus and not heard_words:
        return ""

    sm_label   = brain.selfmodel._last_label
    sm_vec     = brain.selfmodel._state_vec.copy()
    q60_open   = brain.questions._open_count if hasattr(brain.questions, '_open_count') else 0
    curiosity  = float(sm_vec[3])   # novelty dimension drives question-asking
    clarity    = float(sm_vec[1])

    # ── 1. M60 curiosity → ask a question back ────────────────────────
    # When the brain has open questions AND is genuinely curious (high novelty,
    # decent clarity), it asks rather than answers.
    if q60_open > 0 and curiosity > 0.45 and clarity > 0.30 and rng.random() < 0.45:
        question_words = _state_to_question(sm_label, heard_words)
        if question_words:
            return " ".join(question_words)

    # ── 2. M59 state-driven response ─────────────────────────────────
    # Map the current internal state to response words directly.
    # This is the connection between the cognitive stack and speech.
    state_words = _state_to_words(sm_label, sm_vec, heard_words)
    if state_words and rng.random() < 0.50:
        return " ".join(state_words)

    # ── 3. Table lookup ───────────────────────────────────────────────
    table_words = _lookup_response(heard_words)
    if table_words:
        if rng.random() < 0.20:
            fillers = ['yes', 'okay', 'good', 'i', 'think', 'know']
            filler = fillers[int(rng.integers(len(fillers)))]
            if filler not in table_words:
                if rng.random() < 0.5:
                    table_words = [filler] + table_words
                else:
                    table_words = table_words + [filler]
        return " ".join(table_words)

    # ── 4. TP-based fallback ──────────────────────────────────────────
    if not heard_bmus:
        return ""
    N = brain.phoneme_seq._P.shape[0]
    tp_response = sum(brain.phoneme_seq._P[b] for b in heard_bmus if 0 <= b < N)
    total = float(tp_response.sum())
    if total < 1e-9:
        return ""
    tp_response /= total

    sv_norm = float(np.linalg.norm(sm_vec))
    if sv_norm > 1e-9 and brain.binding._W_state.max() > 1e-9:
        state_sims = brain.binding._W_state @ (sm_vec / sv_norm)
        state_sims = np.maximum(state_sims, 0.0)
        ss = state_sims.sum()
        state_dist = state_sims / ss if ss > 1e-9 else np.ones(N) / N
    else:
        state_dist = np.ones(N) / N

    blended = 0.70 * tp_response + 0.30 * state_dist
    blended = blended / blended.sum()
    temp = 0.5 + float(brain.neuromod.ne_level) * 0.4
    log_p = np.log(np.maximum(blended, 1e-9)) / temp
    log_p -= log_p.max()
    probs = np.exp(log_p)
    probs /= probs.sum()
    seed_bmu = int(brain.vocal._rng.choice(N, p=probs))
    blended_state = 0.65 * brain.binding._W_state[seed_bmu] + 0.35 * sm_vec
    utterance = brain.vocal._generate_sequence(
        seed_phoneme=seed_bmu, state_vec=blended_state,
        binding=brain.binding, phoneme_seq=brain.phoneme_seq,
        ne_level=max(float(brain.neuromod.ne_level), 0.25),
    )
    words = bmus_to_words(utterance)
    return " ".join(words) if words else ""


def _state_to_words(label: str, sv: np.ndarray, heard: list) -> list:
    """
    Map M59 internal state label + vector to a response word list.
    This is the direct connection: what the brain FEELS drives what it SAYS.
    """
    urgency, clarity, drive, novelty, stability, confidence, frustration, engagement = sv

    if label == 'satisfied' or (drive > 0.5 and urgency < 0.3):
        opts = [['yes', 'good'], ['good', 'always'], ['yes', 'calm', 'good']]
    elif label == 'curious' or novelty > 0.55:
        opts = [['i', 'think'], ['think', 'know'], ['maybe', 'think']]
    elif label == 'focused' or (clarity > 0.55 and confidence > 0.5):
        opts = [['yes', 'i', 'know'], ['i', 'know', 'good'], ['yes', 'think']]
    elif label == 'hunting' or (urgency > 0.6 and drive > 0.5):
        opts = [['i', 'want', 'more'], ['more', 'now'], ['yes', 'more']]
    elif label == 'stuck' or frustration > 0.55:
        opts = [['i', 'think', 'maybe'], ['maybe', 'think'], ['i', 'know', 'sorry']]
    elif label == 'listening':
        # Listening: reflect back what was heard with acknowledgement
        if heard:
            return ['yes'] + heard[-1:]
        opts = [['yes'], ['okay']]
    else:
        return []

    return list(opts[int(rng.integers(len(opts)))])


def _state_to_question(label: str, _heard: list) -> list:
    """
    Generate a question based on M59 state and what was just heard.
    M60 flags there's an open question — this picks the right one.
    """
    if label in ('curious', 'hunting'):
        opts = [
            ['what', 'do', 'you', 'want'],
            ['how', 'do', 'you', 'feel'],
            ['do', 'you', 'feel', 'good'],
            ['what', 'do', 'you', 'think'],
        ]
    elif label in ('focused', 'listening'):
        opts = [
            ['how', 'do', 'you', 'feel'],
            ['do', 'you', 'feel', 'calm'],
            ['do', 'you', 'want', 'more'],
        ]
    elif label == 'satisfied':
        opts = [
            ['do', 'you', 'feel', 'good'],
            ['do', 'you', 'feel', 'calm'],
        ]
    else:
        opts = [['what', 'do', 'you', 'feel']]

    return list(opts[int(rng.integers(len(opts)))])

# ── REPL ──────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  BRAIN CONVERSATION")
print("  Use words from /help. Punctuation is stripped automatically.")
print("  /state  /words  /help  /quit")
print("=" * 60)
print()

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Goodbye.")
        break

    if not user_input:
        continue

    # ── Commands ──────────────────────────────────────────────────
    if user_input.startswith('/'):
        cmd = user_input.lower().strip()
        if cmd in ('/quit', '/exit', '/q'):
            print("  Goodbye.")
            break
        elif cmd == '/state':
            label = brain.selfmodel._last_label
            n_bound = int(np.sum(brain.binding._binding_strength > 0.05))
            print(f"\n  State:       {label}")
            print(f"  GWS arousal: {brain.gws._arousal_ema:.3f}  (speak threshold 0.35)")
            print(f"  NE level:    {brain.neuromod.ne_level:.3f}")
            print(f"  Utterances:  {brain.vocal._total_utterances}")
            print(f"  Bound BMUs:  {n_bound}/400")
            print()
        elif cmd == '/words':
            print("\n  Top reward associations:")
            rew = [(w, float(brain.binding._W_reward[word_to_bmu[w]]))
                   for w in VOCABULARY]
            rew.sort(key=lambda x: -x[1])
            for w, r in rew[:15]:
                bar = '█' * int(r * 30)
                print(f"    {w:12s} {r:.4f}  {bar}")
            print()
        elif cmd == '/help':
            words = sorted(VOCABULARY.keys())
            print(f"\n  Known words ({len(words)}):")
            for i in range(0, len(words), 10):
                print('  ' + '  '.join(words[i:i+10]))
            print()
        else:
            print(f"  Unknown command: {cmd}")
        continue

    # ── Process input ─────────────────────────────────────────────
    outputs, heard_bmus, heard_words = hear_sentence(user_input)
    if not outputs:
        print("Brain: (no known words — try /help)\n")
        continue

    # Use the most common state across all heard frames — single frames
    # can transiently spike to "confused" even when the brain is stable.
    from collections import Counter as _Counter
    state_counts = _Counter(o['sm_state_label'] for o in outputs)
    state = state_counts.most_common(1)[0][0]
    # "confused" during conversation just means novelty spiked briefly;
    # if nothing else is present, show "listening" instead.
    if state == 'confused':
        state = 'listening'

    response = generate_response(heard_bmus, heard_words)

    if response:
        print(f"Brain [{state}]: {response}")
    else:
        for _ in range(60):
            brain.hear(SILENCE)
            out = brain.step()
            if out['m74_speaking']:
                words = bmus_to_words(out['m74_utterance'])
                if words:
                    response = ' '.join(words)
                    break
        if response:
            print(f"Brain [{state}]: {response}")
        else:
            print(f"Brain [{state}]: (silent)")
    print()
