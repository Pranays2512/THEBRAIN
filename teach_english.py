"""
TEACH ENGLISH — Basic vocabulary with semantic context
=======================================================

Teaches the brain ~80 common English words by exposing it to each word
repeatedly in context. No grammar rules, no tokens, no labels. Words are
encoded as distinct MFCC spectral shapes. Meaning emerges from which
words co-occur with which internal states and reward signals.

HOW IT WORKS
------------
Each word is a unique acoustic fingerprint (MFCC vector).
The brain hears each word hundreds of times in the right context:
  - Positive words ('yes', 'good', 'nice') → during reward signal
  - Question words ('what', 'how', 'why') → during curious state
  - Feeling words ('happy', 'sad', 'hurt') → during matching internal states
  - Object words ('water', 'light', 'sound') → during neutral exposure
  - Action words ('come', 'go', 'stop', 'wait') → with urgency signals

After running this, the brain knows ~80 words well enough to:
  - Recognize which word cluster is which (M71 SOM)
  - Know word boundaries (M72 transition stats)
  - Know what each word means emotionally (M73 binding)
  - Produce relevant words given its current internal state (M74)

AFTER THIS: run converse.py to talk to the brain.

Run:   python teach_english.py
Time:  ~8-10 minutes (500k steps, the minimum for real word learning)
"""

import sys, time
import numpy as np
from collections import defaultdict

sys.dont_write_bytecode = True
from brain import Brain

print("=" * 65)
print("  TEACHING ENGLISH — Brain v13, 400 neurons")
print("  80 words × semantic context × Hebbian binding")
print("=" * 65)

SEED  = 42
rng   = np.random.default_rng(SEED)
N_MFCC = 13

# ═══════════════════════════════════════════════════════════════
# VOCABULARY — 80 words with distinct MFCC shapes
# ═══════════════════════════════════════════════════════════════
# Each word is a unique point in 13-dim MFCC space.
# We use a seeded deterministic layout so word shapes are
# stable across runs. Words are grouped by semantic category —
# the grouping affects training context, not the acoustics.

rng_words = np.random.default_rng(1337)   # fixed seed for word shapes

def _word(coeffs):
    """Build a 13-dim MFCC mean vector from compact specification."""
    v = np.zeros(13, dtype=np.float32)
    v[0] = 3.0   # energy: voiced
    for i, c in enumerate(coeffs):
        if i + 1 < 13:
            v[i + 1] = c
    return v

# 80 words across 8 semantic groups (10 words each)
VOCABULARY = {

    # ── GREETINGS / SOCIAL ────────────────────────────────────
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

    # ── FEELINGS / INTERNAL STATES ───────────────────────────
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

    # ── QUESTIONS ─────────────────────────────────────────────
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

    # ── BASIC OBJECTS / ENVIRONMENT ───────────────────────────
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

    # ── ACTIONS ───────────────────────────────────────────────
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

    # ── DESCRIPTORS ───────────────────────────────────────────
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

    # ── SELF / OTHER ──────────────────────────────────────────
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

    # ── SIMPLE SENTENCES / PHRASES ────────────────────────────
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

def say(word, noise_std=0.20):
    """Produce one spoken instance of a word as a list of MFCC frames."""
    mean_vec, n_frames = VOCABULARY[word]
    return [(mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
            for _ in range(n_frames)]

# ═══════════════════════════════════════════════════════════════
# INTERNAL STATE VECTORS — what each word context feels like
# ═══════════════════════════════════════════════════════════════
# 8-dim: [urgency, calm, curiosity, confusion, drive, clarity, frustration, engagement]

STATES = {
    'positive':   np.array([0.1, 0.9, 0.3, 0.0, 0.5, 0.8, 0.0, 0.9], dtype=np.float32),
    'negative':   np.array([0.7, 0.1, 0.1, 0.5, 0.2, 0.3, 0.8, 0.2], dtype=np.float32),
    'curious':    np.array([0.4, 0.4, 0.9, 0.3, 0.7, 0.5, 0.1, 0.8], dtype=np.float32),
    'confused':   np.array([0.8, 0.1, 0.2, 0.9, 0.3, 0.1, 0.7, 0.3], dtype=np.float32),
    'calm':       np.array([0.0, 0.9, 0.2, 0.0, 0.3, 0.8, 0.0, 0.5], dtype=np.float32),
    'urgent':     np.array([0.9, 0.1, 0.5, 0.2, 0.8, 0.4, 0.3, 0.7], dtype=np.float32),
    'neutral':    np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], dtype=np.float32),
    'satisfied':  np.array([0.0, 0.8, 0.1, 0.0, 0.2, 0.9, 0.0, 0.6], dtype=np.float32),
}

# ═══════════════════════════════════════════════════════════════
# TEACHING CONTEXT — word → (state, reward, n_exposures)
# ═══════════════════════════════════════════════════════════════

TEACHING = {
    # Greetings / social
    'hello':    ('positive',  0.4, 400),
    'hi':       ('positive',  0.4, 400),
    'bye':      ('calm',      0.2, 300),
    'please':   ('positive',  0.5, 400),
    'thank':    ('positive',  0.8, 500),
    'sorry':    ('negative',  0.0, 300),
    'yes':      ('positive',  0.9, 600),
    'no':       ('negative',  0.0, 600),
    'okay':     ('positive',  0.5, 400),
    'good':     ('positive',  1.0, 600),

    # Feelings
    'happy':    ('positive',  0.9, 500),
    'sad':      ('negative',  0.0, 400),
    'confused': ('confused',  0.0, 500),
    'curious':  ('curious',   0.3, 400),
    'scared':   ('negative',  0.0, 300),
    'calm':     ('calm',      0.3, 400),
    'angry':    ('negative',  0.0, 300),
    'tired':    ('negative',  0.0, 300),
    'hurt':     ('negative',  0.0, 300),
    'love':     ('positive',  1.0, 500),

    # Questions
    'what':     ('curious',   0.1, 500),
    'who':      ('curious',   0.1, 400),
    'where':    ('curious',   0.1, 400),
    'when':     ('curious',   0.1, 400),
    'why':      ('curious',   0.2, 500),
    'how':      ('curious',   0.2, 500),
    'which':    ('curious',   0.1, 300),
    'can':      ('curious',   0.2, 400),
    'do':       ('neutral',   0.1, 300),
    'is':       ('neutral',   0.1, 300),

    # Objects
    'water':    ('positive',  0.6, 400),
    'light':    ('curious',   0.3, 300),
    'dark':     ('confused',  0.0, 300),
    'sound':    ('curious',   0.3, 400),
    'heat':     ('neutral',   0.1, 300),
    'cold':     ('negative',  0.0, 300),
    'body':     ('neutral',   0.2, 300),
    'mind':     ('curious',   0.3, 400),
    'world':    ('curious',   0.4, 400),
    'time':     ('curious',   0.2, 400),

    # Actions
    'come':     ('urgent',    0.3, 500),
    'go':       ('urgent',    0.2, 400),
    'stop':     ('urgent',    0.2, 400),
    'wait':     ('calm',      0.1, 400),
    'listen':   ('curious',   0.3, 500),
    'speak':    ('positive',  0.5, 500),
    'think':    ('curious',   0.4, 400),
    'feel':     ('neutral',   0.2, 400),
    'know':     ('positive',  0.5, 400),
    'want':     ('urgent',    0.3, 400),

    # Descriptors
    'big':      ('neutral',   0.1, 300),
    'small':    ('neutral',   0.1, 300),
    'fast':     ('urgent',    0.2, 300),
    'slow':     ('calm',      0.1, 300),
    'near':     ('positive',  0.3, 300),
    'far':      ('confused',  0.1, 300),
    'new':      ('curious',   0.4, 300),
    'old':      ('neutral',   0.1, 300),
    'more':     ('positive',  0.4, 400),
    'less':     ('negative',  0.1, 300),

    # Self / other
    'i':        ('neutral',   0.2, 500),
    'you':      ('positive',  0.4, 500),
    'me':       ('neutral',   0.2, 400),
    'we':       ('positive',  0.3, 400),
    'here':     ('calm',      0.3, 400),
    'there':    ('curious',   0.2, 300),
    'this':     ('neutral',   0.1, 300),
    'that':     ('neutral',   0.1, 300),
    'same':     ('calm',      0.2, 300),
    'different':('curious',   0.3, 300),

    # Phrases
    'not':      ('negative',  0.0, 400),
    'now':      ('urgent',    0.3, 400),
    'again':    ('neutral',   0.2, 300),
    'always':   ('positive',  0.4, 300),
    'never':    ('negative',  0.0, 300),
    'maybe':    ('confused',  0.1, 300),
    'help':     ('urgent',    0.5, 500),
    'ready':    ('positive',  0.6, 400),
    'done':     ('satisfied', 0.8, 400),
    'together': ('positive',  0.6, 400),
}

# ═══════════════════════════════════════════════════════════════
# BRAIN
# ═══════════════════════════════════════════════════════════════

brain = Brain(seed=SEED)
t_total = time.time()
total_steps = sum(v[2] * len(VOCABULARY[w][0]) // 1 for w, v in TEACHING.items())
print(f"\n  Vocabulary: {len(VOCABULARY)} words")
print(f"  Neurons:    {brain.speech_cortex._weights.shape[0]} (20×20 SOM)")
print(f"  Teaching context: state + reward co-occurrence")
print()

# ═══════════════════════════════════════════════════════════════
# PHASE 1 — SOM WARMUP (hear all words to initialize the map)
# ═══════════════════════════════════════════════════════════════
print("Phase 1: SOM warmup — cycling through all words 50 times...")
t0 = time.time()

word_list = list(VOCABULARY.keys())
for cycle in range(50):
    rng.shuffle(word_list)
    for word in word_list:
        for frame in say(word, noise_std=0.3):
            brain.hear(frame)
            brain.step()
        brain.hear(SILENCE)
        brain.step()

print(f"  Done. {brain.t:,} steps, {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════
# PHASE 2 — CONTEXTUAL TEACHING (bind each word to its meaning)
# ═══════════════════════════════════════════════════════════════
print("\nPhase 2: Contextual binding — teaching word meanings...")
print(f"  {'word':12s} {'state':12s} {'reward':6s} {'exposures':10s} {'done':5s}")
print(f"  {'-'*50}")

word_list = list(TEACHING.keys())
t0 = time.time()
completed = 0

for word, (state_name, reward_val, n_exp) in TEACHING.items():
    state_vec = STATES[state_name]

    for _ in range(n_exp):
        frames = say(word, noise_std=0.20)
        for i, frame in enumerate(frames):
            brain.hear(frame)
            brain.step(reward=reward_val if i == len(frames)-1 else 0.0)

        # Between-word silence in the right internal state
        # (We inject state by timing reward to co-occur with the word)
        brain.hear(SILENCE)
        brain.step(reward=0.0)

        # Occasionally add a filler word for natural variation
        if rng.random() < 0.15:
            filler = rng.choice(word_list)
            for frame in say(filler, noise_std=0.30):
                brain.hear(frame)
                brain.step()
            brain.hear(SILENCE)
            brain.step()

    completed += 1
    strength = float(brain.binding._binding_strength.max())
    print(f"  {word:12s} {state_name:12s} {reward_val:6.2f} {n_exp:10d}  [{completed:2d}/{len(TEACHING)}]  "
          f"max_bind={strength:.3f}")

print(f"\n  Phase 2 done. {brain.t:,} steps, {time.time()-t0:.0f}s")

# ═══════════════════════════════════════════════════════════════
# PHASE 3 — SENTENCE-LEVEL EXPOSURE
# ═══════════════════════════════════════════════════════════════
# Feed simple word sequences so M72 learns inter-word transitions.
# These are not grammar rules — just recurring patterns.
# The brain will learn that "what is" tends to follow each other, etc.

print("\nPhase 3: Sentence-level exposure — teaching word sequences...")

SENTENCES = [
    # question patterns
    ['what', 'is', 'this'],
    ['what', 'do', 'you', 'want'],
    ['what', 'is', 'that'],
    ['how', 'do', 'you', 'feel'],
    ['how', 'are', 'you'] if 'are' in VOCABULARY else ['how', 'is', 'this'],
    ['who', 'is', 'there'],
    ['where', 'do', 'you', 'go'],
    ['why', 'is', 'this', 'different'],
    ['can', 'you', 'listen'],
    ['can', 'you', 'speak'],

    # feeling expressions
    ['i', 'feel', 'happy'],
    ['i', 'feel', 'confused'],
    ['i', 'feel', 'curious'],
    ['i', 'feel', 'calm'],
    ['i', 'want', 'more'],
    ['i', 'know', 'this'],
    ['i', 'think', 'so'],
    ['i', 'am', 'here'] if 'am' in VOCABULARY else ['i', 'am', 'ready'],

    # responses
    ['yes', 'good'],
    ['yes', 'okay'],
    ['no', 'not', 'now'],
    ['no', 'stop'],
    ['maybe', 'later'],
    ['okay', 'good'],
    ['thank', 'you'],
    ['help', 'me'],
    ['come', 'here'],
    ['wait', 'please'],

    # observations
    ['this', 'is', 'new'],
    ['this', 'is', 'different'],
    ['you', 'and', 'me'] if 'and' in VOCABULARY else ['you', 'listen'],
    ['we', 'are', 'together'] if 'are' in VOCABULARY else ['we', 'together'],
    ['i', 'hear', 'you'] if 'hear' in VOCABULARY else ['i', 'listen'],
    ['done', 'now'],
    ['not', 'again'],
    ['always', 'together'],
    ['never', 'stop'],
    ['ready', 'now'],
]

# Filter sentences to only use words we know
def valid_sentence(s):
    return all(w in VOCABULARY for w in s)

SENTENCES = [s for s in SENTENCES if valid_sentence(s)]
print(f"  {len(SENTENCES)} sentence patterns to learn")

N_SENTENCE_CYCLES = 300
for cycle in range(N_SENTENCE_CYCLES):
    rng.shuffle(SENTENCES)
    for sentence in SENTENCES:
        # Determine context from first word
        state_name, reward_val, _ = TEACHING.get(sentence[0], ('neutral', 0.0, 0))
        state_vec = STATES[state_name]
        for w_idx, word in enumerate(sentence):
            frames = say(word, noise_std=0.18)
            is_last_word = (w_idx == len(sentence) - 1)
            for i, frame in enumerate(frames):
                brain.hear(frame)
                brain.step(
                    reward=reward_val if (is_last_word and i == len(frames)-1) else 0.0
                )
        # Sentence-end silence
        brain.hear(SILENCE)
        brain.step()

print(f"  Phase 3 done. {brain.t:,} total steps")

# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  LEARNING REPORT")
print("=" * 65)

print(f"\n  Brain steps:      {brain.t:,}")
print(f"  SOM trained on:   {brain.speech_cortex.t:,} frames")
print(f"  Proto-words:      {len(brain.phoneme_seq._word_candidates):,}")
n_bound = int(np.sum(brain.binding._binding_strength > 0.05))
print(f"  Bound phonemes:   {n_bound}/400")
print(f"  Utterances made:  {brain.vocal._total_utterances}")

print("\n  Top word associations (reward signal):")
# Find the top-BMU for each word and its reward association
word_rewards = []
word_bmu_map = {}
for word in VOCABULARY:
    frames = say(word, noise_std=0.05)   # clean probe
    bmus = []
    for frame in frames:
        brain.hear(frame)
        out = brain.step()
        bmus.append(out['m71_phoneme_bmu'])
    top_bmu = max(set(bmus), key=bmus.count)
    word_bmu_map[word] = top_bmu
    rew = float(brain.binding._W_reward[top_bmu])
    word_rewards.append((word, top_bmu, rew))

word_rewards.sort(key=lambda x: -x[2])
print(f"  {'word':12s} {'BMU':5s} {'reward-assoc':12s}")
print(f"  {'-'*32}")
for word, bmu, rew in word_rewards[:15]:
    bar = "█" * int(rew * 20)
    print(f"  {word:12s} {bmu:5d} {rew:.4f}   {bar}")

print("\n  Bottom (neutral/negative words):")
for word, bmu, rew in word_rewards[-8:]:
    print(f"  {word:12s} {bmu:5d} {rew:.4f}")

print(f"\n  Total training time: {time.time() - t_total:.0f}s")

# ── Save ──────────────────────────────────────────────────────────
brain.save('brain_trained.pkl')
print()
print("  Brain is ready. Run:  python converse.py")
