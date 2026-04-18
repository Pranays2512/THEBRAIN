"""
vocab_core.py — Grounded vocabulary: only words with real experiential referents.

56 words selected by one rule: does the brain have a simulation event that
directly grounds this word?

  World6 events → food, eat, drink, wall, hurt, push, open, door, button, go
  Drive states  → hungry, full, tired, awake, calm, warm, sleep, pain
  Reward signal → good, bad, happy, sad, strong, soft, hard
  Teacher/social → hi, hello, bye, yes, no, sorry, help
  Expression    → i, me, you, we, feel, want, need, know, and, not, is, here, stop

Abstract words (love, truth, memory, thought, beautiful, soul...) are deliberately
excluded. The brain will earn them one at a time as new grounded experiences
are added to the simulation.

Every word the brain can say should correspond to something it has felt.
"""

import numpy as np
from vocab import VOCABULARY as _BASE, SILENCE, N_MFCC, _w  # noqa: F401

# ── Pull grounded words from base vocabulary ───────────────────────────────────
_KEEP = {
    # Social basics — teacher-groundable via consistent feedback
    'hi', 'hello', 'bye', 'yes', 'no', 'sorry', 'help',
    # Pronouns / function — needed for minimal expression
    'i', 'me', 'you', 'we', 'and', 'not', 'is', 'here', 'now', 'stop',
    # Cognitive expression — used to articulate states
    'feel', 'want', 'need', 'know', 'good', 'happy', 'sad', 'calm', 'warm',
    # Body drives — World6 hunger/tiredness cycles
    'eat', 'drink', 'sleep', 'tired', 'awake', 'full', 'alive',
    # World events — directly observed in World6
    'food', 'water', 'hurt', 'open', 'go', 'move', 'come', 'wait', 'push',
    # Evaluations — reward-grounded
    'strong', 'soft', 'new', 'more',
}

VOCABULARY = {w: v for w, v in _BASE.items() if w in _KEEP}

# ── Add words not present in base vocab but needed for World6 grounding ────────
# Design: v[1]=valence, v[2]=activity, v[4]=concreteness, v[6]=intensity,
#         v[10]=embodiment — see vocab.py header for full legend.

VOCABULARY['hungry'] = _w([-0.6,  0.3, -0.1,  0.9,  0.4,  0.5,  0.1,  0.1, -0.1,  0.1,  0.8,  0.0], 3)
VOCABULARY['wall']   = _w([-0.5, -0.4,  0.0,  1.0,  0.7,  0.5,  0.0, -0.1,  0.0,  0.2,  0.3,  0.0], 3)
VOCABULARY['door']   = _w([ 0.4,  0.2,  0.0,  0.9,  0.6,  0.2,  0.1,  0.1,  0.0,  0.2,  0.1,  0.0], 3)
VOCABULARY['button'] = _w([ 0.2,  0.4,  0.0,  0.9,  0.5,  0.3,  0.1,  0.2,  0.1,  0.2,  0.5,  0.0], 3)
VOCABULARY['pain']   = _w([-1.2,  0.3, -0.1,  0.8,  0.6,  0.6,  0.1, -0.1,  0.0,  0.2,  0.9,  0.0], 3)
VOCABULARY['bad']    = _w([-0.9, -0.4,  0.0, -0.2, -0.4,  0.1, -0.3,  0.0, -0.2,  0.0,  0.1,  0.0], 2)
VOCABULARY['hard']   = _w([ 0.0,  0.4,  0.0,  0.8,  0.6,  0.5,  0.0,  0.1,  0.1,  0.1,  0.4,  0.0], 3)

# ── Danger zone words — grounded to persistent negative reward experience ───────
# Distinct MFCC profile from pain (sudden shock) — danger is creeping dread:
# low arousal (frozen, not fighting), negative valence, low familiarity/certainty
VOCABULARY['afraid']  = _w([-0.8, -0.4,  0.0,  0.6, -0.5,  0.4,  0.0, -0.2, -0.2,  0.1,  0.8,  0.0], 3)
VOCABULARY['careful'] = _w([-0.2, -0.3,  0.0,  0.5,  0.3,  0.2,  0.0,  0.3, -0.1,  0.1,  0.3,  0.0], 3)
VOCABULARY['danger']  = _w([-0.9,  0.4,  0.0,  0.8, -0.4,  0.5,  0.0, -0.1, -0.3,  0.1,  0.7,  0.0], 3)
VOCABULARY['safe']    = _w([ 0.6, -0.4,  0.0,  0.5,  0.6, -0.1,  0.0,  0.2,  0.3,  0.1,  0.2,  0.0], 3)
VOCABULARY['run']     = _w([-0.3,  1.0,  0.0,  0.7, -0.2,  0.3,  0.1,  0.4, -0.2,  0.2,  0.5,  0.0], 2)

# ── Biographical Identity ─────────────────────────────────────────────────────
VOCABULARY['fastbrain'] = _w([ 0.1,  0.5,  0.0,  0.7, -0.4,  0.4,  0.1,  0.1,  0.0,  0.3,  0.4,  0.0], 3)
VOCABULARY['am']        = _w([ 0.0,  0.1,  0.0,  0.5,  0.2,  0.1,  0.0,  0.2, -0.1,  0.2,  0.2,  0.0], 2)
VOCABULARY['name']      = _w([ 0.2,  0.3,  0.0,  0.6, -0.1,  0.2,  0.1,  0.3, -0.2,  0.1,  0.3,  0.0], 2)

print(f"[vocab_core] {len(VOCABULARY)} grounded words loaded.")
