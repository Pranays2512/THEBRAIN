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

# ── Cognitive/self-identity words — social zone ──────────────────────────────
# think: cognitive act — neutral valence, low arousal, low embodiment
VOCABULARY['think']  = _w([ 0.1,  0.0,  0.0,  0.5,  0.0,  0.3,  0.1,  0.2,  0.0,  0.3,  0.1,  0.0], 2)
# learn: positive cognitive act — mild positive valence, low arousal
VOCABULARY['learn']  = _w([ 0.3,  0.1,  0.0,  0.5,  0.1,  0.3,  0.1,  0.2,  0.0,  0.3,  0.1,  0.0], 2)
# brain: self-concept — near fastbrain on SOM, social zone
VOCABULARY['brain']  = _w([ 0.1,  0.3,  0.0,  0.6,  0.1,  0.3,  0.1,  0.2,  0.0,  0.3,  0.2,  0.0], 3)

# ── Relation/attitude words — social and pain zone ────────────────────────────
# like: social zone — positive valence, low embodiment, moderate arousal
VOCABULARY['like']     = _w([ 0.5,  0.2,  0.0,  0.6,  0.2,  0.4,  0.1,  0.3,  0.0,  0.2,  0.2,  0.0], 2)
# hate: pain-adjacent — strong negative valence, high arousal, moderate embodiment
VOCABULARY['hate']     = _w([-0.8,  0.5,  0.0,  0.7,  0.3,  0.3,  0.0, -0.1, -0.1,  0.1,  0.5,  0.0], 2)
# remember: social/cognitive — neutral valence, low arousal, low embodiment
VOCABULARY['remember'] = _w([ 0.1, -0.1,  0.0,  0.5,  0.1,  0.3,  0.1,  0.2,  0.0,  0.3,  0.1,  0.0], 2)
# helps: social — positive valence, moderate arousal
VOCABULARY['helps']    = _w([ 0.4,  0.2,  0.0,  0.6,  0.2,  0.3,  0.1,  0.2,  0.1,  0.2,  0.2,  0.0], 2)
# hurts: pain zone — negative valence, high embodiment (physical sensation)
VOCABULARY['hurts']    = _w([-0.7,  0.4,  0.0,  0.7,  0.5,  0.5,  0.0, -0.1,  0.0,  0.1,  0.7,  0.0], 3)
# pranay: social — person, positive valence, low embodiment
VOCABULARY['pranay']   = _w([ 0.3,  0.2,  0.0,  0.6,  0.3,  0.3,  0.1,  0.2,  0.0,  0.2,  0.2,  0.0], 3)

# ── Causal / connective words — cognitive zone (close to think/learn on SOM) ───
# why:     questioning — neutral valence, low arousal, inquisitive
VOCABULARY['why']      = _w([ 0.0, -0.1,  0.0,  0.4,  0.0,  0.3,  0.1,  0.2,  0.0,  0.3,  0.1,  0.0], 2)
# because: explanatory — slight positive valence (explanations feel satisfying)
VOCABULARY['because']  = _w([ 0.1,  0.0,  0.0,  0.5,  0.1,  0.3,  0.1,  0.2,  0.0,  0.3,  0.1,  0.0], 2)
# so:      consequence marker — mild positive, lower arousal than because
VOCABULARY['so']       = _w([ 0.2,  0.1,  0.0,  0.4,  0.1,  0.2,  0.1,  0.2,  0.0,  0.3,  0.1,  0.0], 2)
# cause:   causal link noun/verb — neutral, factual, low embodiment
VOCABULARY['cause']    = _w([ 0.0,  0.1,  0.0,  0.5,  0.1,  0.3,  0.0,  0.2,  0.0,  0.3,  0.1,  0.0], 2)
# then:    sequential / consequence — low arousal, mild positive (order is reassuring)
VOCABULARY['then']     = _w([ 0.1,  0.1,  0.0,  0.4,  0.0,  0.2,  0.1,  0.3,  0.0,  0.3,  0.1,  0.0], 2)

# ── Prediction error words — surprise/mismatch signal ─────────────────────────
# wrong:  expectation violated — negative valence, high arousal (surprise)
VOCABULARY['wrong']  = _w([-0.5,  0.4,  0.0,  0.6, -0.2,  0.4,  0.1, -0.1,  0.0,  0.2,  0.4,  0.0], 2)
# search: active re-exploration after mismatch — mild positive (curiosity)
VOCABULARY['search'] = _w([ 0.2,  0.5,  0.0,  0.6,  0.2,  0.3,  0.1,  0.3,  0.1,  0.3,  0.2,  0.0], 2)

# ── Environmental words — grounded in sensory channels v[13–16] ───────────────
# Vector format: 12 semantic dims + 4 environmental dims
#   [valence, activity, social, concreteness, certainty, intensity,
#    temporality, agency, familiarity, embodiment, questioning, fine,
#    vegetation, moisture, temperature, wind]
#
# Vegetation zone (high v[13])
VOCABULARY['plant']  = _w([ 0.5, -0.2,  0.0,  0.9,  0.4,  0.2,  0.0,  0.0,  0.2,  0.3,  0.0,  0.0,  0.9,  0.3,  0.5,  0.1], 3)
VOCABULARY['tree']   = _w([ 0.6, -0.1,  0.0,  1.0,  0.5,  0.4,  0.0, -0.1,  0.3,  0.2,  0.0,  0.1,  1.0,  0.2,  0.4,  0.2], 3)
VOCABULARY['grass']  = _w([ 0.4, -0.2,  0.0,  0.9,  0.4,  0.1,  0.0,  0.0,  0.3,  0.4,  0.0,  0.0,  0.8,  0.3,  0.5,  0.3], 3)
VOCABULARY['green']  = _w([ 0.5,  0.0,  0.0,  0.8,  0.4,  0.2,  0.0,  0.0,  0.2,  0.2,  0.0,  0.1,  0.9,  0.3,  0.5,  0.1], 2)
# Water/moisture zone (high v[14])
VOCABULARY['river']  = _w([ 0.4,  0.3,  0.0,  1.0,  0.5,  0.3,  0.0,  0.0,  0.2,  0.3,  0.0,  0.0,  0.2,  0.9,  0.4,  0.3], 3)
VOCABULARY['rain']   = _w([ 0.2,  0.2,  0.0,  0.9,  0.4,  0.2,  0.1,  0.0,  0.1,  0.3,  0.0,  0.0,  0.3,  0.9,  0.2,  0.4], 2)
VOCABULARY['wet']    = _w([-0.1,  0.0,  0.0,  0.9,  0.5,  0.2,  0.0,  0.0,  0.1,  0.5,  0.0,  0.0,  0.2,  0.8,  0.3,  0.2], 2)
# Air/wind zone (high v[16])
VOCABULARY['air']    = _w([ 0.5,  0.3,  0.0,  0.8,  0.4,  0.1,  0.1,  0.0,  0.2,  0.3,  0.0,  0.0,  0.1,  0.1,  0.5,  0.7], 2)
VOCABULARY['wind']   = _w([ 0.2,  0.6,  0.0,  0.8,  0.3,  0.3,  0.1,  0.1,  0.1,  0.3,  0.0,  0.0,  0.1,  0.2,  0.4,  0.9], 2)
VOCABULARY['sky']    = _w([ 0.7,  0.1,  0.0,  0.9,  0.5,  0.3,  0.0,  0.0,  0.2,  0.1,  0.0,  0.1,  0.0,  0.1,  0.6,  0.5], 3)
# Temperature zone (v[15] low=cold, high=hot)
VOCABULARY['cold']   = _w([-0.3, -0.3,  0.0,  0.8,  0.5,  0.3,  0.0,  0.0,  0.1,  0.5,  0.0,  0.0,  0.0,  0.2,  0.0,  0.4], 2)
# 'warm' already exists in base vocab — update with environmental signal
VOCABULARY['warm']   = _w([ 0.7,  0.1,  0.3,  0.5,  0.3,  0.2,  0.0,  0.0,  0.0,  0.1,  0.5,  0.0,  0.2,  0.1,  0.9,  0.1], 3)
# Sun/light — high temperature, open sky
VOCABULARY['sun']    = _w([ 0.8,  0.2,  0.0,  0.9,  0.6,  0.4,  0.1,  0.0,  0.3,  0.1,  0.0,  0.1,  0.1,  0.0,  1.0,  0.3], 3)

# ── Grammar function words — agent structure ──────────────────────────────────
# 'i': first-person agent — self-reference, social zone, high agency dim
# In _HOLLOW so it never votes for zones, but Word TP uses it as subject prefix.
VOCABULARY['i']  = _w([ 0.1,  0.0,  0.2,  0.5,  0.1,  0.1,  0.0,  0.8,  0.3,  0.2,  0.0,  0.0], 2)
# 'is': copula — links subject to predicate. Neutral, low arousal, functional.
VOCABULARY['is'] = _w([ 0.0,  0.0,  0.1,  0.3,  0.2,  0.0,  0.0,  0.2,  0.2,  0.0,  0.0,  0.0], 1)

print(f"[vocab_core] {len(VOCABULARY)} grounded words loaded.")
