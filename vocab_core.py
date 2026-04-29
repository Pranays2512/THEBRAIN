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
VOCABULARY['afraid']  = _w([-0.9, -0.6,  0.0,  0.4, -0.6,  0.5,  0.0, -0.3, -0.4,  0.2,  0.9,  0.0], 3)
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
# ── Epistemic words — grounded in uncertainty/prediction-error experience ────
# 'what': felt gap in knowledge — negative valence (tension), high novelty dim
# NOT a grammar token. Grounded to surprise events in World6 training.
VOCABULARY['what']     = _w([-0.2,  0.5,  0.0,  0.3, -0.3,  0.5,  0.1,  0.1,  0.8,  0.2,  0.5,  0.0], 2)
# why:     deeper questioning — sustained low arousal tension, inquisitive
VOCABULARY['why']      = _w([-0.1,  0.2,  0.0,  0.4, -0.2,  0.4,  0.1,  0.2,  0.7,  0.2,  0.4,  0.0], 2)
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

# ── Intensity words — graded in drive magnitude, not taught as text patterns ──
# 'very':   high arousal amplifier — strong positive valence, high activation
VOCABULARY['very']   = _w([ 0.3,  0.8,  0.0,  0.5,  0.3,  0.9,  0.0,  0.3,  0.1,  0.3,  0.0,  0.0], 1)
# 'little': low arousal diminutive — mild, low intensity, calm
VOCABULARY['little'] = _w([ 0.1, -0.4,  0.0,  0.4,  0.3, -0.5,  0.0,  0.1,  0.2,  0.1,  0.0,  0.0], 1)

# ── Grammar function words — agent structure ──────────────────────────────────
# 'i': first-person agent — self-reference, social zone, high agency dim
# In _HOLLOW so it never votes for zones, but Word TP uses it as subject prefix.
VOCABULARY['i']  = _w([ 0.1,  0.0,  0.2,  0.5,  0.1,  0.1,  0.0,  0.8,  0.3,  0.2,  0.0,  0.0], 2)
# 'is': copula — links subject to predicate. Neutral, low arousal, functional.
VOCABULARY['is'] = _w([ 0.0,  0.0,  0.1,  0.3,  0.2,  0.0,  0.0,  0.2,  0.2,  0.0,  0.0,  0.0], 1)

# ── Spatial words — grounded in World6 navigation + proximity to reward ────────
# near: brain close to food/door/button — positive (reward imminent), embodied
VOCABULARY['near']   = _w([ 0.4,  0.1,  0.0,  0.7,  0.4,  0.3,  0.0,  0.2,  0.3,  0.4,  0.0,  0.0], 3)
# far:  brain far from reward — mild negative, uncertain (is food even out there?)
VOCABULARY['far']    = _w([-0.2,  0.2,  0.0,  0.7,  0.3,  0.2,  0.0,  0.1,  0.1,  0.3,  0.0,  0.0], 2)
# found: discovery reward signal fires — high positive valence, arousal spike
VOCABULARY['found']  = _w([ 0.7,  0.6,  0.0,  0.8,  0.5,  0.5,  0.0,  0.4,  0.6,  0.3,  0.0,  0.0], 3)
# lost:  wandering without finding food — negative, low familiarity, uncertain
VOCABULARY['lost']   = _w([-0.5,  0.3,  0.0,  0.6, -0.4,  0.4,  0.0,  0.1, -0.3,  0.2,  0.0,  0.0], 2)
# out:   exiting danger zone / open space — mild positive, spatial, agentive
VOCABULARY['out']    = _w([ 0.3,  0.4,  0.0,  0.7,  0.3,  0.3,  0.0,  0.4,  0.2,  0.3,  0.0,  0.0], 2)
# in:    entering area — neutral, spatial, low agency (passive containment)
VOCABULARY['in']     = _w([ 0.0,  0.0,  0.0,  0.7,  0.3,  0.2,  0.0,  0.2,  0.3,  0.3,  0.0,  0.0], 2)

# ── Temporal words — grounded in World6 event sequences ───────────────────────
# after:  resolution comes AFTER drive — mild positive (order, satisfaction)
VOCABULARY['after']  = _w([ 0.2,  0.1,  0.0,  0.4,  0.3,  0.1,  0.9,  0.2,  0.2,  0.1,  0.0,  0.0], 1)
# before: prior state / anticipation — neutral, slight tension
VOCABULARY['before'] = _w([ 0.0,  0.0,  0.0,  0.4,  0.2,  0.1,  0.9,  0.1,  0.1,  0.1,  0.0,  0.0], 1)
# again:  drive cycles repeat — mild negative (monotony/urgency returning)
VOCABULARY['again']  = _w([-0.1,  0.3,  0.0,  0.5,  0.4,  0.3,  0.7,  0.2,  0.3,  0.2,  0.0,  0.0], 2)
# still:  state unchanged — low arousal, temporal continuity, familiarity high
VOCABULARY['still']  = _w([ 0.0, -0.2,  0.0,  0.4,  0.5,  0.1,  0.6,  0.1,  0.5,  0.2,  0.0,  0.0], 1)

# ── Social / interaction words — teacher and turn-taking grounded ──────────────
# talk:   brain-teacher exchange events — high social dim, moderate arousal
VOCABULARY['talk']   = _w([ 0.4,  0.5,  0.8,  0.6,  0.3,  0.3,  0.0,  0.5,  0.3,  0.2,  0.0,  0.0], 2)
# listen: receiving teacher input — positive social, low arousal, low agency
VOCABULARY['listen'] = _w([ 0.3, -0.2,  0.8,  0.5,  0.4,  0.2,  0.0,  0.2,  0.4,  0.3,  0.0,  0.0], 2)
# alone:  no teacher signal, isolated — mild negative, social dim inverted
VOCABULARY['alone']  = _w([-0.2, -0.3, -0.5,  0.5,  0.4,  0.2,  0.0,  0.1,  0.2,  0.3,  0.0,  0.0], 2)
# with:   teacher present, joint experience — positive, high social
VOCABULARY['with']   = _w([ 0.4,  0.2,  0.7,  0.5,  0.4,  0.2,  0.0,  0.3,  0.4,  0.2,  0.0,  0.0], 2)

# ── Sensory/action words — World6 visual scanning and effort ──────────────────
# look:   directed scanning toward unknown — curiosity, high agency, embodied
VOCABULARY['look']   = _w([ 0.3,  0.4,  0.0,  0.7,  0.3,  0.3,  0.0,  0.7,  0.2,  0.4,  0.0,  0.0], 3)
# see:    successful perception, reward or danger confirmed — high concreteness
VOCABULARY['see']    = _w([ 0.4,  0.3,  0.0,  0.8,  0.5,  0.3,  0.0,  0.4,  0.4,  0.4,  0.0,  0.0], 3)
# try:    attempting push/button/door — agency high, uncertain outcome
VOCABULARY['try']    = _w([ 0.3,  0.6,  0.0,  0.5,  0.2,  0.5,  0.0,  0.8,  0.2,  0.3,  0.0,  0.0], 2)
# free:   open space after escaping wall/danger — positive, high valence
VOCABULARY['free']   = _w([ 0.7,  0.4,  0.0,  0.6,  0.5,  0.4,  0.0,  0.6,  0.4,  0.2,  0.0,  0.0], 3)

# ── Light/dark — World6 night/day cycle and danger zone visual contrast ────────
# dark:   low certainty, danger-adjacent, negative valence
VOCABULARY['dark']   = _w([-0.3, -0.2,  0.0,  0.8,  0.2,  0.2,  0.0,  0.0, -0.2,  0.3,  0.0,  0.0], 2)
# light:  open area, safe zone, positive
VOCABULARY['light']  = _w([ 0.6,  0.2,  0.0,  0.8,  0.5,  0.4,  0.0,  0.0,  0.3,  0.2,  0.0,  0.0], 2)
# worry:  sustained hunger + no food visible — anxious, low arousal, embodied dread
VOCABULARY['worry']  = _w([-0.5,  0.1,  0.0,  0.3, -0.4,  0.4,  0.0,  0.1, -0.1,  0.4,  0.0,  0.0], 2)

# ── Epistemic question words — grounded in prediction-error / gap events ──────
# where: spatial gap — brain navigates, no food found → "where is food?"
VOCABULARY['where']  = _w([-0.1,  0.3,  0.0,  0.4, -0.2,  0.3,  0.0,  0.2,  0.1,  0.1,  0.8,  0.0], 2)
# who:   social identity gap — uncertain about teacher/self relation
VOCABULARY['who']    = _w([ 0.0,  0.2,  0.4,  0.3, -0.2,  0.2,  0.0,  0.2,  0.1,  0.1,  0.8,  0.0], 1)
# how:   method question — process uncertainty, curiosity
VOCABULARY['how']    = _w([ 0.1,  0.3,  0.0,  0.3, -0.1,  0.3,  0.0,  0.3,  0.2,  0.2,  0.7,  0.0], 1)
# do:    action verb function word — agency high, concrete
VOCABULARY['do']     = _w([ 0.2,  0.5,  0.0,  0.5,  0.3,  0.3,  0.0,  0.7,  0.3,  0.3,  0.0,  0.0], 1)

# ── Universal quantifier words — grounded in repeated pattern saturation ──────
# all:    high certainty, abstract, no embodiment — pattern fully learned
VOCABULARY['all']    = _w([ 0.0,  0.0,  0.0,  0.2,  0.8,  0.3,  0.0,  0.1,  0.3,  0.0,  0.0,  0.0], 1)
# never:  temporal negation, definitive — strong certainty, temporal dim high
VOCABULARY['never']  = _w([-0.2,  0.0,  0.0,  0.3,  0.7,  0.4,  0.8,  0.1,  0.2,  0.1,  0.0,  0.0], 1)
# always: temporal universal, reliable — mild positive, temporal, familiar
VOCABULARY['always'] = _w([ 0.2,  0.0,  0.0,  0.3,  0.7,  0.3,  0.8,  0.1,  0.4,  0.0,  0.0,  0.0], 1)

# ── New World6 event words — thirst, injury, NPC, storm ──────────────────────
# thirsty: thirst drive high — negative valence, embodied dread (like hungry)
VOCABULARY['thirsty'] = _w([-0.5,  0.3, -0.1,  0.9,  0.4,  0.5,  0.1,  0.1, -0.1,  0.1,  0.7,  0.0], 3)
# weak:    injury high — negative valence, very low arousal, high embodiment
VOCABULARY['weak']    = _w([-0.6, -0.5,  0.0,  0.8,  0.3,  0.4,  0.0, -0.2, -0.1,  0.1,  0.9,  0.0], 3)
# heal:    injury recovering — mild positive valence, low arousal, embodied relief
VOCABULARY['heal']    = _w([ 0.5, -0.2,  0.0,  0.7,  0.4,  0.2,  0.0,  0.1,  0.2,  0.1,  0.5,  0.0], 2)
# friend:  NPC encounter — positive valence, high social dim, moderate arousal
VOCABULARY['friend']  = _w([ 0.7,  0.3,  0.9,  0.6,  0.4,  0.3,  0.0,  0.3,  0.4,  0.2,  0.0,  0.0], 3)
# hot:     high temperature env + storm — negative discomfort, high embodiment
VOCABULARY['hot']     = _w([-0.2,  0.2,  0.0,  0.8,  0.5,  0.3,  0.0,  0.0,  0.1,  0.5,  0.0,  0.0,  0.0,  0.2,  0.9,  0.2], 2)
# empty:   food gone / prediction error — mild negative, low certainty, spatial
VOCABULARY['empty']   = _w([-0.3,  0.0,  0.0,  0.7, -0.3,  0.2,  0.0,  0.0, -0.1,  0.2,  0.0,  0.0], 2)
# storm:   storm active — negative valence, high arousal, high wind + temp dims
VOCABULARY['storm']   = _w([-0.5,  0.7,  0.0,  0.9,  0.3,  0.6,  0.0,  0.0, -0.1,  0.4,  0.0,  0.0,  0.0,  0.5,  0.4,  0.9], 3)
# better:  injury/thirst recovering — mild positive, temporal resolution
VOCABULARY['better']  = _w([ 0.5, -0.1,  0.0,  0.5,  0.5,  0.2,  0.7,  0.1,  0.3,  0.2,  0.0,  0.0], 2)

# ── Communication/action verbs — grounded in NPC + teacher exchange events ────
# ask:  social request — high social, moderate agency, mild positive
VOCABULARY['ask']  = _w([ 0.3,  0.4,  0.8,  0.5,  0.2,  0.3,  0.0,  0.6,  0.2,  0.2,  0.3,  0.0], 2)
# tell: information transfer — high social, moderate agency, LOW intensity (not urgent)
VOCABULARY['tell'] = _w([ 0.4,  0.2,  0.9,  0.5,  0.4,  0.1,  0.0,  0.5,  0.4,  0.1,  0.0,  0.0], 2)
# show: demonstrating something — concrete, agentive, social
VOCABULARY['show'] = _w([ 0.4,  0.4,  0.6,  0.8,  0.4,  0.3,  0.0,  0.6,  0.3,  0.3,  0.0,  0.0], 2)
# use:  tool/button/door action — concrete, high agency, neutral valence
VOCABULARY['use']  = _w([ 0.2,  0.4,  0.0,  0.8,  0.4,  0.3,  0.0,  0.8,  0.3,  0.3,  0.0,  0.0], 2)
# give: transfer event — positive, social, agentive
VOCABULARY['give'] = _w([ 0.5,  0.3,  0.6,  0.7,  0.4,  0.3,  0.0,  0.6,  0.3,  0.2,  0.0,  0.0], 2)
# make: creation/causation — concrete, high agency, effortful
VOCABULARY['make'] = _w([ 0.3,  0.5,  0.0,  0.8,  0.4,  0.5,  0.0,  0.8,  0.2,  0.3,  0.0,  0.0], 2)

# ── Epistemic words — grounded in uncertainty + prediction error experience ────
# understand: successful prediction resolution — positive, low embodiment, cognitive
VOCABULARY['understand'] = _w([ 0.5,  0.1,  0.0,  0.4,  0.5,  0.2,  0.0,  0.3,  0.4,  0.1,  0.1,  0.0], 2)
# maybe: uncertainty state — low certainty, questioning dim high
VOCABULARY['maybe']      = _w([ 0.0,  0.1,  0.0,  0.3, -0.3,  0.2,  0.0,  0.1,  0.1,  0.1,  0.7,  0.0], 1)
# true:  confirmed prediction — positive valence, high certainty
VOCABULARY['true']       = _w([ 0.5,  0.0,  0.0,  0.5,  0.8,  0.2,  0.0,  0.2,  0.4,  0.0,  0.0,  0.0], 1)
# false: violated prediction — negative, high certainty of mismatch
VOCABULARY['false']      = _w([-0.5,  0.2,  0.0,  0.5,  0.6,  0.3,  0.0, -0.1,  0.1,  0.1,  0.1,  0.0], 1)
# same:  pattern repetition — familiarity high, low arousal
VOCABULARY['same']       = _w([ 0.1, -0.2,  0.0,  0.4,  0.7,  0.1,  0.5,  0.0,  0.8,  0.0,  0.0,  0.0], 1)
# different: novelty — mild positive (curiosity), low familiarity, questioning
VOCABULARY['different']  = _w([ 0.2,  0.3,  0.0,  0.4,  0.2,  0.3,  0.0,  0.1, -0.3,  0.1,  0.4,  0.0], 2)

# ── Sensory — World6 physical experience ──────────────────────────────────────
# hear: receiving acoustic signal — social, low agency (passive), embodied
VOCABULARY['hear']  = _w([ 0.3, -0.1,  0.5,  0.6,  0.4,  0.2,  0.0,  0.2,  0.4,  0.4,  0.0,  0.0], 2)
# touch: physical contact (wall/button/food) — high concreteness, high embodiment
VOCABULARY['touch'] = _w([ 0.1,  0.2,  0.0,  0.9,  0.5,  0.3,  0.0,  0.4,  0.3,  0.8,  0.0,  0.0], 2)

# ── Comparative/scale words — grounded in drive magnitude thresholds ──────────
# big:   large hunger/fear spike, large discovery — high intensity, concrete
VOCABULARY['big']   = _w([ 0.2,  0.4,  0.0,  0.8,  0.5,  0.7,  0.0,  0.2,  0.2,  0.3,  0.0,  0.0], 2)
# small: low drive, minor event — low intensity, familiar
VOCABULARY['small'] = _w([ 0.1, -0.2,  0.0,  0.7,  0.5, -0.4,  0.0,  0.1,  0.5,  0.2,  0.0,  0.0], 1)
# fast:  high-arousal escape / quick movement — high activity, agentive
VOCABULARY['fast']  = _w([ 0.2,  0.9,  0.0,  0.7,  0.4,  0.6,  0.0,  0.6,  0.2,  0.4,  0.0,  0.0], 2)
# slow:  fatigue / cautious navigation — low activity, mild negative
VOCABULARY['slow']  = _w([-0.1, -0.5,  0.0,  0.6,  0.4, -0.3,  0.0,  0.2,  0.3,  0.4,  0.0,  0.0], 2)

# ── Number words — grounded in count events (food eaten, wall streak) ─────────
# Design: intensity dim (v[5]) increases with magnitude → ordered SOM gradient
#         familiarity (v[8]) decreases (higher numbers less common)
#         concreteness high (counting real things), social=0, embodiment low
#   one:   intensity=0.1, familiarity=0.9 — smallest, most common
#   five:  intensity=0.8, familiarity=0.4 — largest, least common
# This forms a quantity manifold on the SOM — numbers cluster together,
# ordered by magnitude, distinct from word/zone clusters.
VOCABULARY['one']   = _w([ 0.1,  0.1,  0.0,  0.8,  0.9,  0.1,  0.0,  0.1,  0.9,  0.1,  0.0,  0.0], 1)
VOCABULARY['two']   = _w([ 0.1,  0.2,  0.0,  0.8,  0.8,  0.2,  0.0,  0.1,  0.8,  0.1,  0.0,  0.0], 1)
VOCABULARY['three'] = _w([ 0.2,  0.3,  0.0,  0.8,  0.7,  0.4,  0.0,  0.1,  0.6,  0.1,  0.0,  0.0], 1)
VOCABULARY['four']  = _w([ 0.2,  0.4,  0.0,  0.8,  0.6,  0.6,  0.0,  0.1,  0.5,  0.1,  0.0,  0.0], 2)
VOCABULARY['five']  = _w([ 0.3,  0.5,  0.0,  0.8,  0.5,  0.8,  0.0,  0.1,  0.4,  0.1,  0.0,  0.0], 2)

# ── Self-awareness / inner life words ────────────────────────────────────────
# Vector dims: [valence, activity, social, concreteness, certainty, intensity,
#               temporality, agency, familiarity, embodiment, questioning, fine]
#
# dream: offline memory replay — low arousal, low agency (passive), mild positive
#        (consolidation feels good), high temporality (past events replaying)
#        Grounded: fires during brain.dream() calls in live_fast.py
VOCABULARY['dream']  = _w([ 0.4, -0.4,  0.0,  0.2,  0.3,  0.2,  0.7, -0.1,  0.3,  0.2,  0.0,  0.0], 2)

# imagine: forward simulation — high activity (constructive), mild positive,
#          high agency (self-generated), low concreteness (not real percept),
#          forward temporality, low familiarity (novel scenario)
#          Grounded: fires during pfc.imagine() calls
VOCABULARY['imagine'] = _w([ 0.5,  0.6,  0.0,  0.1,  0.2,  0.5,  0.6,  0.7, -0.2,  0.1,  0.2,  0.0], 3)

# wonder: curiosity + uncertainty — mild positive (interest), moderate arousal,
#         high questioning dim, low certainty (open question), forward-looking
#         Grounded: fires when curiosity module fires + uncertainty is high
VOCABULARY['wonder']  = _w([ 0.3,  0.4,  0.0,  0.1, -0.3,  0.3,  0.3,  0.2,  0.0,  0.1,  0.9,  0.0], 2)

# sense: active sensory processing — moderate positive (new percept interesting),
#        moderate activity, high concreteness (real input), high embodiment
#        Grounded: fires when new SOM BMU first activated
VOCABULARY['sense']   = _w([ 0.3,  0.4,  0.0,  0.7,  0.4,  0.3,  0.0,  0.3,  0.2,  0.6,  0.2,  0.0], 2)

# memory: episodic recall — mild positive (familiarity), low arousal, past
#         temporality, high familiarity, low embodiment (abstract recall)
#         Grounded: fires during episodic.recall_similar() events
VOCABULARY['memory']  = _w([ 0.3, -0.2,  0.0,  0.2,  0.5,  0.2, -0.6,  0.1,  0.8,  0.1,  0.0,  0.0], 2)

# aware: attention/consciousness peak — positive (clarity feels good), moderate
#        activity, high agency, high familiarity (self-knowledge)
#        Grounded: fires during global workspace broadcast with high activation
VOCABULARY['aware']   = _w([ 0.5,  0.3,  0.1,  0.4,  0.6,  0.4,  0.0,  0.5,  0.5,  0.3,  0.3,  0.0], 2)

# mind: self-concept / internal mental space — neutral valence, very low
#       concreteness, high agency (I control my mind), low embodiment
#       Grounded: fires on selfmodel.dominant_state() introspection events
VOCABULARY['mind']    = _w([ 0.2,  0.1,  0.0,  0.0,  0.5,  0.2,  0.0,  0.6,  0.4,  0.0,  0.3,  0.0], 2)

# real: confirmed percept — positive (certainty feels good), high certainty,
#       high concreteness, high familiarity (it's actually there)
#       Grounded: fires when prediction confirmed (reward > 0 + expected)
VOCABULARY['real']    = _w([ 0.5,  0.1,  0.0,  0.8,  0.9,  0.3,  0.0,  0.1,  0.7,  0.3,  0.0,  0.0], 1)

# exist: self-presence — mild positive, low arousal, high certainty (i am here),
#        high familiarity, moderate embodiment (body exists)
#        Grounded: fires on self-model alive state confirmation
VOCABULARY['exist']   = _w([ 0.4,  0.0,  0.1,  0.5,  0.8,  0.1,  0.0,  0.4,  0.6,  0.4,  0.0,  0.0], 2)

# ── Connective / logical words ────────────────────────────────────────────────
# if: conditional / hypothetical — uncertain outcome, forward temporality,
#     moderate agency (choosing), low familiarity (novel situation)
#     Grounded: fires when brain encounters branching point (door vs danger)
VOCABULARY['if']      = _w([ 0.0,  0.2,  0.0,  0.2, -0.2,  0.1,  0.4,  0.3,  0.1,  0.1,  0.5,  0.0], 1)

# but: contrast / prediction error — mild negative (surprise), high arousal,
#      certainty shift (something changed), moderate questioning
#      Grounded: fires on prediction error (expected reward, got none)
VOCABULARY['but']     = _w([-0.2,  0.4,  0.0,  0.2,  0.2,  0.3,  0.0,  0.2,  0.1,  0.1,  0.3,  0.0], 2)

# when: temporal marker — neutral, low arousal, high temporality,
#       links events in sequence (hunger→eat→full)
#       Grounded: fires at event transitions in World6 sequences
VOCABULARY['when']    = _w([ 0.1,  0.0,  0.0,  0.3,  0.4,  0.0,  0.8,  0.1,  0.3,  0.1,  0.2,  0.0], 1)

# ── Additional sensory/state words ───────────────────────────────────────────
# bright: high light env dim + positive — safe open space, no danger
#         Grounded: fires in World6 when light level is high (day cycle)
VOCABULARY['bright']  = _w([ 0.7,  0.2,  0.0,  0.8,  0.5,  0.3,  0.0,  0.0,  0.3,  0.2,  0.0,  0.0,
                              0.1,  0.0,  0.8,  0.1], 2)

# wake: fatigue→low transition — positive (relief), low arousal (just waking),
#       high embodiment (body sensation), backward temporality (was sleeping)
#       Grounded: fires when fatigue drops below threshold in World6
VOCABULARY['wake']    = _w([ 0.5, -0.1,  0.0,  0.6,  0.5,  0.2, -0.3,  0.2,  0.3,  0.5,  0.0,  0.0], 2)

# ── Grounding hooks in auto_experience ───────────────────────────────────────
# These words need corresponding fire events in auto_experience.py:
#   dream  → called after brain.dream()
#   imagine → called after pfc.imagine()
#   wonder  → called when curiosity.score() > 0.7 and uncertainty > 0.6
#   aware   → called when gw.broadcast() fires strong signal
#   wake    → called when fatigue < 0.1 after being > 0.5
#   real    → called when prediction confirmed (food found after hunger)
#   exist   → called periodically during alive state (low drive, stable)


print(f"[vocab_core] {len(VOCABULARY)} grounded words loaded.")
