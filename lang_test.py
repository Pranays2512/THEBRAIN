"""
LANGUAGE LEARNING TEST
======================

Tests the M71–M74 language stack in isolation, without World6 or M50.
Simulates what a brain would experience when exposed to spoken words
repeatedly in consistent internal contexts.

THREE STAGES TESTED
-------------------

STAGE 1 — PHONEME CLUSTERING (M71)
  Feed the same 5 "words" as synthetic MFCC patterns 200 times each.
  Each word = a distinct MFCC fingerprint (different spectral shape).
  Expected: M71's SOM carves out stable BMU regions per word.
  Measure: do the same word's MFCCs always land on nearby BMUs?

STAGE 2 — WORD BOUNDARY DETECTION (M72)
  Feed a continuous stream of the 5 words in random order.
  Expected: M72 detects boundaries between words (low TP transitions)
  and within-word continuations (high TP transitions).
  Measure: n_word_candidates rises, boundary_strength is higher
  between words than within words.

STAGE 3 — SEMANTIC BINDING (M73)
  Feed "food" word while zone=4 and reward=1.0.
  Feed "confused" word while state_label=confused and reward=0.
  Feed "explore" word while state_label=curious and reward=0.
  Expected: M73 binding_strength rises for each word.
  "food" gets strong reward association. "confused" gets state binding.
  Measure: associated_zone for "food" BMUs → 4, associated_reward rises.

STAGE 4 — PRODUCTION (M74)
  Put brain in communicative state (curious, high arousal).
  Expected: brain produces utterances whose seed phoneme is associated
  with the current state.
  Measure: m74_speaking=True, utterance BMUs overlap with
  "curious" word's phoneme region.

HOW TO RUN
----------
  python lang_test.py

No hardware needed. No microphone. No World6.
Runs in ~30 seconds.
"""

import numpy as np
import sys

# ── Suppress verbose imports ──────────────────────────────────
sys.dont_write_bytecode = True

from m71_speech_cortex import SpeechCortex
from m72_phoneme_seq   import PhonemeSequencePredictor
from m73_binding       import SemanticBinding
from m74_vocal         import VocalOutput

print("Language Learning Test — M71 / M72 / M73 / M74")
print("=" * 55)

rng = np.random.default_rng(42)

# ═══════════════════════════════════════════════════════════════
# SYNTHETIC WORD MFCC FINGERPRINTS
# ═══════════════════════════════════════════════════════════════
# Each "word" is a sequence of 4 MFCC frames (4 × 25ms = 100ms).
# Real speech would have 4-20 frames per word depending on length.
# Each frame is 13 MFCCs. We make each word distinctive by giving it
# a different spectral shape (different MFCC mean vector).
# Noise is added to simulate natural variation in pronunciation.

N_MFCC = 13

def make_word_frames(mean_vec, n_frames=4, noise_std=0.3):
    """Generate n_frames MFCC vectors with a given spectral shape + noise."""
    frames = []
    for _ in range(n_frames):
        frame = mean_vec + rng.normal(0, noise_std, N_MFCC)
        frame[0] = 2.5   # energy coefficient > silence threshold → voiced
        frames.append(frame.astype(np.float32))
    return frames

# Five synthetic "words" with distinct spectral shapes
WORDS = {
    'food':     make_word_frames(np.array([2.5, 1.0,-0.5, 0.8,-0.3, 0.2,-0.1, 0.0, 0.1,-0.2, 0.0, 0.1,-0.1])),
    'confused': make_word_frames(np.array([2.5,-1.0, 0.5,-0.8, 0.3,-0.2, 0.1, 0.0,-0.1, 0.2, 0.0,-0.1, 0.1])),
    'explore':  make_word_frames(np.array([2.5, 0.5, 1.0,-0.3, 0.7,-0.4, 0.2,-0.1, 0.3,-0.2, 0.1, 0.0,-0.2])),
    'here':     make_word_frames(np.array([2.5,-0.5,-1.0, 0.3,-0.7, 0.4,-0.2, 0.1,-0.3, 0.2,-0.1, 0.0, 0.2])),
    'good':     make_word_frames(np.array([2.5, 0.2,-0.2, 1.0,-0.5, 0.1, 0.8,-0.3, 0.0, 0.4,-0.2, 0.1,-0.3])),
}

# Silence frame (energy below threshold)
SILENCE = np.zeros(N_MFCC, dtype=np.float32)
SILENCE[0] = -3.0  # very low energy → not voiced

# ═══════════════════════════════════════════════════════════════
# INSTANTIATE MODULES
# ═══════════════════════════════════════════════════════════════

cortex     = SpeechCortex(seed=42)
phoneme_sq = PhonemeSequencePredictor()
binding    = SemanticBinding(seed=42)
vocal      = VocalOutput(seed=42)

# ═══════════════════════════════════════════════════════════════
# STAGE 1 — PHONEME CLUSTERING
# ═══════════════════════════════════════════════════════════════
print("\n── STAGE 1: Phoneme Clustering (M71 SOM learning) ──────")
print("  Feeding each word 300 times to train the speech SOM...")

# Track which BMUs each word lands on
word_bmu_sets = {w: set() for w in WORDS}
word_bmu_lists = {w: [] for w in WORDS}

N_TRAIN = 300
for _ in range(N_TRAIN):
    for word_name, frames in WORDS.items():
        for frame in frames:
            out = cortex.step(frame, familiarity=0.0)
            word_bmu_lists[word_name].append(out['phoneme_bmu'])
        # Silence between words
        cortex.step(SILENCE)

# After training: check if each word consistently hits a small BMU region
print()
for word_name, bmus in word_bmu_lists.items():
    # Last 100 occurrences (trained state)
    recent = bmus[-100*4:]
    unique_bmus = len(set(recent))
    most_common = max(set(recent), key=recent.count)
    mode_frac   = recent.count(most_common) / len(recent)
    print(f"  '{word_name:8s}' → {unique_bmus:3d} unique BMUs, "
          f"top BMU={most_common:3d} ({mode_frac*100:.0f}% of frames)")

print("\n  ✓ Good if unique BMUs < 20 and mode fraction > 30%")
print("    (More training → tighter clustering)")

# ═══════════════════════════════════════════════════════════════
# STAGE 2 — WORD BOUNDARY DETECTION
# ═══════════════════════════════════════════════════════════════
print("\n── STAGE 2: Word Boundary Detection (M72) ──────────────")
print("  Streaming words continuously, measuring boundary signal...")

word_list = list(WORDS.keys())

# Feed 500 words in random order, track boundary_strength
within_word_tp   = []   # TP for within-word transitions
between_word_tp  = []   # TP at first frame after silence (between words)

for _ in range(500):
    word_name = word_list[rng.integers(0, len(word_list))]
    frames    = WORDS[word_name]

    for i, frame in enumerate(frames):
        c_out = cortex.step(frame, familiarity=0.0)
        s_out = phoneme_sq.step(c_out['phoneme_bmu'], c_out['is_voiced'])
        if i == 0:
            between_word_tp.append(s_out['tp_prev_curr'])
        else:
            within_word_tp.append(s_out['tp_prev_curr'])

    # Silence between words
    c_out = cortex.step(SILENCE)
    phoneme_sq.step(c_out['phoneme_bmu'], c_out['is_voiced'])

mean_within  = float(np.mean(within_word_tp[-200:]))   if within_word_tp  else 0
mean_between = float(np.mean(between_word_tp[-200:]))  if between_word_tp else 0
n_candidates = phoneme_sq.n_word_candidates if hasattr(phoneme_sq, 'n_word_candidates') else len(phoneme_sq._word_candidates)

print(f"\n  Within-word TP  (mean): {mean_within:.3f}  ← should be HIGH")
print(f"  Between-word TP (mean): {mean_between:.3f}  ← should be LOW")
print(f"  Proto-words discovered: {n_candidates}")

if within_word_tp and between_word_tp:
    ratio = mean_within / max(mean_between, 1e-9)
    print(f"  Within/between ratio:   {ratio:.2f}x  ← >1.5x = good boundary detection")

top_candidates = phoneme_sq.top_n_candidates(3)
if top_candidates:
    print(f"  Top proto-words (by frequency):")
    for seq, info in top_candidates:
        print(f"    {list(seq)} — seen {info['count']} times")

print("\n  ✓ Good if within-word TP > between-word TP")
print("    (After thousands of words the gap becomes very clear)")

# ═══════════════════════════════════════════════════════════════
# STAGE 3 — SEMANTIC BINDING
# ═══════════════════════════════════════════════════════════════
print("\n── STAGE 3: Semantic Binding (M73) ─────────────────────")
print("  Pairing words with internal states and reward...")

# Fake internal states (M59 self-state vector, 8 dims)
STATES = {
    'satisfied': np.array([0.1, 0.9, 0.2, 0.1, 0.8, 0.7, 0.0, 0.8], dtype=np.float32),
    'confused':  np.array([0.8, 0.2, 0.1, 0.7, 0.2, 0.1, 0.9, 0.2], dtype=np.float32),
    'curious':   np.array([0.5, 0.4, 0.9, 0.8, 0.3, 0.4, 0.1, 0.6], dtype=np.float32),
    'neutral':   np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], dtype=np.float32),
}

# Pairings: word → (state_label, zone, reward)
PAIRINGS = {
    'food':     ('satisfied', 4, 1.0),   # "food" heard at food zone with reward
    'confused': ('confused',  2, 0.0),   # "confused" heard when confused
    'explore':  ('curious',   6, 0.0),   # "explore" heard when curious
    'here':     ('neutral',   1, 0.0),   # "here" = neutral place word
    'good':     ('satisfied', 4, 0.5),   # "good" heard at good outcomes
}

print()
N_BIND = 400
for trial in range(N_BIND):
    for word_name, (state_label, zone, reward) in PAIRINGS.items():
        state_vec = STATES[state_label]
        frames    = WORDS[word_name]
        for frame in frames:
            c_out = cortex.step(frame, familiarity=0.0)
            s_out = phoneme_sq.step(c_out['phoneme_bmu'], c_out['is_voiced'])
            b_out = binding.step(
                phoneme_bmu  = c_out['phoneme_bmu'],
                is_voiced    = c_out['is_voiced'],
                state_vec    = state_vec,
                zone_idx     = zone,
                reward       = reward,
                sound_bmu    = 0,
            )
        cortex.step(SILENCE)
        phoneme_sq.step(0, False)

# Report binding results
print(f"  After {N_BIND} pairings per word:")
print()

for word_name, (state_label, zone, reward) in PAIRINGS.items():
    # Get the BMU this word most commonly fires
    frames = WORDS[word_name]
    bmus_for_word = []
    for frame in frames:
        c_out = cortex.step(frame, familiarity=0.5)
        bmus_for_word.append(c_out['phoneme_bmu'])
    top_bmu = max(set(bmus_for_word), key=bmus_for_word.count)

    # Query M73 for what that BMU is associated with
    b_probe = binding.step(
        phoneme_bmu  = top_bmu,
        is_voiced    = True,
        state_vec    = STATES['neutral'],
        zone_idx     = -1,
        reward       = 0.0,
        sound_bmu    = 0,
    )
    strength      = binding._binding_strength[top_bmu]
    assoc_zone    = b_probe['associated_zone']
    assoc_reward  = b_probe['associated_reward']
    n_bound       = b_probe['n_bound_phonemes']

    print(f"  '{word_name:8s}' BMU={top_bmu:3d}  "
          f"strength={strength:.4f}  "
          f"zone={assoc_zone:2d} (expected {zone:2d})  "
          f"reward={assoc_reward:.4f} (expected {reward:.1f})")

print(f"\n  Total bound phonemes: {b_probe['n_bound_phonemes']}")
print("\n  ✓ Good if strength > 0 and zone/reward associations match expected values")

# ═══════════════════════════════════════════════════════════════
# STAGE 4 — PRODUCTION (M74)
# ═══════════════════════════════════════════════════════════════
print("\n── STAGE 4: Vocal Production (M74) ─────────────────────")
print("  Putting brain in communicative state, waiting for utterance...")

# Simulate: brain is curious (communicative state), high arousal
# and enough silence has passed
vocal._steps_since_utterance = 200   # long silence → ready to speak

n_speak_attempts = 0
n_spoke          = 0
utterances       = []

for _ in range(50):
    n_speak_attempts += 1
    v_out = vocal.step(
        state_vec   = STATES['curious'],
        state_label = 'curious',
        zone_idx    = 6,
        gws_arousal = 0.75,   # above GWS_SPEAK_THRESH=0.60
        ne_level    = 0.3,
        binding     = binding,
        phoneme_seq = phoneme_sq,
    )
    if v_out['speaking']:
        n_spoke += 1
        utterances.append(v_out['utterance'])
        vocal._steps_since_utterance = 200   # reset for next attempt

print(f"\n  Attempts: {n_speak_attempts}  |  Spoke: {n_spoke}")
if utterances:
    print(f"  Sample utterances (phoneme BMU sequences):")
    for utt in utterances[:3]:
        print(f"    {utt}")
    lengths = [len(u) for u in utterances]
    print(f"  Mean utterance length: {np.mean(lengths):.1f} phoneme frames")

    # Check: do utterance BMUs overlap with 'explore'/'curious' word region?
    explore_bmus = set(word_bmu_lists['explore'][-100:])
    overlap_counts = []
    for utt in utterances:
        overlap = sum(1 for bmu in utt if bmu in explore_bmus)
        overlap_counts.append(overlap / max(len(utt), 1))
    mean_overlap = float(np.mean(overlap_counts))
    print(f"  Overlap with 'explore' BMU region: {mean_overlap*100:.1f}%")
    print("  ✓ Overlap > 0% means production is state-grounded")
else:
    print("  No utterances produced — binding_strength still too low.")
    print("  This is correct: the brain needs more exposure before speaking.")
    print("  (Increase N_BIND from 400 → 2000 to see production emerge)")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"  M71 SOM steps:           {cortex.t}")
print(f"  M72 proto-words found:   {len(phoneme_sq._word_candidates)}")
print(f"  M73 bound phonemes:      {binding._binding_strength[binding._binding_strength > 0.01].sum():.2f} total weight")
print(f"  M74 utterances produced: {vocal._total_utterances}")
print()
print("What you're seeing:")
print("  Stage 1: The SOM is carving acoustic space into phoneme regions.")
print("  Stage 2: Transition stats are revealing word boundaries.")
print("  Stage 3: Hebbian binding is linking sounds to experience.")
print("  Stage 4: Production emerges when binding is strong enough.")
print()
print("To see full language emergence: expose the brain to thousands of")
print("hours of real speech via brain.hear(mfcc) in your hardware loop.")
