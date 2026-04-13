"""
SYNTHETIC SPEECH LEARNING TEST — FULL BRAIN (v3)
=================================================

Feeds synthetic speech to Brain v13 and watches it learn language
from scratch. No grid. No zones. No food. No walls. No actions.

The brain sits in one place. A caregiver speaks words in context.
Words get meaning only through co-occurrence with the brain's internal
state. No labels. No grammar. No reward for "correct" answers.

Reward here means what it means biologically: dopamine. Positive
feedback from the environment — a caregiver saying "good", a pleasant
sound, anything that triggers a real reward signal. Here it's simulated
as periodic positive events (like praise or pleasant stimuli).

WHAT YOU'LL SEE
---------------
  proto-words   — recurring phoneme sequences the brain has noticed
  bound-ph      — phoneme patterns with learned associations
  utterances    — times M74 produced a phoneme sequence (speaking)
  reward signal — how strongly key words bind to positive events
  within-TP     — transition probability within words (ideally > between)

Run:   python speech_learn_test.py
Time:  ~2 minutes
"""

import sys, time
import numpy as np
from collections import defaultdict, deque, Counter

sys.dont_write_bytecode = True

from brain import Brain

print("=" * 60)
print("  SYNTHETIC SPEECH LEARNING TEST  (Brain v13)")
print("  Just a brain. Listening. Learning. Speaking.")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SEED         = 42
N_STEPS      = 50_000
REPORT_EVERY = 2_000
N_MFCC       = 13
rng          = np.random.default_rng(SEED)

# ═══════════════════════════════════════════════════════════════
# VOCABULARY
# ═══════════════════════════════════════════════════════════════
# Eight words with distinct acoustic shapes (different MFCC patterns).
# The names are just labels for us — the brain only sees the acoustics.

WORDS = {
    # name       : (MFCC mean vector, n_frames)
    'yes':        (np.array([3.0, 1.2,-0.8, 0.9,-0.4, 0.3,-0.2, 0.1, 0.0,-0.1, 0.1, 0.0,-0.1]), 5),
    'good':       (np.array([3.0, 0.9,-0.5, 0.6,-0.2, 0.5,-0.1, 0.2, 0.1, 0.0,-0.1, 0.1, 0.0]), 4),
    'here':       (np.array([3.0,-0.8, 1.0,-0.5, 0.8,-0.3, 0.4,-0.1, 0.3,-0.1, 0.2, 0.0,-0.1]), 4),
    'confused':   (np.array([3.0,-1.2, 0.7,-0.9, 0.5,-0.4, 0.3,-0.2, 0.1,-0.2, 0.1,-0.1, 0.1]), 5),
    'more':       (np.array([3.0, 0.3, 1.2,-0.4, 0.9,-0.5, 0.3,-0.1, 0.4,-0.2, 0.1, 0.1,-0.1]), 5),
    'no':         (np.array([3.0,-0.4,-1.0, 0.6,-0.8, 0.3,-0.4, 0.1,-0.3, 0.1,-0.2, 0.0, 0.2]), 4),
    'what':       (np.array([3.0, 0.6, 0.4,-1.0, 0.2,-0.6, 0.5,-0.2, 0.2,-0.1, 0.3,-0.1, 0.0]), 4),
    'come':       (np.array([3.0,-0.2, 0.8, 0.4,-1.0, 0.2,-0.5, 0.3,-0.2, 0.4,-0.1, 0.2,-0.1]), 3),
}

SILENCE = np.zeros(N_MFCC, dtype=np.float32)
SILENCE[0] = -4.0   # below M71 silence threshold → unvoiced

def say(word_name, noise_std=0.25):
    """Generate one utterance as a list of MFCC frames."""
    mean_vec, n_frames = WORDS[word_name]
    return [(mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
            for _ in range(n_frames)]

# ═══════════════════════════════════════════════════════════════
# CAREGIVER
# ═══════════════════════════════════════════════════════════════
# Caregiver speaks in context. No labels — just timing.
# Reward events (praise/positive stimulus): caregiver says 'yes'/'good'.
# Brain confused/stuck: caregiver says 'confused'/'no'.
# Caregiver curious: says 'more'/'what'.
# Otherwise: random word (vocabulary exposure).

POSITIVE_WORDS = ['yes', 'good']
STATE_WORDS    = {
    'confused': 'confused',
    'curious':  'more',
    'stuck':    'no',
    'hunting':  'what',
}
SPEAK_PROB = 0.12   # probability of speaking any given step

def caregiver(just_reward, sm_label):
    """Returns a word to say this step, or None."""
    if just_reward:
        return rng.choice(POSITIVE_WORDS)
    if rng.random() < SPEAK_PROB:
        if sm_label in STATE_WORDS:
            return STATE_WORDS[sm_label]
        if rng.random() < 0.3:
            return rng.choice(list(WORDS.keys()))
    return None

# ═══════════════════════════════════════════════════════════════
# BRAIN
# ═══════════════════════════════════════════════════════════════

brain = Brain(seed=SEED)
print(f"\n  Brain initialised — {N_STEPS:,} steps")
print(f"  Vocabulary: {list(WORDS.keys())}\n")

# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

phoneme_bmus    = defaultdict(list)   # word → list of BMUs
within_tp       = deque(maxlen=500)
between_tp      = deque(maxlen=500)
utterance_states= []
prev_bmu        = None

# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

t_start            = time.time()
steps_since_reward = 0
total_rewards      = 0

print("step    | proto-words | bound-ph | utterances | reward-assoc | within-TP | notes")
print("-" * 90)

for step in range(1, N_STEPS + 1):

    # ── Reward signal (periodic positive stimulus) ────────────
    steps_since_reward += 1
    just_reward = (steps_since_reward > 70 and rng.random() < 0.15)
    reward      = 1.0 if just_reward else 0.0
    if just_reward:
        steps_since_reward = 0
        total_rewards += 1

    # ── Caregiver speaks ──────────────────────────────────────
    word = caregiver(just_reward, brain.selfmodel._last_label)

    if word is not None:
        frames      = say(word)
        prev_before = prev_bmu
        for i, frame in enumerate(frames):
            brain.hear(frame)
            out = brain.step(reward=reward if i == len(frames) - 1 else 0.0)
            phoneme_bmus[word].append(out['m71_phoneme_bmu'])
            if i == 0 and prev_before is not None:
                between_tp.append(out['m72_tp'])
            elif i > 0:
                within_tp.append(out['m72_tp'])
            prev_before = out['m71_phoneme_bmu']
        prev_bmu = prev_before

        # Silence between words
        brain.hear(SILENCE)
        out = brain.step(reward=0.0)
    else:
        # Silent step — brain processes its own internal state
        brain.hear(SILENCE)
        out = brain.step(reward=reward)
        prev_bmu = out['m71_phoneme_bmu']

    # ── Utterances ────────────────────────────────────────────
    if out['m74_speaking']:
        utterance_states.append(out['sm_state_label'])

    # ── Report ────────────────────────────────────────────────
    if step % REPORT_EVERY == 0:
        n_cands = out['m72_n_word_candidates']
        n_bound = out['m73_n_bound_phonemes']
        n_utts  = out['m74_total_utterances']

        # Reward association for 'yes' (positive word)
        pos_assoc = "n/a"
        if phoneme_bmus['yes']:
            recent = phoneme_bmus['yes'][-50:]
            top    = max(set(recent), key=recent.count)
            rew    = float(brain.binding._W_reward[top])
            pos_assoc = f"yes={rew:.3f}"

        wtp   = f"{np.mean(within_tp):.3f}" if within_tp else "n/a"
        notes = ""
        if n_cands > 50:  notes += "words↑ "
        if n_bound > 10:  notes += "bound↑ "
        if n_utts  > 0:   notes += "SPEAKS "

        print(f"{step:6d}  | {n_cands:11d} | {n_bound:8d} | {n_utts:10d} | "
              f"{pos_assoc:12s} | {wtp:9s} | {notes}")

# ═══════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  FINAL REPORT")
print("=" * 60)
elapsed = time.time() - t_start

# [1] Clustering
print("\n[1] PHONEME CLUSTERING (M71)")
print("  Consistency: what fraction of utterances land on the same BMU?\n")
for word in WORDS:
    bmus = phoneme_bmus.get(word, [])
    if not bmus:
        print(f"  '{word:10s}'  never heard")
        continue
    recent = bmus[-min(200, len(bmus)):]
    top    = max(set(recent), key=recent.count)
    frac   = recent.count(top) / len(recent)
    unique = len(set(recent))
    bar    = "█" * int(frac * 20)
    print(f"  '{word:10s}'  top_bmu={top:3d}  {frac*100:4.0f}%  unique={unique:3d}  {bar}")

# [2] Word boundary detection
print("\n[2] WORD BOUNDARY DETECTION (M72)")
wtp = float(np.mean(within_tp))  if within_tp  else 0
btp = float(np.mean(between_tp)) if between_tp else 0
print(f"  Within-word TP:   {wtp:.3f}  ← should be HIGH")
print(f"  Between-word TP:  {btp:.3f}  ← should be LOW")
print(f"  Ratio (w/b):      {wtp/max(btp,1e-9):.2f}x")
print(f"  Proto-words:      {len(brain.phoneme_seq._word_candidates)}")
for seq, info in brain.phoneme_seq.top_n_candidates(5):
    seq_str = "[" + ", ".join(str(x) for x in seq) + "]"
    print(f"    {seq_str:<40s}  x{info['count']}")

# [3] Semantic binding
print("\n[3] SEMANTIC BINDING (M73)")
print("  What each word means to the brain (learned from co-occurrence):\n")
for word in WORDS:
    bmus = phoneme_bmus.get(word, [])
    if not bmus:
        continue
    top     = max(set(bmus[-100:]), key=bmus[-100:].count)
    strength= float(brain.binding._binding_strength[top])
    reward  = float(np.clip(brain.binding._W_reward[top], 0, 1))
    print(f"  '{word:10s}'  BMU={top:3d}  strength={strength:.3f}  reward={reward:.3f}")

# [4] Production
print("\n[4] VOCAL PRODUCTION (M74)")
n_utts = brain.vocal._total_utterances
print(f"  Total utterances:    {n_utts}")
print(f"  Rate:                {n_utts / (N_STEPS/1000):.1f} utterances/1000 steps")
if utterance_states:
    top_states = Counter(utterance_states).most_common(3)
    print(f"  Top speaking states: {top_states}")

# [5] Summary
print("\n[5] SUMMARY")
print(f"  Positive events:  {total_rewards}  (periodic reward signal)")
print(f"  Brain steps:      {brain.t:,}")
print(f"  Wall time:        {elapsed:.1f}s  ({N_STEPS/elapsed:.0f} steps/sec)")
print(f"  Current state:    {brain.selfmodel._last_label}")
print()
print("No grid. No navigation. No predefined meanings.")
print("The brain heard sounds and learned what they go with.")
