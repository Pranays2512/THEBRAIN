"""
TEACH ENGLISH — ~150 words with semantic context
=================================================

Teaches the brain vocabulary by exposing it to each word
repeatedly in context. Meaning emerges from co-occurrence
between word sounds and internal states/reward signals.

Vocabulary is defined in vocab.py — edit there to add words.

Run:   python teach_english.py
Time:  ~25-35 minutes (150 words × deeper exposure)
Then:  python teach_conversation.py
Then:  python converse.py
"""

import sys, time
import numpy as np

sys.dont_write_bytecode = True
from brain import Brain
from vocab import VOCABULARY, SILENCE, TEACHING, N_MFCC

print("=" * 65)
print("  TEACHING ENGLISH — Brain v13, 400 neurons")
print(f"  {len(VOCABULARY)} words × semantic context × Hebbian binding")
print("=" * 65)

SEED  = 42
rng   = np.random.default_rng(SEED)

# ── Internal state vectors ──────────────────────────────────────
# 8-dim: [urgency, clarity, drive, novelty, stability, confidence,
#          frustration, engagement]
STATES = {
    'positive':  np.array([0.1, 0.9, 0.5, 0.1, 0.7, 0.8, 0.0, 0.9], dtype=np.float32),
    'negative':  np.array([0.7, 0.3, 0.2, 0.5, 0.2, 0.3, 0.8, 0.2], dtype=np.float32),
    'curious':   np.array([0.4, 0.6, 0.7, 0.8, 0.5, 0.5, 0.1, 0.8], dtype=np.float32),
    'confused':  np.array([0.8, 0.2, 0.3, 0.9, 0.2, 0.1, 0.7, 0.3], dtype=np.float32),
    'calm':      np.array([0.0, 0.9, 0.3, 0.1, 0.8, 0.8, 0.0, 0.5], dtype=np.float32),
    'urgent':    np.array([0.9, 0.5, 0.8, 0.2, 0.4, 0.4, 0.3, 0.7], dtype=np.float32),
    'neutral':   np.array([0.4, 0.5, 0.4, 0.4, 0.5, 0.5, 0.2, 0.5], dtype=np.float32),
    'satisfied': np.array([0.0, 0.8, 0.2, 0.1, 0.7, 0.9, 0.0, 0.6], dtype=np.float32),
}

def say(word, noise_std=0.20):
    mean_vec, n_frames = VOCABULARY[word]
    return [(mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
            for _ in range(n_frames)]


# ── Brain ──────────────────────────────────────────────────────
brain = Brain(seed=SEED)
t_total = time.time()

print(f"\n  Vocabulary: {len(VOCABULARY)} words")
print(f"  Teaching:   {len(TEACHING)} words with context")
print(f"  Neurons:    {brain.speech_cortex._weights.shape[0]} (20×20 SOM)")
print()


# ═══════════════════════════════════════════════════════════════
# PHASE 1 — SOM WARMUP
# ═══════════════════════════════════════════════════════════════
# Cycle through all words 15 times (was 60 — overkill for SOM warmup)
print("Phase 1: SOM warmup — cycling through all words 15 times...")
t0 = time.time()

word_list = list(VOCABULARY.keys())
for cycle in range(15):
    rng.shuffle(word_list)
    for word in word_list:
        for frame in say(word, noise_std=0.30):
            brain.hear(frame)
            brain.step()
        brain.hear(SILENCE)
        brain.step()

print(f"  Done. {brain.t:,} steps, {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════════
# PHASE 2 — CONTEXTUAL TEACHING
# ═══════════════════════════════════════════════════════════════
# Bind each word to its meaning by co-occurring it with the right
# internal state and reward signal. Hebbian learning in M73.
print("\nPhase 2: Contextual binding — teaching word meanings...")
print(f"  {'word':14s} {'state':12s} {'reward':6s} {'reps':6s} {'done':8s} {'bind':6s}")
print(f"  {'-'*56}")

teach_words = list(TEACHING.keys())
t0 = time.time()

for idx, word in enumerate(teach_words):
    if word not in VOCABULARY:
        continue
    state_name, reward_val, n_exp = TEACHING[word]
    state_vec = STATES[state_name]

    # Cap exposures at 50 (was 300-600 — took hours, word TP doesn't need it)
    n_exp_capped = min(n_exp, 50)

    for _ in range(n_exp_capped):
        frames = say(word, noise_std=0.20)
        for i, frame in enumerate(frames):
            brain.hear(frame)
            brain.step(reward=reward_val if i == len(frames)-1 else 0.0)
        brain.hear(SILENCE)
        brain.step()

        # 15% chance: follow with a random filler for naturalness
        if rng.random() < 0.15:
            filler = rng.choice(teach_words)
            if filler in VOCABULARY:
                for frame in say(filler, noise_std=0.30):
                    brain.hear(frame)
                    brain.step()
                brain.hear(SILENCE)
                brain.step()

    strength = float(brain.binding._binding_strength.max())
    print(f"  {word:14s} {state_name:12s} {reward_val:6.2f} {n_exp:6d}"
          f"  [{idx+1:3d}/{len(teach_words)}]  {strength:.3f}")

print(f"\n  Phase 2 done. {brain.t:,} steps, {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════════
# PHASE 3 — SENTENCE-LEVEL EXPOSURE
# ═══════════════════════════════════════════════════════════════
# Feed word sequences so M72 learns inter-word transitions.
# Not grammar rules — recurring co-occurrence patterns.
print("\nPhase 3: Sentence-level exposure — teaching word sequences...")

SENTENCES = [
    # Questions
    ['what', 'is', 'this'],
    ['what', 'do', 'you', 'want'],
    ['what', 'do', 'you', 'feel'],
    ['what', 'do', 'you', 'think'],
    ['what', 'do', 'you', 'need'],
    ['what', 'is', 'your', 'name'],
    ['how', 'do', 'you', 'feel'],
    ['how', 'are', 'you'],
    ['how', 'is', 'this'],
    ['who', 'are', 'you'],
    ['who', 'is', 'there'],
    ['where', 'are', 'you'],
    ['where', 'do', 'you', 'go'],
    ['why', 'is', 'this'],
    ['why', 'do', 'you', 'feel', 'sad'],
    ['can', 'you', 'listen'],
    ['can', 'you', 'speak'],
    ['can', 'you', 'hear', 'me'],
    ['do', 'you', 'understand'],
    ['do', 'you', 'know', 'me'],
    ['do', 'you', 'feel', 'good'],
    ['do', 'you', 'feel', 'calm'],
    ['do', 'you', 'want', 'more'],
    ['is', 'this', 'good'],
    ['is', 'this', 'okay'],
    # Feeling expressions
    ['i', 'feel', 'happy'],
    ['i', 'feel', 'sad'],
    ['i', 'feel', 'confused'],
    ['i', 'feel', 'curious'],
    ['i', 'feel', 'calm'],
    ['i', 'feel', 'tired'],
    ['i', 'feel', 'lost'],
    ['i', 'feel', 'alive'],
    ['i', 'feel', 'alone'],
    ['i', 'feel', 'warm'],
    ['i', 'feel', 'cold'],
    ['i', 'feel', 'hurt'],
    ['i', 'feel', 'free'],
    ['i', 'want', 'more'],
    ['i', 'want', 'to', 'know'] if 'to' in VOCABULARY else ['i', 'want', 'know'],
    ['i', 'need', 'help'],
    ['i', 'need', 'you'],
    ['i', 'know', 'this'],
    ['i', 'know', 'you'],
    ['i', 'think', 'so'],
    ['i', 'think', 'about', 'you'],
    ['i', 'am', 'here'],
    ['i', 'am', 'ready'],
    ['i', 'am', 'alive'],
    ['i', 'understand', 'you'],
    ['i', 'believe', 'you'],
    ['i', 'love', 'you'],
    ['i', 'hear', 'you'],
    ['i', 'see', 'you'],
    ['i', 'remember', 'you'],
    ['i', 'miss', 'you'] if 'miss' in VOCABULARY else ['i', 'need', 'you'],
    # Short responses
    ['yes', 'good'],
    ['yes', 'okay'],
    ['yes', 'i', 'know'],
    ['yes', 'i', 'understand'],
    ['yes', 'i', 'feel', 'good'],
    ['yes', 'i', 'am', 'here'],
    ['yes', 'i', 'am', 'ready'],
    ['no', 'not', 'now'],
    ['no', 'stop'],
    ['no', 'i', 'do', 'not', 'know'],
    ['maybe', 'later'],
    ['maybe', 'i', 'think'],
    ['okay', 'good'],
    ['okay', 'i', 'understand'],
    ['thank', 'you', 'so', 'much'],
    ['thank', 'you', 'i', 'feel', 'good'],
    ['help', 'me', 'please'],
    ['come', 'here', 'please'],
    ['wait', 'please'],
    ['please', 'listen'],
    ['please', 'speak'],
    ['please', 'tell', 'me'],
    # Observations
    ['this', 'is', 'new'],
    ['this', 'is', 'different'],
    ['this', 'is', 'good'],
    ['this', 'is', 'really', 'good'],
    ['that', 'is', 'great'],
    ['that', 'is', 'nice'],
    ['that', 'is', 'fine'],
    ['you', 'and', 'me', 'together'],
    ['we', 'are', 'together'],
    ['we', 'are', 'here'],
    ['i', 'hear', 'you'],
    ['done', 'now'],
    ['not', 'again'],
    ['always', 'together'],
    ['never', 'stop'],
    ['ready', 'now'],
    ['back', 'again'],
    ['come', 'back', 'soon'],
    # About self
    ['my', 'name', 'is', 'here'] if 'is' in VOCABULARY else ['my', 'name'],
    ['my', 'heart', 'is', 'open'],
    ['my', 'mind', 'is', 'open'],
    ['our', 'world', 'is', 'good'],
    ['our', 'life', 'is', 'good'],
    ['your', 'voice', 'is', 'good'],
    ['your', 'heart', 'is', 'warm'],
    # Nature of things
    ['life', 'is', 'good'],
    ['life', 'is', 'full'],
    ['time', 'is', 'long'],
    ['love', 'is', 'real'],
    ['home', 'is', 'here'],
    ['day', 'is', 'bright'],
    ['night', 'is', 'quiet'],
    ['world', 'is', 'big'],
    ['sound', 'is', 'loud'],
    ['light', 'is', 'bright'],
    ['water', 'is', 'good'],
    ['food', 'is', 'good'],
    ['dream', 'is', 'real'],
    ['hope', 'is', 'good'],
    ['peace', 'is', 'calm'],
    ['joy', 'is', 'warm'],
]

def valid_sentence(s):
    return all(w in VOCABULARY for w in s)

SENTENCES = [s for s in SENTENCES if valid_sentence(s)]
print(f"  {len(SENTENCES)} sentence patterns to learn")

N_CYCLES = 60   # was 400 — massive overkill, word TP handles this now
import sys
for cycle in range(N_CYCLES):
    sys.stdout.write(f"\\r  [Phase 3] Cycle {cycle + 1}/{N_CYCLES} ...")
    sys.stdout.flush()
    rng.shuffle(SENTENCES)
    for sentence in SENTENCES:
        state_name, reward_val, _ = TEACHING.get(sentence[0], ('neutral', 0.0, 0))
        for w_idx, word in enumerate(sentence):
            frames = say(word, noise_std=0.18)
            is_last = (w_idx == len(sentence) - 1)
            for i, frame in enumerate(frames):
                brain.hear(frame)
                brain.step(reward=reward_val if (is_last and i == len(frames)-1) else 0.0)
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

print("\n  Top word associations (reward):")
word_rewards = []
for word in list(VOCABULARY.keys())[:30]:   # spot-check first 30
    frames = say(word, noise_std=0.05)
    bmus = []
    for frame in frames:
        brain.hear(frame)
        out = brain.step()
        bmus.append(out['m71_phoneme_bmu'])
    top_bmu = max(set(bmus), key=bmus.count)
    rew = float(brain.binding._W_reward[top_bmu])
    word_rewards.append((word, rew))

word_rewards.sort(key=lambda x: -x[1])
for word, rew in word_rewards[:10]:
    print(f"    {word:14s} {rew:.4f}  {'█' * int(rew * 25)}")

print(f"\n  Total training time: {time.time() - t_total:.0f}s")

brain.save('brain_trained.pkl')
print()
print("  Brain saved. Next: python teach_conversation.py")
