"""
TEACH CONVERSATION — Stimulus-Response Pairing
===============================================

Teaches the brain conversation patterns by playing stimulus phrases
followed immediately by response phrases, with reward at the end.

This trains M72's transition probability matrix so that after hearing
"hello", the brain's sequence predictor expects "hi" or "good" to follow.
That expectation drives M74 production — the brain produces words that
statistically follow what it just heard.

HOW IT WORKS
------------
For each pair (stimulus, response):
  1. Play stimulus words (as if the user is speaking)
  2. Short silence (turn boundary)
  3. Play response words (as if the brain is hearing itself respond)
  4. Give reward

Repeat 400-600 times per pair. After training:
  M72._P[stimulus_bmu, response_bmu] is high → response follows naturally.
  M73 binding also strengthens the stimulus→response state chain.

Run: python teach_conversation.py
Then: python converse.py
"""

import sys, time
import numpy as np

sys.dont_write_bytecode = True
from brain import Brain

SEED = 42
rng  = np.random.default_rng(SEED)
N_MFCC = 13

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
    'calm':     (_word([ 0.2, 0.1,-0.1, 0.2,-0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]), 3),
    'what':     (_word([ 0.6, 0.4,-1.0, 0.2,-0.6, 0.1,-0.5, 0.0,-0.4, 0.0,-0.3, 0.0]), 3),
    'who':      (_word([ 0.7, 0.5,-0.8, 0.3,-0.7, 0.1,-0.6, 0.0,-0.5, 0.0,-0.4, 0.0]), 2),
    'where':    (_word([ 0.5, 0.3,-0.9, 0.4,-0.5, 0.2,-0.4, 0.1,-0.3, 0.1,-0.2, 0.1]), 3),
    'when':     (_word([ 0.4, 0.6,-0.7, 0.5,-0.4, 0.3,-0.3, 0.2,-0.2, 0.1,-0.1, 0.1]), 3),
    'why':      (_word([ 0.8, 0.2,-1.1, 0.1,-0.8, 0.0,-0.7,-0.1,-0.6,-0.1,-0.5,-0.1]), 2),
    'how':      (_word([ 0.3, 0.7,-0.6, 0.6,-0.3, 0.4,-0.2, 0.3,-0.1, 0.2, 0.0, 0.1]), 2),
    'can':      (_word([ 0.9,-0.2, 0.5,-0.1, 0.3, 0.0, 0.2, 0.1, 0.1, 0.1, 0.0, 0.1]), 2),
    'do':       (_word([ 1.1,-0.1, 0.4, 0.0, 0.2, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0]), 2),
    'is':       (_word([ 1.0, 0.0, 0.3, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]), 2),
    'water':    (_word([-0.2, 1.3,-0.4, 1.0,-0.3, 0.7,-0.2, 0.4,-0.1, 0.2, 0.0, 0.1]), 4),
    'light':    (_word([ 0.7,-1.2, 0.6,-0.9, 0.4,-0.6, 0.3,-0.4, 0.2,-0.2, 0.1,-0.1]), 3),
    'sound':    (_word([-0.3, 1.0,-0.2, 0.8,-0.1, 0.5, 0.0, 0.3, 0.1, 0.1, 0.1, 0.0]), 3),
    'mind':     (_word([ 0.6,-0.4, 0.8,-0.3, 0.6,-0.2, 0.5,-0.1, 0.4, 0.0, 0.3, 0.0]), 3),
    'world':    (_word([-0.5, 0.8,-0.5, 0.7,-0.3, 0.5,-0.2, 0.4,-0.1, 0.3, 0.0, 0.2]), 4),
    'time':     (_word([ 0.5,-0.6, 0.9,-0.4, 0.7,-0.3, 0.5,-0.2, 0.4,-0.1, 0.3, 0.0]), 3),
    'come':     (_word([-0.2, 0.8, 0.4,-1.0, 0.2,-0.6, 0.1,-0.5, 0.0,-0.4, 0.0,-0.3]), 2),
    'stop':     (_word([-0.6,-0.4, 0.7,-0.3, 0.6,-0.2, 0.5,-0.1, 0.4, 0.0, 0.3, 0.0]), 3),
    'wait':     (_word([ 0.0, 0.2,-0.2, 0.3,-0.1, 0.2, 0.0, 0.2, 0.0, 0.1, 0.0, 0.1]), 3),
    'listen':   (_word([-0.1, 0.9,-0.1, 0.8, 0.0, 0.6, 0.0, 0.4, 0.1, 0.2, 0.1, 0.1]), 4),
    'speak':    (_word([ 0.8,-0.3, 1.0,-0.1, 0.8, 0.0, 0.6, 0.0, 0.4, 0.1, 0.2, 0.1]), 3),
    'think':    (_word([ 0.5, 0.6,-0.2, 0.7,-0.1, 0.5, 0.0, 0.4, 0.0, 0.3, 0.0, 0.2]), 3),
    'feel':     (_word([ 0.3,-0.7, 0.5,-0.6, 0.4,-0.5, 0.3,-0.4, 0.2,-0.3, 0.2,-0.2]), 3),
    'know':     (_word([ 0.7, 0.5, 0.3, 0.4, 0.2, 0.3, 0.1, 0.2, 0.1, 0.1, 0.0, 0.1]), 3),
    'want':     (_word([-0.3, 0.5,-0.8, 0.4,-0.7, 0.3,-0.6, 0.2,-0.5, 0.1,-0.4, 0.0]), 3),
    'near':     (_word([ 0.4, 1.0, 0.2, 0.9, 0.1, 0.7, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2]), 3),
    'new':      (_word([ 0.6, 0.9, 0.5, 0.7, 0.4, 0.5, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1]), 2),
    'more':     (_word([ 0.3, 1.1,-0.1, 0.9, 0.0, 0.7, 0.0, 0.5, 0.0, 0.3, 0.0, 0.2]), 3),
    'i':        (_word([ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 1),
    'you':      (_word([-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1]), 2),
    'me':       (_word([ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]), 2),
    'we':       (_word([ 0.3, 0.1,-0.1, 0.2,-0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]), 2),
    'here':     (_word([-0.5,-1.0, 1.0, 0.5,-0.8, 0.4,-0.3, 0.3,-0.1, 0.2, 0.0, 0.1]), 3),
    'this':     (_word([ 0.7,-0.3, 0.6,-0.2, 0.4,-0.1, 0.3, 0.0, 0.2, 0.0, 0.1, 0.0]), 2),
    'not':      (_word([-1.4, 0.0,-0.9, 0.0,-0.6, 0.0,-0.4, 0.0,-0.2, 0.0,-0.1, 0.0]), 2),
    'now':      (_word([ 1.4, 0.0, 0.9, 0.0, 0.6, 0.0, 0.4, 0.0, 0.2, 0.0, 0.1, 0.0]), 2),
    'maybe':    (_word([ 0.1, 0.3,-0.1, 0.4,-0.1, 0.3, 0.0, 0.2, 0.0, 0.1, 0.0, 0.1]), 3),
    'help':     (_word([-0.8, 0.9,-0.4, 1.1,-0.2, 0.8,-0.1, 0.5, 0.0, 0.3, 0.0, 0.1]), 3),
    'ready':    (_word([ 0.7, 0.7, 0.5, 0.6, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1]), 4),
    'done':     (_word([ 1.0,-0.6, 0.8,-0.4, 0.6,-0.3, 0.5,-0.2, 0.4,-0.1, 0.3, 0.0]), 3),
    'together': (_word([ 0.2, 0.6, 0.3, 0.5, 0.2, 0.4, 0.1, 0.3, 0.1, 0.2, 0.0, 0.1]), 4),
    'always':   (_word([ 0.6, 0.4, 0.5, 0.3, 0.4, 0.2, 0.3, 0.1, 0.2, 0.1, 0.1, 0.0]), 3),
    'love':     (_word([ 1.4, 0.6,-0.2, 0.5,-0.1, 0.4, 0.0, 0.3, 0.1, 0.2, 0.1, 0.1]), 3),
}

SILENCE = np.zeros(N_MFCC, dtype=np.float32)
SILENCE[0] = -4.0

def say(word, noise_std=0.18):
    mean_vec, n_frames = VOCABULARY[word]
    return [(mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
            for _ in range(n_frames)]

def play(words, reward=0.0, noise_std=0.18):
    """Feed a word sequence to the brain. Reward given on last frame of last word."""
    for w_i, word in enumerate(words):
        frames = say(word, noise_std)
        is_last_word = (w_i == len(words) - 1)
        for f_i, frame in enumerate(frames):
            brain.hear(frame)
            r = reward if (is_last_word and f_i == len(frames) - 1) else 0.0
            brain.step(reward=r)
        brain.hear(SILENCE)
        brain.step()

def silence(n=3):
    """Short inter-turn pause."""
    for _ in range(n):
        brain.hear(SILENCE)
        brain.step()


# ── Load brain ─────────────────────────────────────────────────────────
print("=" * 60)
print("  TEACHING CONVERSATION")
print("=" * 60)
print()
brain = Brain.load('brain_trained.pkl')
print(f"  Starting at step {brain.t:,}")

# CRITICAL: freeze the SOM so BMU assignments don't drift during conversation
# training. Without this, M72's transition stats are built on moving targets.
# The SOM is mature after teach_english.py — no more structural learning needed.
brain.speech_cortex._eta = 0.0   # freeze SOM weights
print("  SOM frozen (BMU assignments locked)")
t_start = time.time()


# ═══════════════════════════════════════════════════════════════════════
# CONVERSATION PAIRS
# ═══════════════════════════════════════════════════════════════════════
# Format: (stimulus_words, response_words, reward, n_reps)
# stimulus = what the user says
# response = what the brain should learn to say back
# reward   = how positive this exchange is (higher = faster learning)

PAIRS = [
    # Greetings
    (['hello'],              ['hi', 'good'],         0.9, 600),
    (['hi'],                 ['hello', 'good'],       0.9, 600),
    (['hello', 'good'],      ['yes', 'good'],         0.9, 500),
    (['bye'],                ['bye', 'okay'],         0.7, 400),

    # "How are you?" → positive response
    (['how', 'are', 'you'] if 'are' in VOCABULARY else ['how', 'is', 'this'],
                             ['good', 'calm'],        0.9, 500),
    (['how', 'do', 'you', 'feel'],
                             ['good', 'calm'],        0.9, 500),
    (['how', 'is', 'this'],  ['good', 'okay'],        0.8, 400),

    # "Who are you?" → self-response
    (['who', 'is', 'this'],  ['i', 'here'],           0.8, 500),
    (['who', 'are', 'you'] if 'are' in VOCABULARY else ['who', 'is', 'this'],
                             ['i', 'here'],           0.8, 500),
    (['what', 'is', 'this'], ['i', 'think'],          0.7, 400),
    (['what', 'do', 'you', 'want'],
                             ['i', 'want', 'more'],   0.8, 500),

    # Questions → curious response
    (['why', 'is', 'this'],  ['i', 'think', 'new'],   0.7, 400),
    (['why'],                ['maybe', 'think'],      0.6, 400),
    (['what'],               ['i', 'think'],          0.6, 400),
    (['where'],              ['here', 'now'],         0.6, 400),
    (['when'],               ['now', 'ready'],        0.6, 400),

    # Positive exchanges
    (['thank', 'you'],       ['yes', 'good'],         1.0, 600),
    (['you', 'are', 'good'] if 'are' in VOCABULARY else ['you', 'good'],
                             ['yes', 'good'],         0.9, 500),
    (['good'],               ['yes', 'good'],         0.9, 500),
    (['yes'],                ['yes', 'good'],         0.9, 600),
    (['okay'],               ['okay', 'good'],        0.8, 500),
    (['yes', 'good'],        ['yes', 'always'],       0.9, 500),
    (['love'],               ['yes', 'love', 'good'], 1.0, 500),
    (['together'],           ['yes', 'together'],     0.9, 400),

    # "I feel X" → acknowledge
    (['i', 'feel', 'happy'], ['yes', 'good', 'happy'],0.9, 500),
    (['i', 'feel', 'sad'],   ['i', 'know', 'sorry'],  0.7, 400),
    (['i', 'feel', 'confused'],
                             ['i', 'know', 'think'],  0.7, 400),
    (['i', 'feel', 'calm'],  ['yes', 'calm', 'good'], 0.8, 400),
    (['i', 'am', 'here'] if 'am' in VOCABULARY else ['i', 'here'],
                             ['yes', 'good', 'near'], 0.9, 400),

    # Commands
    (['come', 'here'],       ['yes', 'here', 'now'],  0.8, 400),
    (['listen'],             ['yes', 'listen'],       0.8, 400),
    (['stop'],               ['okay', 'wait'],        0.7, 400),
    (['wait', 'please'],     ['okay', 'wait'],        0.7, 400),
    (['help', 'me'],         ['yes', 'here', 'okay'], 0.8, 400),
    (['help'],               ['yes', 'here'],         0.8, 400),
    (['please'],             ['yes', 'okay'],         0.8, 400),

    # Emotional support
    (['sorry'],              ['okay', 'good'],        0.7, 400),
    (['i', 'want', 'more'],  ['yes', 'more', 'good'], 0.8, 400),
    (['i', 'think'],         ['yes', 'think'],        0.7, 400),
    (['i', 'know'],          ['yes', 'know', 'good'], 0.8, 400),
    (['i', 'am', 'ready'] if 'am' in VOCABULARY else ['ready'],
                             ['yes', 'ready', 'good'],0.9, 400),
    (['done'],               ['yes', 'done', 'good'], 0.9, 400),

    # World/environment talk
    (['what', 'is', 'that'], ['i', 'think', 'new'],   0.7, 400),
    (['listen', 'this'],     ['yes', 'listen'],       0.7, 300),
    (['do', 'you', 'know'],  ['i', 'think', 'know'],  0.7, 400),
    (['can', 'you', 'speak'],['yes', 'speak', 'now'], 0.8, 400),
    (['speak'],              ['yes', 'speak'],        0.8, 400),
    (['think'],              ['i', 'think'],          0.7, 400),

    # Continuations (teach multi-turn rhythm)
    (['more'],               ['yes', 'more'],         0.7, 400),
    (['now'],                ['yes', 'now'],          0.7, 400),
    (['always', 'together'], ['yes', 'together'],     0.9, 400),
    (['not', 'now'],         ['okay', 'wait'],        0.6, 300),
    (['maybe'],              ['maybe', 'think'],      0.6, 300),

    # Question-asking — brain asks back after a simple exchange.
    # After "okay/yes/good", brain follows up with a question.
    # This trains M72 that a question word can follow a short ack.
    (['okay'],               ['how', 'do', 'you', 'feel'],  0.8, 400),
    (['yes'],                ['what', 'do', 'you', 'want'], 0.7, 400),
    (['good'],               ['do', 'you', 'feel', 'good'], 0.8, 400),
    (['calm'],               ['do', 'you', 'feel', 'calm'], 0.8, 400),
    (['i', 'know'],          ['what', 'do', 'you', 'think'],0.7, 400),
    (['i', 'think'],         ['what', 'do', 'you', 'know'], 0.7, 400),
    (['done'],               ['what', 'do', 'you', 'want'], 0.7, 350),
    (['ready'],              ['do', 'you', 'want', 'more'], 0.8, 350),
    (['i', 'feel', 'happy'], ['what', 'do', 'you', 'want'], 0.8, 350),
    (['i', 'feel', 'sad'],   ['what', 'do', 'you', 'feel'], 0.7, 350),
    (['i', 'here'],          ['do', 'you', 'feel', 'good'], 0.8, 350),
    (['hello', 'good'],      ['how', 'do', 'you', 'feel'],  0.8, 400),
    (['hi', 'good'],         ['do', 'you', 'feel', 'calm'], 0.8, 350),
    (['think'],              ['what', 'do', 'you', 'think'],0.7, 350),
    (['love'],               ['do', 'you', 'feel', 'good'], 0.8, 350),
]

# Filter out pairs with words not in VOCABULARY
def valid(words):
    return all(w in VOCABULARY for w in words)

PAIRS = [(s, r, rew, n) for s, r, rew, n in PAIRS if valid(s) and valid(r)]
print(f"  Teaching {len(PAIRS)} conversation pairs\n")


# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

total_pairs = len(PAIRS)
for pair_idx, (stimulus, response, reward, n_reps) in enumerate(PAIRS):
    stim_str = ' '.join(stimulus)
    resp_str = ' '.join(response)
    t0 = time.time()

    for rep in range(n_reps):
        # Add small random word before/after for natural variation (~20% chance)
        s = stimulus.copy()
        r = response.copy()
        if rng.random() < 0.2:
            filler = rng.choice(list(VOCABULARY.keys()))
            if valid([filler]):
                if rng.random() < 0.5:
                    s = [filler] + s
                else:
                    r = r + [filler]

        play(s, reward=0.0, noise_std=0.20)    # stimulus — no reward yet
        silence(n=2)                           # turn boundary
        play(r, reward=reward, noise_std=0.20) # response — reward here
        silence(n=3)                           # post-turn pause

    elapsed = time.time() - t0
    print(f"  [{pair_idx+1:2d}/{total_pairs}] '{stim_str}' → '{resp_str}'"
          f"   r={reward:.1f}  n={n_reps}  {elapsed:.0f}s")


# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════

print(f"\n  Total steps: {brain.t:,}")
print(f"  Total time:  {time.time()-t_start:.0f}s")
print()

n_bound = int(__import__('numpy').sum(brain.binding._binding_strength > 0.05))
print(f"  Bound phonemes: {n_bound}/400")
print(f"  Proto-words:    {len(brain.phoneme_seq._word_candidates):,}")

brain.save('brain_trained.pkl')
print()
print("  Conversation training done. Run:  python converse.py")
