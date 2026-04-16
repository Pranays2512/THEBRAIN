"""
train_grounded.py — Grounded experiential training for the FastBrain
=====================================================================

Two-phase training that gives the brain real meaning, not just statistics.

PHASE 1 — EMBODIED EXPERIENCE (body-first)
-------------------------------------------
The body runs for N_BODY_STEPS steps. Each step:
  1. body.sense() → get current drive levels (hunger, tiredness, pain, etc.)
  2. Feed the body's MFCC sensation through brain.hear() + brain.step()
     → the SOM learns to map felt states to distinct neurons
  3. When a drive crosses its threshold, feed matching words with reward
     proportional to drive intensity
     → "hungry" trained while actually hungry → word binds to the felt state

Drive → word mappings (only active when drive is high enough):
  hunger  > 0.5  → "hungry", "want", "food", "eat", "need"      (reward ∝ hunger)
  hunger  < 0.2  → "good", "full", "happy", "satisfied"          (relief reward)
  tired   > 0.6  → "tired", "sleep", "rest", "need"              (reward ∝ tired)
  tired   < 0.2  → "rested", "good", "awake", "okay"             (relief reward)
  pain    > 0.3  → "pain", "hurt", "bad", "stop", "no"           (reward ∝ pain)
  pain    < 0.05 → "good", "safe", "okay", "fine"                (relief reward)
  see food       → "food", "see", "eat", "want"
  face visible   → "you", "hello", "hi", "there", "see"
  baseline       → "i", "am", "feel", "here", "me", "think"      (low reward)

The brain also takes random actions (eat, rest, look) to cycle through
drive states — so it experiences hunger AND satisfaction, tired AND rested.

PHASE 2 — LANGUAGE (dialogue corpus)
--------------------------------------
After the body establishes grounded BMU clusters, run 50 dialogue epochs.
These social patterns now build on top of felt meanings:
  "how are you" → TP walks near the "feel good" neurons → "i feel good"
  "are you hungry" → TP walks near hunger neurons → "i want food"

Usage:
    python train_grounded.py            # full training (~2-3 min)
    python train_grounded.py --quick    # short test run (~30s)
    python live_fast.py                 # talk to trained brain
"""

import sys
import os
import time
import argparse
import numpy as np

sys.dont_write_bytecode = True

from brain_fast import FastBrain, SILENCE, N_MFCC
from vocab_extra import VOCABULARY
from body import SimBody

BRAIN_FILE = 'brain_fast.pkl'
rng = np.random.default_rng(int(time.time()) % (2**31))


# ── Grounded word sets — only feed when matching drive is active ──────────────

# Each entry: (word_list, min_reward, drive_scale)
# reward = min_reward + drive_value * drive_scale  (capped at 1.0)
DRIVE_WORDS = {
    'hunger_high': {
        'words':       ['hungry', 'want', 'food', 'eat', 'need', 'more'],
        'base_reward': 0.4,
        'scale':       0.6,
    },
    'hunger_low': {
        'words':       ['good', 'full', 'happy', 'satisfied', 'okay', 'now'],
        'base_reward': 0.5,
        'scale':       0.0,
    },
    'tired_high': {
        'words':       ['tired', 'sleep', 'rest', 'need', 'stop', 'slow'],
        'base_reward': 0.35,
        'scale':       0.55,
    },
    'tired_low': {
        'words':       ['good', 'awake', 'okay', 'ready', 'feel', 'now'],
        'base_reward': 0.45,
        'scale':       0.0,
    },
    'pain_high': {
        'words':       ['pain', 'hurt', 'bad', 'stop', 'no', 'help'],
        'base_reward': 0.5,
        'scale':       0.8,   # pain is a strong teacher
    },
    'pain_low': {
        'words':       ['good', 'safe', 'okay', 'fine', 'calm', 'feel'],
        'base_reward': 0.4,
        'scale':       0.0,
    },
    'see_food': {
        'words':       ['food', 'see', 'eat', 'want', 'there', 'look'],
        'base_reward': 0.45,
        'scale':       0.0,
    },
    'social': {
        'words':       ['you', 'hello', 'hi', 'there', 'see', 'here', 'talk'],
        'base_reward': 0.4,
        'scale':       0.0,
    },
    'baseline': {
        'words':       ['i', 'am', 'feel', 'here', 'me', 'think', 'know',
                        'brain', 'yes', 'no', 'good', 'okay'],
        'base_reward': 0.15,
        'scale':       0.0,
    },
}

# Filter to only words that exist in the vocabulary
for k in DRIVE_WORDS:
    DRIVE_WORDS[k]['words'] = [w for w in DRIVE_WORDS[k]['words'] if w in VOCABULARY]


def feed_word(brain: FastBrain, word: str, reward: float, noise_std: float = 0.08):
    """Feed one word's MFCC frames through the brain with given reward."""
    if word not in VOCABULARY:
        return None
    mean_vec, n_frames = VOCABULARY[word]
    last_bmu = None
    for i, _ in enumerate(range(n_frames)):
        frame = (mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
        brain.hear(frame)
        r = reward if i == n_frames - 1 else 0.0
        out = brain.step(reward=r)
        last_bmu = out['bmu']
    brain.hear(SILENCE)
    brain.step()

    # Update BMU→word map from clean MFCC
    clean_bmu = brain.som.find_bmu(mean_vec.astype(np.float32))
    if word not in brain.word_to_bmu:
        brain.word_to_bmu[word] = clean_bmu
        brain.bmu_to_word[clean_bmu] = word

    return last_bmu


def feed_word_set(brain: FastBrain, drive_key: str, drive_value: float = 0.0):
    """
    Feed all words in a drive category through the brain.
    Reward scales with drive_value so stronger drives teach harder.
    """
    entry = DRIVE_WORDS[drive_key]
    reward = min(1.0, entry['base_reward'] + drive_value * entry['scale'])
    bmus = []
    for word in entry['words']:
        bmu = feed_word(brain, word, reward)
        if bmu is not None:
            bmus.append(bmu)
    return bmus


def senses_to_mfcc(senses: dict) -> np.ndarray:
    """
    Convert an already-obtained senses dict to an MFCC vector.
    Avoids calling body.sense() a second time (which would double-advance
    the simulation clock and prevent drive cycles from completing).
    """
    import numpy as np
    v = np.zeros(N_MFCC, dtype=np.float32)
    v[0] = 3.0

    pain      = senses.get('pain', 0.0)
    hunger    = senses.get('hunger', 0.0)
    tiredness = senses.get('tiredness', 0.0)
    touch     = senses.get('touch', 0.0)
    warmth    = senses.get('warmth', 0.5)
    vision    = senses.get('vision', [])

    if pain > 0.1:
        v[1] = -0.9; v[2] = 0.9; v[6] = pain; v[10] = 1.0
        return v
    if hunger > 0.5:
        v[1] = -0.4; v[2] = 0.6; v[6] = hunger; v[10] = 1.0
        return v
    if tiredness > 0.6:
        v[1] = -0.2; v[2] = -0.7; v[6] = tiredness; v[10] = 0.9
        return v
    if touch > 0.3:
        v[1] = 0.4 if warmth > 0.5 else -0.2
        v[2] = -0.2 if warmth > 0.5 else 0.4
        v[6] = touch; v[10] = 1.0
        return v
    if 'face' in vision:
        v[1] = 0.3; v[2] = 0.4; v[3] = 0.9; v[5] = 0.6
        return v
    if 'food' in vision:
        v[1] = 0.6; v[2] = 0.5; v[6] = 0.7; v[4] = 0.9
        return v
    # neutral baseline
    v[5] = 0.4; v[7] = 0.5
    return v


def pick_action(senses: dict) -> str:
    """
    Choose a body action based on current drive state.
    Drives the body through full experience cycles:
    hungry → look for food → eat → satisfied → tired → rest → rested → repeat
    """
    hunger    = senses.get('hunger', 0.0)
    tiredness = senses.get('tiredness', 0.0)
    vision    = senses.get('vision', [])
    pain      = senses.get('pain', 0.0)

    if pain > 0.2:
        return 'idle'       # freeze when hurt

    # Tiredness beats hunger — a totally exhausted body can't eat anyway
    if tiredness > 0.7:
        return 'rest'

    if hunger > 0.6:
        if 'food' in vision:
            return 'eat'    # eat if food visible
        else:
            return 'look'   # look for food

    # Random exploration otherwise
    r = rng.random()
    if r < 0.3:
        return 'look'
    elif r < 0.5:
        return 'move'
    elif r < 0.6:
        return 'touch'
    elif r < 0.7:
        return 'rest'
    else:
        return 'idle'


# ── Phase 1: Body experience ──────────────────────────────────────────────────

def run_body_phase(brain: FastBrain, n_steps: int, report_every: int = 10_000):
    """
    Run the body simulation for n_steps.
    The brain hears body sensations and learns grounded word meanings.
    """
    body = SimBody(seed=int(time.time()) % (2**31))

    # Randomize starting drives so we get variety early
    body.hunger    = rng.uniform(0.3, 0.8)
    body.tiredness = rng.uniform(0.4, 0.8)   # start tired so cycles happen early

    WORD_FEED_EVERY = 6    # feed words every N body steps when drive active
    ACTION_EVERY    = 5    # take a body action every N steps (fast cycling)

    # Occasionally trigger pain (hard object collision)
    PAIN_EVERY = 800

    # Track relief events (drive drops below threshold)
    _prev_hunger   = body.hunger
    _prev_tired    = body.tiredness
    _prev_pain     = 0.0

    total_words_fed = 0
    n_hunger_cycles = 0
    n_tired_cycles  = 0
    n_pain_events   = 0

    t_start = time.time()

    for t in range(n_steps):
        senses = body.sense()
        hunger    = senses['hunger']
        tiredness = senses['tiredness']
        pain      = senses['pain']
        vision    = senses['vision']

        # ── Step 1: Brain feels body state ────────────────────────────
        # Use senses dict we already have — don't call sense() again,
        # that would double-advance the world and prevent drive cycles.
        body_mfcc = senses_to_mfcc(senses)
        brain.hear(body_mfcc)
        brain.step(reward=0.0)

        # ── Step 2: Take action every ACTION_EVERY steps ──────────────
        if t % ACTION_EVERY == 0:
            action = pick_action(senses)
            body.act(action)
            # Eat repeatedly when very hungry until satisfied
            if action == 'eat' and hunger > 0.6:
                for _ in range(2):
                    body.act('eat')
            # Rest repeatedly when very tired until rested enough
            if action == 'rest' and tiredness > 0.65:
                for _ in range(4):
                    body.act('rest')

        # ── Step 3: Trigger pain occasionally ─────────────────────────
        if t % PAIN_EVERY == 0 and pain < 0.1:
            body._pain = rng.uniform(0.3, 0.7)

        # ── Step 4: Feed grounded words when drives are active ─────────
        if t % WORD_FEED_EVERY == 0:
            bmus = []

            # Hunger
            if hunger > 0.5:
                bmus += feed_word_set(brain, 'hunger_high', hunger)
                total_words_fed += len(DRIVE_WORDS['hunger_high']['words'])
            elif hunger < 0.15 and _prev_hunger >= 0.15:
                # Relief event: just satisfied hunger
                bmus += feed_word_set(brain, 'hunger_low')
                total_words_fed += len(DRIVE_WORDS['hunger_low']['words'])
                n_hunger_cycles += 1

            # Tiredness
            if tiredness > 0.6:
                bmus += feed_word_set(brain, 'tired_high', tiredness)
                total_words_fed += len(DRIVE_WORDS['tired_high']['words'])
            elif tiredness < 0.30 and _prev_tired >= 0.30:
                # Relief event: just rested
                bmus += feed_word_set(brain, 'tired_low')
                total_words_fed += len(DRIVE_WORDS['tired_low']['words'])
                n_tired_cycles += 1

            # Pain
            if pain > 0.3:
                bmus += feed_word_set(brain, 'pain_high', pain)
                total_words_fed += len(DRIVE_WORDS['pain_high']['words'])
                n_pain_events += 1
            elif pain < 0.05 and _prev_pain >= 0.05:
                bmus += feed_word_set(brain, 'pain_low')
                total_words_fed += len(DRIVE_WORDS['pain_low']['words'])

            # Visual grounding
            if 'food' in vision:
                bmus += feed_word_set(brain, 'see_food')
                total_words_fed += len(DRIVE_WORDS['see_food']['words'])

            # Social presence (simulated)
            if t % 300 < 30:
                bmus += feed_word_set(brain, 'social')
                total_words_fed += len(DRIVE_WORDS['social']['words'])

            # Baseline identity words — always
            bmus += feed_word_set(brain, 'baseline')
            total_words_fed += len(DRIVE_WORDS['baseline']['words'])

            # Buffer for dreaming
            if bmus:
                brain.record_turn(bmus, 0.4)

            # Snapshot drive levels AT feed intervals — this is what the
            # relief detection compares against next feed check.
            # Updating every step loses the transition (action happens between checks).
            _prev_hunger = hunger
            _prev_tired  = tiredness
            _prev_pain   = pain

        # ── Step 5: Dream every 5000 steps ────────────────────────────
        if t > 0 and t % 5000 == 0:
            brain.dream(n_sequences=20)

        # ── Report progress ────────────────────────────────────────────
        if t > 0 and t % report_every == 0:
            elapsed = time.time() - t_start
            steps_per_sec = t / elapsed
            eta = (n_steps - t) / steps_per_sec
            print(f"  [{t:>7,}/{n_steps:,}]  "
                  f"hunger={hunger:.2f}  tired={tiredness:.2f}  pain={pain:.2f}  "
                  f"words={total_words_fed:,}  "
                  f"hunger_cycles={n_hunger_cycles}  tired_cycles={n_tired_cycles}  "
                  f"pain_events={n_pain_events}  "
                  f"[{elapsed:.0f}s, ETA {eta:.0f}s]")

    elapsed = time.time() - t_start
    print(f"\n  Body phase complete: {n_steps:,} steps in {elapsed:.1f}s")
    print(f"  Words fed: {total_words_fed:,}")
    print(f"  Hunger cycles: {n_hunger_cycles}")
    print(f"  Tired cycles:  {n_tired_cycles}")
    print(f"  Pain events:   {n_pain_events}")
    print(f"  BMU map:       {len(brain.bmu_to_word)} words")


# ── Phase 2: Dialogue training ────────────────────────────────────────────────

def run_dialogue_phase(brain: FastBrain, dialogues: list, n_epochs: int,
                       report_every: int = 5):
    """
    Train on the dialogue corpus. Same as train_fast.py but more epochs.
    Social patterns build on top of the grounded word meanings.
    """
    from train_fast import train_epoch, evaluate, REWARD_RESPONSE, REWARD_INPUT
    from train_fast import NOISE_INPUT, NOISE_SELF, PATIENCE, SAVE_EVERY

    EVAL_EVERY = report_every
    best_score   = -1.0
    patience_ctr = 0
    t_start = time.time()

    for epoch in range(1, n_epochs + 1):
        stats = train_epoch(brain, dialogues, epoch)
        elapsed = time.time() - t_start

        print(f"  Epoch {epoch:3d}/{n_epochs}  "
              f"exchanges={stats['exchanges']}  "
              f"learned={stats['words_learned']}  "
              f"reward={stats['mean_reward']:.3f}  "
              f"word_tp={brain.word_tp.n_transitions()} transitions  "
              f"[{elapsed:.0f}s]")

        if epoch % SAVE_EVERY == 0:
            brain.save(BRAIN_FILE)
            print(f"    → Saved to {BRAIN_FILE}")

        if epoch % EVAL_EVERY == 0:
            score = evaluate(brain, dialogues, n_sample=20)
            print(f"    → Word TP overlap: {score:.3f}")

            if score > best_score:
                best_score   = score
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"    → Early stopping (no improvement for {PATIENCE} evals)")
                    break


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 20k body steps + 10 dialogue epochs')
    parser.add_argument('--reset', action='store_true',
                        help='Start from scratch, ignore existing checkpoint')
    parser.add_argument('--body-only', action='store_true',
                        help='Run only the body phase')
    parser.add_argument('--lang-only', action='store_true',
                        help='Run only the dialogue phase')
    args = parser.parse_args()

    # ── Parameters ────────────────────────────────────────────────────
    if args.quick:
        N_BODY_STEPS    = 20_000
        N_DIALOGUE_EPOCHS = 10
        REPORT_EVERY    = 5_000
        print("Quick mode: 20k body steps + 10 dialogue epochs")
    else:
        N_BODY_STEPS      = 250_000    # ~75 full hunger cycles
        N_DIALOGUE_EPOCHS = 50         # deep language learning
        REPORT_EVERY      = 25_000
        print("Full training: 250k body steps + 50 dialogue epochs")

    from dialogue_corpus import DIALOGUES

    # ── Load or create brain ───────────────────────────────────────────
    if not args.reset and os.path.exists(BRAIN_FILE):
        print(f"Resuming from {BRAIN_FILE}...")
        brain = FastBrain.load(BRAIN_FILE)
    else:
        print("Creating new FastBrain...")
        brain = FastBrain()

    print(brain.status())
    print(f"  Vocabulary:  {len(VOCABULARY)} words")
    print(f"  Corpus:      {len(DIALOGUES)} dialogues")

    # Show active word sets
    print("\n  Drive→word mappings:")
    for key, entry in DRIVE_WORDS.items():
        print(f"    {key:15s} → {entry['words'][:5]} (base_reward={entry['base_reward']})")

    print()
    t_total = time.time()

    # ── Phase 1: Body experience ───────────────────────────────────────
    if not args.lang_only:
        print("=" * 60)
        print("PHASE 1: Body Experience")
        print("=" * 60)
        run_body_phase(brain, N_BODY_STEPS, report_every=REPORT_EVERY)

        brain.save(BRAIN_FILE)
        print(f"  Saved after body phase → {BRAIN_FILE}")
        print()

    # ── Phase 2: Dialogue training ─────────────────────────────────────
    if not args.body_only:
        print("=" * 60)
        print("PHASE 2: Language Training")
        print("=" * 60)
        run_dialogue_phase(brain, DIALOGUES, N_DIALOGUE_EPOCHS)

    # ── Final save + report ────────────────────────────────────────────
    brain.save(BRAIN_FILE)
    total_time = time.time() - t_total

    print()
    print("=" * 60)
    print(f"Training complete in {total_time:.1f}s")
    print(brain.status())
    print(f"Brain saved → {BRAIN_FILE}")
    print(f"Run: python live_fast.py")


if __name__ == '__main__':
    main()
