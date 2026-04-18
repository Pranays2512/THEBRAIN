#!/usr/bin/env python3
"""
train_world6_fast.py — Grounded training for FastBrain
=======================================================

PHASE 1 — EMBODIED EXPERIENCE (500k steps)
-------------------------------------------
The brain navigates World6. Every event (food, wall, button, door, hunger)
is encoded as a distinct MFCC and the corresponding grounded words are fed
with that MFCC blended in (30%). This places word BMUs topographically
near their referent experience clusters on the SOM.

PHASE 2 — GROUNDED EXPRESSION (30 epochs)
------------------------------------------
30 simple exchanges using only grounded words. Teaches the WordTP basic
expression patterns: hungry → want food, hurt → stop pain, tired → need sleep.
No pre-written corpus. Every dialogue pair is derived from a real drive state.

Usage:
    python train_world6_fast.py           # full training (~15 min)
    python train_world6_fast.py --quick   # 10k steps + 5 epochs (test)
    python train_world6_fast.py --lang-only  # skip Phase 1
"""

import sys
import os
import time
import argparse
import numpy as np

sys.dont_write_bytecode = True

from brain_fast import FastBrain, SILENCE, N_MFCC
from vocab_core import VOCABULARY
from world6 import (
    World6, DOOR_NODES, BUTTON_NODES,
    bfs_optimal_actions
)

BRAIN_FILE = 'brain_fast.pkl'
rng = np.random.default_rng(42)

# ── Grounded dialogue pairs ────────────────────────────────────────────────────
# These are NOT a corpus — they are simple expression patterns derived from
# drive states. Every pair reflects something the brain can actually experience.
# "hungry → want food" is grounded because the brain WAS hungry in World6.

GROUNDED_DIALOGUES = [
    # ═══ FOOD / HUNGER — paraphrases + slot variants ═══════════════════════
    ("hungry",           "want food eat"),
    ("hungry",           "need eat now"),
    ("i am hungry",      "hungry need food"),
    ("you hungry",       "yes hungry want food"),
    ("feel hungry",      "hungry want eat"),
    ("need food now",    "hungry eat food"),
    ("hungry now",       "eat food want"),
    ("hungry need",      "want food eat now"),
    ("hungry want",      "hungry need eat now"),
    ("food",             "eat food want"),
    ("food here",        "eat food"),
    ("want food",        "eat food now"),
    ("need food",        "eat food here"),
    ("eat",              "eat food"),
    ("eat food",         "full calm"),
    ("i want eat",       "eat food now"),
    ("go food",          "go eat food"),
    ("need go food",     "go eat food"),
    ("drink",            "water drink"),
    ("water",            "drink water"),
    ("need water",       "drink water now"),
    ("full",             "full calm"),
    ("feel full",        "full calm stop eat"),
    ("not hungry",       "full calm"),
    ("hungry is",        "hungry is need eat food"),
    ("food is",          "food is eat hungry"),
    # ═══ PAIN / WALL — paraphrases + cause/effect ══════════════════════════
    ("hurt",             "stop pain"),
    ("hurt",             "hurt stop bad"),
    ("pain",             "pain bad stop"),
    ("i am hurt",        "stop pain bad"),
    ("feel pain",        "pain bad hurt"),
    ("hurt bad",         "hurt stop bad"),
    ("wall",             "hurt wall stop"),
    ("wall",             "stop pain bad"),
    ("wall hurt",        "hurt stop bad"),
    ("hurt wall",        "stop pain bad"),
    ("stop",             "stop no"),
    ("bad",              "stop bad"),
    ("feel bad",         "hurt stop bad"),
    ("hurt wall pain",   "wall hurt pain"),
    ("hurt is",          "hurt is pain bad"),
    ("pain is",          "pain is hurt bad stop"),
    ("wall is",          "wall is hard hurt stop"),
    ("no",               "stop no bad"),
    ("help",             "help stop hurt"),
    ("need help",        "help stop hurt"),
    ("i need help",      "help stop pain"),
    ("sad",              "sad stop bad"),
    # ═══ REST / TIREDNESS — paraphrases + circadian ═════════════════════════
    ("tired",            "need sleep"),
    ("tired",            "sleep calm"),
    ("i am tired",       "need sleep calm"),
    ("feel tired",       "tired need sleep"),
    ("tired now",        "tired sleep now"),
    ("tired need",       "tired need sleep now"),
    ("tired feel",       "tired need sleep"),
    ("tired is",         "tired is need sleep"),
    ("sleep",            "sleep calm"),
    ("sleep now",        "calm sleep"),
    ("need sleep",       "sleep calm"),
    ("i want sleep",     "sleep calm"),
    ("calm",             "calm sleep"),
    ("feel calm",        "calm sleep"),
    ("warm",             "warm calm sleep"),
    ("soft",             "calm sleep soft"),
    ("awake",            "awake alive"),
    ("feel awake",       "awake alive"),
    ("alive",            "alive awake"),
    ("feel alive",       "alive awake"),
    ("not tired",        "awake alive calm"),
    ("sleep is",         "sleep is calm tired"),
    # ═══ ACTION / NAVIGATION — compositional sequences ═══════════════════════
    ("go",               "go move"),
    ("move",             "move go"),
    ("come",             "come here"),
    ("open",             "open door go"),
    ("open door",        "go open door"),
    ("go door",          "go door open"),
    ("push",             "push open go"),
    ("push button",      "push button open"),
    ("button",           "push button open"),
    ("open button",      "push button open door"),
    ("door",             "open door go"),
    ("door open",        "go open door"),
    ("push button open door", "push button open door go"),
    ("button open door", "push button open door"),
    ("go open",          "go open door move"),
    ("go door",          "go open door"),
    ("move go",          "go move open"),
    ("here",             "here go"),
    ("go here",          "go move here"),
    ("come here",        "come here go"),
    ("strong",           "strong go move"),
    ("hard",             "hard go push"),
    ("new",              "new go move"),
    ("button is",        "button is push open door"),
    ("door is",          "door is open go"),
    # ═══ DANGER / SAFETY — cause/effect + resolution ═════════════════════════
    ("afraid",           "afraid run careful"),
    ("afraid",           "danger afraid run"),
    ("i am afraid",      "afraid careful run"),
    ("feel afraid",      "afraid careful run"),
    ("afraid danger",    "danger afraid run"),
    ("afraid is",        "afraid is danger run"),
    ("danger",           "careful run afraid"),
    ("danger",           "danger stop careful"),
    ("danger here",      "run careful stop"),
    ("danger now",       "run careful stop"),
    ("careful",          "careful stop"),
    ("careful now",      "careful stop run"),
    ("run",              "run careful"),
    ("run now",          "run go careful"),
    ("run safe",         "run careful stop"),
    ("safe",             "safe calm"),
    ("feel safe",        "safe calm"),
    ("not safe",         "danger afraid run"),
    ("safe is",          "no danger run careful"),
    ("safe now",         "safe calm awake"),
    ("danger stop",      "safe calm"),
    # ═══ SOCIAL — multi-turn greetings + identity ════════════════════════════
    ("hi",               "hi hello"),
    ("hi",               "hello hi"),
    ("hello",            "hello hi"),
    ("hello",            "hi hello"),
    ("hi hello",         "hello hi"),
    ("hello you",        "hi hello"),
    ("bye",              "bye calm"),
    ("bye now",          "bye calm"),
    ("yes",              "yes"),
    ("sorry",            "sorry calm"),
    ("i am sorry",       "sorry calm stop"),
    ("know",             "yes know"),
    ("you",              "yes hi"),
    ("you fastbrain",    "hi fastbrain"),
    ("you name",         "fastbrain"),
    ("name you",         "fastbrain name"),
    ("name",             "fastbrain name"),
    ("name is",          "name fastbrain"),
    # ═══ MULTI-TURN PAIRS — sequential context chains ════════════════════════
    ("hi",               "hi hello"),
    ("hello hi",         "hi hello yes"),
    ("food yes",         "eat food"),
    ("hungry yes",       "eat food now"),
    ("danger afraid",    "run careful stop"),
    ("run stop",         "stop careful"),
    ("hurt stop",        "stop pain bad"),
    ("sleep calm",       "calm sleep"),
    ("open door",        "go open door move"),
    ("button door",      "push button open door"),
    ("food hungry",      "eat food want"),
    ("eat food",         "full calm"),
    ("pain bad",         "stop hurt"),
    ("careful run",      "run careful"),
    ("awake alive",      "alive awake calm"),
    ("hello yes",        "yes hi hello"),
    ("safe calm",        "calm safe awake"),
    ("tired sleep",      "sleep calm"),
    ("eat full",         "full calm stop"),
]


# ── Word feeding ───────────────────────────────────────────────────────────────

def feed_word(brain: FastBrain, word: str, reward: float,
              noise_std: float = 0.08,
              context_mfcc: np.ndarray | None = None,
              blend: float = 0.0):
    """
    Feed a word through the brain's acoustic pipeline.

    context_mfcc + blend: tints the word's MFCC with the current
    experience, placing its BMU near the referent cluster on the SOM.
    """
    if word not in VOCABULARY:
        return None
    mean_vec, n_frames = VOCABULARY[word]
    last_bmu = None
    for i in range(n_frames):
        frame = mean_vec + rng.normal(0, noise_std, N_MFCC)
        if context_mfcc is not None and blend > 0:
            frame = (1.0 - blend) * frame + blend * context_mfcc
        brain.hear(frame.astype(np.float32))
        r = reward if i == n_frames - 1 else 0.0
        out = brain.step(reward=r)
        last_bmu = out['bmu']
    brain.hear(SILENCE)
    brain.step()

    clean_bmu = brain.som.find_bmu(mean_vec.astype(np.float32))
    if word not in brain.word_to_bmu:
        brain.word_to_bmu[word] = clean_bmu
        brain.bmu_to_word[clean_bmu] = word

    return last_bmu


def feed_words_if_appropriate(brain, info, grid_mfcc: np.ndarray):
    """
    Feed grounded words during World6 events.
    Words are blended with the current grid MFCC so their BMUs land
    topographically near the experience that gives them meaning.
    """
    BLEND = 0.30   # 70% word sound, 30% experience tint
    bmus  = []

    # ── Wall collision → hurt / pain / stop / bad
    if info['wall_hit']:
        words = ['wall', 'hurt', 'stop', 'pain', 'bad']
        for w in rng.choice([x for x in words if x in VOCABULARY], size=2, replace=False):
            bmus.append(feed_word(brain, w, 0.5, context_mfcc=grid_mfcc, blend=BLEND))

    # ── Food → eat / food / good / happy
    elif info['is_food']:
        words = ['food', 'eat', 'good', 'happy']
        for w in rng.choice([x for x in words if x in VOCABULARY], size=2, replace=False):
            bmus.append(feed_word(brain, w, 0.7, context_mfcc=grid_mfcc, blend=BLEND))

    # ── Button → push / button / open
    elif info['is_button']:
        words = ['push', 'button', 'open', 'new']
        available = [x for x in words if x in VOCABULARY]
        if available:
            for w in rng.choice(available, size=min(2, len(available)), replace=False):
                bmus.append(feed_word(brain, w, 0.4, context_mfcc=grid_mfcc, blend=BLEND))

    # ── Danger zone → afraid / careful / danger / run
    elif info.get('is_danger'):
        words = ['afraid', 'careful', 'danger', 'run']
        available = [x for x in words if x in VOCABULARY]
        if available:
            for w in rng.choice(available, size=min(2, len(available)), replace=False):
                bmus.append(feed_word(brain, w, 0.0, context_mfcc=grid_mfcc, blend=BLEND))

    # ── Door → door / open / go
    elif info['is_door'] and info['door_open']:
        words = ['door', 'open', 'go']
        available = [x for x in words if x in VOCABULARY]
        if available:
            bmus.append(feed_word(brain, rng.choice(available), 0.3,
                                  context_mfcc=grid_mfcc, blend=BLEND))

    # ── Just left danger (safe = not danger, not wall, not special node)
    # Feed 'safe' occasionally when none of the threat conditions are active
    if (not info['wall_hit'] and not info.get('is_danger')
            and not info['is_food'] and not info['is_button']
            and rng.random() < 0.03):   # rare — safety is background, not an event
        if 'safe' in VOCABULARY:
            bmus.append(feed_word(brain, 'safe', 0.15, context_mfcc=grid_mfcc, blend=BLEND))

    # ── Hunger states → hungry / need / food / want (or full / good / calm)
    hunger = info['hunger']
    if hunger > 0.75 and not info['is_food']:
        words = ['hungry', 'need', 'food', 'want']
        available = [x for x in words if x in VOCABULARY]
        if available:
            bmus.append(feed_word(brain, rng.choice(available), 0.4,
                                  context_mfcc=grid_mfcc, blend=BLEND))
    elif hunger < 0.25 and not info['is_food'] and not info['wall_hit']:
        words = ['full', 'good', 'happy', 'calm']
        available = [x for x in words if x in VOCABULARY]
        if available:
            bmus.append(feed_word(brain, rng.choice(available), 0.25,
                                  context_mfcc=grid_mfcc, blend=BLEND))

    # ── Texture (full 3-band: soft / neutral / rough)
    if not info['wall_hit'] and not info['is_food'] and not info['is_button']:
        tex = info['texture']
        if tex < 0.33:
            words = ['soft', 'calm', 'good']
            available = [x for x in words if x in VOCABULARY]
            if available:
                bmus.append(feed_word(brain, rng.choice(available), 0.2,
                                      context_mfcc=grid_mfcc, blend=BLEND))
        elif tex < 0.67:
            words = ['warm', 'feel', 'here']
            available = [x for x in words if x in VOCABULARY]
            if available:
                bmus.append(feed_word(brain, rng.choice(available), 0.15,
                                      context_mfcc=grid_mfcc, blend=BLEND))
        else:
            words = ['hard', 'strong', 'stop']
            available = [x for x in words if x in VOCABULARY]
            if available:
                bmus.append(feed_word(brain, rng.choice(available), 0.2,
                                      context_mfcc=grid_mfcc, blend=BLEND))

    # ── Fatigue states → tired / sleep / need (or awake / strong / alive)
    fatigue = info.get('fatigue', 0.3)
    if fatigue > 0.70 and not info['is_food'] and not info['wall_hit']:
        words = ['tired', 'sleep', 'need', 'soft']
        available = [x for x in words if x in VOCABULARY]
        if available:
            bmus.append(feed_word(brain, rng.choice(available), 0.35,
                                  context_mfcc=grid_mfcc, blend=BLEND))
    elif fatigue < 0.15 and not info['is_food'] and not info['wall_hit']:
        words = ['awake', 'alive', 'strong', 'good']
        available = [x for x in words if x in VOCABULARY]
        if available:
            bmus.append(feed_word(brain, rng.choice(available), 0.20,
                                  context_mfcc=grid_mfcc, blend=BLEND))

    # ── Day/Night cycle → awake/tired/sleep (grounded in circadian rhythm) ─────
    if 'is_night' in info:
        if info['is_night'] and fatigue > 0.50 and rng.random() < 0.04:
            # Nighttime + tired → feed sleep/tired
            words = ['sleep', 'tired', 'calm']
            available = [x for x in words if x in VOCABULARY]
            if available:
                bmus.append(feed_word(brain, rng.choice(available), 0.3,
                                      context_mfcc=grid_mfcc, blend=BLEND))
        elif not info['is_night'] and fatigue < 0.30 and rng.random() < 0.04:
            # Daytime + fresh → feed awake/alive
            words = ['awake', 'alive', 'feel']
            available = [x for x in words if x in VOCABULARY]
            if available:
                bmus.append(feed_word(brain, rng.choice(available), 0.25,
                                      context_mfcc=grid_mfcc, blend=BLEND))


    return [b for b in bmus if b is not None]


def world6_to_mfcc(info) -> np.ndarray:
    """
    Encode the current World6 state as a 13-dim MFCC vector.
    Each event type has a distinct, non-overlapping coefficient profile
    so they cluster in different SOM regions without aliasing.

    Coefficient roles:
      v[0]  — energy (always 3.0)
      v[1]  — valence (positive=reward, negative=threat)
      v[2]  — arousal (high=alert/push, low=tired/rest)
      v[3]  — texture low-band
      v[4]  — texture high-band
      v[5]  — fatigue level (high = tired, low = energised)
      v[6]  — hunger/pain signal
      v[7]  — food reward signal
      v[8]  — novelty/button signal
      v[9]  — action/push signal
      v[10] — stress/urgency
      v[11] — satiation/rest signal
      v[12] — energy reserve (high fatigue → low energy)
    """
    v = np.zeros(N_MFCC, dtype=np.float32)
    v[0] = 3.0  # energy (always present)

    if info['wall_hit']:
        # Pain profile: strong negative valence, high arousal
        v[1] = -0.9; v[2] = 0.9; v[6] = 1.0; v[10] = 1.0
        return v

    if info['is_food']:
        # Reward profile: positive valence, satiation, energy restored
        v[1] = 0.8; v[2] = -0.5; v[7] = 1.0; v[11] = 1.0
        return v

    if info['is_button']:
        # Novelty/push profile: neutral valence, strong activity
        v[1] = 0.0; v[2] = 0.8; v[8] = 1.0; v[9] = 1.0
        return v

    if info.get('is_danger'):
        # Danger profile: creeping dread — distinct from wall (sudden shock)
        # Low arousal (frozen), negative valence, low certainty, strong embodiment
        v[1] = -0.6; v[2] = -0.5; v[5] = -0.6; v[9] = -0.4; v[12] = 0.9
        return v

    # General navigation: modulated by hunger, texture, and fatigue
    hunger  = info['hunger']
    tex     = info['texture']
    fatigue = info.get('fatigue', 0.3)

    # Texture: full 3-band encoding
    if tex < 0.33:       # soft ground
        v[3] = -0.6; v[4] = tex
    elif tex < 0.67:     # neutral ground
        v[3] = 0.0;  v[4] = tex
    else:                # rough/hard ground
        v[3] = 0.6;  v[4] = tex

    # Hunger modulation (occupies same coefficients as before)
    if hunger > 0.6:
        v[1] = -0.4; v[2] = 0.6; v[6] = hunger; v[10] = 1.0
    elif hunger < 0.2:
        v[1] = 0.4; v[2] = -0.2; v[6] = 0.3

    # Fatigue modulation (v[5] and v[12] — distinct from hunger/pain)
    v[5]  = fatigue * 1.2 - 0.6   # high fatigue → positive, low → negative
    v[12] = -fatigue * 0.8         # energy reserve (depletes with fatigue)
    if fatigue > 0.65:
        v[2] -= 0.5   # tiredness suppresses arousal
        v[11] = 0.6   # rest-seeking signal
    elif fatigue < 0.20:
        v[2] += 0.3   # fresh/energised raises arousal

    return v


# ── Phase 1: World6 embodiment ─────────────────────────────────────────────────

def run_world6_phase(brain: FastBrain, n_steps: int = 500_000):
    print(f"\n{'='*60}")
    print(f"PHASE 1: World 6 Embodiment ({n_steps:,} steps)")
    print(f"{'='*60}")

    world = World6(seed=42)
    world.reset()
    opt_policy = bfs_optimal_actions(world._active_food, BUTTON_NODES, DOOR_NODES)

    t0 = time.time()
    events = {'wall': 0, 'food': 0, 'button': 0, 'door': 0, 'danger': 0}

    for step in range(n_steps):
        # Semi-guided: 40% optimal (ensures diverse events), 60% random
        if rng.random() < 0.4:
            opt_a = opt_policy.get(world.current_node, -1)
            action = opt_a if opt_a >= 0 else rng.integers(0, 4)
        else:
            action = rng.integers(0, 4)

        _, _, reward, info = world.step(int(action))

        if info['wall_hit']:          events['wall']   += 1
        if info['is_food']:           events['food']   += 1
        if info['is_button']:         events['button'] += 1
        if info['is_door']:           events['door']   += 1
        if info.get('is_danger'):     events['danger'] += 1

        if info.get('food_moved', False):
            opt_policy = bfs_optimal_actions(world._active_food, BUTTON_NODES, DOOR_NODES)

        # 1. Brain experiences the world state
        grid_mfcc = world6_to_mfcc(info)
        brain.hear(grid_mfcc)
        brain.step(reward=max(reward, 0))

        # 2. Feed grounded words blended with this experience
        feed_words_if_appropriate(brain, info, grid_mfcc)

        if (step + 1) % 50_000 == 0:
            elapsed = time.time() - t0
            hunger  = info['hunger']
            fatigue = info['fatigue']
            print(f"  Step {step+1:7,}/{n_steps:,} | "
                  f"Food: {events['food']:5,} | Walls: {events['wall']:5,} | "
                  f"Danger: {events['danger']:5,} | Buttons: {events['button']:4,} | "
                  f"Hunger: {hunger:.2f} | Fatigue: {fatigue:.2f} | "
                  f"[{elapsed:.0f}s]")

    mapped = len(brain.bmu_to_word)
    print(f"\n  Done. {mapped} words grounded in SOM.")


# ── Phase 2: Grounded expression ───────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return [w for w in text.lower().split() if w in VOCABULARY]


def feed_sequence(brain: FastBrain, words: list[str], reward: float) -> list[int]:
    bmus = []
    for word in words:
        bmu = feed_word(brain, word, reward)
        if bmu is not None:
            bmus.append(bmu)
    return bmus


def train_expression_epoch(brain: FastBrain, dialogues: list) -> dict:
    """
    Train one epoch of grounded expression.
    Each pair: hear input (low reward) → separator → say response (high reward).
    Only uses grounded vocabulary — no abstract corpus.
    """
    n_exchanges = 0
    words_learned = 0
    order = rng.permutation(len(dialogues))

    for idx in order:
        you_text, brain_text = dialogues[idx]
        you_words   = tokenize(you_text)
        brain_words = tokenize(brain_text)
        if not you_words or not brain_words:
            continue

        # Teach WordTP the input→response pattern
        for w in you_words:
            brain.word_tp.observe(w)
        brain.hear_word_separator()
        for w in brain_words:
            brain.word_tp.observe(w)
        brain.hear_word_end()

        # Feed through acoustic pipeline
        feed_sequence(brain, you_words, 0.1)
        for _ in range(4):
            brain.hear(SILENCE)
            brain.step()

        bmus = feed_sequence(brain, brain_words, 0.8)
        words_learned += len(brain_words)

        if bmus:
            brain.record_turn(bmus, 0.8)

        brain.hear(SILENCE)
        brain.step()
        n_exchanges += 1

    brain.dream(n_sequences=min(30, len(brain._dream_buffer)))
    return {'exchanges': n_exchanges, 'words_learned': words_learned}


def run_expression_phase(brain: FastBrain, epochs: int = 30):
    print(f"\n{'='*60}")
    print(f"PHASE 2: Grounded Expression ({epochs} epochs, "
          f"{len(GROUNDED_DIALOGUES)} pairs)")
    print(f"{'='*60}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        stats = train_expression_epoch(brain, GROUNDED_DIALOGUES)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{epochs}  "
              f"exchanges={stats['exchanges']}  "
              f"word_tp={brain.word_tp.n_transitions()} trans  "
              f"[{elapsed:.0f}s]")

    print(f"\n  Done. Word TP: {brain.word_tp.n_words()} words, "
          f"{brain.word_tp.n_transitions()} transitions.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',     action='store_true', help='10k steps + 5 epochs')
    parser.add_argument('--lang-only', action='store_true', help='Skip Phase 1')
    args = parser.parse_args()

    if args.lang_only and os.path.exists(BRAIN_FILE):
        print(f"Loading {BRAIN_FILE} for expression update...")
        brain = FastBrain.load(BRAIN_FILE)
    else:
        print("Creating fresh FastBrain...")
        brain = FastBrain()

    n_steps  = 10_000 if args.quick else 500_000
    n_epochs = 5      if args.quick else 30

    if not args.lang_only:
        run_world6_phase(brain, n_steps=n_steps)

    run_expression_phase(brain, epochs=n_epochs)

    brain.save(BRAIN_FILE)
    print("\nTraining complete.")
    print(brain.status())
    print(f"\n  Vocabulary: {len(VOCABULARY)} grounded words")
    print("  Run: python live_fast.py")


if __name__ == '__main__':
    main()
