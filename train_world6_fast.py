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
    bfs_optimal_actions,
    ENV_VEGETATION_HIGH, ENV_MOISTURE_HIGH,
    ENV_COLD_THRESHOLD, ENV_WARM_THRESHOLD, ENV_WIND_HIGH,
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
    # ═══ SOCIAL REINFORCEMENT — approval/correction chains ══════════════════
    ("good",             "good happy calm"),
    ("yes",              "yes good calm"),
    ("great",            "good happy"),
    ("right",            "yes good calm"),
    ("no",               "bad stop sad"),
    ("wrong",            "bad stop"),
    ("bad",              "bad stop sad"),
    ("good yes",         "good happy calm"),
    ("no bad",           "bad stop sad"),
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

    # ═══ ACTION GRAMMAR — verb-object pairs grounded in actual events ════════
    # These teach Word TP the action→object word order from real World6 events.
    # NOT state parroting — these are agent actions that the brain actually did.
    ("i eat food",       "eat food full"),
    ("i push button",    "push button open door"),
    ("i open door",      "open door go"),
    ("i go",             "go move"),
    # "X is Y" — grounded conceptual relations
    ("food is",          "food is eat hungry good"),
    ("tree is",          "tree is plant calm green"),
    ("river is",         "river is water drink"),
    ("danger is",        "danger is afraid run"),
    ("wall is",          "wall is hurt stop bad"),
    ("button is",        "button is push open door"),
    ("door is",          "door is open go"),
    ("hunger is",        "hunger is need eat food"),
    ("pain is",          "pain is hurt bad stop"),

    # ═══ ENVIRONMENTAL — vegetation, water, air, temperature ════════════════
    ("plant",            "plant calm green"),
    ("tree",             "tree green calm"),
    ("grass",            "grass green calm"),
    ("green",            "plant green calm"),
    ("plant tree",       "calm green plant"),
    ("river",            "river water drink"),
    ("river water",      "water drink river"),
    ("rain",             "rain wet cold"),
    ("rain wet",         "wet cold rain"),
    ("wet",              "wet water river"),
    ("wind",             "wind air go"),
    ("air",              "air wind sky"),
    ("sky",              "sky air go"),
    ("wind air",         "go air wind"),
    ("cold",             "cold tired sleep"),
    ("cold wind",        "cold tired stop"),
    ("warm",             "warm sun calm"),
    ("sun",              "sun warm good"),
    ("warm sun",         "good warm calm"),
    ("what is plant",    "plant green calm"),
    ("what is tree",     "tree plant calm"),
    ("what is river",    "river water drink"),
    ("what is wind",     "wind air go"),
    ("what is cold",     "cold hurt stop"),
    ("what is warm",     "warm calm good"),
    ("what is sun",      "sun warm good"),
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


def feed_sequence(brain: FastBrain, words: list, rewards,
                  context_mfcc: np.ndarray | None = None,
                  blend: float = 0.30,
                  silence_steps: int = 1) -> list:
    """
    Feed words as an ordered causal chain with brief silence gaps.
    Each word gets its own reward so the brain learns:
      cause (low/no reward) → effect → resolution (reward)
    The silence gaps preserve temporal order in WordTP transitions.
    """
    bmus = []
    available = [w for w in words if w in VOCABULARY]
    if not available:
        return bmus
    # rewards can be a single float (apply to all) or a list per word
    if isinstance(rewards, (int, float)):
        reward_list = [float(rewards)] * len(available)
    else:
        reward_list = list(rewards)
    for word, reward in zip(available, reward_list):
        bmu = feed_word(brain, word, reward, context_mfcc=context_mfcc, blend=blend)
        if bmu is not None:
            bmus.append(bmu)
        for _ in range(silence_steps):
            brain.hear(SILENCE)
            brain.step()
    return bmus


def feed_words_if_appropriate(brain, info, grid_mfcc: np.ndarray):
    """
    Feed grounded words during World6 events as causal chains.
    Words are blended with the current grid MFCC so their BMUs land
    topographically near the experience that gives them meaning.

    Causal chain order matters: cause → intermediate → resolution.
    WordTP learns these as temporal sequences, not isolated words.
    """
    BLEND = 0.30
    bmus  = []

    # ── Wall collision: world acts on agent — "wall hurt stop" + agent recoils
    if info['wall_hit']:
        bmus += feed_sequence(brain,
            ['wall', 'hurt', 'stop'],
            [0.0,    0.0,    0.5],
            grid_mfcc, BLEND)

    # ── Food: agent eats — "i eat food full" (subject → verb → object → state)
    elif info['is_food']:
        bmus += feed_sequence(brain,
            ['i', 'eat', 'food', 'full'],
            [0.0,  0.4,   0.5,   0.7],
            grid_mfcc, BLEND)

    # ── Button: agent acts — "i push button open" (subject → verb → object → result)
    elif info['is_button']:
        bmus += feed_sequence(brain,
            ['i', 'push', 'button', 'open'],
            [0.0,  0.2,    0.2,     0.4],
            grid_mfcc, BLEND)

    # ── Danger: world threatens — "danger i am afraid run"
    elif info.get('is_danger'):
        bmus += feed_sequence(brain,
            ['danger', 'i', 'am', 'afraid', 'run'],
            [0.0,       0.0,  0.0,  0.0,     0.0],
            grid_mfcc, BLEND)

    # ── Door: agent opens — "i open door go" (subject → verb → object → action)
    elif info['is_door'] and info['door_open']:
        bmus += feed_sequence(brain,
            ['i', 'open', 'door', 'go'],
            [0.0,  0.2,    0.2,   0.3],
            grid_mfcc, BLEND)

    # ── Just left danger (safe = not danger, not wall, not special node)
    # Feed 'safe' occasionally when none of the threat conditions are active
    if (not info['wall_hit'] and not info.get('is_danger')
            and not info['is_food'] and not info['is_button']
            and rng.random() < 0.03):   # rare — safety is background, not an event
        if 'safe' in VOCABULARY:
            bmus.append(feed_word(brain, 'safe', 0.15, context_mfcc=grid_mfcc, blend=BLEND))

    # ── Hunger: agent state — "i am hungry want food" (subject → copula → state → goal)
    hunger = info['hunger']
    if hunger > 0.75 and not info['is_food']:
        bmus += feed_sequence(brain,
            ['i', 'am', 'hungry', 'want', 'food'],
            [0.0,  0.0,  0.0,     0.1,    0.4],
            grid_mfcc, BLEND)
    elif hunger < 0.25 and not info['is_food'] and not info['wall_hit']:
        bmus += feed_sequence(brain,
            ['i', 'am', 'full', 'calm'],
            [0.0,  0.0,  0.2,   0.25],
            grid_mfcc, BLEND)

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

    # ── Fatigue: agent state — "i am tired sleep" / "i am awake alive"
    fatigue = info.get('fatigue', 0.3)
    if fatigue > 0.70 and not info['is_food'] and not info['wall_hit']:
        bmus += feed_sequence(brain,
            ['i', 'am', 'tired', 'sleep'],
            [0.0,  0.0,  0.0,    0.35],
            grid_mfcc, BLEND)
    elif fatigue < 0.15 and not info['is_food'] and not info['wall_hit']:
        bmus += feed_sequence(brain,
            ['i', 'am', 'awake', 'alive'],
            [0.0,  0.0,  0.1,    0.20],
            grid_mfcc, BLEND)

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

    # ── Discovery → new → think → learn (novel event → cognition chain)
    if info.get('is_novel'):
        bmus += feed_sequence(brain,
            ['new', 'think', 'learn'],
            [0.1,   0.2,     0.4],
            grid_mfcc, BLEND)

    # ── Recall → remember → know → good (memory retrieval → success)
    if info.get('is_recall'):
        bmus += feed_sequence(brain,
            ['remember', 'know', 'good'],
            [0.1,         0.2,    0.5],
            grid_mfcc, BLEND)

    # ── Social zone → hello → like → calm (greeting chain)
    if info.get('is_social') and rng.random() < 0.15:
        bmus += feed_sequence(brain,
            ['hello', 'like', 'calm'],
            [0.1,      0.2,    0.3],
            grid_mfcc, BLEND)

    # ── Satiation → eat → full → calm (eating chain resolved)
    if info.get('is_satiated'):
        bmus += feed_sequence(brain,
            ['eat', 'full', 'calm'],
            [0.2,   0.4,    0.5],
            grid_mfcc, BLEND)

    # ── Frustration → wall → stop → bad (repeated failure chain)
    if info.get('is_frustrated'):
        bmus += feed_sequence(brain,
            ['wall', 'stop', 'bad'],
            [0.0,    0.0,    0.0],
            grid_mfcc, BLEND)

    # ── Anticipation → go → open → door (navigation chain en route to door)
    if info.get('is_anticipating') and rng.random() < 0.08:
        bmus += feed_sequence(brain,
            ['go', 'open', 'door'],
            [0.1,   0.2,    0.2],
            grid_mfcc, BLEND)

    # ── Rest event → tired → sleep → warm (fatigue resolution chain)
    if info.get('is_resting'):
        bmus += feed_sequence(brain,
            ['tired', 'sleep', 'warm'],
            [0.0,     0.2,     0.35],
            grid_mfcc, BLEND)

    # ── Boredom → new → go → move (restlessness → exploration chain)
    if info.get('is_bored') and rng.random() < 0.2:
        bmus += feed_sequence(brain,
            ['new', 'go', 'move'],
            [0.0,   0.05, 0.1],
            grid_mfcc, BLEND)

    # ── Prediction error → wrong → no → new (expectation violated → reorient)
    if info.get('is_surprise') and rng.random() < 0.4:
        bmus += feed_sequence(brain,
            ['wrong', 'no',  'new'],
            [0.0,     0.0,   0.05],
            grid_mfcc, BLEND)

    # ── Environmental sensory grounding ───────────────────────────────────────
    # Feed environmental word chains only at low frequency (3% per step) so
    # they accumulate over 500k steps without drowning out action events.
    env = info.get('env', {})
    if not info.get('wall_hit') and not info.get('is_food'):
        veg   = env.get('vegetation',  0.0)
        moist = env.get('moisture',    0.0)
        temp  = env.get('temperature', 0.5)
        wind  = env.get('wind',        0.0)
        tod   = info.get('time_of_day', 0.5)

        # Vegetation zone — plant / tree / grass / green
        if veg > ENV_VEGETATION_HIGH and rng.random() < 0.12:
            bmus += feed_sequence(brain,
                ['plant', 'tree', 'green'],
                [0.05,     0.05,   0.05],
                grid_mfcc, BLEND)

        # Lush vegetation with moisture — grass
        if veg > 0.5 and moist > 0.4 and rng.random() < 0.02:
            bmus += feed_sequence(brain,
                ['grass', 'green', 'calm'],
                [0.05,     0.05,   0.08],
                grid_mfcc, BLEND)

        # Water/moisture zone — river / rain / wet
        if moist > ENV_MOISTURE_HIGH and rng.random() < 0.12:
            bmus += feed_sequence(brain,
                ['river', 'water', 'wet'],
                [0.05,     0.1,    0.05],
                grid_mfcc, BLEND)

        # Rain: high moisture + high wind
        if moist > 0.6 and wind > 0.5 and rng.random() < 0.02:
            bmus += feed_sequence(brain,
                ['rain', 'wet', 'cold'],
                [0.0,    0.0,  -0.02],
                grid_mfcc, BLEND)

        # Open air / wind zone — air / wind / sky
        if wind > ENV_WIND_HIGH and rng.random() < 0.12:
            bmus += feed_sequence(brain,
                ['air', 'wind', 'go'],
                [0.05,   0.05,  0.05],
                grid_mfcc, BLEND)

        # Cold: low temperature (especially night)
        is_night = tod > 0.75 or tod < 0.25
        if temp < ENV_COLD_THRESHOLD and rng.random() < 0.12:
            bmus += feed_sequence(brain,
                ['cold', 'tired', 'sleep'],
                [-0.02,   0.0,    0.05],
                grid_mfcc, BLEND)

        # Warm sunny day — warm / sun / good
        if temp > ENV_WARM_THRESHOLD and not is_night and rng.random() < 0.12:
            bmus += feed_sequence(brain,
                ['warm', 'sun', 'good'],
                [0.08,   0.05,  0.08],
                grid_mfcc, BLEND)

        # Sky: open areas with low vegetation + daytime
        if wind > 0.4 and veg < 0.3 and not is_night and rng.random() < 0.02:
            bmus += feed_sequence(brain,
                ['sky', 'air', 'free'],
                [0.05,  0.05,  0.06],
                grid_mfcc, BLEND)

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

    # Environmental sensory channels v[13–16]
    env = info.get('env', {})
    time_of_day = info.get('time_of_day', 0.5)
    # Temperature modulated by day/night cycle
    base_temp = env.get('temperature', 0.5)
    day_factor = np.sin(time_of_day * 2 * np.pi)   # warm at noon, cold at night
    v[13] = float(env.get('vegetation',  0.0))
    v[14] = float(env.get('moisture',    0.0))
    v[15] = float(np.clip(base_temp + day_factor * 0.2, 0.0, 1.0))
    v[16] = float(env.get('wind',        0.0))

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
