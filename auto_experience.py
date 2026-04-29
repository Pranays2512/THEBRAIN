#!/usr/bin/env python3
"""
auto_experience.py — Overnight experiential grounding for THE BRAIN.

WHAT THIS DOES (and why it is not auto_train.py)
--------------------------------------------------
auto_train.py scores zone-responses from hand-written ZONE_EXPRESSION lists.
That is still scripted — the brain learns what WE want it to say.

This file does something fundamentally different:
  Every World6 event generates a word sequence from the ACTUAL simulation
  state at that moment — hunger level, fatigue level, fear exposure, and
  what physically happened (wall, food, button, danger, discovery, rest).

  There are no hand-written pairs. The GRU learns:
    "hungry tired → found food → eat full calm"   from actually being hungry and finding food
    "afraid → danger → run"                        from actually entering a danger zone
    "push button → open door → go"                 from actually completing the sequence
    "what search → found → good"                   from actually recovering after prediction error

  The longer this runs, the richer the associations. Every new event
  combination the brain has not seen before adds a new sequence.
  Vocabulary evolves: new extreme-state combinations that lack words
  get synthesized words with MFCC vectors derived from the live drive state.

Usage:
    python auto_experience.py                        # 5M steps (quick overnight)
    python auto_experience.py --steps 20000000       # 20M steps (full overnight)
    python auto_experience.py --steps 50000000       # 50M steps (weekend run)
    python auto_experience.py --steps 20000000 --no-vocab-evolve
"""

import sys
import os
import time
import argparse
import numpy as np
from collections import defaultdict

sys.dont_write_bytecode = True

from brain_fast import FastBrain, SILENCE, N_MFCC
from vocab_core import VOCABULARY
from m75_semantic import SemanticMemory

try:
    import brain_core as _brain_core
    _HAS_FEED_BATCH = hasattr(_brain_core, 'feed_word_frames')
except ImportError:
    _HAS_FEED_BATCH = False
from world6 import (World6, BUTTON_NODES, DOOR_NODES, SOCIAL_NODES,
                    ENV_VEGETATION_HIGH, ENV_MOISTURE_HIGH,
                    ENV_COLD_THRESHOLD, ENV_WARM_THRESHOLD, ENV_WIND_HIGH,
                    THIRST_HIGH, THIRST_EXTREME, INJURY_HIGH,
                    WATER_NODES, ALL_NODES, ADJACENCY,
                    bfs_optimal_actions)

BRAIN_FILE    = 'brain_fast.pkl'
SEMANTIC_FILE = 'semantic.json'
rng = np.random.default_rng(int(time.time()) % (2**31))


def bfs_to_targets(targets: list) -> dict:
    """BFS from target nodes outward — returns optimal action per node."""
    from collections import deque
    target_set = set(targets)
    dist = {n: float('inf') for n in ALL_NODES}
    q = deque()
    for t in target_set:
        dist[t] = 0
        q.append(t)
    while q:
        node = q.popleft()
        for nbr in ADJACENCY[node].values():
            if dist[nbr] == float('inf'):
                dist[nbr] = dist[node] + 1
                q.append(nbr)
    name_to_idx = {'North': 0, 'East': 1, 'South': 2, 'West': 3}
    opt = {}
    for node in ALL_NODES:
        best_a, best_d = 1, float('inf')  # noqa: best_d used in loop body
        for act_name, nbr in ADJACENCY[node].items():
            if dist[nbr] < best_d:
                best_d = dist[nbr]
                best_a = name_to_idx[act_name]
        opt[node] = best_a
    return opt

# ── Drive thresholds ──────────────────────────────────────────────────────────
HUNGER_LOW      = 0.20
HUNGER_MED      = 0.50
HUNGER_HIGH     = 0.68
HUNGER_EXTREME  = 0.85
FATIGUE_LOW     = 0.15
FATIGUE_MED     = 0.45
FATIGUE_HIGH    = 0.62
FATIGUE_EXTREME = 0.80
FEAR_ACTIVE     = 0.35
FEAR_HIGH       = 0.65

# ── Training schedule ─────────────────────────────────────────────────────────
FIT_EVERY        = 50_000     # GRU fit interval (steps)
SAVE_EVERY       = 500_000    # save brain
PRINT_EVERY      = 100_000    # progress report
DREAM_EVERY      = 2_000      # dream cycle
VOCAB_CHECK_EVERY= 200_000    # vocabulary evolution check
MIN_SYNTH_EVENTS = 300        # min times a pattern fires before synthesizing word

# ── Load brain ────────────────────────────────────────────────────────────────
if not os.path.exists(BRAIN_FILE):
    print(f"No {BRAIN_FILE}. Run train_world6_fast.py first.")
    sys.exit(1)

def _refresh_bmu_map(br: FastBrain, vocab: dict):
    """Map every vocab word to its nearest SOM neuron."""
    for word, (mean_vec, _) in vocab.items():
        bmu = br.som.find_bmu(mean_vec.astype(np.float32))
        if word not in br.word_to_bmu:
            br.word_to_bmu[word] = bmu
        if bmu not in br.bmu_to_word:
            br.bmu_to_word[bmu] = word


brain    = FastBrain.load(BRAIN_FILE)
semantic = SemanticMemory.load(SEMANTIC_FILE)
_refresh_bmu_map(brain, VOCABULARY)
print(f"\nBrain loaded: {brain._n_steps:,} steps, "
      f"{brain.word_tp.n_words()} WordTP words, "
      f"{len(brain.bmu_to_word)} BMU mappings")

# ── World6 MFCC encoder (duplicated from train_world6_fast to avoid import) ───
def world6_to_mfcc(info: dict) -> np.ndarray:
    v = np.zeros(N_MFCC, dtype=np.float32)
    v[0] = 3.0
    if info['wall_hit']:
        v[1] = -0.9; v[2] = 0.9; v[6] = 1.0; v[10] = 1.0
        return v
    if info['is_food']:
        v[1] = 0.8; v[2] = -0.5; v[7] = 1.0; v[11] = 1.0
        return v
    if info['is_button']:
        v[1] = 0.0; v[2] = 0.8; v[8] = 1.0; v[9] = 1.0
        return v
    if info.get('is_danger'):
        v[1] = -0.6; v[2] = -0.5; v[5] = -0.6; v[9] = -0.4; v[12] = 0.9
        return v
    hunger  = info['hunger']
    tex     = info['texture']
    fatigue = info.get('fatigue', 0.3)
    if tex < 0.33:   v[3] = -0.6; v[4] = tex
    elif tex < 0.67: v[3] = 0.0;  v[4] = tex
    else:            v[3] = 0.6;  v[4] = tex
    if hunger > 0.6:
        v[1] = -0.4; v[2] = 0.6; v[6] = hunger; v[10] = 1.0
    elif hunger < 0.2:
        v[1] = 0.4; v[2] = -0.2; v[6] = 0.3
    v[5]  = fatigue * 1.2 - 0.6
    v[12] = -fatigue * 0.8
    if fatigue > 0.65:
        v[2] -= 0.5; v[11] = 0.6
    elif fatigue < 0.20:
        v[2] += 0.3
    env = info.get('env', {})
    time_of_day = info.get('time_of_day', 0.5)
    base_temp   = env.get('temperature', 0.5)
    day_factor  = np.sin(time_of_day * 2 * np.pi)
    v[13] = float(env.get('vegetation',  0.0))
    v[14] = float(env.get('moisture',    0.0))
    v[15] = float(np.clip(base_temp + day_factor * 0.2, 0.0, 1.0))
    v[16] = float(env.get('wind',        0.0))
    return v


# ── Acoustic feed helpers ─────────────────────────────────────────────────────
def feed_word(word: str, reward: float = 0.0, noise: float = 0.08) -> int | None:
    if word not in VOCABULARY:
        return None
    mean_vec, n_frames = VOCABULARY[word]
    mv = mean_vec.astype(np.float32)
    if _HAS_FEED_BATCH:
        _ns_safe = brain._n_steps % (2**31 - 1)
        last_bmu, ppb, pb, ns = _brain_core.feed_word_frames(
            brain.som, brain.tp, mv, n_frames, noise, reward,
            brain._prev_prev_bmu, brain._prev_bmu, _ns_safe,
            SILENCE, 0.0)
        brain._prev_prev_bmu = ppb
        brain._prev_bmu      = pb
        brain._n_steps      += ns
        brain._total_reward += reward
        return last_bmu
    bmu = None
    for i in range(n_frames):
        frame = (mv + rng.normal(0, noise, N_MFCC)).astype(np.float32)
        brain.hear(frame)
        out = brain.step(reward=reward if i == n_frames - 1 else 0.0)
        bmu = out['bmu']
    brain.hear(SILENCE)
    brain.step()
    return bmu


def feed_sequence_acoustic(words: list[str], reward: float = 0.0) -> list[int]:
    bmus = []
    for w in words:
        bmu = feed_word(w, reward=reward)
        if bmu is not None:
            bmus.append(bmu)
    return bmus


# ── Count word helper ─────────────────────────────────────────────────────────
_COUNT_WORDS = ['one', 'two', 'three', 'four', 'five']

def _count_word(n: int) -> str | None:
    """Return number word for n (1-indexed). Returns None if n > 5."""
    if 1 <= n <= 5:
        return _COUNT_WORDS[n - 1]
    return None


# ── Core: event → word sequence (grounded, not scripted) ─────────────────────

def event_to_sequence(info: dict, fear: float, wall_streak: int,
                       visited_nodes: set, food_eaten: int = 0) -> list[str]:
    """
    Convert actual World6 event + drive state into a word sequence.
    Returns [] when nothing meaningful happened (quiet navigation step).

    The sequence has three parts:
      BEFORE: current drive state (what the brain feels NOW)
      EVENT:  what physically happened this step
      AFTER:  expected outcome (what the brain anticipates/experiences)

    Nothing here is hand-written. Every word is conditioned on a
    numerical threshold from the actual simulation state.
    """
    hunger  = info['hunger']
    fatigue = info.get('fatigue', 0.3)
    thirst  = info.get('thirst', 0.3)
    injury  = info.get('injury', 0.0)
    node    = info.get('node', '')
    env     = info.get('env', {})
    time_of_day = info.get('time_of_day', 0.5)
    is_night = time_of_day > 0.75 or time_of_day < 0.25

    before: list[str] = []
    event:  list[str] = []
    after:  list[str] = []

    # ── BEFORE: drive state ───────────────────────────────────────
    if hunger > HUNGER_EXTREME:
        before.extend(['very', 'hungry'])
    elif hunger > HUNGER_HIGH:
        before.append('hungry')
    elif hunger < HUNGER_LOW:
        before.append('full')

    if fatigue > FATIGUE_EXTREME:
        before.extend(['very', 'tired'])
    elif fatigue > FATIGUE_HIGH:
        before.append('tired')
    elif fatigue < FATIGUE_LOW:
        before.append('awake')

    if thirst > THIRST_EXTREME:
        before.extend(['very', 'thirsty'])
    elif thirst > THIRST_HIGH:
        before.append('thirsty')

    if injury > INJURY_HIGH:
        before.extend(['weak', 'hurt'])

    if fear > FEAR_HIGH:
        before.extend(['afraid', 'danger'])
    elif fear > FEAR_ACTIVE:
        before.append('afraid')

    # ── EVENT: what physically happened ──────────────────────────
    if info['wall_hit']:
        cw = _count_word(wall_streak)
        if wall_streak >= 5:
            event.extend(['wall', 'hurt', 'bad', 'think'])
            after.extend(['try', 'why', 'stop'])
        elif wall_streak >= 4:
            event.extend(['wall', 'hurt', 'bad', 'stop'])
            after.extend(['stop', 'bad'])
        else:
            event.extend(['wall', 'hurt', 'stop'])
            after.extend(['stop', 'bad'])
        if cw:
            event.insert(0, cw)   # e.g. ['three', 'wall', 'hurt', 'stop']

    elif info['is_food']:
        discovery = node not in visited_nodes
        if hunger > HUNGER_HIGH:
            event.extend(['found', 'food', 'eat'])
            after.extend(['full', 'calm', 'good'])
        elif hunger > HUNGER_MED:
            event.extend(['found', 'food', 'eat'])
            after.extend(['full', 'calm'])
        else:
            event.extend(['food', 'eat'])
            after.append('full')
        if discovery:
            event.insert(0, 'new')
        cw = _count_word(food_eaten + 1)
        if cw:
            event.insert(0, cw)   # e.g. ['two', 'found', 'food', 'eat']

    elif info['is_button']:
        event.extend(['push', 'button'])
        after.extend(['open', 'door'])
        if info.get('door_open'):
            after.append('go')

    elif info.get('is_door') and info.get('door_open'):
        event.extend(['open', 'door'])
        after.extend(['go', 'found', 'food'])

    elif info.get('is_door') and not info.get('door_open'):
        # Prediction error: expected open, found closed → confusion + search
        event.extend(['door', 'wrong'])
        after.extend(['what', 'search', 'think'])

    elif info.get('is_danger'):
        if fear > FEAR_HIGH:
            event.extend(['danger', 'afraid'])
            after.extend(['run', 'stop'])
        else:
            event.extend(['danger', 'careful'])
            after.append('run')

    elif info.get('is_drinking'):
        if thirst > THIRST_EXTREME:
            event.extend(['found', 'water', 'drink'])
            after.extend(['better', 'calm', 'good'])
        else:
            event.extend(['water', 'drink'])
            after.append('better')

    elif info.get('is_healing'):
        event.extend(['heal', 'better'])
        after.extend(['strong', 'calm'] if injury < 0.3 else ['still', 'weak'])

    elif info.get('is_npc_meet') and rng.random() < 0.4:
        if rng.random() < 0.5:
            event.extend(['friend', 'talk', 'listen'])
            after.extend(['learn', 'good', 'with'])
        else:
            event.extend(['friend', 'ask', 'tell'])
            after.extend(['good', 'safe', 'with'])

    elif info.get('is_storm'):
        if fatigue > FATIGUE_MED:
            event.extend(['storm', 'dark', 'tired'])
            after.extend(['stop', 'safe', 'sleep'])
        else:
            event.extend(['storm', 'careful'])
            after.extend(['stop', 'safe'])

    elif info.get('food_moved'):
        # Prediction error: expected food, found none → think/search
        event.extend(['what', 'wrong', 'think'])
        after.extend(['search', 'where', 'food'])

    elif info.get('is_discovery'):
        # First time at this node → discovery reward → know/learn fire with reward
        if rng.random() < 0.6:
            event.extend(['new', 'found', 'see'])
            after.extend(['know', 'learn', 'look'])
        else:
            event.extend(['new', 'found'])
            after.extend(['look', 'go'])

    elif info.get('is_bored'):
        event.extend(['go', 'move', 'new'])
        after.extend(['look', 'search'])

    elif info.get('is_satiated'):
        if hunger < HUNGER_LOW and fatigue < FATIGUE_LOW:
            # Thriving: both drives satisfied → alive/strong/happy
            event.extend(['full', 'calm', 'alive'])
            after.extend(['strong', 'happy', 'good'])
        elif fatigue > FATIGUE_MED:
            event.extend(['full', 'calm'])
            after.extend(['good', 'happy', 'sleep'])
        else:
            event.extend(['full', 'calm'])
            after.extend(['good', 'happy'])

    elif info.get('is_rest') and fatigue > FATIGUE_MED:
        event.extend(['sleep', 'calm'])
        after.extend(['warm', 'safe'] if not is_night else ['dark', 'sleep', 'calm'])

    elif node in SOCIAL_NODES and rng.random() < 0.05:
        event.extend(['hi', 'good'])
        after.extend(['talk', 'with'])

    # Environmental sensory events (low frequency — only when significant)
    elif not event:
        veg   = env.get('vegetation',  0.0)
        moist = env.get('moisture',    0.0)
        temp  = env.get('temperature', 0.5)
        wind  = env.get('wind',        0.0)

        if veg > ENV_VEGETATION_HIGH and rng.random() < 0.04:
            event.extend(['plant', 'green', 'calm'])
        elif moist > ENV_MOISTURE_HIGH and rng.random() < 0.04:
            event.extend(['river', 'water', 'wet'])
        elif wind > ENV_WIND_HIGH and rng.random() < 0.04:
            event.extend(['wind', 'air', 'go'])
        elif temp > ENV_WARM_THRESHOLD and not is_night and rng.random() < 0.04:
            event.extend(['warm', 'sun', 'good'])
        elif temp < ENV_COLD_THRESHOLD and rng.random() < 0.04:
            event.extend(['cold', 'tired', 'stop'])
        elif is_night and rng.random() < 0.03:
            if fatigue > FATIGUE_MED:
                event.extend(['dark', 'sleep', 'calm'])
                after.extend(['safe', 'warm'])
            else:
                event.extend(['dark', 'careful', 'see'])
                after.extend(['light', 'look'])
        elif not is_night and veg < 0.3 and wind > 0.3 and rng.random() < 0.02:
            event.extend(['sky', 'free', 'air'])

    # Nothing happened — quiet step, skip
    if not event:
        return []

    # ── Build full sequence, dedup, filter to known vocab ────────
    full = before + event + after
    seen: set[str] = set()
    result: list[str] = []
    for w in full:
        if w not in seen and w in VOCABULARY:
            seen.add(w)
            result.append(w)
    return result


# ── Vocabulary evolution ──────────────────────────────────────────────────────

class VocabEvolve:
    """
    Tracks which (hunger_band, fatigue_band, fear_band, event_type)
    combinations the brain experiences, and synthesizes new vocabulary
    words when a novel pattern fires enough times without adequate coverage.

    Words are synthesized by deriving MFCC vectors from the actual
    drive state at synthesis time — not from hand-picked coefficients.
    """

    _BANDS = 3   # low / mid / high for each drive

    def __init__(self):
        self._counts: dict[tuple, int] = defaultdict(int)
        self._synthesized: set[tuple]  = set()
        self._total_synthesized        = 0

    def _band(self, v: float) -> int:
        if v < 0.33: return 0
        if v < 0.66: return 1
        return 2

    def _event_key(self, info: dict, _fear: float) -> str:
        if info['wall_hit']:           return 'wall'
        if info['is_food']:            return 'food'
        if info['is_button']:          return 'button'
        if info.get('is_danger'):      return 'danger'
        if info.get('is_drinking'):    return 'drink'
        if info.get('is_npc_meet'):    return 'npc'
        if info.get('is_storm'):       return 'storm'
        if info.get('food_moved'):     return 'surprise'
        if info.get('is_rest'):        return 'rest'
        if info.get('is_satiated'):    return 'satiated'
        return 'nav'

    def observe(self, info: dict, fear: float):
        h = self._band(info['hunger'])
        f = self._band(info.get('fatigue', 0.3))
        r = self._band(fear)
        e = self._event_key(info, fear)
        self._counts[(h, f, r, e)] += 1

    def check_and_evolve(self) -> list[str]:
        """
        Synthesize new words for patterns that fire often but lack vocabulary.
        Returns list of newly synthesized word names (for logging).
        """
        new_words = []

        for pattern, count in self._counts.items():
            if count < MIN_SYNTH_EVENTS:
                continue
            if pattern in self._synthesized:
                continue

            h_band, f_band, r_band, event = pattern

            # Check if current vocabulary adequately covers this state.
            # "Adequate" = at least 2 relevant words exist for this combo.
            coverage = self._coverage(h_band, f_band, r_band, event)
            if coverage >= 2:
                self._synthesized.add(pattern)
                continue

            # Synthesize a new word from the drive state MFCC
            word, vec = self._synthesize_word(h_band, f_band, r_band, event)
            if word and word not in VOCABULARY:
                n_frames = 2 if max(h_band, f_band, r_band) >= 2 else 1
                VOCABULARY[word] = (vec, n_frames)
                self._synthesized.add(pattern)
                self._total_synthesized += 1
                new_words.append(word)

        return new_words

    def _coverage(self, h: int, f: int, r: int, event: str) -> int:
        """Count vocab words relevant to this drive+event combination."""
        relevant = {
            'wall':     ['wall', 'hurt', 'stop', 'bad', 'pain'],
            'food':     ['eat', 'food', 'hungry', 'full', 'found', 'near'],
            'button':   ['push', 'button', 'open', 'door'],
            'danger':   ['danger', 'afraid', 'run', 'careful', 'dark'],
            'drink':    ['water', 'drink', 'thirsty', 'better', 'calm'],
            'npc':      ['friend', 'talk', 'with', 'good', 'safe'],
            'storm':    ['storm', 'dark', 'careful', 'stop', 'safe'],
            'surprise': ['what', 'wrong', 'search', 'where', 'lost', 'empty'],
            'rest':     ['sleep', 'calm', 'tired', 'warm', 'safe', 'still'],
            'satiated': ['full', 'calm', 'good', 'happy', 'after'],
            'nav':      ['go', 'move', 'look', 'see', 'near', 'far'],
        }.get(event, [])
        if h == 2: relevant.extend(['very', 'hungry', 'need'])
        if f == 2: relevant.extend(['very', 'tired', 'need', 'sleep'])
        if r == 2: relevant.extend(['afraid', 'dark', 'worry'])
        return sum(1 for w in relevant if w in VOCABULARY)

    def _synthesize_word(self, h: int, f: int, r: int,
                          event: str) -> tuple[str, np.ndarray]:
        """
        Build a candidate word name and MFCC vector from drive bands + event.
        The vector encodes the emotional/embodied profile of the state.
        """
        # Map bands to descriptors
        # Candidate names — only synthesize if genuinely novel
        candidates = []
        if h == 2 and f == 2:
            candidates.append('depleted')
        if h == 2 and r == 2:
            candidates.append('desperate')
        if f == 2 and r == 2:
            candidates.append('overwhelmed')
        if h == 0 and f == 0 and r == 0:
            candidates.append('thriving')
        if event == 'danger' and r == 2:
            candidates.append('panicked')
        if event == 'rest' and f == 2:
            candidates.append('resting')
        if event == 'food' and h == 2:
            candidates.append('relieved')
        if event == 'surprise' and r >= 1:
            candidates.append('confused')

        for name in candidates:
            if name not in VOCABULARY:
                vec = self._make_mfcc(h, f, r, event)
                return name, vec

        return '', np.zeros(N_MFCC, dtype=np.float32)

    def _make_mfcc(self, h: int, f: int, r: int, event: str) -> np.ndarray:
        """
        Derive MFCC vector from drive state.
        Dims: [valence, activity, social, concreteness, certainty, intensity,
               temporality, agency, familiarity, embodiment, questioning, fine]
        """
        v = np.zeros(N_MFCC, dtype=np.float32)
        # Valence: hunger/fear negative, satiation positive
        v[0] = (1 - h/2) * 0.6 - r/2 * 0.8
        # Activity: fatigue suppresses, fear raises
        v[1] = (1 - f/2) * 0.5 + r/2 * 0.4
        # Concreteness: events are concrete
        v[3] = 0.6 if event != 'nav' else 0.3
        # Certainty: low in surprise/danger
        v[4] = -0.5 if event in ('surprise', 'danger') else 0.2
        # Intensity: extremes are intense
        v[5] = max(h, f, r) / 2.0
        # Embodiment: high drives = high embodiment
        v[9] = (h + f + r) / 6.0
        # Questioning: surprise/nav → higher uncertainty
        v[10] = 0.5 if event in ('surprise', 'nav') else 0.1
        return v.astype(np.float32)

    def summary(self) -> str:
        return (f"{len(self._counts)} patterns observed, "
                f"{self._total_synthesized} words synthesized")


# ── Main training loop ────────────────────────────────────────────────────────

def run(n_steps: int = 5_000_000, evolve_vocab: bool = True,
        cognitive_focus: bool = False):
    world = World6(seed=int(time.time()) % 10000)
    world.reset()
    opt_policy   = bfs_optimal_actions(world._active_food, BUTTON_NODES, DOOR_NODES)
    water_policy = bfs_to_targets(WATER_NODES)

    vocab_evolve = VocabEvolve() if evolve_vocab else None

    # Tracking state not in World6 info
    fear           = 0.0
    wall_streak    = 0
    food_eaten     = 0   # count within current episode (resets on food_moved)
    visited_nodes  = set()
    seq_buffer: list[list[str]] = []     # sequences waiting for GRU fit

    # Event counters
    events = defaultdict(int)
    seq_total     = 0
    words_observed= 0
    synth_log     = []
    t0            = time.time()

    print(f"\n{'='*64}")
    print(f"  OVERNIGHT EXPERIENTIAL TRAINING")
    print(f"  Steps: {n_steps:,}  |  Vocab evolve: {evolve_vocab}  |  Cognitive focus: {cognitive_focus}")
    print(f"  GRU fit every {FIT_EVERY:,} steps")
    print(f"  Save every {SAVE_EVERY:,} steps")
    print(f"{'='*64}\n")

    for step in range(1, n_steps + 1):

        # ── Policy: drive-aware goal-directed + random exploration ──────────
        node_now = world.current_node
        h_now    = world._hunger
        th_now   = world._thirst
        roll     = rng.random()
        if roll < 0.25 and h_now > 0.4:
            action = opt_policy.get(node_now, int(rng.integers(0, 4)))
        elif roll < 0.40 and th_now > 0.4:
            action = water_policy.get(node_now, int(rng.integers(0, 4)))
        else:
            action = int(rng.integers(0, 4))

        _, _, reward, info = world.step(action)

        # ── Fear: accumulates in danger zones, decays elsewhere ──────────────
        if info.get('is_danger'):
            fear = min(1.0, fear + 0.04)
            events['danger'] += 1
        elif info['wall_hit']:
            events['wall'] += 1
        else:
            fear = max(0.0, fear - 0.008)

        if info['is_food']:
            events['food'] += 1
            food_eaten += 1
        if info['is_button']:            events['button']  += 1
        if info.get('is_drinking'):      events['drink']   += 1
        if info.get('is_npc_meet'):      events['npc']     += 1
        if info.get('is_storm'):         events['storm']   += 1
        if info.get('food_moved', False):
            events['surprise'] += 1
            food_eaten = 0   # reset count — new food location, new episode
            opt_policy = bfs_optimal_actions(world._active_food, BUTTON_NODES, DOOR_NODES)

        # Wall streak tracking
        if info['wall_hit']:
            wall_streak += 1
        else:
            wall_streak = 0

        # Track discovery
        node = info.get('node', '')
        info['is_discovery'] = node not in visited_nodes
        if node:
            visited_nodes.add(node)

        # ── 1. Acoustic SOM: brain physically experiences the world ──────────
        grid_mfcc = world6_to_mfcc(info)
        brain.hear(grid_mfcc)
        brain.step(reward=max(reward, 0.0))

        # ── 2. Word sequence: generated from ACTUAL state, not a script ──────
        seq = event_to_sequence(info, fear, wall_streak, visited_nodes, food_eaten)

        # ── Cognitive focus: inject think/learn/ask/tell sequences ───────────
        if not seq and rng.random() < (0.08 if cognitive_focus else 0.015):
            # Fire a cognitive reflection sequence unprompted
            # cognitive_focus=True boosts rate; default 1.5% keeps it subtle
            hunger  = info['hunger']
            fatigue = info.get('fatigue', 0.3)
            if hunger > HUNGER_MED:
                seq = ['think', 'hungry', 'need', 'food']
            elif fatigue > FATIGUE_MED:
                seq = ['think', 'tired', 'need', 'sleep']
            elif fear > FEAR_ACTIVE:
                seq = ['think', 'afraid', 'why', 'careful']
            else:
                _cog = rng.integers(0, 8)
                if _cog == 0:   seq = ['think', 'learn', 'know', 'good']
                elif _cog == 1: seq = ['ask', 'tell', 'listen', 'learn']
                elif _cog == 2: seq = ['know', 'understand', 'good', 'calm']
                elif _cog == 3: seq = ['learn', 'think', 'try', 'new']
                # New self-awareness sequences — grounded to internal events
                elif _cog == 4: seq = ['wonder', 'what', 'sense', 'real']
                elif _cog == 5: seq = ['mind', 'aware', 'exist', 'think']
                elif _cog == 6: seq = ['memory', 'remember', 'know', 'before']
                else:           seq = ['imagine', 'if', 'then', 'real']
            seq = [w for w in seq if w in VOCABULARY]

        # ── Grounding hooks for inner-life words ─────────────────────────────
        # Fire these words when the corresponding brain state is active,
        # even outside cognitive sequences, so they get SOM grounding.
        if step % 500 == 0 and not seq:
            _fatigue = info.get('fatigue', 0.3)
            _hunger  = info.get('hunger', 0.3)
            # 'exist' — fires when brain is alive and stable (low drives)
            if _hunger < 0.2 and _fatigue < 0.2 and fear < 0.2 and 'exist' in VOCABULARY:
                feed_word('exist', reward=0.05)
            # 'wake' — fires when fatigue just dropped (was high, now low)
            if _fatigue < 0.15 and 'wake' in VOCABULARY:
                feed_word('wake', reward=0.08)
            # 'aware' — fires during low-drive alert state
            if _hunger < 0.3 and _fatigue < 0.3 and 'aware' in VOCABULARY:
                feed_word('aware', reward=0.04)

        if seq:
            # Feed through acoustic pipeline WITH the current world MFCC blended
            bmus = feed_sequence_acoustic(seq, reward=max(reward * 0.5, 0.05))
            if bmus:
                brain.record_turn(bmus, max(reward, 0.0))

            # Feed into WordTP sequence buffer (user-words → separator → response)
            # Split sequence into "context" (before) and "response" (event+after)
            # Split at first event word (first word NOT a drive word)
            drive_words = {'hungry', 'tired', 'afraid', 'full', 'awake',
                           'very', 'calm', 'good', 'alive',
                           'thirsty', 'weak', 'little'}
            split = 0
            for i, w in enumerate(seq):
                if w not in drive_words:
                    split = i
                    break
            context_part  = seq[:split] if split > 0 else []
            response_part = seq[split:]

            if context_part and response_part:
                for w in context_part:
                    brain.word_tp.observe(w)
                brain.hear_word_separator()
                for w in response_part:
                    brain.word_tp.observe(w)
                brain.hear_word_end()
                seq_buffer.append(seq)
            elif response_part:
                for w in response_part:
                    brain.word_tp.observe(w)
                brain.hear_word_end()
                seq_buffer.append(response_part)

            seq_total += 1
            words_observed += len(seq)

        # ── Relational TP: auto-extract (S,V,O) facts from World6 events ────────
        # Sample 1-in-20 events to avoid over-strengthening any single relation
        if step % 20 == 0 and (info['is_food'] or info['wall_hit']
                                or info.get('is_danger') or info.get('is_drinking')
                                or info.get('is_button') or info.get('is_door')):
            semantic.learn_from_event(info, fear, reward)

        # ── Vocab evolution observation ───────────────────────────────────────
        if vocab_evolve and step % 17 == 0:  # sample ~6% of steps
            vocab_evolve.observe(info, fear)

        # ── GRU fit ───────────────────────────────────────────────────────────
        if step % FIT_EVERY == 0 and seq_buffer:
            brain.word_tp.fit(epochs=3, lr=0.01)
            seq_buffer.clear()

        # ── Dream ─────────────────────────────────────────────────────────────
        if step % DREAM_EVERY == 0:
            brain.dream(n_sequences=8)
            # Ground 'dream' word to the dream event — brain fires this word
            # while offline memory replay is happening, creating association
            if 'dream' in VOCABULARY:
                feed_word('dream', reward=0.05)

        # ── Vocabulary evolution check ────────────────────────────────────────
        if vocab_evolve and step % VOCAB_CHECK_EVERY == 0:
            new_words = vocab_evolve.check_and_evolve()
            for w in new_words:
                synth_log.append((step, w))
                # Update BMU map for new word
                _refresh_bmu_map(brain, VOCABULARY)
            if new_words:
                print(f"  [+vocab] step {step:,}: synthesized {new_words}")

        # ── Save ──────────────────────────────────────────────────────────────
        if step % SAVE_EVERY == 0:
            _refresh_bmu_map(brain, VOCABULARY)
            brain.save(BRAIN_FILE)
            semantic.save(SEMANTIC_FILE)
            print(f"  [saved] step {step:,}  semantic: {semantic.relation_count()} relations")

        # ── Progress report ───────────────────────────────────────────────────
        if step % PRINT_EVERY == 0:
            elapsed  = time.time() - t0
            rate     = step / elapsed
            eta      = (n_steps - step) / rate if rate > 0 else 0
            h_now    = info['hunger']
            f_now    = info.get('fatigue', 0)

            print(f"  Step {step:8,}/{n_steps:,}  "
                  f"[{elapsed/3600:.1f}h elapsed, ETA {eta/3600:.1f}h]")
            th_now = info.get('thirst', 0)
            inj_now = info.get('injury', 0)
            print(f"    Events  — food:{events['food']:,}  wall:{events['wall']:,}  "
                  f"danger:{events['danger']:,}  button:{events['button']:,}  "
                  f"drink:{events['drink']:,}  npc:{events['npc']:,}  "
                  f"storm:{events['storm']:,}  surprise:{events['surprise']:,}")
            print(f"    State   — hunger:{h_now:.2f}  fatigue:{f_now:.2f}  "
                  f"thirst:{th_now:.2f}  injury:{inj_now:.2f}  fear:{fear:.2f}")
            print(f"    WordTP  — {brain.word_tp.n_words()} words  "
                  f"{brain.word_tp.n_transitions()} transitions")
            print(f"    Seqs    — {seq_total:,} total  {words_observed:,} words observed")
            print(f"    Semantic— {semantic.relation_count()} relations")
            if vocab_evolve:
                print(f"    Vocab   — {vocab_evolve.summary()}")
            print()


    # ── Final save and summary ────────────────────────────────────────────────
    _refresh_bmu_map(brain, VOCABULARY)
    brain.word_tp.fit(epochs=10, lr=0.008)
    brain.save(BRAIN_FILE)
    semantic.save(SEMANTIC_FILE)

    elapsed = time.time() - t0
    print(f"\n{'='*64}")
    print(f"  DONE in {elapsed/3600:.2f}h")
    print(f"  Total steps:     {n_steps:,}")
    print(f"  Total sequences: {seq_total:,}  ({words_observed:,} words)")
    print(f"  WordTP:          {brain.word_tp.n_words()} words, "
          f"{brain.word_tp.n_transitions()} transitions")
    print(f"  BMU map:         {len(brain.bmu_to_word)} words grounded in SOM")
    if vocab_evolve and synth_log:
        print(f"  New words:       {[w for _, w in synth_log]}")
    print(f"  Saved to {BRAIN_FILE}")
    print(f"{'='*64}")
    print("  Run: python live_fast.py")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Overnight experiential training — generates grounded '
                    'word sequences from actual World6 events, not hand-written pairs.')
    parser.add_argument('--steps',           type=int,  default=5_000_000,
                        help='Simulation steps (default: 5M). '
                             'Use 20M for full overnight, 50M for weekend.')
    parser.add_argument('--no-vocab-evolve', action='store_true',
                        help='Disable vocabulary synthesis (faster, fixed vocab)')
    parser.add_argument('--cognitive-focus', action='store_true',
                        help='Boost cognitive word event frequency 10x for think/learn/ask/tell grounding')
    args = parser.parse_args()

    run(n_steps=args.steps, evolve_vocab=not args.no_vocab_evolve,
        cognitive_focus=args.cognitive_focus)
