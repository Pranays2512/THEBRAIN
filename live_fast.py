"""
live_fast.py — Grounded conversation loop for FastBrain
========================================================

How generation works:
  1. Your words → acoustic SOM → BMUs
  2. BMUs → which semantic zone (food / pain / rest / action / social)
  3. Zone → expression seeds (what the brain wants to say, from experience)
  4. Seeds + context → WordTP generates the expression (how to say it)

The zone is determined by grounded SOM topology, not text statistics.
WordTP only handles grammar — the meaning comes from experience.

Commands:
  /state  — brain status
  /zones  — show zone centers and word mapping
  /dream  — manual dream cycle
  /help   — known words
  /quit   — save and exit

Feedback:
  good / yes / great  → reward (+1.0)
  no / wrong / bad    → penalty
"""

import sys
import os
import time
import threading
import queue
import numpy as np
from collections import deque

sys.dont_write_bytecode = True

from brain_fast import FastBrain, SILENCE, N_MFCC, SOM_COLS
from vocab_core import VOCABULARY
from m75_semantic import SemanticMemory
from brain_modules import (WorkingMemory, EpisodicMemory, SelfModel,
                            CuriosityModule, NeuromodSystem, TemporalContext,
                            PrefrontalCortex)

BRAIN_FILE = 'brain_fast.pkl'
rng = np.random.default_rng(int(time.time()) % (2**31))

# ── Load brain ────────────────────────────────────────────────────────────────
if not os.path.exists(BRAIN_FILE):
    print(f"\n  No trained brain found ({BRAIN_FILE}).")
    print("  Run:  python train_world6_fast.py  first.")
    sys.exit(1)

print(f"\nLoading FastBrain from {BRAIN_FILE}...")
brain = FastBrain.load(BRAIN_FILE)
print(brain.status())

# ── Semantic memory ───────────────────────────────────────────────────────────
semantic = SemanticMemory()
print(f"  Semantic memory: {len(semantic._facts)} facts")

wm        = WorkingMemory(n_neurons=5000)
episodic  = EpisodicMemory(maxlen=50)
selfmodel = SelfModel()
curiosity = CuriosityModule(n_neurons=5000)
neuromod  = NeuromodSystem()
temporal  = TemporalContext()
pfc       = PrefrontalCortex()

# ── BMU → word map ────────────────────────────────────────────────────────────
print("  Updating BMU→word map from clean MFCCs...", end='', flush=True)
for word in VOCABULARY:
    mean_vec, _ = VOCABULARY[word]
    bmu = brain.som.find_bmu(mean_vec.astype(np.float32))
    if word not in brain.word_to_bmu:
        brain.word_to_bmu[word] = bmu
    if bmu not in brain.bmu_to_word:
        brain.bmu_to_word[bmu] = word
print(f" {len(brain.bmu_to_word)} BMU→word entries.")

# ── Conversation state ────────────────────────────────────────────────────────
_recent_responses: list[str] = []
_MAX_RECENT       = 2
_turn_history: deque = deque(maxlen=5)
_total_turns  = 0
_DREAM_EVERY  = 10
_IMAGINATION_ENABLED = False

_FEEDBACK_POSITIVE = {'good', 'yes', 'great', 'nice', 'right', 'correct',
                      'perfect', 'well', 'okay', 'fine'}
_FEEDBACK_NEGATIVE = {'no', 'wrong', 'bad', 'not', 'never'}

_PUNCT = str.maketrans('', '', '.,?!;:\'"()-/\\')

_last_response_words: list[str] = []
_last_heard_bmus:     list[int] = []

# ── Conversation context buffer (lightweight turn memory) ─────────────────────
# Stores (heard_words, response_words) for the last N turns.
# Fed back into generate_response so the brain can track topic continuity
# and avoid repeating the same response to the same input.
_context_buffer: deque = deque(maxlen=3)   # last 3 turns of (heard, response)


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT BRIDGE — grounded zone detection + expression
# ═══════════════════════════════════════════════════════════════════════════════

# Zone definitions: which words anchor each zone's center on the SOM.
# These are grounded words — their BMU positions were shaped by World6 events.
ZONE_ANCHORS = {
    'food':   ['food', 'eat', 'drink', 'hungry', 'full', 'water'],
    'pain':   ['hurt', 'pain', 'wall', 'bad', 'stop', 'no'],
    'rest':   ['tired', 'sleep', 'calm', 'warm', 'soft', 'awake'],
    'action': ['go', 'move', 'push', 'open', 'come', 'door', 'button'],
    'social': ['hi', 'hello', 'bye', 'yes', 'happy', 'help', 'sorry'],
    'danger': ['afraid', 'danger', 'careful', 'run'],
}

# Zone → what the brain expresses when in this zone.
# RULE: each word appears in AT MOST ONE zone's expression list.
# Generic words (good, feel, here, want) are deliberately removed
# to force the brain to speak in specific grounded vocabulary.
ZONE_EXPRESSION = {
    'food':   ['eat', 'food', 'hungry', 'want', 'drink', 'water'],
    'pain':   ['stop', 'hurt', 'pain', 'bad', 'no'],
    'rest':   ['sleep', 'calm', 'tired', 'warm', 'awake', 'safe'],
    'action': ['go', 'open', 'move', 'push', 'door', 'button', 'come'],
    'social': ['hello', 'hi', 'happy', 'help', 'yes', 'know', 'sorry'],
    'danger': ['afraid', 'careful', 'run', 'danger'],
}


# Inject Identity
semantic.learn_from_sentence(['i', 'am', 'fastbrain'])
if 'fastbrain' not in ZONE_ANCHORS['social']:
    ZONE_ANCHORS['social'].append('fastbrain')
if 'fastbrain' not in ZONE_EXPRESSION['social']:
    ZONE_EXPRESSION['social'].append('fastbrain')


# Direct word→zone lookup (highest-confidence path: anchor words have known zones).
# Extended to cover all vocabulary words so BMU fallback is rarely needed.
_WORD_TO_ZONE: dict[str, str] = {
    w: z for z, ws in ZONE_ANCHORS.items() for w in ws
}
_WORD_TO_ZONE.update({
    # Pronouns map to social but don't anchor it
    'i': 'social', 'me': 'social', 'you': 'social', 'we': 'social',
    'and': 'social', 'is': 'social', 'not': 'social', 'now': 'social',
    # Explicit single-zone assignments for ambiguous words
    'feel': 'rest',    # body-state → rest zone, not social
    'good': 'social',  # reward signal → social
    'want': 'food',    'need': 'food',    'more': 'food',
    'alive': 'rest',   'wait': 'rest',    'safe': 'rest',
    'happy': 'social', 'sad': 'pain',
    'strong': 'action', 'hard': 'action', 'new': 'action', 'push': 'action',
    'know': 'social',  'sorry': 'social',
})

INTENT_SPECS = {
    'action_open_door': {
        'zone': 'action',
        'required': {'open', 'door'},
        'seeds': ['open', 'door', 'go', 'move'],
        'template': ['open', 'door', 'go'],
    },
    'action_push_button': {
        'zone': 'action',
        'required': {'push', 'button'},
        'seeds': ['push', 'button', 'open', 'door'],
        'template': ['push', 'button', 'open'],
    },
    'action_button_door_chain': {
        'zone': 'action',
        'required': {'button', 'door'},
        'seeds': ['push', 'button', 'open', 'door'],
        'template': ['push', 'button', 'open', 'door'],
    },
    'danger_escape': {
        'zone': 'danger',
        'required': {'danger'},
        'any_of': {'run', 'afraid', 'careful'},
        'seeds': ['run', 'careful', 'danger'],
        'template': ['run', 'careful'],
    },
    'danger_here': {
        'zone': 'danger',
        'required': {'danger', 'here'},
        'seeds': ['run', 'careful', 'danger'],
        'template': ['run', 'careful'],
    },
    'food_seek': {
        'zone': 'food',
        'required': {'hungry'},
        'any_of': {'food', 'eat', 'want', 'need'},
        'seeds': ['eat', 'food', 'want', 'hungry'],
        'template': ['want', 'food', 'eat'],
    },
    'food_water': {
        'zone': 'food',
        'required': {'water'},
        'any_of': {'drink', 'need'},
        'seeds': ['drink', 'water', 'need'],
        'template': ['drink', 'water'],
    },
    'rest_sleep': {
        'zone': 'rest',
        'required': {'tired'},
        'any_of': {'sleep', 'calm', 'need'},
        'seeds': ['sleep', 'calm', 'tired'],
        'template': ['sleep', 'calm'],
    },
    'social_greeting': {
        'zone': 'social',
        'required': {'hi'},
        'any_of': {'hello', 'you', 'yes'},
        'seeds': ['hi', 'hello', 'yes'],
        'template': ['hello'],
    },
    'identity_name': {
        'zone': 'social',
        'required': {'name'},
        'any_of': {'you', 'fastbrain', 'is'},
        'seeds': ['name', 'fastbrain'],
        'template': ['name', 'fastbrain'],
    },
    'identity_self': {
        'zone': 'social',
        'required': {'fastbrain'},
        'any_of': {'you', 'name', 'hi'},
        'seeds': ['hi', 'fastbrain', 'name'],
        'template': ['hi', 'fastbrain'],
    },
}

_ZONE_NEUTRAL = {'i', 'me', 'you', 'we', 'and', 'is', 'not', 'now', 'here'}
_HOLLOW = {'i', 'me', 'we', 'you', 'and', 'is', 'not'}



def _zone_center(zone_name: str) -> tuple[float, float] | None:
    """Average SOM (row, col) of the zone's anchor words — used by /zones command."""
    positions = []
    for w in ZONE_ANCHORS[zone_name]:
        bmu = brain.word_to_bmu.get(w)
        if bmu is not None:
            positions.append((bmu // SOM_COLS, bmu % SOM_COLS))
    if not positions:
        return None
    return (sum(p[0] for p in positions) / len(positions),
            sum(p[1] for p in positions) / len(positions))


def _detect_zone(heard_bmus: list[int]) -> str:
    """
    Find which zone the heard BMUs belong to.

    Uses Voronoi-style nearest-anchor classification: compare each heard BMU
    directly to every anchor word's BMU and assign to the zone of the closest
    single anchor (not the zone centroid, which degrades when anchor words
    are spread across the SOM).
    """
    # Build (zone, bmu, row, col) for all mapped anchor words
    anchors: list[tuple[str, int, int, int]] = []
    for zone, words in ZONE_ANCHORS.items():
        for w in words:
            bmu = brain.word_to_bmu.get(w)
            if bmu is not None:
                anchors.append((zone, bmu, bmu // SOM_COLS, bmu % SOM_COLS))

    if not anchors:
        return 'social'

    votes: dict[str, int] = {z: 0 for z in ZONE_ANCHORS}
    for bmu in heard_bmus:
        row, col = bmu // SOM_COLS, bmu % SOM_COLS
        best_zone = min(anchors, key=lambda a: (row - a[2]) ** 2 + (col - a[3]) ** 2)[0]
        votes[best_zone] += 1

    return max(votes, key=votes.get)


def _detect_intent(heard_words: list[str]) -> str | None:
    """
    Detect a small set of grounded intents before falling back to zone-only
    generation. This gives multi-word phrases like "open door" a stable
    semantic target instead of only contributing to the broad action zone.
    """
    heard = set(heard_words)
    for intent_name, spec in INTENT_SPECS.items():
        required = spec.get('required', set())
        if not required.issubset(heard):
            continue
        any_of = spec.get('any_of')
        if any_of and not (heard & set(any_of)):
            continue
        return intent_name
    return None


def _template_allowed(template_words: list[str], zone: str, avoid: set[str]) -> str | None:
    """Return a deterministic grounded template if it is zone-valid and not repetitive."""
    clipped = _clip_to_zone(template_words, zone)
    if not clipped:
        return None
    resp = ' '.join(clipped)
    if resp in avoid:
        return None
    return resp


def _resolve_zone(heard_words: list[str], heard_bmus: list[int]) -> str:
    """Resolve the current semantic zone, giving explicit intents priority."""
    intent = _detect_intent(heard_words)
    if intent is not None:
        return INTENT_SPECS[intent]['zone']

    zone_votes: dict[str, int] = {z: 0 for z in ZONE_ANCHORS}
    for w in heard_words:
        if w in _WORD_TO_ZONE:
            zone_votes[_WORD_TO_ZONE[w]] += 1

    if any(zone_votes.values()):
        return max(zone_votes, key=zone_votes.get)

    clean_bmus = [brain.word_to_bmu[w] for w in heard_words if w in brain.word_to_bmu]
    return _detect_zone(clean_bmus if clean_bmus else heard_bmus)


def _clip_to_zone(words: list[str], zone: str) -> list[str]:
    """
    Truncate a generated word list at the first word that clearly belongs
    to a *different* zone. Neutral words pass through freely.

    This prevents Markov chain drift across zone boundaries mid-response
    (e.g. 'run → stop → pain → bad' when zone is 'danger').
    """
    _TRAILING_DROP = {'and', 'or', 'is', 'not', 'a', 'the', 'to', 'of'}
    zone_ok = set(ZONE_EXPRESSION[zone]) | set(ZONE_ANCHORS[zone])
    result = []
    seen: set[str] = set()
    for w in words:
        if w in zone_ok or w in _ZONE_NEUTRAL:
            if w not in seen:        # no repeated words in one response
                result.append(w)
                seen.add(w)
        else:
            break   # first out-of-zone word — stop here
    # Drop trailing conjunctions/function words ("i feel and" → "i feel")
    while result and result[-1] in _TRAILING_DROP:
        result.pop()
    return result


def generate_response(heard_bmus: list[int], heard_words: list[str]) -> str:
    """
    Intent-driven grounded response generation with turn memory.

    Step 1: Detect which semantic zone the input activated (grounded in SOM).
    Step 2: Select expression seeds from that zone (what the brain wants to say).
    Step 3: WordTP generates the actual word sequence (how to say it).
    Step 4: Clip output to stay within the detected zone.
    """
    if not heard_bmus and not heard_words:
        return ''

    # Step 1a: detect explicit grounded intent before generic zone routing.
    intent = _detect_intent(heard_words)

    if intent is not None:
        zone = INTENT_SPECS[intent]['zone']
        seeds = [w for w in INTENT_SPECS[intent]['seeds'] if w in VOCABULARY]
    else:
        # Step 1b: direct word→zone lookup for anchor words (zero-distance).
        zone_votes: dict[str, int] = {z: 0 for z in ZONE_ANCHORS}
        for w in heard_words:
            if w in _WORD_TO_ZONE:
                zone_votes[_WORD_TO_ZONE[w]] += 1

        if any(zone_votes.values()):
            zone = max(zone_votes, key=zone_votes.get)
        else:
            # Step 1c: fall back to nearest-anchor BMU comparison.
            clean_bmus = [brain.word_to_bmu[w] for w in heard_words if w in brain.word_to_bmu]
            zone = _detect_zone(clean_bmus if clean_bmus else heard_bmus)

        # Zone seeds are always the base when no specific intent fired.
        seeds = [w for w in ZONE_EXPRESSION[zone] if w in VOCABULARY]

    # PFC goal appends urgency words IF they belong to the same zone.
    # PFC never replaces zone seeds — it only reinforces them when relevant.
    pfc.check_goal(selfmodel)
    goal_seeds = [w for w in pfc.get_goal_seeds()
                  if _WORD_TO_ZONE.get(w, zone) == zone]
    seeds = seeds + goal_seeds

    # Self-model words only reinforce the zone — never override it.
    self_words = [w for w in selfmodel.state_words()
                  if _WORD_TO_ZONE.get(w, zone) == zone]
    seeds = seeds + self_words

    # Step 2b: Episodic memory — if topic changed, seed with what brain said last time
    # in this zone. This gives genuine conversational memory.
    if episodic.topic_changed():
        past = episodic.recall_similar(zone, n=1)
        if past and past[0]['brain']:
            prev_word = past[0]['brain'][0]
            if prev_word in VOCABULARY:
                seeds = [prev_word] + seeds

    # Build the set of responses to avoid (last 3 turns from buffer).
    avoid: set[str] = set(_recent_responses)
    for _, past_said in _context_buffer:
        if past_said:
            avoid.add(' '.join(past_said))

    def _is_meaningful(clipped: list[str]) -> bool:
        """True if response has at least one non-hollow word."""
        return any(w not in _HOLLOW for w in clipped) if clipped else False

    # Deterministic boundary: strong grounded intents can answer directly.
    # This keeps the system expressive without letting generic WordTP override
    # a correctly detected sensorimotor meaning.
    if intent is not None:
        template = INTENT_SPECS[intent].get('template')
        if template:
            direct = _template_allowed(template, zone, avoid)
            if direct is not None:
                return direct

    # Context: zone seeds first, then heard words
    context = seeds[:3] + heard_words[:3]
    temperature = neuromod.temperature()

    # Primary attempt
    words = brain.word_tp_generate(context, max_len=5, temperature=temperature)
    if words:
        clipped = _clip_to_zone(words, zone)
        if _is_meaningful(clipped):
            resp = ' '.join(clipped)
            if resp not in avoid:
                return resp

    # Second attempt — slightly higher temperature for a different path
    words = brain.word_tp_generate(context, max_len=5, temperature=temperature + 0.15)
    if words:
        clipped = _clip_to_zone(words, zone)
        if _is_meaningful(clipped):
            resp = ' '.join(clipped)
            if resp not in avoid:
                return resp

    # Fall back to zone seeds directly (always zone-correct, always meaningful)
    available = [w for w in seeds if w in brain.word_to_bmu and w not in _HOLLOW]
    return ' '.join(available[:3]) if available else ''


# ── Acoustic helpers ──────────────────────────────────────────────────────────

def say_frames(word: str, noise_std: float = 0.10) -> list[np.ndarray]:
    if word not in VOCABULARY:
        return []
    mean_vec, n_frames = VOCABULARY[word]
    return [
        (mean_vec + rng.normal(0, noise_std, N_MFCC)).astype(np.float32)
        for _ in range(n_frames)
    ]


def hear_own_response(words: list[str], reward: float = 0.0):
    """Feed the brain's own spoken words back through its acoustic SOM."""
    bmus = []
    for word in words:
        frames = say_frames(word, noise_std=0.05)
        if not frames:
            continue
        for i, frame in enumerate(frames):
            brain.hear(frame)
            out = brain.step(reward=reward if i == len(frames) - 1 else 0.0)
        bmus.append(out['bmu'])
        brain.hear(SILENCE)
        brain.step()
    if bmus:
        brain.record_turn(bmus, max(reward, 0.0))
    return bmus


def feed_input(words: list[str]) -> list[int]:
    """Feed the user's words through the brain. Returns BMUs."""
    bmus = []
    for word in words:
        frames = say_frames(word, noise_std=0.10)
        if not frames:
            continue
        for i, frame in enumerate(frames):
            brain.hear(frame)
            out = brain.step(reward=0.05)
        bmus.append(out['bmu'])
        brain.hear(SILENCE)
        brain.step()
    return bmus


def apply_feedback(positive: bool):
    global _last_response_words
    if not _last_response_words:
        return
    hear_own_response(_last_response_words, reward=1.0 if positive else 0.0)
    if positive:
        print("  [brain: +reward]")
    else:
        print("  [brain: noted]")


# ── Main loop (only runs when executed directly, not when imported) ───────────

def main():
    global _total_turns, _last_response_words, _last_heard_bmus, _IMAGINATION_ENABLED

    print("\n" + "=" * 55)
    print("  FAST BRAIN — grounded vocabulary")
    print(f"  {len(VOCABULARY)} words | {len(brain.bmu_to_word)} BMU mappings")
    print("  Type to talk. /quit to exit.")
    print(f"  (Async Mode: Brain ticks every 3 seconds, imagination {'on' if _IMAGINATION_ENABLED else 'off'})")
    print("=" * 55 + "\n")

    input_queue = queue.Queue()
    print_lock  = threading.Lock()
    _last_printed_goal = [None]   # mutable so inner tick can update it

    def input_thread():
        while True:
            try:
                with print_lock:
                    sys.stdout.write("you: ")
                    sys.stdout.flush()
                line = sys.stdin.readline().rstrip('\n')
                input_queue.put(line)
            except EOFError:
                input_queue.put("/quit")
                break
            except Exception:
                pass
    threading.Thread(target=input_thread, daemon=True).start()

    ticks_idle = 0
    _last_turn_time = [time.time()]  # mutable for closure
    interrupted = False
    try:
        while True:
            try:
                raw = input_queue.get(timeout=3.0)
                ticks_idle = 0
                _last_turn_time[0] = time.time()   # user spoke — reset silence clock
            except queue.Empty:
                raw = None
                ticks_idle += 1

            if raw is None:

                # --- ASYNC BACKGROUND TICK ---
                brain.step()  # Pass time organically

                # Passive state drift
                selfmodel.update('rest', 0.0, wm.activation_strength(), episodic.count())

                # Re-evaluate goal
                pfc.check_goal(selfmodel)

                # Check how long since user last spoke
                silence_s = time.time() - _last_turn_time[0]

                # Drive monologue: only after 15s silence, once per new goal
                # Imagination: only after 30s silence, once per 27s
                if ticks_idle % 3 == 0 and silence_s >= 15.0:
                    goal = pfc._active_goal
                    with print_lock:
                        sys.stdout.write("\r\033[K")  # clear the dangling "you: " line
                        if goal and goal != _last_printed_goal[0]:
                            # Drive-based monologue — fires ONCE per new goal
                            print(f"[{temporal.session_phase()}-thought]: i want {goal}")
                            _last_printed_goal[0] = goal
                        elif (_IMAGINATION_ENABLED and not goal and
                              ticks_idle % 9 == 0 and silence_s >= 30.0):
                            # Curiosity imagination — only during genuine idle periods
                            _last_printed_goal[0] = None
                            thought = pfc.imagine(brain, episodic,
                                                  curiosity=curiosity,
                                                  zone_anchors=ZONE_ANCHORS)
                            if thought:
                                print(f"[{temporal.session_phase()}-imagine]: {thought}")
                        sys.stdout.write("you: ")
                        sys.stdout.flush()
                continue

            # --- USER INPUT PROCESSING ---
            raw = raw.strip()
            if not raw:
                continue

            # ── Slash commands ────────────────────────────────────────────
            if raw.startswith('/'):
                cmd = raw[1:].strip().lower()

                if cmd in ('quit', 'exit', 'q'):
                    break

                elif cmd == 'state':
                    print(brain.status())
                    print(f"  Semantic facts: {len(semantic._facts)}")
                    print(f"  Recent responses: {_recent_responses[-3:]}")
                    dom = selfmodel.dominant_state()
                    print(f"  Self state:    {dom}  {selfmodel._state}")
                    print(f"  WM strength:   {wm.activation_strength():.3f}")
                    print(f"  WM top words:  {wm.top_active_words(getattr(brain, 'bmu_to_word', {}))}")
                    print(f"  Neuromod:      ACh={neuromod._ach:.2f}  NE={neuromod._ne:.2f}  5HT={neuromod._serotonin:.2f}")
                    print(f"  Temperature:   {neuromod.temperature():.3f}")
                    print(f"  Curiosity:     least zone = {curiosity.least_visited_zone(list(ZONE_ANCHORS.keys()))}")
                    print(f"  Temporal:      turn={temporal._turn}  phase={temporal.session_phase()}")
                    print(f"  Episodes:      {episodic.count()} recorded")

                elif cmd == 'zones':
                    print("  Zone centers (SOM row, col):")
                    for zone in ZONE_ANCHORS:
                        c = _zone_center(zone)
                        anchors = [w for w in ZONE_ANCHORS[zone] if w in brain.word_to_bmu]
                        if c:
                            print(f"    {zone:8s} → row={c[0]:.0f}, col={c[1]:.0f}"
                                  f"  anchors: {anchors}")
                        else:
                            print(f"    {zone:8s} → no anchors mapped yet")

                elif cmd == 'context':
                    print(f"  Context buffer ({len(_context_buffer)} turns):")
                    for i, (hw, rw) in enumerate(_context_buffer):
                        print(f"    [{i+1}] you={hw}  brain={rw}")

                elif cmd == 'imagination':
                    _IMAGINATION_ENABLED = not _IMAGINATION_ENABLED
                    state = 'on' if _IMAGINATION_ENABLED else 'off'
                    print(f"  Imagination: {state}")

                elif cmd == 'dream':
                    print("  Dreaming...", end='', flush=True)
                    brain.dream(n_sequences=50)
                    print(" done.")

                elif cmd == 'help':
                    words = sorted(VOCABULARY.keys())
                    print(f"  {len(words)} grounded words:")
                    for i in range(0, len(words), 10):
                        print("   ", ' '.join(words[i:i + 10]))

                elif cmd == 'save':
                    brain.save(BRAIN_FILE)
                    print(f"  Saved to {BRAIN_FILE}")

                else:
                    print("  Commands: /state /zones /context /dream /imagination /help /save /quit")

                continue

            # ── Parse input ───────────────────────────────────────────────
            tokens = raw.lower().translate(_PUNCT).split()

            # ── Feedback shortcut ─────────────────────────────────────────
            if len(tokens) == 1 and tokens[0] in _FEEDBACK_POSITIVE:
                apply_feedback(positive=True)
                _total_turns += 1
                continue

            if len(tokens) == 1 and tokens[0] in _FEEDBACK_NEGATIVE:
                apply_feedback(positive=False)
                _total_turns += 1
                continue

            if tokens and tokens[0] in _FEEDBACK_POSITIVE:
                apply_feedback(positive=True)
            elif tokens and tokens[0] in _FEEDBACK_NEGATIVE:
                apply_feedback(positive=False)

            heard_words = [w for w in tokens if w in VOCABULARY]

            # ── Semantic learning ─────────────────────────────────────────
            semantic.learn_from_sentence(tokens)

            # ── Feed input through acoustic SOM ──────────────────────────
            _last_heard_bmus = feed_input(heard_words)

            for _ in range(4):
                brain.hear(SILENCE)
                brain.step()

            # ── Generate response via intent bridge ───────────────────────
            response = generate_response(_last_heard_bmus, heard_words)

            if response:
                with print_lock:
                    sys.stdout.write(f"\rbrain: {response}\n")
                    sys.stdout.flush()
                response_words = response.split()
                _last_response_words = response_words

                hear_own_response(response_words, reward=0.0)

                _recent_responses.append(response)
                if len(_recent_responses) > _MAX_RECENT:
                    _recent_responses.pop(0)

                _turn_history.append({'you': raw, 'brain': response})
                _context_buffer.append((heard_words, response_words))
            else:
                with print_lock:
                    sys.stdout.write("\rbrain: ...\n")
                    sys.stdout.flush()
                _last_response_words = []
                _context_buffer.append((heard_words, []))

            # Recompute zone for module updating
            zone = _resolve_zone(heard_words, _last_heard_bmus)

            # Module updates — order matters
            heard_bmu = _last_heard_bmus[0] if _last_heard_bmus else 0
            novelty   = curiosity.novelty_score(heard_bmu)
            surprise  = abs(novelty - 0.5)  # proxy: very novel OR very familiar = surprise

            wm.update(bmu=heard_bmu, heard_words=heard_words,
                      word_to_bmu=getattr(brain, 'word_to_bmu', {}))

            curiosity.update(bmu=heard_bmu, zone=zone)

            neuromod.update(novelty=novelty, surprise=surprise, reward=0.0)

            selfmodel.update(zone=zone, reward=0.0,
                             wm_strength=wm.activation_strength(),
                             episode_count=episodic.count())

            temporal.update(zone=zone, reward=0.0)

            episodic.record(
                you_words   = heard_words,
                brain_words = response.split() if response else [],
                zone        = zone,
                drives      = {'hunger': selfmodel._state['hunger'],
                               'fatigue': selfmodel._state['fatigue']},
                reward      = 0.0,
            )

            _total_turns += 1

            if _total_turns % _DREAM_EVERY == 0:
                brain.dream(n_sequences=10)
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopping live session...")

    # ── Exit ──────────────────────────────────────────────────────────
    if _total_turns > 0:
        print("\nSaving brain...", end='', flush=True)
        brain.save(BRAIN_FILE)
        print(" done.")
    else:
        print("\n(No turns played — brain unchanged.)")
    if interrupted:
        print("Exited on Ctrl+C.")
    print(f"Total turns: {_total_turns}")


if __name__ == '__main__':
    main()
