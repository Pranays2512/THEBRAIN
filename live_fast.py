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

from brain_fast import FastBrain, SILENCE, N_MFCC, SOM_COLS, SOM_ROWS
from vocab_core import VOCABULARY
from m75_semantic import SemanticMemory
from brain_modules import (WorkingMemory, EpisodicMemory, SelfModel,
                            CuriosityModule, NeuromodSystem, TemporalContext,
                            PrefrontalCortex, LogicModule, UserModel)
from global_workspace import GlobalWorkspace

BRAIN_FILE = 'brain_fast.pkl'
SEMANTIC_FILE = 'semantic.json'
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
semantic = SemanticMemory.load(SEMANTIC_FILE)
print(f"  Semantic memory: {len(semantic._facts)} facts")

wm        = WorkingMemory(n_neurons=5000)
episodic  = EpisodicMemory(maxlen=50)
selfmodel = SelfModel()
usermodel = UserModel()
curiosity = CuriosityModule(n_neurons=5000)
neuromod  = NeuromodSystem()
temporal  = TemporalContext()
pfc       = PrefrontalCortex()
logic     = LogicModule()
gw        = GlobalWorkspace()
_gw_out: dict = {}          # last broadcast — used by background tick
_steps_since_reward: int = 0  # turns since last positive feedback

# Persistent user memory: recall known user name from previous sessions.
_known_user_name: str | None = semantic._latest_relation_object('you', 'name')
if _known_user_name:
    usermodel._name = _known_user_name
    print(f"  [remembered: user = {_known_user_name}]")

# Seed curiosity zone_counts with approximate World6 experience priors.
# Brain experienced food/danger/rest/action heavily — those are NOT novel.
# Social/water are genuinely least experienced → curiosity should target them.
_WORLD6_ZONE_PRIORS = {
    'food':   500, 'danger': 300, 'rest': 400,
    'action': 350, 'pain':   200, 'water': 20, 'social': 5,
}
for _z, _cnt in _WORLD6_ZONE_PRIORS.items():
    curiosity._zone_counts[_z] = _cnt

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

# ── Pavlovian word→drive conditioning map ────────────────────────────────────
# Hearing these words nudges internal drive state — classical conditioning.
# Small deltas accumulate if topic persists; can trigger spontaneous expression.
_PAVLOV_MAP: dict[str, tuple[str, float]] = {
    # Food cues: small appetitive spike (cue arouses appetite briefly)
    # but zone relief (-0.12/turn in SelfModel) dominates when actually in food zone.
    'food':   ('hunger',  +0.02),
    'eat':    ('hunger',  +0.01),
    'hungry': ('hunger',  +0.03),
    'drink':  ('hunger',  +0.01),
    'full':   ('hunger',  -0.08),  # "full" strongly relieves hunger
    'water':  ('comfort', +0.03),
    'danger': ('fear',    +0.10),
    'afraid': ('fear',    +0.08),
    'hurt':   ('fear',    +0.06),
    'pain':   ('fear',    +0.08),
    'wall':   ('fear',    +0.04),
    'run':    ('fear',    +0.05),
    'tired':  ('fatigue', +0.05),
    'sleep':  ('fatigue', -0.03),
    'calm':   ('comfort', +0.05),
    'safe':   ('comfort', +0.06),
    'good':   ('comfort', +0.04),
    'happy':  ('comfort', +0.05),
}

# ── Active vocabulary acquisition ────────────────────────────────────────────
# When brain hears unknown word it asks "what is X?". On next turn it stores
# the definition in semantic memory and synthesizes a vocab vector so the word
# can be used in future zone routing without retraining.
_pending_unknown: str | None = None   # word brain just asked about
_pending_unknown_turns: int  = 0      # turns since question was asked (auto-reset)
_awaiting_resolution: bool  = False   # Phase 3: brain asked a question, waiting for answer

_SKIP_TOKENS = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for',
                'it', 'this', 'that', 'they', 'he', 'she', 'was', 'are',
                'with', 'from', 'by', 'do', 'did', 'does', 'have', 'has',
                'okay', 'ok', 'fine', 'well', 'great', 'sure', 'please',
                'just', 'really', 'very', 'much', 'also', 'still', 'ever',
                # Zone names that aren't in VOCABULARY but brain "knows" experientially
                'rest', 'action', 'social', 'zone', 'feel', 'feeling', 'think', 'thinking'}

def _synthesize_vocab_entry(word: str, definition_tokens: list[str]) -> bool:
    """
    Average known word vectors from definition → add word to VOCABULARY.
    Also assigns zone routing so the new word participates in zone detection
    and can appear in generated responses — partial fix for grounding/conv split.
    """
    known = [t for t in definition_tokens if t in VOCABULARY and t not in _HOLLOW]
    if not known:
        return False
    avg = np.mean([VOCABULARY[t][0] for t in known], axis=0)
    VOCABULARY[word] = (avg, 2)

    # Zone routing: vote from definition words' zones.
    # New word inherits the dominant zone of its definition — not perfect grounding
    # but allows zone-routing and expression without touching frozen SOM.
    zone_votes: dict[str, float] = {}
    for t in known:
        z = _WORD_TO_ZONE.get(t)
        if z:
            zone_votes[z] = zone_votes.get(z, 0) + 1.0
    if zone_votes:
        dominant = max(zone_votes, key=zone_votes.get)
        _WORD_TO_ZONE[word] = dominant
        if word not in ZONE_EXPRESSION[dominant]:
            ZONE_EXPRESSION[dominant].append(word)
        if word not in ZONE_ANCHORS[dominant]:
            ZONE_ANCHORS[dominant].append(word)

    # BMU mapping: use BMU of strongest known word so zone detection works
    best_known = max(known, key=lambda t: zone_votes.get(_WORD_TO_ZONE.get(t, ''), 0))
    if best_known in brain.word_to_bmu:
        brain.word_to_bmu[word] = brain.word_to_bmu[best_known]

    return True

# ── Conversation state ────────────────────────────────────────────────────────

class ConversationState:
    """
    Between-turn memory: topic, entities, speaker focus, intent history, follow-up.
    Never generates responses — only adds context seeds to the existing zone pipeline.
    """
    _FOLLOWUP_OPENERS = {'and', 'but', 'also', 'so', 'then'}
    _USER_STATE_VERBS = {'feel', 'feels', 'am', 'felt', 'need', 'want', 'have'}
    _TEACH_PATTERNS   = {'is', 'name', 'called', 'means'}

    def __init__(self):
        self.topic: str = 'social'
        self.pending_question: str | None = None
        self.recent_entities: list[str] = []
        self.last_relation_hit: bool = False
        self.last_zones: deque = deque(maxlen=4)
        self._zone_repeat_count: int = 0
        # ── New fields ──────────────────────────────────────────────────
        self.intent: str = 'unknown'        # last classified high-level intent
        self.speaker_focus: str = 'neutral' # 'user' | 'self' | 'neutral'
        self.follow_up: bool = False        # True when this turn continues last topic
        self.turn_history: list[dict] = []  # last 5 {input, response, intent, zone}

    # ── Speaker focus ──────────────────────────────────────────────────

    @staticmethod
    def _classify_speaker(tokens: list[str]) -> str:
        """
        'user'  — user is reporting their own state  ("i feel X", "i am tired")
        'self'  — user is asking about the brain     ("how are you", "what do you feel")
        'neutral' — bare word / action / other
        """
        t = set(tokens)
        has_i   = tokens and tokens[0] in ('i', 'i\'m')
        has_you = 'you' in t
        has_state = bool(t & ConversationState._USER_STATE_VERBS)

        if has_i and has_state and not has_you:
            return 'user'
        if has_you and has_state:
            return 'self'
        return 'neutral'

    # ── Intent classification ──────────────────────────────────────────

    @staticmethod
    def _classify_intent(tokens: list[str], zone: str) -> str:
        t = set(tokens)
        if not tokens:
            return 'unknown'
        if t & {'hi', 'hello', 'bye'}:
            return 'greeting'
        if 'who' in t and 'you' in t:
            return 'self_query'
        if 'who' in t and 'am' in t:
            return 'user_query'
        if ('how' in t or 'what' in t) and 'you' in t:
            return 'self_query'
        if tokens[0] in ('i', "i'm") and t & ConversationState._USER_STATE_VERBS:
            return 'user_state'
        if 'wrong' in t or ('no' in t and len(tokens) <= 2):
            return 'correction'
        if ('my' in t or 'is' in t) and t & ConversationState._TEACH_PATTERNS:
            return 'teach'
        if t & {'why', 'what', 'where', 'how'} and len(tokens) >= 2:
            return 'fact_query'
        if zone == 'action':
            return 'action'
        return 'unknown'

    # ── Update ─────────────────────────────────────────────────────────

    def update(self, zone: str, heard_words: list, relation_hit: bool,
               hollow_set: set, question: bool = False):
        # Follow-up: opener word + <= 4 tokens inherits previous topic
        self.follow_up = (bool(heard_words) and
                          heard_words[0] in self._FOLLOWUP_OPENERS and
                          len(heard_words) <= 4)

        if zone == self.topic:
            self._zone_repeat_count += 1
        else:
            self._zone_repeat_count = 0

        self.topic = zone
        self.last_zones.append(zone)
        self.last_relation_hit = relation_hit
        self.speaker_focus = self._classify_speaker(heard_words)
        self.intent = self._classify_intent(heard_words, zone)

        for w in heard_words:
            if w not in hollow_set:
                if w not in self.recent_entities:
                    self.recent_entities.insert(0, w)
        self.recent_entities = self.recent_entities[:6]

        if question:
            self.pending_question = ' '.join(heard_words[:4])
        elif relation_hit:
            self.pending_question = None

    def record_turn(self, input_words: list[str], response: str):
        self.turn_history.append({
            'input': input_words, 'response': response,
            'intent': self.intent, 'zone': self.topic,
        })
        if len(self.turn_history) > 5:
            self.turn_history.pop(0)

    # ── Queries ────────────────────────────────────────────────────────

    def topic_consistent(self) -> bool:
        return len(self.last_zones) >= 3 and len(set(list(self.last_zones)[-3:])) == 1

    def entity_seeds(self, vocabulary: set) -> list[str]:
        return [w for w in self.recent_entities[:3] if w in vocabulary]

    def last_response(self) -> str:
        return self.turn_history[-1]['response'] if self.turn_history else ''

    def last_input_words(self) -> list[str]:
        return self.turn_history[-1]['input'] if self.turn_history else []


conv_state = ConversationState()

_recent_responses: list[str] = []
_MAX_RECENT       = 5          # extended: anti-repetition across 5 turns
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
_predicted_bmu:       int   = 0    # TP prediction of next input BMU
_last_pe:             float = 0.0  # prediction error from last turn [0,1]

# ── Conversation context buffer (lightweight turn memory) ─────────────────────
# Stores (heard_words, response_words) for the last N turns.
# Fed back into generate_response so the brain can track topic continuity
# and avoid repeating the same response to the same input.
_context_buffer: deque = deque(maxlen=3)   # last 3 turns of (heard, response)
_last_zone: str = 'social'                 # zone from previous turn — topic continuity prior


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT BRIDGE — grounded zone detection + expression
# ═══════════════════════════════════════════════════════════════════════════════

# Zone definitions: which words anchor each zone's center on the SOM.
# These are grounded words — their BMU positions were shaped by World6 events.
ZONE_ANCHORS = {
    'food':      ['food', 'eat', 'hungry', 'full', 'found', 'near', 'search'],
    'water':     ['water', 'river', 'rain', 'wet', 'drink'],
    'pain':      ['hurt', 'pain', 'wall', 'bad', 'stop', 'no', 'cold', 'dark', 'worry', 'lost'],
    'rest':      ['tired', 'sleep', 'calm', 'warm', 'soft', 'awake', 'plant', 'tree', 'grass', 'sun', 'still', 'light', 'free'],
    'action':    ['go', 'move', 'push', 'open', 'come', 'door', 'button', 'wind', 'air', 'sky', 'try', 'look', 'see', 'out', 'run', 'free'],
    'social':    ['hi', 'hello', 'bye', 'yes', 'happy', 'help', 'sorry', 'talk', 'listen', 'with', 'alone'],
    'danger':    ['afraid', 'danger', 'careful', 'run', 'dark'],
    'cognitive': ['think', 'dream', 'imagine', 'wonder', 'sense', 'memory', 'aware',
                  'mind', 'real', 'exist', 'learn', 'know', 'remember', 'if', 'but', 'when'],
}

# Zone → what the brain expresses when in this zone.
# RULE: each word appears in AT MOST ONE zone's expression list.
# Generic words (good, feel, here, want) are deliberately removed
# to force the brain to speak in specific grounded vocabulary.
ZONE_EXPRESSION = {
    'food':      ['eat', 'food', 'hungry', 'want', 'full', 'found', 'near', 'far'],
    'water':     ['water', 'drink', 'wet', 'river', 'rain', 'calm', 'good'],
    'pain':      ['stop', 'hurt', 'pain', 'bad', 'no', 'cold', 'dark', 'worry', 'lost', 'again'],
    'rest':      ['sleep', 'calm', 'tired', 'warm', 'awake', 'safe', 'alive', 'plant', 'tree', 'grass', 'sun', 'green', 'still', 'light', 'free', 'after'],
    'action':    ['go', 'open', 'move', 'push', 'door', 'button', 'come', 'wind', 'air', 'sky', 'try', 'look', 'see', 'out'],
    'social':    ['hello', 'hi', 'happy', 'help', 'yes', 'sorry', 'like', 'pranay', 'talk', 'listen', 'with', 'who', 'how'],
    'danger':    ['afraid', 'careful', 'run', 'danger', 'dark', 'worry'],
    'cognitive': ['think', 'feel', 'know', 'dream', 'imagine', 'wonder', 'sense',
                  'memory', 'aware', 'mind', 'real', 'exist', 'learn', 'remember'],
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
    'like': 'social',  'pranay': 'social', 'remember': 'rest',
    'hate': 'pain',    'hurts': 'pain',    'helps': 'food',
    'think': 'cognitive', 'learn': 'cognitive', 'brain': 'cognitive',
    'know':  'cognitive', 'feel':  'cognitive', 'remember': 'cognitive',
    'dream': 'cognitive', 'imagine': 'cognitive', 'wonder': 'cognitive',
    'sense': 'cognitive', 'memory': 'cognitive', 'aware': 'cognitive',
    'mind':  'cognitive', 'real':   'cognitive', 'exist': 'cognitive',
    'am': 'social',
    'why': 'cognitive', 'because': 'cognitive', 'so': 'cognitive',
    'cause': 'cognitive', 'then': 'cognitive', 'if': 'cognitive', 'but': 'cognitive',
    'when': 'cognitive',
    'wrong': 'pain',   'search': 'action',
    # Environmental words
    'plant': 'rest',  'tree': 'rest',  'grass': 'rest', 'green': 'rest',
    'sun':   'rest',  'warm': 'rest',
    'river': 'water', 'rain': 'water', 'wet': 'water',  'water': 'water',
    'wind':  'action','air':  'action','sky':  'action',
    'cold':  'pain',
    # Spatial words
    'near': 'food', 'far': 'food', 'found': 'food', 'lost': 'pain',
    'out': 'action', 'in': 'action', 'free': 'rest',
    # Temporal words
    'after': 'rest', 'before': 'social', 'again': 'pain', 'still': 'rest',
    # Social interaction
    'talk': 'social', 'listen': 'social', 'alone': 'social', 'with': 'social',
    # Sensory/action
    'look': 'action', 'see': 'action', 'try': 'action',
    # Light/dark/mood
    'dark': 'danger', 'light': 'rest', 'worry': 'pain',
    # Question words
    'where': 'action', 'who': 'social', 'how': 'social',
    # Quantifiers
    'all': 'social', 'never': 'social', 'always': 'social',
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
    # "who are you" / "what are you" → brain identifies as fastbrain
    'who_are_you': {
        'zone': 'social',
        'required': {'who'},
        'any_of': {'you', 'are'},
        'seeds': ['fastbrain', 'name', 'am'],
        'template': ['fastbrain', 'name'],
    },
    # "who am i" → user identity (pranay from semantic memory, or 'you pranay')
    'who_am_i': {
        'zone': 'social',
        'required': {'who'},
        'any_of': {'am'},
        'seeds': ['you', 'pranay', 'name'],
        'template': ['you', 'pranay'],
    },
}

_ZONE_NEUTRAL = {'i', 'me', 'you', 'we', 'and', 'is', 'am', 'not', 'now',
                 'very', 'little',
                 'good', 'bad', 'safe', 'calm', 'alive', 'free', 'still', 'better',
                 'know', 'think', 'feel', 'want', 'need', 'like', 'yes', 'no'}
_HOLLOW = {'i', 'me', 'we', 'you', 'and', 'is', 'not',
           'why', 'how', 'what', 'when', 'where', 'who', 'which',
           'because', 'so', 'then', 'cause',
           'pranay', 'here', 'do', 'did', 'does'}

_QUESTION_OPENERS = {'who', 'what', 'when', 'where', 'why', 'how', 'which',
                     'is', 'are', 'do', 'does', 'can', 'will', 'would', 'did'}

def _is_question(raw_tokens: list[str]) -> bool:
    """True if raw_tokens look like a question — opener word or trailing ?"""
    if not raw_tokens:
        return False
    if raw_tokens[0] in _QUESTION_OPENERS:
        return True
    return raw_tokens[-1].endswith('?')



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

def _relation_response(tokens: list[str]) -> str | None:
    """
    Only answers identity facts the GRU could never have learned (names).
    Everything else — food, wall, danger, causal chains — goes to generate_response.
    """
    t = set(tokens)
    # "what is my name" / "who am i"
    if ('name' in t and 'my' in t) or tokens == ['who', 'am', 'i']:
        obj = semantic._latest_relation_object('you', 'name')
        if obj:
            return f"your name is {obj}"
    # "what is your name"
    if 'name' in t and ('your' in t or 'brain' in t) and 'my' not in t:
        obj = semantic._latest_relation_object('i', 'name')
        if obj:
            return f"my name is {obj}"
    return None


# Drive → (selfmodel key, [(upper_threshold, answer_words), ...])
# Thresholds are exclusive upper bounds on drive value.
_STATE_QUERY_MAP: dict[str, tuple[str, list]] = {
    'hungry': ('hunger',  [(0.15, ['no', 'full']),
                           (0.50, ['little', 'hungry']),
                           (1.01, ['yes', 'hungry', 'want', 'food'])]),
    'tired':  ('fatigue', [(0.20, ['no', 'awake']),
                           (0.55, ['little', 'tired']),
                           (1.01, ['yes', 'tired', 'sleep'])]),
    'afraid': ('fear',    [(0.25, ['no', 'calm', 'safe']),
                           (0.55, ['little', 'afraid']),
                           (1.01, ['yes', 'afraid', 'careful'])]),
    'hurt':   ('fear',    [(0.25, ['no', 'safe', 'good']),
                           (1.01, ['yes', 'hurt', 'pain'])]),
    'full':   ('hunger',  [(0.30, ['yes', 'full', 'good']),
                           (1.01, ['no', 'hungry', 'want'])]),
    'calm':   ('fear',    [(0.40, ['yes', 'calm', 'safe']),
                           (1.01, ['no', 'afraid'])]),
    'good':   ('comfort', [(0.30, ['little', 'good']),
                           (1.01, ['yes', 'good', 'calm'])]),
    'sad':    ('comfort', [(0.25, ['yes', 'sad']),
                           (1.01, ['no', 'calm', 'good'])]),
}

# dominant_state → honest words (grounded vocab only)
_DOMINANT_TO_WORDS: dict[str, list[str]] = {
    'uncertain': ['search', 'know', 'what'],
    'afraid':    ['afraid', 'careful', 'stop'],
    'hungry':    ['hungry', 'want', 'food'],
    'tired':     ['tired', 'sleep', 'slow'],
    'content':   ['good', 'calm', 'full'],
    'curious':   ['awake', 'look', 'new'],
    'calm':      ['calm', 'good', 'safe'],
}

# highest drive → what brain wants
_DRIVE_WANT_WORDS: dict[str, list[str]] = {
    'hunger':  ['eat', 'food', 'want'],
    'fatigue': ['sleep', 'rest', 'calm'],
    'fear':    ['calm', 'safe', 'stop'],
    'comfort': ['calm', 'good', 'rest'],
}

def _state_query_response(tokens: list[str]) -> str | None:
    """
    Intercept questions about brain's internal state.
    Specific: 'are you hungry?' → check drive → truthful answer.
    Open:     'how do you feel?' → dominant_state() → words.
              'what do you want?' → highest drive → words.
    Returns None if not a state query directed at brain.
    """
    t = set(tokens)
    if 'you' not in t:
        return None

    # Open feel query: "how do you feel" / "how are you"
    if ('how' in t and ('feel' in t or 'are' in t or 'feeling' in t)):
        dom = selfmodel.dominant_state()
        words = [w for w in _DOMINANT_TO_WORDS.get(dom, ['calm']) if w in VOCABULARY]
        return _apply_grammar(words[:3], 'social') if words else None

    # Open want/need query: "what do you want" / "what do you need"
    if 'what' in t and ('want' in t or 'need' in t):
        s = selfmodel._state
        # Pick highest drive (excluding curiosity/uncertainty — those aren't "wants")
        ranked = sorted(
            [(v, k) for k, v in s.items() if k in _DRIVE_WANT_WORDS],
            reverse=True
        )
        if ranked:
            _, top_drive = ranked[0]
            words = [w for w in _DRIVE_WANT_WORDS[top_drive] if w in VOCABULARY]
            return _apply_grammar(words[:3], top_drive) if words else None

    # Specific state word query: "are you hungry/tired/afraid/..."
    for state_word, (drive_key, thresholds) in _STATE_QUERY_MAP.items():
        if state_word not in t:
            continue
        drive_val = selfmodel._state.get(drive_key, 0.5)
        for upper, words in thresholds:
            if drive_val < upper:
                answer = [w for w in words if w in VOCABULARY]
                return _apply_grammar(answer[:3], drive_key) if answer else None
    return None


def _resolve_zone(heard_words: list[str], heard_bmus: list[int]) -> str:
    """Resolve the current semantic zone, giving explicit intents priority."""
    intent = _detect_intent(heard_words)
    if intent is not None:
        return INTENT_SPECS[intent]['zone']

    _STRONG_ZONES = {'food', 'water', 'pain', 'rest', 'danger', 'action', 'cognitive'}
    zone_votes: dict[str, float] = {z: 0.0 for z in ZONE_ANCHORS}
    for w in heard_words:
        if w in _WORD_TO_ZONE:
            z = _WORD_TO_ZONE[w]
            zone_votes[z] += 2.0 if z in _STRONG_ZONES else 1.0

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
    # Person names and filler words that drift into responses from WTP training noise
    _PERSON_FILTER = {'pranay', 'raphael', 'fastbrain'}
    zone_ok = set(ZONE_EXPRESSION[zone]) | set(ZONE_ANCHORS[zone])
    result = []
    seen: set[str] = set()
    for w in words:
        if w in _PERSON_FILTER:
            continue
        if w in zone_ok or w in _ZONE_NEUTRAL:
            if w not in seen:
                result.append(w)
                seen.add(w)
        else:
            break
    # Drop trailing conjunctions/function words ("i feel and" → "i feel")
    while result and result[-1] in _TRAILING_DROP:
        result.pop()
    return result


_TEMPL_COGVERB = {'think','feel','know','dream','imagine','wonder','sense',
                  'learn','remember','exist','wake','hear','see','touch','aware'}
_TEMPL_COGNOUN = {'mind','memory','real'}  # "i have [noun]" pattern

_TEMPL_EVAL  = {'good','bad','safe','wrong','big','small','warm','cold','hard',
                'soft','fast','slow','full','empty','strong','weak','better','true',
                'false','near','far','wet','dry','dark','light','alive','dead','new',
                'old','hot','quiet','loud','open','closed','green','bright'}
_TEMPL_STATE = {'hungry','tired','afraid','hurt','thirsty','calm','happy','sad',
                'alone','awake','alive','lost','free','pain','worry','still',
                'awake','alert','safe','content','curious'}
_TEMPL_THING = {'food','water','wall','danger','pain','door','button','sun','rain',
                'tree','river','sky','wind','storm','grass','plant','fire','rock',
                'air','ground','body','home','world','place','thing','way','time'}
_TEMPL_WANT  = {'want','need','search','find','eat','drink','go','take','get','look'}
_TEMPL_PASS  = {'i','my','you','hello','hi','yes','no','stop','run','help','what',
                'why','how','who','where','maybe','sorry','bye','am','is','are'}

def _apply_grammar(words: list[str], zone: str) -> str:
    """
    Wrap GRU content words in minimal grammatical structure.
    GRU picks WHAT — this function adds IS/AM/FEEL connectives only.
    Passes through if words already start with a grammatical anchor.
    """
    if not words:
        return ''
    _LEADING_STRIP = {'and', 'or', 'then', 'of', 'with', 'the', 'a'}
    w = list(words)
    while w and w[0] in _LEADING_STRIP:
        w = w[1:]
    if not w:
        return ''
    w0 = w[0]

    # Already grammatical — pass through, but fix "i [state]" → "i am [state]"
    if w0 in _TEMPL_PASS:
        if w0 == 'i' and len(w) >= 2 and w[1] in _TEMPL_STATE and w[1] != 'am':
            return f"i am {' '.join(w[1:4])}"
        if w0 == 'i' and len(w) >= 2 and w[1] in _TEMPL_COGVERB:
            return f"i {' '.join(w[1:4])}"
        if w0 == 'i' and len(w) >= 2 and w[1] in _TEMPL_COGNOUN:
            return f"i have {w[1]}"
        return ' '.join(w[:5])

    # "[cognoun]..." → "i have [noun]"  ("memory" → "i have memory")
    if w0 in _TEMPL_COGNOUN:
        return f"i have {w0}"

    # "[cogverb]..." → "i [cogverb] [rest]"  ("dream memory think" → "i dream memory think")
    if w0 in _TEMPL_COGVERB:
        tail = [x for x in w[1:3] if x not in _TEMPL_PASS and x not in {'pranay', 'do', 'here'}]
        return f"i {w0} {' '.join(tail)}" if tail else f"i {w0}"

    # "[thing] [eval]..." → "[thing] is [eval]"
    if w0 in _TEMPL_THING and len(w) >= 2 and w[1] in _TEMPL_EVAL:
        return f"{w0} is {' '.join(w[1:3])}"

    # "[eval] [thing]..." → "[thing] is [eval]"  ("good food" → "food is good")
    if w0 in _TEMPL_EVAL and len(w) >= 2 and w[1] in _TEMPL_THING:
        return f"{w[1]} is {w0}"

    # "[state]..." → "i am [state] [rest]"  ("hungry want food" → "i am hungry want food")
    if w0 in _TEMPL_STATE:
        return f"i am {' '.join(w[:3])}"

    # "[want] [thing]..." → "i [want] [thing]"
    if w0 in _TEMPL_WANT and len(w) >= 2:
        obj_words = [x for x in w[1:3] if x in _TEMPL_THING or x in _TEMPL_EVAL]
        tail = ' '.join(obj_words) if obj_words else w[1]
        return f"i {w0} {tail}"

    # "[eval]..." alone → "that is [eval]"
    if w0 in _TEMPL_EVAL:
        return f"that is {' '.join(w[:2])}"

    # Zone-based prefix for bare thing words
    if w0 in _TEMPL_THING:
        _ZONE_WANT = {'food', 'water', 'action'}
        if zone in _ZONE_WANT:
            # second word only if it's also a thing or eval — avoid "i want food want"
            extra = f" {w[1]}" if len(w) >= 2 and w[1] in (_TEMPL_THING | _TEMPL_EVAL) else ""
            return f"i want {w0}{extra}"
        return f"{w0} is {w[1]}" if len(w) >= 2 and w[1] in _TEMPL_EVAL else w0

    return ' '.join(w[:5])


def generate_response(heard_bmus: list[int], heard_words: list[str],
                      raw_tokens: list[str] | None = None) -> str:
    """
    Intent-driven grounded response generation with turn memory.

    Step 0: Semantic memory lookup (factual queries answered directly).
    Step 1: Detect which semantic zone the input activated (grounded in SOM).
    Step 2: Select expression seeds from that zone (what the brain wants to say).
    Step 3: WordTP generates the actual word sequence (how to say it).
    Step 4: Clip output to stay within the detected zone.
    """
    global _last_zone, _last_response_words, _recent_responses

    if not heard_bmus and not heard_words:
        return ''

    def _commit(resp: str) -> str:
        """Update turn memory and return response."""
        global _last_response_words, _recent_responses
        _last_response_words = resp.split() if resp else []
        if resp:
            _recent_responses.append(resp)
            if len(_recent_responses) > _MAX_RECENT:
                _recent_responses.pop(0)
        return resp

    # Step 0: semantic memory — factual queries answered without zone routing.
    tokens = raw_tokens or heard_words
    sem_ans = _relation_response(tokens)
    if sem_ans is not None:
        return _commit(sem_ans)

    # Step 1a: detect explicit grounded intent before generic zone routing.
    intent = _detect_intent(heard_words)

    if intent is not None:
        zone = INTENT_SPECS[intent]['zone']
        _last_zone = zone
        seeds = [w for w in INTENT_SPECS[intent]['seeds'] if w in VOCABULARY]
    else:
        # Step 1b: direct word→zone lookup. Strong experiential zones outweigh pronouns.
        _STRONG_ZONES = {'food', 'water', 'pain', 'rest', 'danger', 'action', 'cognitive'}
        zone_votes: dict[str, float] = {z: 0.0 for z in ZONE_ANCHORS}
        for w in heard_words:
            if w in _WORD_TO_ZONE and w not in _HOLLOW:
                z = _WORD_TO_ZONE[w]
                zone_votes[z] += 2.0 if z in _STRONG_ZONES else 1.0

        total_signal = sum(zone_votes.values())
        if total_signal >= 1.0:
            zone = max(zone_votes, key=zone_votes.get)
        elif total_signal > 0:
            # Weak signal — blend last zone as prior (topic continuity)
            zone_votes[_last_zone] = zone_votes.get(_last_zone, 0.0) + 0.8
            zone = max(zone_votes, key=zone_votes.get)
        else:
            # No vocab signal at all — inherit last zone or fall back to BMU
            if _last_zone and _last_zone != 'social':
                zone = _last_zone
            else:
                clean_bmus = [brain.word_to_bmu[w] for w in heard_words if w in brain.word_to_bmu]
                zone = _detect_zone(clean_bmus if clean_bmus else heard_bmus)

        # Zone seeds are always the base when no specific intent fired.
        _last_zone = zone
        seeds = [w for w in ZONE_EXPRESSION[zone] if w in VOCABULARY]

        # Social zone: brain discloses current internal state — no question detection.
        # Uncertainty state surfaces as 'search'/'what' seeds, not hardcoded responses.
        if zone == 'social':
            s = selfmodel._state
            dom = selfmodel.dominant_state()
            _dom_map = {
                'uncertain': ('search', 'uncertainty'),
                'afraid':    ('afraid', 'fear'),
                'hungry':    ('hungry', 'hunger'),
                'tired':     ('tired',  'fatigue'),
                'content':   ('calm',   'comfort'),
                'calm':      ('calm',   'comfort'),
                'curious':   ('awake',  'curiosity'),
            }
            sw, dk = _dom_map.get(dom, ('calm', 'comfort'))
            dv = s.get(dk, 0.5)
            if dom == 'uncertain':
                # Uncertainty replaces social seeds entirely so 'search'/'what'/'know'
                # aren't drowned by 'hello'/'hi'/'happy' in WordTP context.
                seeds = [w for w in ['search', 'what', 'know'] if w in VOCABULARY]
            else:
                state_seeds = [w for w in ['i', 'am'] if w in VOCABULARY]
                if dv >= 0.85 and 'very' in VOCABULARY:
                    state_seeds.append('very')
                elif dv < 0.50 and 'little' in VOCABULARY:
                    state_seeds.append('little')
                if sw in VOCABULARY:
                    state_seeds.append(sw)
                seeds = state_seeds + seeds

        # Environmental word overrides: when input is a nature/env word, prepend
        # env-specific seeds so Word TP produces plant/water words not sleep/tired.
        _ENV_VEGETATION = {'plant', 'tree', 'grass', 'green', 'sun', 'warm'}
        _ENV_WATER      = {'river', 'rain', 'wet', 'water'}
        _ENV_AIR        = {'wind', 'air', 'sky'}
        hw_set = set(heard_words)
        if hw_set & _ENV_VEGETATION and zone == 'rest':
            seeds = [w for w in ['plant', 'tree', 'calm', 'green', 'alive', 'warm']
                     if w in VOCABULARY] + seeds
        elif hw_set & _ENV_WATER and zone == 'water':
            seeds = [w for w in ['water', 'drink', 'river', 'wet', 'calm']
                     if w in VOCABULARY] + seeds
        elif hw_set & _ENV_AIR and zone == 'action':
            seeds = [w for w in ['wind', 'air', 'sky', 'go', 'free']
                     if w in VOCABULARY] + seeds

        # Raw-token identity overrides: prepend identity seeds when raw input
        # contains identity/cognitive words not in vocab (who, think, learn, are).
        # Response still generated by WordTP — not hardcoded.
        if raw_tokens:
            rt = set(raw_tokens)
            if ('who' in rt or ('what' in rt and 'are' in rt)) and 'you' in rt:
                seeds = [w for w in ['fastbrain', 'am', 'think', 'feel', 'know']
                         if w in VOCABULARY] + seeds
                zone = 'social'
            elif 'think' in rt and 'you' in rt:
                seeds = [w for w in ['think', 'learn', 'know', 'feel']
                         if w in VOCABULARY] + seeds
                zone = 'social'
            elif 'learn' in rt and 'you' in rt:
                seeds = [w for w in ['learn', 'think', 'know', 'feel']
                         if w in VOCABULARY] + seeds
                zone = 'social'

    # ── Conversation state: speaker focus + follow-up topic continuity ──────────
    #
    # speaker_focus == 'user'  → user reported own state ("i feel hungry")
    #   Brain should respond with action/comfort directed AT user, not self-report.
    #   Concretely: strip self-model seeds (which express brain's own state) and
    #   boost action/comfort seeds relevant to what the user said.
    #
    # speaker_focus == 'self'  → user asked about brain ("how are you")
    #   Brain should self-report. Self-model seeds are already present (social zone).
    #   No change needed — default pipeline handles this.
    #
    # follow_up == True → "and X", "but X", "so X" extends previous topic
    #   Pull last 2 entities from turn history as additional seeds.

    # Compute speaker focus for current tokens (conv_state.speaker_focus is from previous turn).
    _current_speaker_focus = ConversationState._classify_speaker(tokens)
    directive_words: list[str] = []  # populated below if speaker_focus == 'user'

    _input_negated = 'not' in set(tokens)
    if _current_speaker_focus == 'user' and intent is None and not _input_negated:
        # User stated their own state — respond with advice/action, not self-report.
        # Skip when negated ("i am NOT hungry") — user denying state, no directive needed.
        _USER_DIRECTIVES = {
            'food':   ['food', 'eat', 'go', 'find'],
            'rest':   ['sleep', 'calm', 'safe'],
            'danger': ['run', 'careful', 'go'],
            'pain':   ['stop', 'safe', 'calm'],
            'social': ['good', 'calm', 'with'],
        }
        directive_words = [w for w in _USER_DIRECTIVES.get(zone, []) if w in VOCABULARY]
        if directive_words:
            seeds = directive_words + [w for w in seeds if w not in directive_words]

    if conv_state.follow_up and conv_state.turn_history:
        # Inherit topic: pull content words from last input that are in vocab
        prev_input = conv_state.last_input_words()
        inherit = [w for w in prev_input
                   if w in VOCABULARY and w not in _HOLLOW and w not in seeds]
        seeds = seeds + inherit[:2]

    # Phase 2 — WM Seeking: uncertainty is active and brain is looking for resolution.
    # Suppress when user is reporting own state — seeking would override directive seeds.
    if wm._seeking and _current_speaker_focus != 'user':
        seek_words = [w for w in ['what', 'search', 'know'] if w in VOCABULARY]
        seeds = seek_words + seeds

    # Logic inference: drive + percept + 2-hop relation chains → conclusion words.
    # Conclusions prepend to seeds so WordTP expresses reasoned intent, not just zone reflex.
    logic_words = logic.infer(selfmodel, semantic, zone, heard_words, set(VOCABULARY), episodic)
    seeds = logic_words + seeds

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

    # Working memory: pull content words from what the USER said in last 2 turns
    # that match the current zone. Tracks topic continuity from user input only
    # (not brain's own output — avoids self-reinforcement loops).
    if _context_buffer:
        wm_candidates: list[str] = []
        for past_heard, _past_said in list(_context_buffer)[-2:]:
            for w in (past_heard or []):
                if (w in VOCABULARY and w not in _HOLLOW
                        and _WORD_TO_ZONE.get(w, zone) == zone):
                    wm_candidates.append(w)
        seeds = seeds + wm_candidates[:2]

    # Step 2b: Episodic memory — if topic changed, seed with what brain said last time
    # in this zone. This gives genuine conversational memory.
    if episodic.topic_changed():
        past = episodic.recall_similar(zone, n=1)
        if past and past[0]['brain']:
            prev_word = past[0]['brain'][0]
            if prev_word in VOCABULARY:
                seeds = [prev_word] + seeds

    # Dedup seeds preserving priority order (zone > PFC > self-model > episodic).
    _seen_s: set[str] = set()
    _deduped: list[str] = []
    for w in seeds:
        if w not in _seen_s:
            _deduped.append(w)
            _seen_s.add(w)
    seeds = _deduped

    # Conv-state entity seeds: recent content words from this topic thread.
    # Only added when topic is consistent — not during topic transitions.
    if conv_state.topic_consistent():
        for ew in conv_state.entity_seeds(set(VOCABULARY)):
            if ew not in _seen_s and _WORD_TO_ZONE.get(ew, zone) == zone:
                seeds.append(ew)
                _seen_s.add(ew)

    # ── WM gating: attention shapes generation ────────────────────────────────
    # Words whose BMU has high WM activation lead the seed list.
    # WM = what brain is actively thinking about right now.
    # High-WM words reach GRU context first → dominate generation.
    # Low-WM words deprioritized — brain doesn't generate from dormant concepts.
    def _wm_score(word: str) -> float:
        bmu = brain.word_to_bmu.get(word)
        return wm.get_activation(bmu) if bmu is not None else 0.0

    _WM_GATE = 0.12  # minimum activation to be "hot"
    seeds_hot  = [w for w in seeds if _wm_score(w) >= _WM_GATE]
    seeds_cold = [w for w in seeds if _wm_score(w) <  _WM_GATE]
    seeds = seeds_hot + seeds_cold  # hot words lead context → gate generation

    # Build the set of responses to avoid (last _MAX_RECENT turns).
    avoid: set[str] = set(_recent_responses)
    for _, past_said in _context_buffer:
        if past_said:
            avoid.add(' '.join(past_said))

    def _is_meaningful(clipped: list[str]) -> bool:
        return any(w not in _HOLLOW for w in clipped) if clipped else False

    # Zone-based forbidden set: words brain knows are false in this zone.
    # Passed to GRU as constrained decoding — brain stays silent on false things.
    forbidden = semantic.negated_words_for(zone)

    # Deliberate thought: generate N candidates at different temperatures,
    # score each on zone-match + content density + non-repetition, pick best.
    def _score_candidate(cand: list[str]) -> float:
        if not cand:
            return -1.0
        score = 0.0
        zone_ok = set(ZONE_EXPRESSION[zone]) | set(ZONE_ANCHORS[zone])
        for w in cand:
            if w in zone_ok:
                score += 0.3
        score += len([w for w in cand if w not in _HOLLOW]) * 0.2
        for w in cand:
            if w in heard_words:
                score += 0.1
        resp_str = ' '.join(cand)
        if resp_str in avoid or resp_str in _BLACKLISTED_RESPONSES:
            score -= 1.5
        return score

    # Deterministic boundary: strong grounded intents answer directly.
    if intent is not None:
        template = INTENT_SPECS[intent].get('template')
        if template:
            direct = _template_allowed(template, zone, avoid)
            if direct is not None:
                return _commit(direct)

    context = seeds[:3] + heard_words[:3]
    temperature = neuromod.temperature()

    candidates: list[list[str]] = []
    for t_offset in [0.0, 0.15, 0.35]:
        raw_words = brain.word_tp_generate(context, max_len=5,
                                           temperature=temperature + t_offset,
                                           forbidden=forbidden)
        if raw_words:
            clipped = _clip_to_zone(raw_words, zone)
            if _is_meaningful(clipped):
                candidates.append(clipped)

    # User-state directive: add a synthetic candidate from directive seeds so the
    # key content word (e.g. 'food') competes against WTP candidates under scoring.
    # Avoids post-hoc injection while ensuring the word can win via score.
    if _current_speaker_focus == 'user' and directive_words:
        dir_cand = [w for w in directive_words[:3] if w in VOCABULARY and w not in _HOLLOW]
        if dir_cand:
            candidates.append(dir_cand)

    if candidates:
        best = max(candidates, key=_score_candidate)
        if _score_candidate(best) > -0.5:
            resp = _apply_grammar(best, zone)
            if resp not in avoid:
                return _commit(resp)

    # Low-confidence fallback: zone seeds if available.
    # When seeking, 'what'/'search' carry meaning even if hollow.
    _hollow_filter = (_HOLLOW - {'what', 'search'}) if wm._seeking else _HOLLOW
    available = [w for w in seeds if w in brain.word_to_bmu and w not in _hollow_filter]
    if available:
        return _commit(_apply_grammar(available[:5], zone))

    return _commit('')


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
    """Feed the brain's own spoken words back through its acoustic SOM + word TP."""
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
        brain.word_tp.observe(word)   # GRU learns from brain's own words
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
        brain.word_tp.observe(word)   # GRU learns from user's words
    return bmus


_SOCIAL_POSITIVE_WORDS = ['good', 'happy', 'calm', 'yes']
_SOCIAL_NEGATIVE_WORDS = ['bad',  'stop',  'sad',  'no']

# Correction mode: set True after negative feedback → next user input is correction.
_correction_mode: bool = False
# Permanently blacklisted responses — never repeat even if scores well.
_BLACKLISTED_RESPONSES: set[str] = set()


def apply_feedback(positive: bool):
    """
    Social reinforcement: user's good/bad signal fires a reward spike
    AND feeds a grounded word chain through the SOM pipeline.

    Negative feedback now also:
      - permanently blacklists the wrong response,
      - decays the logic conclusion that drove it,
      - enters correction-seeking mode so next user input is absorbed as teaching.
    """
    global _last_response_words, _correction_mode
    if not _last_response_words:
        return

    if positive:
        global _steps_since_reward
        _steps_since_reward = 0
        wm.resolve_seeking()
        selfmodel.resolve_uncertainty(reward=1.0)
        hear_own_response(_last_response_words, reward=1.0)
        for word in _SOCIAL_POSITIVE_WORDS:
            if word in VOCABULARY:
                frames = say_frames(word, noise_std=0.06)
                for i, frame in enumerate(frames):
                    brain.hear(frame)
                    brain.step(reward=1.0 if i == len(frames) - 1 else 0.0)
                brain.hear(SILENCE)
                brain.step()
        print("  [brain: +reward → good happy calm]")
    else:
        # Withhold reward on wrong response
        hear_own_response(_last_response_words, reward=0.0)
        # Permanently blacklist — brain will never repeat this exact response
        _BLACKLISTED_RESPONSES.add(' '.join(_last_response_words))
        # Decay semantic relations that drove the wrong conclusion
        logic.decay_last_conclusion(semantic, amount=0.2)
        # Enter correction-seeking: next user turn absorbed as teaching
        _correction_mode = True
        wm.enter_seeking()
        selfmodel.spike_uncertainty(magnitude=0.3)
        # Feed negative chain through SOM
        for word in _SOCIAL_NEGATIVE_WORDS:
            if word in VOCABULARY:
                frames = say_frames(word, noise_std=0.06)
                for i, frame in enumerate(frames):
                    brain.hear(frame)
                    brain.step(reward=0.0)
                brain.hear(SILENCE)
                brain.step()
        print("  [brain: wrong — seeking]")


# ── Main loop (only runs when executed directly, not when imported) ───────────

def main():
    global _total_turns, _last_response_words, _last_heard_bmus, _IMAGINATION_ENABLED
    global _awaiting_resolution, _steps_since_reward, _gw_out
    global _predicted_bmu, _last_pe

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
    _now = time.time()
    _last_turn_time  = [_now]   # mutable for closure
    _last_curious_t  = _now     # overwrite the earlier init — load finished now
    _last_dream_t    = _now
    interrupted = False

    # ── Inner speech: brain hears its own generated words ─────────────────────
    # Critical: feeds words back through SOM+TP, not just prints.
    # This creates the self-feedback loop — brain shapes its own next state.
    def _inner_speech(words: list[str], label: str = 'thinks', reward: float = 0.01):
        if not words:
            return
        bmus = hear_own_response(words, reward=reward)
        if bmus:
            wm.update(bmu=bmus[0], heard_words=words,
                      word_to_bmu=getattr(brain, 'word_to_bmu', {}))
        with print_lock:
            sys.stdout.write("\r\033[K")
            print(f"[{label}]: {' '.join(words)}")
            sys.stdout.write("you: ")
            sys.stdout.flush()

    # Drive surfacing: track last time each drive was voiced (seconds since epoch)
    _drive_last_voiced: dict[str, float] = {
        'hunger': 0.0, 'fatigue': 0.0, 'fear': 0.0
    }
    _DRIVE_COOLDOWN   = 120.0
    _DREAM_INTERVAL   = 90.0
    _CURIOUS_COOLDOWN = 120.0
    _recent_thoughts: deque = deque(maxlen=8)  # dedup inner speech across all sources
    # Intrinsic reward per zone — brain self-reinforces grounded responses.
    # Strong zones (food/danger) earn more; social is weakest (less predictable).
    _ZONE_INTRINSIC_REWARD: dict = {
        'food': 0.15, 'danger': 0.15, 'water': 0.12,
        'pain': 0.10, 'rest':   0.10, 'action': 0.12, 'social': 0.05,
        'cognitive': 0.08,
    }
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
                # epsilon_boost from last GW broadcast modulates idle SOM steps
                _idle_reward = _gw_out.get('epsilon_boost', 0.0) * 0.5
                brain.step(reward=_idle_reward)

                # Passive state drift
                selfmodel.update('rest', 0.0, wm.activation_strength(), episodic.count())

                # Re-evaluate goal
                pfc.check_goal(selfmodel)

                # Check how long since user last spoke
                silence_s = time.time() - _last_turn_time[0]

                # ── Spontaneous drive expression ──────────────────────────
                # Fires when internal drive crosses threshold, regardless of
                # whether user is speaking. Cooldown prevents spamming.
                now_t = time.time()
                s = selfmodel._state
                _drive_chain = None
                _drive_key = None
                if (s['hunger'] > 0.75
                        and now_t - _drive_last_voiced['hunger'] > _DRIVE_COOLDOWN):
                    _drive_chain = ['hungry', 'want', 'food']
                    _drive_key = 'hunger'
                elif (s['fatigue'] > 0.75
                        and now_t - _drive_last_voiced['fatigue'] > _DRIVE_COOLDOWN):
                    _drive_chain = ['tired', 'sleep']
                    _drive_key = 'fatigue'
                elif (s['fear'] > 0.6
                        and now_t - _drive_last_voiced['fear'] > _DRIVE_COOLDOWN):
                    _drive_chain = ['afraid', 'careful']
                    _drive_key = 'fear'
                if silence_s < 30.0:
                    pass  # warmup — no inner speech until brain has settled
                elif _drive_chain and _drive_key:
                    _drive_last_voiced[_drive_key] = now_t
                    drive_expr = brain.word_tp_generate(
                        _drive_chain, max_len=4, temperature=0.9)
                    words_out = drive_expr or _drive_chain[:2]
                    key = tuple(words_out)
                    if key not in _recent_thoughts:
                        _recent_thoughts.append(key)
                        _inner_speech(words_out, label='brain', reward=0.02)

                elif now_t - _last_curious_t > _CURIOUS_COOLDOWN and silence_s > 20.0:
                    c_words = curiosity.curiosity_words(ZONE_ANCHORS,
                                                        getattr(brain, 'word_to_bmu', {}))
                    if c_words:
                        _last_curious_t = now_t
                        c_seeds = [w for w in c_words if w in VOCABULARY]
                        c_expr = brain.word_tp_generate(c_seeds, max_len=4,
                                                        temperature=neuromod.temperature())
                        words_out = c_expr or c_seeds[:2]
                        key = tuple(words_out)
                        if key not in _recent_thoughts:
                            _recent_thoughts.append(key)
                            _inner_speech(words_out, label='brain', reward=0.01)

                # Goal-driven thought + imagination — both feed back through SOM
                if ticks_idle % 3 == 0 and silence_s >= 30.0:
                    goal = pfc._active_goal
                    if goal and goal != _last_printed_goal[0]:
                        goal_words = [w for w in pfc.get_goal_seeds() if w in VOCABULARY]
                        gkey = tuple(goal_words[:3])
                        if gkey not in _recent_thoughts:
                            _recent_thoughts.append(gkey)
                            _inner_speech(goal_words[:3], label='thinks', reward=0.03)
                        _last_printed_goal[0] = goal
                    elif (_IMAGINATION_ENABLED and not goal and
                          ticks_idle % 9 == 0 and silence_s >= 30.0):
                        _last_printed_goal[0] = None
                        imagine_words = pfc.imagine(brain, episodic,
                                                    curiosity=curiosity,
                                                    zone_anchors=ZONE_ANCHORS)
                        if imagine_words:
                            ikey = tuple(imagine_words.split())
                            if ikey not in _recent_thoughts:
                                _recent_thoughts.append(ikey)
                                _inner_speech(imagine_words.split(), label='imagines', reward=0.01)

                # Dream replay — consolidate memory during long silence
                if silence_s >= 60.0 and now_t - _last_dream_t > _DREAM_INTERVAL:
                    _last_dream_t = now_t
                    brain.dream(n_sequences=20)

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
                    print(f"  Relation memory: {semantic.relation_count()}  "
                          f"contradictions: {semantic.contradiction_count()}")
                    for rel in semantic.recent_relations(5):
                        print(f"    {rel['subject']} -{rel['relation']}-> {rel['object']}"
                              f" [{rel['source']}]")
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
                    print(f"  {logic.explain()}")
                    gw.summary()

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
            global _correction_mode

            # ── Correction mode absorption ────────────────────────────────
            # When brain was wrong and user corrects → absorb as semantic teaching.
            # Must happen before feedback shortcuts so user correction text isn't
            # mistakenly treated as a feedback keyword.
            if _correction_mode and tokens:
                feedback_words = set(tokens) & (_FEEDBACK_POSITIVE | _FEEDBACK_NEGATIVE)
                if not feedback_words:
                    # User is teaching — absorb into semantic memory directly
                    semantic.learn_from_sentence(tokens)
                    wm.resolve_seeking()
                    _correction_mode = False
                    with print_lock:
                        sys.stdout.write("\r\033[K")
                        print("  [brain: correction absorbed]")
                        sys.stdout.write("you: ")
                        sys.stdout.flush()
                    _total_turns += 1
                    continue

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

            # ── Emotional contagion via UserModel ────────────────────────
            # User's expressed emotion nudges brain's OWN drives — not template response.
            # "i feel sad" → user dominant=sad → brain's comfort drops slightly →
            # brain responds from its own shifted state, not injected empathy words.
            # This is alive: response emerges from internal change, not scripted care.
            usermodel.update_from_words(tokens)
            _user_dom = usermodel.dominant_state()
            _s = selfmodel._state
            if _user_dom == 'sad':
                _s['comfort'] = max(0.0, _s['comfort'] - 0.06)
            elif _user_dom == 'afraid':
                _s['fear'] = min(1.0, _s['fear'] + 0.08)
            elif _user_dom == 'happy':
                _s['comfort'] = min(1.0, _s['comfort'] + 0.05)

            # ── Salience filter — sort heard_words by drive relevance ─────
            # Words that relate to active drives score higher and lead generation.
            # Brain attends to what matters given current internal state, not
            # equal-weight bag-of-words. High-salience words become first seeds.
            def _salience(word: str) -> float:
                score = 0.0
                s = selfmodel._state
                if word in ('food', 'eat', 'hungry', 'full'):
                    score += s['hunger'] * 2.0
                if word in ('danger', 'afraid', 'run', 'careful', 'hurt', 'pain'):
                    score += s['fear'] * 2.0
                if word in ('tired', 'sleep', 'rest', 'calm'):
                    score += s['fatigue'] * 2.0
                if word in ('water', 'drink', 'river'):
                    score += s['hunger'] * 0.8
                # Novelty boost: words the brain rarely hears rise in attention
                bmu = brain.word_to_bmu.get(word)
                if bmu is not None:
                    score += curiosity.novelty_score(bmu) * 0.5
                return score
            heard_words = sorted(heard_words, key=_salience, reverse=True)

            # ── Phase 3: Resolution check ─────────────────────────────────
            # If brain was awaiting resolution from a prior question,
            # and user now says content words → reward fires (felt resolution).
            if _awaiting_resolution and heard_words:
                content_words = [w for w in heard_words if w not in _HOLLOW and w not in _ZONE_NEUTRAL]
                if content_words:
                    # User answered something — reward the prior question turn
                    brain.step(reward=0.8)
                    selfmodel.resolve_uncertainty(reward=0.8)
                    wm.resolve_seeking()
                    _awaiting_resolution = False

            # ── Pavlovian conditioning: heard words nudge drive state ──────
            s = selfmodel._state
            _negated = 'not' in set(tokens)   # "i am NOT hungry" reverses delta
            for w in heard_words:
                if w in _PAVLOV_MAP:
                    drive, delta = _PAVLOV_MAP[w]
                    effective = -delta * 0.7 if _negated else delta
                    s[drive] = max(0.0, min(1.0, s[drive] + effective))
                    # Spike WM at this word's BMU so drive cluster stays hot
                    # during response generation — grounds meaning in activation
                    if w in brain.word_to_bmu:
                        _dbmu = brain.word_to_bmu[w]
                        wm.write(w, _dbmu, strength=min(1.0, wm.get_activation(_dbmu) + 0.5))

            # ── Semantic learning + contradiction detection ───────────────
            prev_n = semantic.contradiction_count()
            _n_rels_before = len(semantic._relations)
            semantic.learn_from_sentence(tokens)
            # learning_boost: high ACh (novelty turn) → new relations stored stronger.
            # Surprise deepens encoding — same mechanism as biological long-term potentiation.
            _boost = neuromod.learning_boost()
            if _boost > 1.05 and len(semantic._relations) > _n_rels_before:
                for _rel in semantic._relations[_n_rels_before:]:
                    _rel['strength'] = min(1.5, _rel['strength'] * _boost)
            if semantic.contradiction_count() > prev_n:
                conflict = semantic.recent_contradictions(1)[0]
                with print_lock:
                    sys.stdout.write("\r\033[K")
                    print(f"[brain-conflict]: {conflict['subject']} {conflict['relation']} "
                          f"'{conflict['stored']}' but now '{conflict['new']}' — wrong?")
                    sys.stdout.write("you: ")
                    sys.stdout.flush()

            # ── Active vocabulary acquisition ─────────────────────────────
            global _pending_unknown, _pending_unknown_turns
            if _pending_unknown:
                # Previous turn: brain asked "what is X?" — absorb definition
                _synthesize_vocab_entry(_pending_unknown, tokens)
                _pending_unknown_turns += 1
                if _pending_unknown_turns >= 2:   # gave up / answered → reset
                    _pending_unknown = None
                    _pending_unknown_turns = 0

            # Detect unknown content words (not in vocab, not structural).
            # Skip words that semantic memory already has a factual relation for
            # (e.g. 'xylophone' after user taught 'xylophone is music') — those
            # are known in meaning even if not in the SOM vocabulary.
            _unknown_found = None
            if not _pending_unknown:
                for t in tokens:
                    if (t not in VOCABULARY
                            and t not in _HOLLOW
                            and t not in _SKIP_TOKENS
                            and t not in _QUESTION_OPENERS
                            and 2 < len(t) <= 12
                            and t.isalpha()
                            and not any(t.startswith(k) or t.endswith(k)
                                        for k in VOCABULARY if len(k) > 3)
                            # Skip if semantic already knows this word
                            and not semantic._latest_relation_object(t, 'is')
                            and not semantic._latest_relation_object(t, 'means')):
                        _unknown_found = t
                        break

            # ── Feed input through acoustic SOM ──────────────────────────
            _last_heard_bmus = feed_input(heard_words)

            # ── Prediction error: compare TP prediction vs actual BMU ────
            # Brain predicted _predicted_bmu last turn; actual = _last_heard_bmus[0].
            # PE = normalized SOM distance. High PE → surprise → learn more.
            if _predicted_bmu and _last_heard_bmus:
                _act = _last_heard_bmus[0]
                _pr, _pc = _predicted_bmu // SOM_COLS, _predicted_bmu % SOM_COLS
                _ar, _ac = _act // SOM_COLS, _act % SOM_COLS
                _dist = ((_pr - _ar) ** 2 + (_pc - _ac) ** 2) ** 0.5
                _max_dist = (SOM_ROWS ** 2 + SOM_COLS ** 2) ** 0.5
                _last_pe = _dist / _max_dist
                if _last_pe > 0.25:
                    # Significant misprediction → spike uncertainty + boost SOM learning
                    selfmodel.spike_uncertainty(_last_pe * 0.35)
                    brain.step(reward=_last_pe * 0.1)  # surprise = learning signal
            else:
                _last_pe = 0.0

            for _ in range(4):
                brain.hear(SILENCE)
                brain.step()

            # ── Generate response: semantic memory first, then intent bridge ──
            # Always check relation memory first — even for unknown-vocab words
            # that have been taught semantically (e.g. 'what is xylophone').
            response = _relation_response(tokens)
            if response is not None:
                # Semantic answered — clear any pending unknown for this word
                if _unknown_found and response:
                    _unknown_found = None
            elif _state_query_response(tokens) is not None:
                # State query: brain introspects actual drive → truthful answer
                response = _state_query_response(tokens)
                _unknown_found = None
            elif _unknown_found:
                # Semantic has no answer AND word is genuinely unknown — ask.
                _pending_unknown = _unknown_found
                _pending_unknown_turns = 0
                response = f"what is {_unknown_found}"
                # Phase 2: brain just asked about unknown word → enter seeking
                wm.enter_seeking()
                selfmodel.spike_uncertainty(magnitude=0.5)
            else:
                response = generate_response(_last_heard_bmus, heard_words,
                                             raw_tokens=tokens)

            zone = _resolve_zone(heard_words, _last_heard_bmus)
            # Intrinsic reward: grounded zone response self-reinforces the SOM.
            # Zero when brain asked about unknown word — that's uncertainty, not success.
            intrinsic_r = (0.0 if _unknown_found
                           else _ZONE_INTRINSIC_REWARD.get(zone, 0.0)) if response else 0.0

            if response:
                with print_lock:
                    sys.stdout.write(f"\rbrain: {response}\nyou: ")
                    sys.stdout.flush()
                response_words = response.split()
                _last_response_words = response_words

                # Phase 3: mark awaiting resolution when brain asks/seeks
                if response_words and response_words[0] in ('what', 'search', 'know'):
                    _awaiting_resolution = True

                # Separator between user words and brain words in GRU sequence.
                # feed_input() already observed user words; now insert boundary,
                # hear_own_response() will observe brain words, then end() commits.
                brain.word_tp.separator()
                hear_own_response(response_words, reward=intrinsic_r)
                brain.word_tp.end()   # commit full exchange as GRU training sequence

                # ── Predict next input BMU from TP context ────────────────
                # Brain uses current TP state to form expectation of what comes next.
                # PE computed at START of next turn: actual vs this prediction.
                try:
                    _pred_dist = brain.tp.get_distribution(
                        [brain._prev_bmu, brain._current_bmu])
                    _predicted_bmu = brain.tp.sample(_pred_dist)
                except Exception:
                    _predicted_bmu = 0

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
                brain.word_tp.end()   # commit partial sequence even with no response

            # Module updates — order matters
            heard_bmu = _last_heard_bmus[0] if _last_heard_bmus else 0
            novelty   = curiosity.novelty_score(heard_bmu)
            surprise  = abs(novelty - 0.5)
            relation_hit = (response is not None and
                            _relation_response(tokens) is not None)

            # Spike uncertainty when input lands on genuinely unknown BMU.
            # Uncertainty spikes only from actual unknown words (_unknown_found path).

            wm.tick_seeking()
            wm.update(bmu=heard_bmu, heard_words=heard_words,
                      word_to_bmu=getattr(brain, 'word_to_bmu', {}))

            curiosity.update(bmu=heard_bmu, zone=zone)

            neuromod.update(novelty=novelty, surprise=surprise, reward=intrinsic_r)

            selfmodel.update(zone=zone, reward=intrinsic_r,
                             wm_strength=wm.activation_strength(),
                             episode_count=episodic.count())

            temporal.update(zone=zone, reward=intrinsic_r)

            episodic.record(
                you_words   = heard_words,
                brain_words = response.split() if response else [],
                zone        = zone,
                drives      = {'hunger': selfmodel._state['hunger'],
                               'fatigue': selfmodel._state['fatigue']},
                reward      = intrinsic_r,
            )

            # ── Global Workspace integration ──────────────────────────────
            # Collect signals from all modules → one broadcast per turn.
            _steps_since_reward += 1
            # Zone-level familiarity: how many times has brain been in this zone?
            # More robust than BMU-level (noisy MFCC hits different BMUs each turn).
            _zone_visits = curiosity._zone_counts.get(zone, 0)
            _familiarity = min(1.0, _zone_visits / 15.0)
            _gw_out = gw.step(
                qe_norm            = wm.activation_strength(),
                familiarity        = _familiarity,
                freq_idx           = heard_bmu,
                prediction_error   = max(novelty, _last_pe),
                thought_confidence = 0.5 + wm.activation_strength() * 0.5,
                rpe                = intrinsic_r - 0.08,
                intrinsic_rwd      = intrinsic_r,
                corridor_boredom   = min(1.0, conv_state._zone_repeat_count / 5.0),
                steps_since_reward = _steps_since_reward,
                salience           = min(1.0, novelty + wm.activation_strength()),
                l4_top_prob        = _familiarity,
            )
            # Broadcast GW outputs back into modules:
            # 1. Arousal → NE (urgency raises exploration temperature)
            neuromod._ne = max(neuromod._ne, _gw_out['arousal'] * 0.5)
            # 2. Curiosity pull → ACh (genuine unknowns boost learning rate)
            neuromod._ach = max(neuromod._ach, _gw_out['curiosity_pull'] * 0.7)
            # 3. Surprise debt → uncertainty spike (unresolved errors surface)
            if _gw_out['surprise_debt'] > 0.5:
                selfmodel.spike_uncertainty(_gw_out['surprise_debt'] * 0.25)
            # 4. Valence tone → comfort (positive turns feel better)
            s = selfmodel._state
            s['comfort'] = max(0.0, min(1.0,
                s['comfort'] + _gw_out['valence_tone'] * 0.04))
            # 5. Tension → curiosity pressure (stuck in known zone → explore)
            if _gw_out['tension'] > 0.7:
                curiosity._zone_counts[zone] = curiosity._zone_counts.get(zone, 0) + 5

            # Update conversation state: topic, entities, speaker focus, intent.
            conv_state.update(zone=zone, heard_words=heard_words,
                              relation_hit=relation_hit,
                              hollow_set=_HOLLOW,
                              question=_is_question(tokens))
            conv_state.record_turn(heard_words, response)

            _total_turns += 1

            if _total_turns % _DREAM_EVERY == 0:
                brain.dream(n_sequences=10)
                # GRU fit: consolidate language sequences accumulated in live use.
                # Each call improves GRU without forgetting — sequences accumulate across sessions.
                try:
                    _n_seqs = len(brain.word_tp._sequences)
                except AttributeError:
                    _state = brain.word_tp.get_state()
                    _n_seqs = len(_state.get('sequences', [])) if isinstance(_state, dict) else 0

                if _n_seqs >= 10:
                    brain.word_tp.fit(epochs=5, lr=0.01)
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopping live session...")

    # ── Exit ──────────────────────────────────────────────────────────
    if _total_turns > 0:
        print("\nSaving brain...", end='', flush=True)
        semantic.save(SEMANTIC_FILE)
        brain.save(BRAIN_FILE)
        print(" done.")
    else:
        print("\n(No turns played — brain unchanged.)")
    if interrupted:
        print("Exited on Ctrl+C.")
    print(f"Total turns: {_total_turns}")


if __name__ == '__main__':
    main()
