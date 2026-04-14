"""
M76: BROCA'S AREA — Grammar Template Engine
=============================================

Simplified analog of Broca's area (inferior frontal gyrus, pars opercularis
+ pars triangularis).

Real Broca's performs recursive Merge — combining words into hierarchical
phrase structure. This module implements the practical subset that matters at
our vocabulary size: slot-filling templates that CONSTRAIN the TP walk so the
brain generates grammatical sentences instead of random word sequences.

Without this:  "home place if"          (free TP walk — statistically legal,
                                          but no grammatical frame)
With this:     "i feel curious"          (SELF_STATE template, slot filled from
                                          TP distribution restricted to STATE words)

HOW IT WORKS
------------
1. Pick a template based on current cognitive state (sm_label + state_vec).
2. For each SLOT in the template, sample from the TP distribution restricted
   to words that are valid for that slot category.
3. Literal words in the template (like 'i', 'is', 'want') are always kept.

This isn't full Merge — it can't handle "the cat that the dog chased ran away."
But it handles all common simple sentences: self-reports, desires, assertions,
questions, and echo-continuations.

TEMPLATES
---------
    SELF_STATE:   i [SELF_VERB] [STATE]        "i feel calm"
    SELF_DESIRE:  i want [OBJECT]              "i want food"
    SELF_SIMPLE:  i [SELF_VERB]                "i know"
    ASSERTION:    [SUBJ] is [ADJ]              "food is good"
    QUESTION:     [Q_WORD] is [SUBJ]           "what is this"
    YES_STATE:    yes [STATE]                  "yes calm"
    ECHO:         [HEARD] is [ADJ]             topic continuation
"""

import numpy as np
from typing import Callable


# ── Slot categories ────────────────────────────────────────────────────
# These define which words can legally fill each template slot.
# The brain samples from the TP distribution RESTRICTED to these sets.

SLOT: dict[str, set] = {
    'SELF_VERB': {
        'feel', 'think', 'know', 'want', 'need', 'am', 'see', 'hear',
        'learn', 'remember', 'understand', 'believe', 'try', 'find',
        'like', 'enjoy', 'miss', 'mean',
    },
    'STATE': {
        'calm', 'confused', 'happy', 'sad', 'tired', 'curious', 'good',
        'okay', 'ready', 'alive', 'awake', 'lost', 'hurt', 'scared',
        'lonely', 'free', 'still', 'open', 'full', 'warm', 'cold', 'safe',
        'sick', 'hungry', 'strong', 'quiet', 'alive', 'bright', 'alone',
        'deep', 'different', 'easy', 'fine', 'great', 'nice', 'right',
    },
    'OBJECT': {
        'food', 'water', 'home', 'more', 'rest', 'help', 'time', 'life',
        'peace', 'love', 'light', 'way', 'thing', 'place', 'music',
        'friend', 'this', 'that', 'warmth', 'joy', 'hope', 'dream',
        'answer', 'truth',
    },
    'SUBJ': {
        'food', 'water', 'life', 'world', 'this', 'that', 'it', 'home',
        'love', 'time', 'light', 'sound', 'pain', 'dream', 'mind', 'body',
        'day', 'night', 'thought', 'memory', 'idea',
    },
    'ADJ': {
        'good', 'bad', 'big', 'small', 'warm', 'cold', 'fast', 'slow',
        'long', 'deep', 'strong', 'soft', 'new', 'old', 'near', 'far',
        'real', 'right', 'wrong', 'calm', 'strange', 'clear', 'beautiful',
        'sweet', 'hot', 'heavy', 'safe', 'important', 'free', 'true',
    },
    'Q_WORD': {
        'what', 'who', 'how', 'why', 'where', 'when',
    },
}

# Template definitions: list of (slot_name_or_literal_word)
# Uppercase = slot to fill from SLOT dict
# Lowercase = literal word, always used as-is

_TEMPLATES: dict[str, list] = {
    'SELF_STATE':  ['i', 'SELF_VERB', 'STATE'],
    'SELF_DESIRE': ['i', 'want',      'OBJECT'],
    'SELF_SIMPLE': ['i', 'SELF_VERB'],
    'ASSERTION':   ['SUBJ', 'is', 'ADJ'],
    'QUESTION':    ['Q_WORD', 'is', 'SUBJ'],
    'YES_STATE':   ['yes', 'STATE'],
    'NO_STATE':    ['no', 'STATE'],
    'ECHO_ASSERT': ['SUBJ', 'is', 'good'],     # default echo
}


class BrocaGrammar:
    """
    Slot-filling grammar engine — simplified Broca's area.

    Usage:
        broca = BrocaGrammar(bmu_to_word_fn, word_to_bmu_dict)
        sentence = broca.generate(heard_bmus, state_vec, sm_label,
                                   phoneme_seq, binding)
    """

    def __init__(self,
                 bmu_to_word_fn: Callable,
                 word_to_bmu_dict: dict):
        self._bmu_to_word  = bmu_to_word_fn
        self._word_to_bmu  = word_to_bmu_dict
        # Precompute: slot_name → set of BMUs for valid words in that slot
        self._slot_bmus: dict[str, dict[int, str]] = {}
        for slot_name, words in SLOT.items():
            bmu_map = {}
            for w in words:
                if w in word_to_bmu_dict:
                    bmu = word_to_bmu_dict[w]
                    # keep highest-frequency word per BMU if collision
                    if bmu not in bmu_map:
                        bmu_map[bmu] = w
            self._slot_bmus[slot_name] = bmu_map   # {bmu: word}

    # ── Template selection ──────────────────────────────────────────

    def _pick_template_name(self,
                            sm_label: str,
                            state_vec: np.ndarray,
                            heard_words: list) -> str:
        """Choose which template frame to use based on cognitive state."""
        urgency, clarity, drive, novelty, stability, confidence, frustration, engagement = \
            state_vec[:8]

        # Extreme drives → desire frame
        if drive > 0.55 or urgency > 0.55 or sm_label == 'hunting':
            return 'SELF_DESIRE'

        # Curious → question frame
        if novelty > 0.5 or sm_label == 'curious':
            return 'QUESTION'

        # Satisfied → affirmation
        if sm_label == 'satisfied' or (clarity > 0.6 and frustration < 0.3):
            return 'YES_STATE'

        # Stuck → minimal
        if sm_label == 'stuck' or frustration > 0.55:
            return 'SELF_SIMPLE'

        # Focused → assertion about something heard
        if (sm_label == 'focused' or confidence > 0.6) and heard_words:
            # Try to use a heard content word as subject
            for w in reversed(heard_words):
                if w in SLOT['SUBJ']:
                    return 'ASSERTION'
            return 'ASSERTION'

        # Default: self-state report
        return 'SELF_STATE'

    # ── Slot filling ────────────────────────────────────────────────

    def _fill_slot(self,
                   slot_name: str,
                   tp_dist: np.ndarray,
                   exclude: set,
                   temperature: float = 1.0) -> str | None:
        """
        Sample a word for the given slot from the TP distribution,
        restricted to words valid in that slot.
        """
        bmu_map = self._slot_bmus.get(slot_name, {})
        if not bmu_map:
            return None

        # Collect (score, word) pairs from TP distribution
        candidates = []
        for bmu, word in bmu_map.items():
            if word in exclude:
                continue
            if bmu < len(tp_dist):
                score = float(tp_dist[bmu])
                candidates.append((score, word))

        if not candidates:
            # Fallback: return any valid slot word not excluded
            for bmu, word in bmu_map.items():
                if word not in exclude:
                    return word
            return None

        # Tempered softmax sampling
        scores = np.array([s for s, _ in candidates], dtype=np.float32)
        log_s = np.log(np.maximum(scores, 1e-9)) / temperature
        log_s -= log_s.max()
        probs = np.exp(log_s)
        probs /= probs.sum()

        idx = int(np.random.choice(len(candidates), p=probs))
        return candidates[idx][1]

    # ── Main generate ───────────────────────────────────────────────

    def generate(self,
                 heard_bmus:   list,
                 heard_words:  list,
                 state_vec:    np.ndarray,
                 sm_label:     str,
                 phoneme_seq,
                 binding,
                 recent_words: set = None,
                 temperature:  float = 1.2) -> str:
        """
        Generate one grammatical sentence via slot-filling.

        Args:
            heard_bmus:   BMU sequence from last heard input
            heard_words:  word sequence from last heard input
            state_vec:    current M59 state vector (8-dim)
            sm_label:     current M59 state label string
            phoneme_seq:  M72 PhonemeSequencer (for TP matrix)
            binding:      M73 SemanticBinding (for state→BMU weights)
            recent_words: words used recently (to avoid repetition)
            temperature:  sampling temperature (higher = more diverse)

        Returns:
            A space-joined sentence string, or "" on failure.
        """
        N = phoneme_seq._P.shape[0]

        # Build TP distribution from heard BMUs
        if heard_bmus:
            tp = sum(phoneme_seq._P[b] for b in heard_bmus if 0 <= b < N)
            tp = tp.astype(np.float64)
        else:
            tp = np.ones(N, dtype=np.float64)

        total = float(tp.sum())
        if total > 1e-9:
            tp /= total

        # Blend with state-binding distribution
        sv_norm = float(np.linalg.norm(state_vec))
        if sv_norm > 1e-9 and binding._W_state.max() > 1e-9:
            state_sims = binding._W_state @ (state_vec / sv_norm)
            state_sims = np.maximum(state_sims, 0.0)
            ss = float(state_sims.sum())
            state_dist = (state_sims / ss) if ss > 1e-9 else np.ones(N) / N
        else:
            state_dist = np.ones(N, dtype=np.float64) / N

        tp_blended = 0.60 * tp + 0.40 * state_dist
        tp_blended /= tp_blended.sum()

        # Choose template
        template_name = self._pick_template_name(sm_label, state_vec, heard_words)
        template = _TEMPLATES[template_name]

        # Fill slots
        used = set(recent_words or [])
        words_out = []

        for slot in template:
            if slot in SLOT:
                word = self._fill_slot(slot, tp_blended, used, temperature)
                if word:
                    words_out.append(word)
                    used.add(word)
                # If slot can't be filled, skip it (shorter sentence)
            else:
                # Literal word
                if slot in self._word_to_bmu or len(slot) <= 3:
                    words_out.append(slot)

        if len(words_out) < 1:
            return ""

        return ' '.join(words_out)

    def summary(self) -> str:
        total_slot_words = sum(len(m) for m in self._slot_bmus.values())
        return (f"  BrocaGrammar: {len(_TEMPLATES)} templates, "
                f"{total_slot_words} slot-word mappings")
