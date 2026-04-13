"""
M72: PHONEME SEQUENCE PREDICTOR — STATISTICAL WORD DISCOVERY
=============================================================

Learns the transition statistics of phoneme sequences from pure exposure.
Word boundaries emerge as dips in transition probability — no dictionary,
no grammar rules, no labels.

BIOLOGICAL BASIS
----------------
This is the computational implementation of what Saffran, Aslin & Newport
(1996, Science) showed: 8-month-old infants exposed to a continuous stream
of nonsense syllables for just 2 minutes learned word boundaries. How?

The only cue was TRANSITION PROBABILITY:
  Within "word" BIDAKU: B→I=1.0, I→D=1.0, D→A=1.0, A→K=1.0, K→U=1.0
  Between words:        U→B=0.33 (three words start with B, or different word starts follow)

The infant brain tracked: "after hearing sound X, how predictable is sound Y?"
High predictability = probably within a word.
Low predictability = probably a word boundary.

This is exactly what L2 does for spatial BMUs (navigation). M72 does the
same for phoneme BMUs (speech). Same architecture, new domain.

NEURAL BASIS
------------
Superior Temporal Gyrus (STG): primary site of phoneme sequence learning.
Left hemisphere STG shows stronger response to transition probability
violations than right — specialized for statistical regularities in
the ~200ms timescale (phoneme sequence) range.

The STG connects to Broca's area (IFG), where sequence predictions are
maintained and syntax-like regularities begin to emerge from statistics
alone after sufficient exposure.

THREE SIGNALS
-------------

1. TRANSITION PROBABILITY (TP)
   P[i, j] = how often does phoneme_bmu j follow phoneme_bmu i?
   Updated online with Hebbian-like ETA: P[prev, curr] increases.
   All rows normalized to sum=1.

2. WORD BOUNDARY DETECTION
   boundary_strength = 1 - TP[prev, curr]
   High boundary_strength = unexpected transition = likely word edge.
   But single transitions are noisy. We smooth: boundary_ema
   rises when recent transitions are consistently unexpected.

3. WORD CANDIDATE EXTRACTION
   When boundary_strength crosses BOUNDARY_THRESH, M72 closes the current
   "syllable chunk" and registers it as a word candidate:
     - The sequence of phoneme_bmus since the last boundary
     - Its frequency of occurrence (how many times seen)
     - Its total length
   Word candidates that recur frequently become "proto-words" — the
   brain's first lexical units.

   These are NOT tokens. They are recurring phoneme subsequences that
   the brain has noticed co-occur. They get meaning through M73 binding.

CALL INTERFACE
--------------
Called every voiced frame (when M71.is_voiced=True).
Silent frames do not update the sequence model.
"""

import numpy as np
from collections import deque, defaultdict

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

N_PHONEMES = 400   # must match M71 N_NEURONS

# Transition probability learning
TP_ETA          = 0.03     # learning rate for P matrix. Slow: tau ~33 updates.
                            # Each phoneme pair needs many observations to
                            # build reliable statistics. Too fast → noise.
TP_DECAY        = 0.0001   # very slow decay — phoneme stats are stable across
                            # hours of exposure. tau ~10000 updates.
TP_MIN          = 0.001    # floor — never predict 0 probability (unseen ≠ impossible)

# Word boundary detection
BOUNDARY_THRESH      = 0.55   # boundary_strength > this → word boundary
                               # Calibrated: at chance TP=0.01 (1/100 phonemes),
                               # boundary_strength=0.99 → always fires.
                               # For common bigrams (TP~0.5), strength=0.5 → near threshold.
BOUNDARY_EMA_ALPHA   = 0.25   # smoothing for boundary signal
                               # tau ~4 frames (~100ms) — phoneme-timescale.

# Word candidate tracking
MAX_WORD_LENGTH   = 12    # maximum phoneme sequence length for a candidate
                           # Corresponds to ~300ms at 25ms/frame — longer than
                           # any English word phoneme sequence.
MIN_WORD_LENGTH   = 2     # single phonemes are not words
MAX_WORD_CANDIDATES = 200 # ring buffer of most recent word candidates

# Sequence prediction error (drives L2-analog learning in M72)
PRED_ERROR_EMA_ALPHA = 0.05


class PhonemeSequencePredictor:
    """
    M72: Statistical phoneme transition learner + word boundary detector.

    Maintains a 100×100 transition probability matrix P over phoneme BMUs.
    Detects word boundaries as dips in transition probability.
    Extracts recurring phoneme sequences as proto-word candidates.
    """

    def __init__(self):
        # ── Transition probability matrix ─────────────────────
        # P[i, j] = P(phoneme j follows phoneme i)
        # Initialise uniform — every transition equally possible.
        # Probabilities will differentiate as the brain hears speech.
        self._P = np.ones((N_PHONEMES, N_PHONEMES), dtype=np.float64) / N_PHONEMES

        # Raw counts (for diagnostics)
        self._counts = np.zeros((N_PHONEMES, N_PHONEMES), dtype=np.int32)

        # ── Current phoneme sequence buffer ───────────────────
        # Stores the phoneme_bmu sequence since the last word boundary.
        self._current_sequence: list = []

        # ── Boundary state ────────────────────────────────────
        self._boundary_ema      = 0.0
        self._prev_phoneme_bmu  = -1
        self._just_boundary     = False

        # ── Word candidate store ──────────────────────────────
        # key: tuple of phoneme_bmu ints (the sequence)
        # value: {'count': int, 'last_seen': int}
        self._word_candidates: dict = {}
        self._candidate_queue = deque(maxlen=MAX_WORD_CANDIDATES)

        # ── Prediction error tracking ─────────────────────────
        self._pred_error_ema = 0.5   # start at maximum uncertainty

        # ── Step counter ──────────────────────────────────────
        self.t = 0

    # ── Transition probability update ─────────────────────────

    def _update_tp(self, prev: int, curr: int):
        """Update P[prev, curr] toward 1 (Hebbian-like) and decay others."""
        # Decay all transitions from prev toward uniform (forgetting)
        self._P[prev] *= (1.0 - TP_DECAY)
        # Strengthen the observed transition
        self._P[prev, curr] += TP_ETA * (1.0 - self._P[prev, curr])
        # Clip to floor
        self._P[prev] = np.maximum(self._P[prev], TP_MIN)
        # Normalize row to sum=1
        row_sum = self._P[prev].sum()
        if row_sum > 1e-9:
            self._P[prev] /= row_sum
        # Update counts
        self._counts[prev, curr] += 1

    # ── Word boundary detection ───────────────────────────────

    def _detect_boundary(self, prev: int, curr: int) -> tuple[float, bool]:
        """
        Compute boundary strength and detect if a word boundary just occurred.

        boundary_strength = 1 - P[prev, curr]
        High strength = this transition is unexpected = likely between words.
        """
        tp = float(self._P[prev, curr])
        boundary_strength = 1.0 - tp

        # Smooth the boundary signal to reduce frame-level noise
        self._boundary_ema = (
            (1.0 - BOUNDARY_EMA_ALPHA) * self._boundary_ema
            + BOUNDARY_EMA_ALPHA * boundary_strength
        )

        # Boundary fires when smoothed strength exceeds threshold
        is_boundary = bool(self._boundary_ema >= BOUNDARY_THRESH)
        return boundary_strength, is_boundary

    # ── Word candidate extraction ─────────────────────────────

    def _extract_candidate(self, sequence: list):
        """Register a phoneme sequence as a word candidate."""
        if len(sequence) < MIN_WORD_LENGTH:
            return
        if len(sequence) > MAX_WORD_LENGTH:
            sequence = sequence[:MAX_WORD_LENGTH]

        key = tuple(sequence)
        if key in self._word_candidates:
            self._word_candidates[key]['count'] += 1
            self._word_candidates[key]['last_seen'] = self.t
        else:
            self._word_candidates[key] = {'count': 1, 'last_seen': self.t}
        self._candidate_queue.append(key)

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             phoneme_bmu: int,
             is_voiced:   bool,
             ) -> dict:
        """
        One phoneme sequence learning step.

        Parameters
        ----------
        phoneme_bmu : int — current phoneme BMU from M71
        is_voiced   : bool — True if this frame is voiced speech

        Returns
        -------
        dict with:
          tp_prev_curr        float [0,1]  — transition prob of this bigram
          boundary_strength   float [0,1]  — how unexpected was this transition?
          boundary_ema        float [0,1]  — smoothed boundary signal
          just_boundary       bool         — word boundary detected this frame
          current_seq_len     int          — phonemes since last boundary
          n_word_candidates   int          — total distinct proto-words found
          top_word_candidate  tuple|None   — most frequent proto-word so far
          phoneme_pred_error  float [0,1]  — prediction error (1 - TP)
        """
        self.t += 1
        self._just_boundary = False

        if not is_voiced:
            # Silence = natural word boundary
            if self._current_sequence:
                self._extract_candidate(self._current_sequence)
                self._current_sequence = []
            self._prev_phoneme_bmu = -1
            return {
                'tp_prev_curr':       0.0,
                'boundary_strength':  1.0,
                'boundary_ema':       float(self._boundary_ema),
                'just_boundary':      True,
                'current_seq_len':    0,
                'n_word_candidates':  len(self._word_candidates),
                'top_word_candidate': self._top_candidate(),
                'phoneme_pred_error': float(self._pred_error_ema),
            }

        # ── Update transition stats ───────────────────────────
        tp_val = 0.5   # default if no previous phoneme
        boundary_strength = 0.5
        is_boundary = False

        if self._prev_phoneme_bmu >= 0:
            self._update_tp(self._prev_phoneme_bmu, phoneme_bmu)
            tp_val = float(self._P[self._prev_phoneme_bmu, phoneme_bmu])
            boundary_strength, is_boundary = self._detect_boundary(
                self._prev_phoneme_bmu, phoneme_bmu
            )

        # ── Prediction error ──────────────────────────────────
        pred_error = 1.0 - tp_val
        self._pred_error_ema = (
            (1.0 - PRED_ERROR_EMA_ALPHA) * self._pred_error_ema
            + PRED_ERROR_EMA_ALPHA * pred_error
        )

        # ── Accumulate current sequence ───────────────────────
        if is_boundary and self._current_sequence:
            self._extract_candidate(self._current_sequence)
            self._current_sequence = []
            self._just_boundary = True
            # Reset boundary EMA after firing
            self._boundary_ema = 0.0

        self._current_sequence.append(phoneme_bmu)
        self._prev_phoneme_bmu = phoneme_bmu

        return {
            'tp_prev_curr':       float(tp_val),
            'boundary_strength':  float(boundary_strength),
            'boundary_ema':       float(self._boundary_ema),
            'just_boundary':      bool(self._just_boundary),
            'current_seq_len':    len(self._current_sequence),
            'n_word_candidates':  len(self._word_candidates),
            'top_word_candidate': self._top_candidate(),
            'phoneme_pred_error': float(self._pred_error_ema),
        }

    def _top_candidate(self):
        """Return the most frequently occurring proto-word candidate."""
        if not self._word_candidates:
            return None
        return max(self._word_candidates, key=lambda k: self._word_candidates[k]['count'])

    def get_candidate_count(self, sequence: tuple) -> int:
        """How many times has this phoneme sequence been observed?"""
        return self._word_candidates.get(sequence, {}).get('count', 0)

    def top_n_candidates(self, n: int = 10) -> list:
        """Return top-n most frequent proto-word candidates."""
        sorted_cands = sorted(
            self._word_candidates.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        return [(seq, info) for seq, info in sorted_cands[:n]]

    def transition_prob(self, prev_bmu: int, curr_bmu: int) -> float:
        """Query learned transition probability directly."""
        if prev_bmu < 0 or prev_bmu >= N_PHONEMES:
            return 1.0 / N_PHONEMES
        return float(self._P[prev_bmu, curr_bmu])
