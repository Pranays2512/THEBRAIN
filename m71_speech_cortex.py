"""
M71: SPEECH CORTEX — ACOUSTIC PHONEME SOM
==========================================

A Self-Organizing Map that develops phoneme-like categories from raw
acoustic features through pure statistical exposure. No labels, no
phoneme dictionary. Categories emerge from the statistics of the input
stream — exactly as primary auditory cortex (A1) and surrounding belt
regions develop in the first months of life.

BIOLOGICAL BASIS
----------------
At birth, the infant auditory cortex responds equally to all phonemes
of all languages. By 6-12 months, the cortex has reorganised to favour
the phoneme contrasts of the native language. This happens through:

  1. Competitive learning — neurons compete to represent each incoming
     sound. The winner updates its weights toward the input. Over time,
     neurons specialise for particular acoustic patterns.

  2. Temporal contiguity — sounds that occur together repeatedly bind
     together in the map's neighborhood structure.

  3. Frequency tuning — tonotopy in A1 extends upward to the speech
     belt cortex (A2, STG), where neurons become selective for formant
     patterns characteristic of vowels and consonants.

  No teacher, no labels. Just statistics + competition.

HOW IT DIFFERS FROM M54 (the spatial frequency SOM)
----------------------------------------------------
M54 detects which of 8 slow frequencies (0.4-2.2 Hz) is present.
That's the world's navigation signal.

M71 processes the SPEECH signal — the 80-8000 Hz range where human
vocalisations live. Input is MFCC (Mel-Frequency Cepstral Coefficients):
the standard acoustic representation that approximates the ear's
non-linear frequency decomposition and the cochlear processing that
M50 doesn't cover.

  M50: cochlear-like, 0.4-2.2 Hz, for spatial navigation
  M71: speech-range, 80-8000 Hz, for vocal communication

MFCC INPUT
----------
MFCCs are computed externally (by the harness or hardware interface)
and passed in as a numpy array. Standard: 13 coefficients per 25ms
frame. M71 also computes delta-MFCCs internally from a buffer of
recent frames, giving the SOM a 26-dim input (13 MFCC + 13 delta).

Delta-MFCCs capture the rate of change of spectral features — they
carry information about transitions between phonemes that static
MFCCs miss. This is analogous to the motion-sensitive V1 cells that
respond to the onset/offset of visual features.

SILENCE DETECTION
-----------------
Speech is not continuous. The brain must distinguish voiced frames
(speech present) from silence. M71 computes a voiced energy estimate
from the MFCC energy coefficient (coefficient 0) and suppresses
learning during silence. The SOM map only updates on voiced frames.

OUTPUTS
-------
  phoneme_bmu       int          — winning phoneme neuron (0-99)
  phoneme_qe_norm   float [0,1]  — quantization error (perceptual surprise)
  phoneme_likelihood (100,)      — soft activation across all neurons
  is_voiced         bool         — True if this frame is speech (not silence)
  phoneme_familiarity float [0,1] — how familiar is this phoneme pattern?

PARAMETERS
----------
N_NEURONS = 100  (10×10 grid)
  English has ~44 phonemes. 100 neurons gives 2.3 neurons/phoneme on
  average — enough for allophonic variation within each phoneme class
  without wasting capacity. Same 8×8 logic as M54 but for richer space.

INPUT_DIM = 26  (13 MFCC + 13 delta-MFCC)
  Standard speech representation. Delta captures dynamics.

SIGMA_MAX = 4.0  (larger than M54's 3.5 — speech space needs more
  initial exploration before the 44-category structure emerges)
SIGMA_MIN = 0.8  (same consolidation floor as M54)

ETA_BASE = 0.12  (slightly lower than M54's 0.15 — speech features
  are richer and more variable, so we learn more carefully)
"""

import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

GRID_H    = 20
GRID_W    = 20
N_NEURONS = GRID_H * GRID_W   # 400  (~9 neurons per English phoneme)

N_MFCC    = 13    # standard MFCC coefficient count
INPUT_DIM = N_MFCC * 2   # 26: MFCCs + delta-MFCCs

# Learning
ETA_BASE    = 0.12
ETA_MIN     = 0.008
ETA_MAX     = ETA_BASE * 2.5

# Neighborhood
SIGMA_MAX   = 4.0
SIGMA_MIN   = 0.8
SIGMA_DECAY = 0.999_95   # slower decay than M54 — more phoneme structure to learn

# Familiarity suppression (same mechanism as M54/M56)
FAM_ETA_SUPPRESS = 1.2

# Conscience (prevents neuron monopoly)
CONSCIENCE_FACTOR = 0.5
CONSCIENCE_LEAK   = 0.002

# QE normalization
QE_EMA_ALPHA = 0.01
QE_EMA_INIT  = 0.5
QE_EMA_EPS   = 1e-4

# Silence detection
# MFCC[0] (energy coefficient) in silence is typically << voiced frames.
# Threshold is in normalized MFCC units after z-score normalization.
# A frame is voiced if its energy coefficient > SILENCE_THRESH.
SILENCE_THRESH = -1.5   # in z-score units; negative because log-energy is negative in silence

# Delta buffer: how many frames to keep for delta computation
DELTA_WINDOW = 3   # delta[t] = mfcc[t] - mfcc[t-2] (central difference)

# Novelty boost — same as M54
NOVELTY_BOOST = 2.0


# ═══════════════════════════════════════════════════════════════
# SPEECH CORTEX
# ═══════════════════════════════════════════════════════════════

class SpeechCortex:
    """
    M71: Self-Organizing Map for acoustic phoneme learning.

    Accepts 13-dim MFCC vectors, computes deltas internally,
    runs competitive learning, returns phoneme_bmu and related signals.

    All learning is suppressed during silence (is_voiced=False).
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)

        # ── SOM weights ───────────────────────────────────────
        # Initialise with small random values.
        # MFCCs are zero-mean after normalisation, so init near zero.
        self._weights = rng.normal(0.0, 0.3, (N_NEURONS, INPUT_DIM)).astype(np.float32)

        # ── Grid positions ────────────────────────────────────
        rows = np.arange(N_NEURONS) // GRID_W
        cols = np.arange(N_NEURONS) % GRID_W
        self._pos = np.stack([rows, cols], axis=1).astype(np.float32)

        # ── Adaptive parameters ───────────────────────────────
        self._sigma   = float(SIGMA_MAX)
        self._eta     = float(ETA_BASE)
        self._qe_ema  = float(QE_EMA_INIT)

        # ── Conscience (win frequency) ────────────────────────
        self._win_freq = np.ones(N_NEURONS, dtype=np.float32) / N_NEURONS

        # ── Online MFCC normalization (running mean + variance) ──
        # Real brains adapt to the statistics of their acoustic environment.
        # We maintain a running mean and variance and z-score normalize.
        # Initialized with zeros — will adapt after first few hundred frames.
        self._mfcc_mean = np.zeros(N_MFCC, dtype=np.float64)
        self._mfcc_var  = np.ones(N_MFCC,  dtype=np.float64)
        self._mfcc_n    = 0   # count for Welford running stats

        # ── Delta computation buffer ──────────────────────────
        self._mfcc_buf = deque(maxlen=DELTA_WINDOW * 2 + 1)

        # ── Per-neuron exposure (familiarity analog) ──────────
        self._exposure = np.zeros(N_NEURONS, dtype=np.float32)

        # ── Step counter ──────────────────────────────────────
        self.t = 0

    # ── Online MFCC normalization (Welford's algorithm) ──────

    def _update_stats(self, mfcc: np.ndarray):
        """Update running mean and variance for z-score normalization."""
        self._mfcc_n += 1
        delta = mfcc - self._mfcc_mean
        self._mfcc_mean += delta / self._mfcc_n
        delta2 = mfcc - self._mfcc_mean
        self._mfcc_var += delta * delta2

    def _normalize(self, mfcc: np.ndarray) -> np.ndarray:
        """Z-score normalize MFCC vector using running statistics."""
        if self._mfcc_n < 10:
            return mfcc.copy()   # not enough data yet — pass through
        std = np.sqrt(self._mfcc_var / max(self._mfcc_n - 1, 1))
        std = np.maximum(std, 1e-6)
        return (mfcc - self._mfcc_mean) / std

    # ── Delta computation ─────────────────────────────────────

    def _compute_delta(self) -> np.ndarray:
        """
        Central-difference delta of MFCC buffer.
        Returns zeros if not enough frames yet.
        """
        buf = list(self._mfcc_buf)
        if len(buf) < DELTA_WINDOW + 1:
            return np.zeros(N_MFCC, dtype=np.float32)
        # central difference: delta[t] ≈ mfcc[t] - mfcc[t - DELTA_WINDOW]
        return (buf[-1] - buf[-1 - DELTA_WINDOW]).astype(np.float32)

    # ── BMU selection with conscience ────────────────────────

    def _find_bmu(self, x: np.ndarray) -> tuple[int, float]:
        """
        Find Best Matching Unit with conscience correction.

        Conscience penalizes neurons that win too frequently,
        preventing any single neuron from monopolizing.
        d_eff[i] = d[i] × (1 + CONSCIENCE_FACTOR × (p[i] - 1/N))
        """
        diff = self._weights - x   # (100, 26)
        d    = np.sum(diff * diff, axis=1)   # (100,) squared L2 distances
        # Conscience correction
        d_eff = d * (1.0 + CONSCIENCE_FACTOR * (self._win_freq - 1.0 / N_NEURONS))
        bmu   = int(np.argmin(d_eff))
        qe    = float(np.sqrt(d[bmu]))   # actual distance (not corrected)
        return bmu, qe

    # ── Neighborhood function ─────────────────────────────────

    def _neighborhood(self, bmu: int) -> np.ndarray:
        """Gaussian neighborhood around BMU on the 2D grid."""
        d_sq = np.sum((self._pos - self._pos[bmu]) ** 2, axis=1)
        return np.exp(-d_sq / (2.0 * self._sigma ** 2)).astype(np.float32)

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             mfcc_vec:   np.ndarray,
             familiarity: float = 0.0,
             ) -> dict:
        """
        One acoustic cortex step.

        Parameters
        ----------
        mfcc_vec    : (13,) float — MFCC features for current 25ms frame.
                      Pass np.zeros(13) if no audio this step.
        familiarity : float [0,1] — M55 cross-modal familiarity (for
                      plasticity suppression during well-known sounds)

        Returns
        -------
        dict with:
          phoneme_bmu         int
          phoneme_qe          float
          phoneme_qe_norm     float [0,1]
          phoneme_likelihood  (100,) float
          is_voiced           bool
          phoneme_familiarity float [0,1]
        """
        mfcc = np.array(mfcc_vec, dtype=np.float64)

        # ── Silence detection (before normalization) ──────────
        # MFCC[0] is the log-energy coefficient. In silence it's
        # much lower than in voiced speech.
        is_voiced = bool(float(mfcc[0]) > SILENCE_THRESH) if len(mfcc) > 0 else False

        # ── Update running statistics on voiced frames ─────────
        if is_voiced:
            self._update_stats(mfcc)

        # ── Normalize ─────────────────────────────────────────
        mfcc_norm = self._normalize(mfcc).astype(np.float32)

        # ── Update buffer for delta ────────────────────────────
        self._mfcc_buf.append(mfcc_norm.copy())
        delta = self._compute_delta()

        # ── Build 26-dim input ────────────────────────────────
        x = np.concatenate([mfcc_norm, delta]).astype(np.float32)

        # ── BMU selection ─────────────────────────────────────
        bmu, qe = self._find_bmu(x)

        # ── QE normalization (EMA-relative surprise) ──────────
        self._qe_ema = (1.0 - QE_EMA_ALPHA) * self._qe_ema + QE_EMA_ALPHA * qe
        qe_norm = float(np.clip(
            (qe - self._qe_ema) / (self._qe_ema + QE_EMA_EPS),
            0.0, 1.0
        ))

        # ── Phoneme likelihood (soft activations) ─────────────
        # Gaussian likelihood over all neurons based on distance to BMU.
        d_sq = np.sum((self._weights - x) ** 2, axis=1)
        scale = max(self._qe_ema ** 2, 1e-6)
        likelihood = np.exp(-d_sq / (2.0 * scale)).astype(np.float32)
        likelihood /= (likelihood.sum() + 1e-9)

        # ── Familiarity from exposure ─────────────────────────
        phoneme_fam = float(np.clip(
            self._exposure[bmu] / max(self._exposure.mean() * 3, 1.0),
            0.0, 1.0
        ))

        # ── Learning (only on voiced frames) ──────────────────
        if is_voiced:
            # Update win frequency (conscience)
            self._win_freq *= (1.0 - CONSCIENCE_LEAK)
            self._win_freq[bmu] += CONSCIENCE_LEAK

            # Effective eta: base + novelty boost - familiarity suppression
            eta_base = ETA_MIN + (ETA_BASE - ETA_MIN) * (0.5 + 0.5 * qe_norm)
            eta_nov  = (ETA_BASE - ETA_MIN) * NOVELTY_BOOST * qe_norm
            eta_fam  = -FAM_ETA_SUPPRESS * (ETA_BASE - ETA_MIN) * float(familiarity)
            eta = float(np.clip(eta_base + eta_nov + eta_fam, ETA_MIN, ETA_MAX))

            # Neighborhood-weighted weight update
            h = self._neighborhood(bmu)
            self._weights += eta * h[:, None] * (x - self._weights)

            # Sigma decay
            self._sigma = max(self._sigma * SIGMA_DECAY, SIGMA_MIN)

            # Exposure increment
            self._exposure[bmu] += 1.0

        self.t += 1

        return {
            'phoneme_bmu':        bmu,
            'phoneme_qe':         float(qe),
            'phoneme_qe_norm':    qe_norm,
            'phoneme_likelihood': likelihood,
            'is_voiced':          is_voiced,
            'phoneme_familiarity': phoneme_fam,
        }
