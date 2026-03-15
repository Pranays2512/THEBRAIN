"""
L2: SEQUENCE PREDICTOR — ELIGIBILITY-WEIGHTED PREDICTION
=========================================================

WHAT THIS IS
------------
Layer 2 sits above M55. It receives the current BMU each step and
does one thing the layers below cannot: it predicts what comes NEXT.

Every step:
  1. Before seeing the new BMU — make a prediction
  2. See the new BMU — measure prediction error
  3. Learn: strengthen context → outcome connections
  4. Update context vector for the next prediction

The prediction error is the system's first genuine cognitive signal.
It is NOT perceptual surprise (M54's QE — "is this pattern familiar?")
It is NOT recognition (M55's familiarity — "have I been here before?")
It IS sequence surprise — "did I expect this to happen NOW?"

HOW IT WORKS
------------
Two data structures. That's the entire system.

CONTEXT VECTOR c  (shape: 64)
  c[i] = how strongly BMU i is influencing the current prediction
  Decays exponentially each step — recent BMUs dominate, distant fade
  Adaptive decay: high prediction error → slow decay → longer memory
  Biologically: the eligibility trace of sequence-predicting neurons

PREDICTION MATRIX P  (shape: 64 × 64)
  P[i, j] = how strongly "BMU i in context" predicts "BMU j fires next"
  Learned online via STDP-inspired rule:
    When BMU k fires: P[:, k] += eta * c   (context → outcome)
    Always:           P        *= (1-decay) (synaptic decay)
  Biologically: cortico-striatal synapses for sequence learning

PREDICTION
  scores    = P.T @ c          (64,) — how strongly each BMU is predicted
  predicted = argmax(scores)   the single most expected next BMU
  confidence = softmax sharpness of scores

PREDICTION ERROR
  When BMU k actually fires:
  error = 1 - scores_normalized[k]
  0 = perfect prediction (k was top prediction with full confidence)
  1 = complete surprise   (k had zero predicted probability)

  This error:
  - Feeds back to M54 as a learning rate modulator
  - Modulates L2's own context decay (high error → longer memory)
  - Drives the curiosity signal (sustained error = novel territory)

ADAPTIVE CONTEXT DECAY
  Same biological principle as M55's adaptive trace:
  High prediction error → slow context decay → longer eligibility window
  Low prediction error  → fast context decay → shorter window
  Rationale: when you're wrong, remember more of the past — something
  in that history is relevant and you haven't figured out what yet.

BIOLOGICAL RULES
----------------
1. STDP-inspired sequence learning
   ΔP[:, k] = eta * c   (context that preceded k gets credit)
   Biologically: pre-synaptic trace × post-synaptic spike

2. Synaptic decay
   P *= (1 - decay)
   Unused predictions fade. The system forgets sequences it hasn't
   seen in a while — exactly like M55 forgets associations.

3. Column normalization (homeostasis)
   Each column of P (predictions for one outcome BMU) is bounded.
   Prevents any single outcome from monopolizing predictions.

4. Adaptive context (neuromodulatory)
   context_decay = f(prediction_error)
   Error-driven context extension mirrors norepinephrine modulation:
   unexpected events trigger wider attention to recent history.

SIGNALS OUTPUT EACH STEP
-------------------------
BEFORE seeing new BMU (prediction phase):
  'predicted_bmu'   int   — which BMU L2 expects next
  'confidence'      float — how certain (0=random, 1=certain)
  'scores'          array — full prediction distribution (64,)

AFTER seeing new BMU (error phase):
  'prediction_error'  float [0,1] — how wrong the prediction was
  'correct'           bool        — was top prediction correct?
  'context_decay'     float       — current window size parameter
  'learning_rate'     float       — current eta (boosted on error)
  'curiosity'         float [0,1] — smoothed sustained error signal

SPACE COST
----------
Context vector c:    64 × float32  =  256 bytes
Prediction matrix P: 64×64 float32 = 16 KB
Everything else:     negligible
Total:               ~16 KB RAM. Zero bytes disk.
Same footprint as M55.

STACK POSITION
--------------
M50 → M54 → M55 → L2
              ↓         ↓
        ExperienceBuffer  prediction_error → M54 eta modulator

INTERFACE
---------
pred = SequencePredictor()

# In your loop, BEFORE cortex.step() — make prediction:
p = pred.predict()
# p['predicted_bmu'], p['confidence'], p['scores']

# AFTER cortex.step() and memory.step() — update with actual:
result = pred.step(bmu_idx, qe_norm=cortex_out['qe_norm'],
                   familiarity=mem_out['familiarity'])
# result['prediction_error']  → feed back to M54
# result['curiosity']         → sustained novelty signal
# result['correct']           → was prediction right?
"""

import numpy as np
from collections import deque
import math


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Grid (must match M54 and M55)
N_NEURONS = 64

# ── Prediction matrix learning ───────────────────────────────
# Learning rate for P update
# Small: sequence structure accumulates over many transitions
ETA_BASE  = 0.05

# Bonus learning rate on high prediction error
# When wrong, learn faster — something important happened
ETA_ERROR_BOOST = 0.10   # added to ETA_BASE when error > ERROR_THRESH

# Error threshold above which learning is boosted
ERROR_THRESH = 0.5

# Synaptic decay on P — unused predictions fade
# τ ≈ 1/P_DECAY ≈ 1000 steps ≈ ~50s
# Fix 7 (P_DECAY=0.005) was reverted. τ=200 cleared the P matrix within
# ~500 steps, causing top_predictions() to return [] for any frequency
# whose modal BMU hadn't fired recently. Rare frequencies (G=10%, H=12%)
# routinely had >500-step gaps, producing BestDist=999 artifacts.
# At τ=1000, P survives long enough for the probe while staying selective.
P_DECAY = 0.001

# Column normalization ceiling (per outcome BMU)
P_MAX = 1.0

# ── Adaptive context decay ───────────────────────────────────
# Base: τ = 1/BASE ≈ 5 steps — only the last few BMUs dominate.
# At 0.05, after 4 steps context still has 0.81 strength — the
# entire cycle is in context simultaneously, making every outcome
# look equally predicted. At 0.30, after 4 steps oldest has 0.24
# strength — genuine recency weighting, only 1-2 BMUs dominate.
CONTEXT_DECAY_BASE  = 0.30

# Minimum: τ = 1/MIN ≈ 10 steps — extended window after high error
CONTEXT_DECAY_MIN   = 0.10

# How strongly prediction error modulates context decay
# decay = BASE - MODULATION * error
# At error=1.0: decay = 0.30 - 0.20 = 0.10 (longest window, ~10 steps)
# At error=0.0: decay = 0.30              (shortest window, ~3 steps)
CONTEXT_ERROR_MODULATION = 0.20

# Minimum context activation to participate in P update.
# At context_decay=0.30, one step back gives c[prev]=0.70.
# Two steps back gives c[prev²]=0.49. Three steps: 0.34.
#
# Setting threshold to 0.5 means ONLY the immediately preceding
# BMU (c=0.70) participates in the P update. BMUs from 2+ steps
# ago are excluded. This has two effects:
#   1. Eliminates the self-prediction artifact in 2-step cycles:
#      A→B→A: when A fires again, c[A]=(0.70)^2=0.49 < 0.5. Excluded.
#      P[A,A] never accumulates for period-2 cycles.
#   2. Keeps P sparse — only direct predecessors write to each column.
#      With min=0.05 (old), 5+ context neurons wrote every step.
#      With min=0.5, only 1-2 write, keeping columns clean and selective.
MIN_CONTEXT_TO_LEARN = 0.50

# ── Curiosity signal ─────────────────────────────────────────
# Curiosity = smoothed sustained prediction error
# EMA of prediction_error — rises when system is consistently wrong
# Falls when predictions improve
CURIOSITY_EMA_ALPHA = 0.05   # τ ≈ 20 steps — medium smoothing

# ── Spatial soft-match ───────────────────────────────────────
# Instead of binary correct/wrong, prediction error is the spatial
# distance between predicted BMU and actual BMU on the 8×8 cortical grid.
#   error = 1 - exp(-dist² / (2 × SPATIAL_SIGMA²))
# At dist=0 (exact match): error = 0.0
# At dist=1 (adjacent):    error ≈ 0.12  (SIGMA=2.0)
# At dist=3 (same cluster): error ≈ 0.64
# At dist=6 (wrong region): error ≈ 0.95
#
# WHY: the SOM fires a REGION of 15-30 BMU indices per frequency,
# not a single locked index. L2 cannot reliably predict the exact
# next index in a region. Binary error treats a 1-cell miss identically
# to a full-grid miss (both score 1.0), so curiosity never falls even
# after L2 has learned the correct frequency zone. Spatial error
# falls when L2 predicts within the right neighbourhood, giving
# the curiosity EMA and Brain's delta feedback meaningful gradients.
#
# SIGMA=2.0 corresponds to ~2 grid cells ≈ one SOM neighbourhood.
# Predictions within the typical firing cluster count as near-correct.
SPATIAL_SIGMA = 2.0

# ── Familiarity modulation ───────────────────────────────────
# High familiarity (M55) scales down prediction error.
#   effective_error = spatial_error × (1 - FAMILIARITY_ERROR_SCALE × familiarity)
# At familiarity=1.0: error is reduced by FAMILIARITY_ERROR_SCALE (30%).
# At familiarity=0.0: no reduction.
#
# WHY: familiarity signals that this context has been visited before.
# Recognised contexts are less surprising — even if the exact sequence
# transition is not predicted, the brain expects something coherent.
# Biologically: hippocampal familiarity suppresses prediction error
# signals in prefrontal prediction circuits.
#
# Effect: curiosity falls faster on frequently-visited sequences,
# and Brain's delta feedback to M54 is smaller for familiar material.
FAMILIARITY_ERROR_SCALE = 0.3

# Grid dimensions (must match M54)
GRID_W = 8   # columns in M54's neuron grid
# Softmax temperature for reading prediction scores.
# Lower = sharper discrimination between predicted and background.
# At T=0.3, even a fully trained correct prediction scores ~0.47
# (error=0.53), so curiosity can't fall below ~0.53 after learning.
# At T=0.15, a correct prediction scores higher → lower error →
# curiosity actually falls after learning → BT-08 passes correctly.
SCORE_TEMPERATURE = 0.15


# ═══════════════════════════════════════════════════════════════
# SEQUENCE PREDICTOR
# ═══════════════════════════════════════════════════════════════

class SequencePredictor:
    """
    Eligibility-weighted sequence predictor for the M54+M55 stack.

    Sits above M55. Reads bmu_idx each step.
    Never modifies M54, M55, or ExperienceBuffer.
    Prediction = P matrix (64×64). Context = c vector (64,).
    Zero bytes on disk.

    Call order each step:
      1. pred.predict()          → get prediction BEFORE new BMU
      2. cortex.step(...)        → M54 fires actual BMU
      3. memory.step(...)        → M55 updates
      4. pred.step(bmu_idx, ...) → L2 learns and outputs error
    """

    def __init__(self):
        # ── Prediction matrix ─────────────────────────────────
        # P[i, j] = strength of prediction: "context i → next BMU j"
        # Initialized to zero — no sequence knowledge at birth
        # NOT symmetric (unlike M55): prediction is directional
        #   "A predicts B" does not mean "B predicts A"
        self._P = np.zeros((N_NEURONS, N_NEURONS), dtype=np.float32)

        # ── Context vector ────────────────────────────────────
        # c[i] = eligibility of BMU i in current context
        # Decays exponentially — recent BMUs dominate
        self._c = np.zeros(N_NEURONS, dtype=np.float32)

        # Current adaptive context decay rate
        self._context_decay = CONTEXT_DECAY_BASE

        # ── Prediction state ──────────────────────────────────
        # Scores computed at last predict() call
        # Held until step() sees the actual BMU and computes error
        self._last_scores      = np.zeros(N_NEURONS, dtype=np.float32)
        self._last_predicted   = 0
        self._last_confidence  = 0.0
        self._prediction_ready = False   # True after first predict()

        # ── Curiosity signal ──────────────────────────────────
        # EMA of prediction error — sustained error = novel territory
        self._curiosity = 0.5   # start at moderate uncertainty

        # ── Diagnostics ───────────────────────────────────────
        self.t               = 0
        self._n_correct      = 0
        self._n_predictions  = 0
        self._error_history  = deque(maxlen=200)
        self._correct_history= deque(maxlen=200)

        # Per-BMU prediction accuracy (how often each BMU was
        # correctly predicted — shows which patterns L2 has learned)
        self._bmu_correct = np.zeros(N_NEURONS, dtype=np.int32)
        self._bmu_total   = np.zeros(N_NEURONS, dtype=np.int32)

    # ── Predict ───────────────────────────────────────────────

    def predict(self) -> dict:
        """
        Make a prediction about the next BMU BEFORE it fires.

        Call this at the start of each step, before cortex.step().

        Returns
        -------
        dict with:
            'predicted_bmu'  int   — most expected next BMU
            'confidence'     float — certainty of prediction [0,1]
            'scores'         array — full (64,) prediction distribution
        """
        # scores[j] = how strongly the current context predicts BMU j
        # = sum over all context neurons i of: P[i,j] * c[i]
        # = P.T @ c
        raw_scores = self._P.T @ self._c   # (64,)

        # Softmax normalization with temperature.
        #
        # WHY NOT sum-normalization:
        #   After the P matrix becomes dense (which happens quickly —
        #   every context BMU writes into nearly every outcome column),
        #   raw_scores has 60+ nonzero entries of similar magnitude.
        #   Dividing by their sum makes every normalized score ≈ 1/64.
        #   argmax picks based on tiny numerical differences.
        #   The correct prediction no longer dominates.
        #
        # WHY softmax:
        #   Softmax exponentiates before normalizing. Small absolute
        #   differences become large relative differences.
        #   If correct score = 0.8 and noise = 0.4:
        #     sum-norm → 0.8/50 vs 0.4/50 — nearly equal
        #     softmax  → exp(0.8/T) >> exp(0.4/T) — correct wins clearly
        #   This is exactly what the brain does — lateral inhibition
        #   creates competitive selection, not proportional weighting.
        score_max  = raw_scores.max()
        exp_scores = np.exp((raw_scores - score_max) / (SCORE_TEMPERATURE + 1e-9))
        norm_scores = exp_scores / (exp_scores.sum() + 1e-9)

        # Confidence: how peaked is the distribution?
        # Use softmax sharpness — ratio of top score to uniform baseline
        top_score = float(norm_scores.max())
        uniform   = 1.0 / N_NEURONS
        confidence = float(np.clip((top_score - uniform) / (1.0 - uniform + 1e-9),
                                   0.0, 1.0))

        predicted_bmu = int(np.argmax(norm_scores))

        # Store for error computation in step()
        self._last_scores     = norm_scores
        self._last_predicted  = predicted_bmu
        self._last_confidence = confidence
        self._prediction_ready = True

        return {
            'predicted_bmu': predicted_bmu,
            'confidence':    confidence,
            'scores':        norm_scores,
        }

    # ── Step ──────────────────────────────────────────────────

    def step(self, bmu_idx: int,
             qe_norm:         float = 0.0,
             familiarity:     float = 0.0,
             prediction_bias: np.ndarray = None) -> dict:
        """
        Update L2 after the actual BMU fires.

        Call this after cortex.step() and memory.step().

        Parameters
        ----------
        bmu_idx          : int      — actual BMU that fired (from M54)
        qe_norm          : float    — perceptual surprise from M54 [0,1]
        familiarity      : float    — recognition signal from M55 [0,1]
        prediction_bias  : ndarray  — (64,) soft distribution from Thought
                                       over expected next BMUs. If provided,
                                       gently pre-warms context toward the
                                       expected region before prediction.
                                       Defaults to None (no top-down bias).

        Returns
        -------
        dict with:
            'prediction_error'  float [0,1]
            'correct'           bool
            'predicted_bmu'     int
            'confidence'        float
            'context_decay'     float
            'eta'               float
            'curiosity'         float [0,1]
            'p_mean'            float
            'p_max'             float
        """
        # ── 1. Compute prediction error ───────────────────────
        # Spatial soft-match: error = 1 - exp(-dist² / 2σ²)
        # where dist is the grid distance between predicted and actual BMU.
        #
        # WHY NOT probability-based error (old: 1 - scores[bmu_idx]):
        # The SOM fires a region of ~20 BMU indices per frequency, so
        # L2 cannot learn to concentrate all prediction mass on one exact
        # index. scores[actual] stays low (~0.02-0.05) even after the
        # correct frequency zone is well-learned. Binary error stays at
        # ~0.97 permanently. Spatial error reflects what L2 actually knows:
        # it can learn the right frequency zone (low spatial error) even
        # if it cannot predict the exact index within that zone.
        #
        # The P matrix learning (below) still uses exact bmu_idx —
        # this change only affects what the error SIGNAL reports.
        if self._prediction_ready:
            predicted = self._last_predicted
            row_p, col_p = predicted // GRID_W, predicted % GRID_W
            row_a, col_a = bmu_idx   // GRID_W, bmu_idx   % GRID_W
            dist2 = float((row_p - row_a)**2 + (col_p - col_a)**2)
            spatial_correct = float(np.exp(-dist2 / (2.0 * SPATIAL_SIGMA**2)))
            error = float(np.clip(1.0 - spatial_correct, 0.0, 1.0))
        else:
            error = 1.0   # cold start — no prediction made yet

        # Familiarity modulation: recognised contexts are less surprising.
        # Scales down error proportionally to how familiar this BMU is.
        error = float(np.clip(
            error * (1.0 - FAMILIARITY_ERROR_SCALE * familiarity),
            0.0, 1.0
        ))

        correct = (self._last_predicted == bmu_idx) and self._prediction_ready

        # ── 2. Adapt context decay to prediction error ────────
        # High error → remember more context (slow decay, long window)
        # Low error  → prune context (fast decay, short window)
        # Same adaptive principle as M55's trace window
        self._context_decay = max(
            CONTEXT_DECAY_MIN,
            CONTEXT_DECAY_BASE - CONTEXT_ERROR_MODULATION * error
        )

        # ── 3. Decay context (all past BMUs fade) ─────────────
        self._c *= (1.0 - self._context_decay)

        # ── 3b. Inject Thought's prediction bias (top-down) ───
        # Thought pre-warms the context toward the BMU region it expects
        # to fire next. This is a gentle nudge — PREDICTION_BIAS_STRENGTH
        # keeps the bias contribution well below a full BMU imprint (1.0).
        #
        # Applied AFTER decay but BEFORE learning, so the bias participates
        # in the P update if it rises above MIN_CONTEXT_TO_LEARN.
        # This means: if Thought correctly anticipates the next BMU, that
        # pathway gets slightly more credit in P — reinforcing the prediction.
        #
        # Biologically: PFC pre-activation of striatal prediction circuits
        # before the actual stimulus arrives.
        if prediction_bias is not None:
            pb = np.asarray(prediction_bias, dtype=np.float32)
            pb_sum = pb.sum()
            if pb_sum > 1e-9:
                pb = pb / pb_sum   # ensure normalized
            self._c = np.clip(self._c + 0.10 * pb, 0.0, 1.0)

        # ── 4. Learn from context BEFORE imprinting current BMU ──
        # CRITICAL ORDER: learning must use the context that existed
        # BEFORE the current BMU fired. That causal context is what
        # should predict this BMU.
        #
        # Bug in original: imprint then learn.
        # c[bmu_idx] = 1.0 was set BEFORE P[:, bmu_idx] += eta * c.
        # This included the current BMU in its own predictive context,
        # strengthening P[bmu_idx, bmu_idx] every single step.
        # After enough training the diagonal dominated everything —
        # every BMU predicted itself. Sequence learning was buried.
        # Accuracy collapsed to zero after rep 1.
        #
        # Correct biological order: the pre-synaptic trace (context)
        # that existed BEFORE the post-synaptic spike (current BMU)
        # is what gets potentiated. The spike itself is the outcome,
        # not part of its own causal context.
        eta = ETA_BASE
        if error > ERROR_THRESH:
            eta += ETA_ERROR_BOOST * error

        active_context = self._c >= MIN_CONTEXT_TO_LEARN
        if active_context.sum() > 0:
            delta = eta * self._c * active_context.astype(np.float32)
            self._P[:, bmu_idx] += delta

        # Suppress self-prediction by excluding current BMU from its own
        # learning context. Zero P[bmu_idx, bmu_idx] after every write.
        #
        # WHY: In a short cycle A→B→A→B (period=2 steps), when A fires
        # at step t+2, context still has c[A] = (1-decay)^2 = 0.49.
        # Without suppression, P[A,A] gets written every cycle — A
        # predicts itself — and with MIN_CONTEXT_TO_LEARN=0.5, c[A]=0.49
        # is exactly below the threshold, so self-writes don't happen for
        # 2-step cycles. But for 1-step cycles (A→A→A), c[A]=0.70 which
        # IS above threshold — that's correct, self-loops should be learnable.
        # Setting the diagonal to zero after each write ensures:
        #   - Self-loops (A→A): P[A,A] gets written (c[A]=0.70 ≥ 0.5),
        #     but then zeroed → self-loops treated as unpredictable.
        #   - 2-step cycles (A→B→A): c[A]=0.49 < 0.5, not written.
        #
        # This is the biologically correct behavior: a neuron's OWN
        # activity should not predict itself — that would be tautological.
        # The eligibility trace of OTHER neurons predicts the current BMU.
        self._P[bmu_idx, bmu_idx] = 0.0

        # ── 5. NOW imprint current BMU into context ───────────
        # Available for the NEXT step's prediction only.
        self._c[bmu_idx] = 1.0

        # ── 6. Synaptic decay on P ────────────────────────────
        self._P *= (1.0 - P_DECAY)

        # ── 7. Column normalization (homeostasis) ─────────────
        # Each column (predictions for one outcome) is bounded
        # Prevents any single outcome from monopolizing the matrix
        # Note: column-wise (axis=0) because P is context × outcome
        col_max = self._P.max(axis=0, keepdims=True)   # (1, 64)
        scale   = np.where(col_max > P_MAX,
                           P_MAX / (col_max + 1e-9), 1.0)
        self._P *= scale

        # ── 8. Update curiosity (EMA of prediction error) ─────
        # Rises when system is consistently surprised by sequences
        # Falls when predictions improve
        # Sustained curiosity = genuinely novel sequence territory
        self._curiosity = ((1.0 - CURIOSITY_EMA_ALPHA) * self._curiosity
                           + CURIOSITY_EMA_ALPHA * error)

        # ── 9. Diagnostics ────────────────────────────────────
        self._error_history.append(error)
        self._correct_history.append(int(correct))
        self._bmu_total[bmu_idx] += 1
        if correct:
            self._n_correct += 1
            self._bmu_correct[bmu_idx] += 1
        self._n_predictions += 1
        self.t += 1

        return {
            'prediction_error': error,
            'correct':          correct,
            'predicted_bmu':    self._last_predicted,
            'confidence':       self._last_confidence,
            'context_decay':    float(self._context_decay),
            'eta':              float(eta),
            'curiosity':        float(self._curiosity),
            'p_mean':           float(self._P.mean()),
            'p_max':            float(self._P.max()),
        }

    # ── Diagnostics ───────────────────────────────────────────

    def accuracy(self) -> float:
        """Overall prediction accuracy so far."""
        if self._n_predictions == 0:
            return 0.0
        return float(self._n_correct / self._n_predictions)

    def recent_accuracy(self, window: int = 100) -> float:
        """Accuracy over last `window` predictions."""
        if not self._correct_history:
            return 0.0
        recent = list(self._correct_history)[-window:]
        return float(sum(recent) / len(recent))

    def recent_error(self, window: int = 100) -> float:
        """Mean prediction error over last `window` steps."""
        if not self._error_history:
            return 1.0
        recent = list(self._error_history)[-window:]
        return float(np.mean(recent))

    def top_predictions(self, bmu_idx: int, k: int = 5) -> list:
        """
        What does L2 predict will follow bmu_idx?
        Returns top-k (predicted_bmu, score) pairs.
        Based on current P matrix — reflects learned sequences.
        """
        # Seed context with just this one BMU at full strength
        c_seed     = np.zeros(N_NEURONS, dtype=np.float32)
        c_seed[bmu_idx] = 1.0
        scores     = self._P.T @ c_seed
        s_sum      = scores.sum()
        if s_sum > 1e-9:
            scores = scores / s_sum
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx
                if scores[i] > 1e-4]

    def get_state(self) -> dict:
        """Full diagnostic snapshot."""
        return {
            't':               self.t,
            'n_predictions':   self._n_predictions,
            'n_correct':       self._n_correct,
            'accuracy':        self.accuracy(),
            'recent_accuracy': self.recent_accuracy(),
            'recent_error':    self.recent_error(),
            'curiosity':       float(self._curiosity),
            'context_decay':   float(self._context_decay),
            'context_peak':    float(self._c.max()),
            'p_mean':          float(self._P.mean()),
            'p_max':           float(self._P.max()),
            'p_nonzero':       float((self._P > 1e-4).mean()),
            'P_snapshot':      self._P.copy(),
            'c_snapshot':      self._c.copy(),
        }

    def prediction_map(self) -> np.ndarray:
        """
        (8×8) array — total outgoing prediction strength per BMU.
        Bright spots = BMUs that strongly predict what comes next.
        Dark spots = BMUs with no learned sequence following them.
        """
        strength = self._P.sum(axis=1)   # (64,) — sum of outgoing weights
        return strength.reshape(8, 8)

    def summary(self):
        """Human-readable state summary."""
        s = self.get_state()
        print(f"  SequencePredictor (L2) — step {s['t']}")
        print(f"  Predictions:    {s['n_predictions']}  "
              f"correct={s['n_correct']}  "
              f"accuracy={s['accuracy']*100:.1f}%")
        print(f"  Recent (100):   accuracy={s['recent_accuracy']*100:.1f}%  "
              f"error={s['recent_error']:.4f}")
        print(f"  Curiosity:      {s['curiosity']:.4f}")
        print(f"  Context decay:  {s['context_decay']:.4f}  "
              f"(window≈{1/s['context_decay']:.0f} steps)")
        print(f"  P mean/max:     {s['p_mean']:.5f} / {s['p_max']:.4f}")
        print(f"  P nonzero:      {s['p_nonzero']*100:.1f}%")

        # BMUs with best prediction accuracy
        mask = self._bmu_total > 10   # only BMUs seen enough times
        if mask.any():
            acc = np.where(mask,
                           self._bmu_correct / (self._bmu_total + 1e-9),
                           -1.0)
            top = np.argsort(acc)[::-1][:3]
            print(f"  Best-predicted: "
                  + "  ".join(
                      f"BMU{i}({acc[i]*100:.0f}%)"
                      for i in top if acc[i] >= 0
                  ))