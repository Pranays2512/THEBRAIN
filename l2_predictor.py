"""
L2: SEQUENCE PREDICTOR — ELIGIBILITY-WEIGHTED PREDICTION
=========================================================
v2: Action-conditioned transition matrix (PA)

WHAT CHANGED FROM v1
--------------------
v1 learned P[i, j] = "BMU i tends to be followed by BMU j."
This is action-blind. The brain hears A, then hears B — but L2
did not know whether the brain moved East or stayed (wall). Both
transitions looked the same to L2 and both updated P[A→B].

The result: P became a passive frequency co-occurrence matrix.
It learned that A tends to follow B in the stream — but not that
"East from A" leads to B while "North from A" stays at A.
M57, reading L2's P, could not distinguish actions. All simulated
futures looked alike. argmax returned action 0 (North) always.

v2 adds PA: action-conditioned prediction matrix.
  PA[action, i, j] = strength of prediction:
    "after taking action `a`, BMU i in context predicts BMU j next"

Shape: (N_ACTIONS, 64, 64) — four 64×64 matrices, one per action.
Learning: when action `a` was taken and BMU k fired,
    PA[a, :, k] += eta * c   (same STDP rule, action-gated)

The un-conditioned P matrix is retained unchanged — it is used by
Thought and M55 for general sequence prediction. PA is used ONLY
by M57 for planning ("if I take action a, what comes next?").

PA LEARNING GATE
----------------
PA is only updated when a real world move happened (world_moved=True).
On wall hits, the brain stays at the same node — the transition
PA[wall_action, prev_bmu → same_bmu] would teach M57 that action
leads nowhere, which is exactly what we want: PA for a wall action
accumulates self-transitions (low value), while PA for the correct
exit action accumulates transitions to the next node's BMU (high value).

BACKWARD COMPATIBILITY
----------------------
All existing L2 outputs are unchanged.
New output key: 'pa_ready' — True once PA has enough data to use.
New method: top_predictions_for_action(bmu_idx, action, k)
  Used by M57 instead of top_predictions() when planning.
  Falls back to top_predictions() if PA is not yet populated.

PA PARAMETERS
-------------
PA_MIN_TRANSITIONS = 5   — minimum writes to any PA[a] slice before
                           M57 is allowed to use it. Below this, the
                           slice is too sparse and top_predictions_for_action
                           falls back to the unconditional P.
"""

import numpy as np
from collections import deque
import math


# ═══════════════════════════════════════════════════════════════
# PARAMETERS (unchanged from v1 except PA additions)
# ═══════════════════════════════════════════════════════════════

N_NEURONS = 64
N_ACTIONS = 4

ETA_BASE             = 0.05
ETA_ERROR_BOOST      = 0.10
ERROR_THRESH         = 0.5
P_DECAY              = 0.001
P_MAX                = 1.0
CONTEXT_DECAY_BASE   = 0.30
CONTEXT_DECAY_MIN    = 0.10
CONTEXT_ERROR_MODULATION = 0.20
MIN_CONTEXT_TO_LEARN = 0.50
CURIOSITY_EMA_ALPHA  = 0.05
SPATIAL_SIGMA        = 2.0
FAMILIARITY_ERROR_SCALE = 0.3
GRID_W               = 8
SCORE_TEMPERATURE    = 0.15

# ── Action-conditioned matrix ─────────────────────────────────
# PA[action, i, j] = context i predicts j when action was taken
# Same decay and normalisation as P.
PA_DECAY             = 0.001   # same as P_DECAY
PA_MAX               = 1.0     # same normalisation ceiling

# Minimum total writes (across all BMU pairs) to a PA[action] slice
# before top_predictions_for_action() trusts it over the unconditional P.
# At 5 writes, the slice has at least seen a handful of real transitions.
PA_MIN_TRANSITIONS   = 5


# ═══════════════════════════════════════════════════════════════
# SEQUENCE PREDICTOR
# ═══════════════════════════════════════════════════════════════

class SequencePredictor:
    """
    Eligibility-weighted sequence predictor for the M54+M55 stack.

    v2 adds PA: action-conditioned prediction matrix used by M57.
    All existing interface is unchanged. New interface:
      pred.step(..., last_action=a, world_moved=True)
      pred.top_predictions_for_action(bmu_idx, action, k)
    """

    def __init__(self):
        # ── Prediction matrix (unconditional) ────────────────
        self._P = np.zeros((N_NEURONS, N_NEURONS), dtype=np.float32)

        # ── Action-conditioned prediction matrix ──────────────
        # PA[a, i, j] — for each action, context i → outcome j
        self._PA = np.zeros((N_ACTIONS, N_NEURONS, N_NEURONS), dtype=np.float32)

        # Track total writes per action slice (for PA_MIN_TRANSITIONS gate)
        self._PA_writes = np.zeros(N_ACTIONS, dtype=np.int32)

        # ── Context vector ────────────────────────────────────
        self._c = np.zeros(N_NEURONS, dtype=np.float32)
        self._context_decay = CONTEXT_DECAY_BASE

        # ── Prediction state ──────────────────────────────────
        self._last_scores      = np.zeros(N_NEURONS, dtype=np.float32)
        self._last_predicted   = 0
        self._last_confidence  = 0.0
        self._prediction_ready = False

        # ── Last action (set by step(), used for PA update) ───
        self._last_action    = -1    # -1 = unknown / wall hit
        self._last_moved     = False # was last step a real world move?

        # ── Curiosity signal ──────────────────────────────────
        self._curiosity = 0.5

        # ── Diagnostics ───────────────────────────────────────
        self.t               = 0
        self._n_correct      = 0
        self._n_predictions  = 0
        self._error_history  = deque(maxlen=200)
        self._correct_history= deque(maxlen=200)
        self._bmu_correct    = np.zeros(N_NEURONS, dtype=np.int32)
        self._bmu_total      = np.zeros(N_NEURONS, dtype=np.int32)

    # ── Predict ───────────────────────────────────────────────

    def predict(self) -> dict:
        """
        Make a prediction about the next BMU BEFORE it fires.
        Unchanged from v1 — uses unconditional P.
        """
        raw_scores = self._P.T @ self._c

        score_max  = raw_scores.max()
        exp_scores = np.exp((raw_scores - score_max) / (SCORE_TEMPERATURE + 1e-9))
        norm_scores = exp_scores / (exp_scores.sum() + 1e-9)

        top_score  = float(norm_scores.max())
        uniform    = 1.0 / N_NEURONS
        confidence = float(np.clip((top_score - uniform) / (1.0 - uniform + 1e-9),
                                   0.0, 1.0))
        predicted_bmu = int(np.argmax(norm_scores))

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
             prediction_bias: np.ndarray = None,
             last_action:     int   = -1,
             world_moved:     bool  = True) -> dict:
        """
        Update L2 after the actual BMU fires.

        Parameters (new in v2)
        ----------------------
        last_action : int
            The action that was taken before arriving at this BMU.
            -1 if unknown. Used to update PA[last_action].
        world_moved : bool
            True if a real world transition occurred.
            False on wall hits. PA is updated regardless — wall hits
            teach PA[wall_action] that action leads to self-transitions
            (same BMU), which correctly depresses its value in M57.
        """
        # ── 1. Prediction error (unchanged) ───────────────────
        if self._prediction_ready:
            predicted = self._last_predicted
            row_p, col_p = predicted // GRID_W, predicted % GRID_W
            row_a, col_a = bmu_idx   // GRID_W, bmu_idx   % GRID_W
            dist2 = float((row_p - row_a)**2 + (col_p - col_a)**2)
            spatial_correct = float(np.exp(-dist2 / (2.0 * SPATIAL_SIGMA**2)))
            error = float(np.clip(1.0 - spatial_correct, 0.0, 1.0))
        else:
            error = 1.0

        error = float(np.clip(
            error * (1.0 - FAMILIARITY_ERROR_SCALE * familiarity),
            0.0, 1.0
        ))
        correct = (self._last_predicted == bmu_idx) and self._prediction_ready

        # ── 2. Adapt context decay ────────────────────────────
        self._context_decay = max(
            CONTEXT_DECAY_MIN,
            CONTEXT_DECAY_BASE - CONTEXT_ERROR_MODULATION * error
        )

        # ── 3. Decay context ──────────────────────────────────
        self._c *= (1.0 - self._context_decay)

        # ── 3b. Thought prediction bias (unchanged) ───────────
        if prediction_bias is not None:
            pb = np.asarray(prediction_bias, dtype=np.float32)
            pb_sum = pb.sum()
            if pb_sum > 1e-9:
                pb = pb / pb_sum
            self._c = np.clip(self._c + 0.10 * pb, 0.0, 1.0)

        # ── 4. Learn P (unconditional) — BEFORE imprinting ────
        eta = ETA_BASE
        if error > ERROR_THRESH:
            eta += ETA_ERROR_BOOST * error

        active_context = self._c >= MIN_CONTEXT_TO_LEARN
        if active_context.sum() > 0:
            delta = eta * self._c * active_context.astype(np.float32)
            self._P[:, bmu_idx] += delta

        self._P[bmu_idx, bmu_idx] = 0.0   # no self-prediction

        # ── 4b. Learn PA (action-conditioned) ─────────────────
        # Update PA[action] when we know which action was just taken.
        # This runs for BOTH real moves and wall hits:
        #   - Real move (world_moved=True):
        #       context was at prev_node BMU, action led to curr_node BMU.
        #       PA[action, prev_bmu_region → curr_bmu_region] strengthened.
        #   - Wall hit (world_moved=False):
        #       context was at node BMU, action led to same BMU.
        #       PA[wall_action, node_bmu → same_bmu] strengthened.
        #       This teaches M57 that this action at this node goes nowhere.
        #
        # We do NOT update PA on the very first step (last_action=-1).
        if last_action >= 0 and last_action < N_ACTIONS:
            if active_context.sum() > 0:
                self._PA[last_action, :, bmu_idx] += delta   # same delta as P
                self._PA[last_action, bmu_idx, bmu_idx] = 0.0  # no self-prediction
                self._PA_writes[last_action] += int(active_context.sum())

        # ── 5. Imprint current BMU into context ───────────────
        self._c[bmu_idx] = 1.0

        # ── 6. Synaptic decay ─────────────────────────────────
        self._P  *= (1.0 - P_DECAY)
        self._PA *= (1.0 - PA_DECAY)

        # ── 7. Column normalisation ───────────────────────────
        # P
        col_max = self._P.max(axis=0, keepdims=True)
        scale   = np.where(col_max > P_MAX, P_MAX / (col_max + 1e-9), 1.0)
        self._P *= scale

        # PA — normalise each action slice independently
        for a in range(N_ACTIONS):
            col_max_a = self._PA[a].max(axis=0, keepdims=True)
            scale_a   = np.where(col_max_a > PA_MAX,
                                 PA_MAX / (col_max_a + 1e-9), 1.0)
            self._PA[a] *= scale_a

        # ── 8. Curiosity ──────────────────────────────────────
        self._curiosity = ((1.0 - CURIOSITY_EMA_ALPHA) * self._curiosity
                           + CURIOSITY_EMA_ALPHA * error)

        # ── 9. Store last action for diagnostics ──────────────
        self._last_action = last_action
        self._last_moved  = world_moved

        # ── 10. Diagnostics ───────────────────────────────────
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
            'pa_ready':         bool(self._PA_writes.min() >= PA_MIN_TRANSITIONS),
        }

    # ── Query ─────────────────────────────────────────────────

    def top_predictions(self, bmu_idx: int, k: int = 5) -> list:
        """
        Unconditional predictions: what tends to follow bmu_idx?
        Unchanged from v1. Used by Thought and M55.
        """
        c_seed = np.zeros(N_NEURONS, dtype=np.float32)
        c_seed[bmu_idx] = 1.0
        scores = self._P.T @ c_seed
        s_sum  = scores.sum()
        if s_sum > 1e-9:
            scores = scores / s_sum
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 1e-4]

    def top_predictions_for_action(self, bmu_idx: int,
                                   action: int,
                                   k: int = 5) -> list:
        """
        Action-conditioned predictions: given I'm at bmu_idx and take
        action `action`, what BMU do I expect to arrive at?

        Uses PA[action] if it has enough data (PA_writes >= PA_MIN_TRANSITIONS).
        Falls back to unconditional top_predictions() otherwise.

        This is the key method M57 calls instead of top_predictions()
        so its planning is action-aware.

        Returns list of (predicted_bmu, score) pairs.
        """
        if (action < 0 or action >= N_ACTIONS or
                self._PA_writes[action] < PA_MIN_TRANSITIONS):
            # PA not ready for this action — fall back to unconditional
            return self.top_predictions(bmu_idx, k)

        c_seed = np.zeros(N_NEURONS, dtype=np.float32)
        c_seed[bmu_idx] = 1.0
        scores = self._PA[action].T @ c_seed
        s_sum  = scores.sum()
        if s_sum > 1e-9:
            scores = scores / s_sum
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 1e-4]

    def pa_ready(self) -> bool:
        """True when all PA action slices have enough data."""
        return bool(self._PA_writes.min() >= PA_MIN_TRANSITIONS)

    def pa_ready_per_action(self) -> list:
        """Per-action readiness."""
        return [bool(self._PA_writes[a] >= PA_MIN_TRANSITIONS)
                for a in range(N_ACTIONS)]

    # ── Diagnostics (unchanged) ───────────────────────────────

    def accuracy(self) -> float:
        if self._n_predictions == 0:
            return 0.0
        return float(self._n_correct / self._n_predictions)

    def recent_accuracy(self, window: int = 100) -> float:
        if not self._correct_history:
            return 0.0
        recent = list(self._correct_history)[-window:]
        return float(sum(recent) / len(recent))

    def recent_error(self, window: int = 100) -> float:
        if not self._error_history:
            return 1.0
        recent = list(self._error_history)[-window:]
        return float(np.mean(recent))

    def get_state(self) -> dict:
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
            'pa_writes':       self._PA_writes.tolist(),
            'pa_ready':        self.pa_ready(),
            'P_snapshot':      self._P.copy(),
            'c_snapshot':      self._c.copy(),
        }

    def prediction_map(self) -> np.ndarray:
        strength = self._P.sum(axis=1)
        return strength.reshape(8, 8)

    def summary(self):
        s = self.get_state()
        print(f"  SequencePredictor (L2 v2) — step {s['t']}")
        print(f"  Predictions:    {s['n_predictions']}  "
              f"correct={s['n_correct']}  "
              f"accuracy={s['accuracy']*100:.1f}%")
        print(f"  Recent (100):   accuracy={s['recent_accuracy']*100:.1f}%  "
              f"error={s['recent_error']:.4f}")
        print(f"  Curiosity:      {s['curiosity']:.4f}")
        print(f"  P mean/max:     {s['p_mean']:.5f} / {s['p_max']:.4f}")
        print(f"  PA writes:      {s['pa_writes']}  ready={s['pa_ready']}")

        mask = self._bmu_total > 10
        if mask.any():
            acc = np.where(mask,
                           self._bmu_correct / (self._bmu_total + 1e-9), -1.0)
            top = np.argsort(acc)[::-1][:3]
            print(f"  Best-predicted: "
                  + "  ".join(
                      f"BMU{i}({acc[i]*100:.0f}%)"
                      for i in top if acc[i] >= 0))