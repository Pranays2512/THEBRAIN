"""
THOUGHT — TOP-DOWN PREDICTION AND VOLUNTARY ATTENTION  (v6)
============================================================

v6: prediction_bias now blends L2 sequence memory with M55 associative memory.
Previously Thought only used L2's top_predictions (temporal: "what fires next
in sequence"). Now it also reads M55's weight row for the attended BMU
(associative: "what tends to co-occur with this BMU") and mixes the two sources.

New parameters:   W_ASSOC_L2, W_ASSOC_M55, MIN_ASSOC_STRENGTH.
New output key:   assoc_weight — fraction of final bias mass from M55.
New step() arg:   memory (AssociativeMemory instance or None).
Backward compat:  memory=None produces identical output to v5 Thought.

WHAT THIS IS
------------
Thought is the highest layer of the current cognitive stack. It sits above
Attention and closes the top-down loop that bottom-up Attention cannot.

Bottom-up attention (Attention module): "something surprising happened →
attend to it." Reactive. Driven by incoming signal properties.

Top-down attention (Thought): "I expect X to happen next → pre-attend to
that region." Predictive. Driven by learned sequence knowledge.

Together they implement the two-component model of biological attention:
  Attention module  ≈  thalamic bottom-up saliency (superior colliculus)
  Thought           ≈  prefrontal cortex top-down bias (PFC → thalamus)

WHAT THOUGHT DOES
-----------------
Each step, Thought:

  1. Reads the attended_bmu from Attention — "where is attention now?"

  2. Queries L2's sequence memory: "given this BMU fired, what does L2
     expect next?" → top_predictions(attended_bmu) → sparse distribution
     over the 64 cortical neurons

  3. Builds a prediction_bias vector (64,): soft probability mass over
     the BMUs L2 most expects next. This encodes what Thought "imagines"
     will happen in the next step.

  4. Computes thought_confidence: how concentrated the prediction is.
     High = "I have a clear expectation." Low = "I have no idea."

  5. Tracks expectation_error: was the PREVIOUS step's expected_bmu
     close to what actually fired? This gives Thought a self-monitoring
     signal — how well is it predicting?

  6. On the NEXT step, Brain feeds:
       prediction_bias → L2: pre-warms the context vector toward expected
                              next BMU, giving L2 a head start on prediction
       thought_confidence_delta → Attention: dampens salience when Thought's
                                  expectation is being met (if I predicted it,
                                  it is less surprising)

BIOLOGICAL BASIS
----------------
Prefrontal cortex (PFC) → thalamus pathway:
  - PFC maintains working memory of current task/context
  - Sends top-down signals to the thalamic reticular nucleus
  - This selectively gates which thalamic relay nuclei forward information
  - Net effect: attended information gets amplified; expected-but-attended
    information gets slightly suppressed (prediction dampening)

The prediction_bias → L2 pathway models the PFC → striatum projection:
  - PFC sends predictions about upcoming stimuli to basal ganglia
  - Basal ganglia use this to "prime" upcoming sequence predictions
  - This is why humans can predict sentence endings, melody completions, etc.

FEEDBACK RULES (Guide Rule 1)
------------------------------
prediction_bias (64,) → L2:
  This is NOT a scalar signal with a floor problem. It is a sparse direction
  vector (soft probability over BMUs). At cold start it is uniform (1/64 each).
  L2 uses it by adding a small fraction to the context vector c before
  prediction:
      c += PREDICTION_BIAS_STRENGTH * prediction_bias
  The bias is gentle — it nudges, doesn't dominate. L2's own context still
  drives prediction. At PREDICTION_BIAS_STRENGTH=0.10, the bias contribution
  is ~10% of context strength. No delta needed — the vector is already
  relative (it sums to 1.0, not a signal with an inflated absolute baseline).

thought_confidence_delta → Attention:
  thought_confidence (raw scalar) has a cold-start floor near 0 and grows
  as L2 learns sequences. Feeding raw confidence to Attention would
  permanently suppress salience after learning — circular collapse. Instead,
  feed the delta: how much ABOVE the recent baseline is confidence this step?
  Near-zero when confidence is stably high. Positive when confidence spikes
  (Thought suddenly knows what comes next). Negative spikes are clipped to 0.

CALL ORDER
----------
  8. thought.step(...)   ← reads Attention output + L2 top_predictions
                            stores prediction_bias and confidence_delta
                            for NEXT step

On the NEXT step, Brain feeds to L2:
    pred.step(..., prediction_bias=thought._last_prediction_bias)
And to Attention:
    attention.step(..., thought_confidence_delta=thought._last_confidence_delta)

Thought is read-only with respect to the current step.
It only affects the NEXT step via the stored outputs.

OUTPUTS
-------
expected_bmu         int          — BMU Thought most expects next
prediction_bias      ndarray(64,) — soft probability over expected next BMUs
thought_confidence   float [0,1]  — how concentrated the prediction is
confidence_ema       float [0,1]  — smoothed confidence (EMA)
confidence_delta     float [0,1]  — spike above EMA (for Attention feedback)
expectation_error    float [0,1]  — was last step's expected_bmu close?
focus_entropy        float [0,1]  — prediction spread (0=certain, 1=uniform)
t                    int          — step counter

INTERFACE
---------
  from thought import Thought

  thought = Thought()

  # Standalone (no Brain needed):
  result = thought.step(
      attended_bmu = 20,
      bmu_idx      = 20,
      pred         = None,   # SequencePredictor — if None, returns uniform bias
      salience     = 0.3,
  )

  # With Brain (called after attention.step inside Brain.step):
  thought_out = thought.step(
      attended_bmu = attn_out['attended_bmu'],
      bmu_idx      = bmu_idx,
      pred         = self.pred,          # L2 instance — for top_predictions
      salience     = attn_out['salience'],
  )

  # Outputs fed NEXT step:
  # pred.step(..., prediction_bias=thought._last_prediction_bias)
  # attention.step(..., thought_confidence_delta=thought._last_confidence_delta)
"""

import numpy as np
import math
from collections import deque


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Grid (must match all other modules)
N_NEURONS = 64
GRID_W    = 8

# ── Prediction bias strength ─────────────────────────────────
# How strongly Thought's prediction_bias nudges L2's context vector.
# c += PREDICTION_BIAS_STRENGTH * prediction_bias
# At 0.10: bias is ~10% of a full BMU imprint (c[bmu]=1.0).
# At 0.30: bias becomes dominant — Thought overrides L2's own context.
# Keep low (≤0.20) until Thought has been validated with feedback.
PREDICTION_BIAS_STRENGTH = 0.10

# How many top predictions to include in the bias vector.
# Top 5 covers the likely-next cluster without spreading mass too thin.
TOP_K_PREDICTIONS = 5

# ── M55 associative memory blend ─────────────────────────
# The final prediction_bias is a weighted blend of:
#   L2 signal  — top_predictions(attended_bmu): temporal sequence memory
#   M55 signal — W[attended_bmu]: Hebbian co-occurrence associations
#
# W_ASSOC_L2 + W_ASSOC_M55 = 1.0 (enforced at runtime).
# L2 is kept dominant (0.70) because it is directional and temporally precise.
# M55 is secondary (0.30) — it adds co-occurrence context but is undirected.
#
# Biological basis:
#   L2  ≈ striatal sequence prediction (basal ganglia → PFC)
#   M55 ≈ hippocampal pattern completion (CA3 → CA1 → neocortex)
#   Blending mirrors the dual-route model of prediction in primate cortex.
#
# Keep W_ASSOC_M55 ≤ 0.50. Above that, undirected co-occurrence dominates
# the temporal sequence signal and L2 stops being the primary driver.
W_ASSOC_L2  = 0.70
W_ASSOC_M55 = 0.30

# Minimum M55 weight to include a neuron in the associative bias.
# Filters out noise from early training when W is near zero.
# Matches the threshold used in memory.recall() top_associations.
MIN_ASSOC_STRENGTH = 1e-4

# ── Confidence EMA ───────────────────────────────────────────
# tau = 1/alpha steps.
# At 0.15: tau ~7 steps — responds within 1 frequency cycle.
# Fast enough to track learning progress, slow enough not to jitter.
CONFIDENCE_EMA_ALPHA = 0.15

# Cold-start EMA value.
# 0.0 means no confidence at birth — no spurious delta on step 1.
CONFIDENCE_EMA_INIT = 0.0

# ── Expectation error ────────────────────────────────────────
# Spatial soft-match for Thought's own prediction (matches L2's formula).
# error = 1 - exp(-dist² / 2σ²), SIGMA = 2.0 grid cells.
# Tracks how well Thought is doing independently of L2.
EXPECTATION_SIGMA = 2.0

# ── Salience gate for building bias ──────────────────────────
# Only build a strong prediction bias when salience is above this threshold.
# Below threshold, prediction_bias defaults to uniform (no strong expectation).
# Prevents Thought from confidently predicting in low-salience / diffuse states
# where attention itself is uncertain.
MIN_SALIENCE_FOR_BIAS = 0.05   # very low floor — Thought is almost always active

# Precomputed grid distances (same pattern as attention.py)
def _build_grid_dist_sq():
    dist_sq = np.zeros((N_NEURONS, N_NEURONS), dtype=np.float32)
    for i in range(N_NEURONS):
        ri, ci = divmod(i, GRID_W)
        for j in range(N_NEURONS):
            rj, cj = divmod(j, GRID_W)
            dist_sq[i, j] = (ri - rj)**2 + (ci - cj)**2
    return dist_sq

_GRID_DIST_SQ = _build_grid_dist_sq()


# ═══════════════════════════════════════════════════════════════
# THOUGHT
# ═══════════════════════════════════════════════════════════════

class Thought:
    """
    Top-down prediction and voluntary attention bias for the Brain stack.

    Reads Attention's output and L2's sequence memory each step.
    Produces a prediction_bias vector (64,) fed to L2 next step,
    and a confidence_delta scalar fed to Attention next step.

    Does NOT modify any module directly — Brain passes its outputs
    as arguments on the following step.

    All Brain-fed parameters default to safe values for standalone operation.
    """

    def __init__(self):
        # ── Confidence EMA ────────────────────────────────────
        self._confidence_ema = float(CONFIDENCE_EMA_INIT)

        # ── One-step-delayed outputs for Brain to pass downward ──
        # Stored after each step, fed into L2 and Attention on the NEXT step.
        self._last_prediction_bias      = np.full(N_NEURONS, 1.0 / N_NEURONS,
                                                  dtype=np.float32)
        self._last_confidence_delta     = 0.0
        self._last_expected_bmu         = 0

        # ── Self-monitoring ───────────────────────────────────
        # Track how well Thought's own predictions are doing
        self._prev_expected_bmu         = -1   # -1 = no prediction yet
        self._expectation_error_history = deque(maxlen=200)
        self._n_predictions             = 0
        self._n_close                   = 0    # within EXPECTATION_SIGMA grid cells

        # ── Diagnostics ───────────────────────────────────────
        self._confidence_history        = deque(maxlen=200)
        self._last_confidence           = 0.0
        self._last_expectation_error    = 0.0
        self._last_focus_entropy        = 1.0
        self._last_assoc_weight         = 0.0
        self.t                          = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             attended_bmu: int,
             bmu_idx:      int,
             pred,                       # SequencePredictor instance or None
             salience:     float = 0.0,
             memory        = None,       # AssociativeMemory instance or None
             ) -> dict:
        """
        One Thought step.

        Parameters
        ----------
        attended_bmu : int
            Which cortical neuron Attention is focused on (from Attention).
        bmu_idx : int
            Which cortical neuron actually fired this step (from M54).
        pred : SequencePredictor or None
            L2 instance — used to query top_predictions. If None, returns
            uniform bias (safe for standalone testing).
        salience : float
            Current salience from Attention [0,1]. Low salience → weaker bias.
        memory : AssociativeMemory or None
            M55 instance — used to read W[attended_bmu] for associative blend.
            If None, only L2 sequence memory contributes (v5 behaviour).
            NOTE: reads memory._W directly (no recall settling) — fast lookup.

        Returns
        -------
        dict with keys:
            expected_bmu          int          — most expected next BMU
            prediction_bias       ndarray(64,) — soft distribution over next BMUs
            thought_confidence    float [0,1]  — how concentrated the prediction
            confidence_ema        float [0,1]  — smoothed confidence
            confidence_delta      float [0,1]  — spike above EMA (for Attention)
            expectation_error     float [0,1]  — was prev expected_bmu close?
            focus_entropy         float [0,1]  — prediction spread (0=focused)
            assoc_weight          float [0,1]  — fraction of bias mass from M55
            t                     int          — step counter
        """
        # ── 1. Compute expectation_error for PREVIOUS prediction ──
        # Did last step's expected_bmu end up close to this step's bmu_idx?
        if self._prev_expected_bmu >= 0:
            row_e, col_e = divmod(self._prev_expected_bmu, GRID_W)
            row_a, col_a = divmod(bmu_idx,                GRID_W)
            dist2 = float((row_e - row_a)**2 + (col_e - col_a)**2)
            expectation_error = float(
                np.clip(1.0 - np.exp(-dist2 / (2.0 * EXPECTATION_SIGMA**2)),
                        0.0, 1.0)
            )
            self._n_predictions += 1
            if dist2 <= (EXPECTATION_SIGMA ** 2):   # within 1σ = close enough
                self._n_close += 1
        else:
            expectation_error = 1.0   # cold start

        self._expectation_error_history.append(expectation_error)
        self._last_expectation_error = expectation_error

        # ── 2. Build prediction_bias: blend L2 + M55 ────────────
        # L2  source: top_predictions(attended_bmu) — temporal sequence memory
        #             "what is likely to fire NEXT in the learned sequence?"
        # M55 source: W[attended_bmu] row             — associative co-occurrence
        #             "what has tended to fire AT THE SAME TIME as this BMU?"
        #
        # Both are normalised to probability vectors before blending.
        # Final bias = W_ASSOC_L2 * l2_bias + W_ASSOC_M55 * m55_bias.
        # If either source is unavailable (None) or below threshold, its weight
        # is redistributed to the other source so the blend always sums to 1.
        #
        # NOTE: we read memory._W directly — a fast weight lookup that does NOT
        # trigger M55's recall settling loop (which runs inside memory.recall()).
        # recall() is already called inside Brain.step() for familiarity; we
        # don't call it again here to avoid double-settling cost.

        bias       = np.full(N_NEURONS, 0.0, dtype=np.float32)
        assoc_weight = 0.0   # fraction of final bias mass that came from M55

        if float(salience) < MIN_SALIENCE_FOR_BIAS:
            # Salience too low — uniform prior, no prediction
            bias = np.full(N_NEURONS, 1.0 / N_NEURONS, dtype=np.float32)
        else:
            # ── L2 bias ──────────────────────────────────────────
            l2_bias = np.zeros(N_NEURONS, dtype=np.float32)
            if pred is not None:
                top = pred.top_predictions(attended_bmu, k=TOP_K_PREDICTIONS)
                for idx, score in top:
                    l2_bias[idx] += float(score)
            l2_sum = float(l2_bias.sum())
            if l2_sum > 1e-9:
                l2_bias = l2_bias / l2_sum
            else:
                l2_bias = np.full(N_NEURONS, 1.0 / N_NEURONS, dtype=np.float32)
                l2_sum  = 0.0   # signal absent

            # ── M55 bias ─────────────────────────────────────────
            m55_bias = np.zeros(N_NEURONS, dtype=np.float32)
            if memory is not None:
                row = memory._W[attended_bmu].copy()
                row[attended_bmu] = 0.0   # exclude self-association
                # Gate on minimum strength — noise floor from early training
                row[row < MIN_ASSOC_STRENGTH] = 0.0
                m55_sum = float(row.sum())
                if m55_sum > 1e-9:
                    m55_bias = (row / m55_sum).astype(np.float32)
                else:
                    m55_bias = np.full(N_NEURONS, 1.0 / N_NEURONS, dtype=np.float32)
                    m55_sum  = 0.0   # signal absent
            else:
                m55_sum = 0.0

            # ── Blend ─────────────────────────────────────────────
            # If a source has no signal (sum was zero), redistribute its weight.
            has_l2  = l2_sum  > 1e-9
            has_m55 = m55_sum > 1e-9

            if has_l2 and has_m55:
                w_l2  = W_ASSOC_L2
                w_m55 = W_ASSOC_M55
            elif has_l2:
                w_l2  = 1.0
                w_m55 = 0.0
            elif has_m55:
                w_l2  = 0.0
                w_m55 = 1.0
            else:
                # Neither source has signal — fall back to uniform
                bias = np.full(N_NEURONS, 1.0 / N_NEURONS, dtype=np.float32)
                w_l2 = w_m55 = 0.0

            if w_l2 > 0.0 or w_m55 > 0.0:
                bias = (w_l2 * l2_bias + w_m55 * m55_bias).astype(np.float32)
                b_sum = float(bias.sum())
                if b_sum > 1e-9:
                    bias = bias / b_sum
                else:
                    bias = np.full(N_NEURONS, 1.0 / N_NEURONS, dtype=np.float32)

            # Fraction of mass contributed by M55 (diagnostic)
            if has_m55 and (w_l2 > 0.0 or w_m55 > 0.0):
                assoc_weight = float(w_m55)   # = W_ASSOC_M55 when both present

        bias = bias.astype(np.float32)

        # ── 3. Compute thought_confidence ────────────────────
        # How concentrated is the prediction?
        # max(bias) - 1/N is the deviation above uniform baseline.
        # 0 = no idea (uniform). ~0.35+ = clear expectation.
        raw_confidence = float(np.clip(
            bias.max() - (1.0 / N_NEURONS),
            0.0, 1.0
        ))

        # ── 4. Compute confidence_delta (Rule 1) ─────────────
        # Only upward spikes propagate to Attention.
        # Near-zero when confidence is stably high.
        # Positive when Thought suddenly forms a strong prediction.
        confidence_delta = float(np.clip(
            raw_confidence - self._confidence_ema,
            0.0, 1.0
        ))

        # Update EMA AFTER computing delta
        self._confidence_ema = ((1.0 - CONFIDENCE_EMA_ALPHA) * self._confidence_ema
                                + CONFIDENCE_EMA_ALPHA * raw_confidence)

        # ── 5. Expected BMU — peak of bias distribution ───────
        expected_bmu = int(np.argmax(bias))

        # ── 6. Focus entropy ──────────────────────────────────
        # How spread out is the prediction distribution?
        # 0 = all mass on one BMU (maximally certain)
        # 1 = uniform (no idea)
        log_bias   = np.log(bias + 1e-9)
        entropy    = float(-np.sum(bias * log_bias))
        max_entropy = math.log(N_NEURONS)
        focus_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))

        # ── 7. Store for NEXT step ────────────────────────────
        self._last_prediction_bias  = bias
        self._last_confidence_delta = confidence_delta
        self._last_expected_bmu     = expected_bmu
        self._prev_expected_bmu     = expected_bmu
        self._last_confidence       = raw_confidence
        self._last_focus_entropy    = focus_entropy
        self._last_assoc_weight     = assoc_weight

        self._confidence_history.append(raw_confidence)
        self.t += 1

        return {
            'expected_bmu':       expected_bmu,
            'prediction_bias':    bias,
            'thought_confidence': raw_confidence,
            'confidence_ema':     self._confidence_ema,
            'confidence_delta':   confidence_delta,
            'expectation_error':  expectation_error,
            'focus_entropy':      focus_entropy,
            'assoc_weight':       assoc_weight,
            't':                  self.t,
        }

    # ── Convenience accessors ─────────────────────────────────

    def expectation_accuracy(self) -> float:
        """Fraction of predictions where expected_bmu landed within σ of actual."""
        if self._n_predictions == 0:
            return 0.0
        return float(self._n_close / self._n_predictions)

    def get_state(self) -> dict:
        """Full diagnostic snapshot."""
        return {
            't':                     self.t,
            'thought_confidence':    self._last_confidence,
            'confidence_ema':        self._confidence_ema,
            'confidence_delta':      self._last_confidence_delta,
            'expected_bmu':          self._last_expected_bmu,
            'expectation_error':     self._last_expectation_error,
            'focus_entropy':         self._last_focus_entropy,
            'assoc_weight':          self._last_assoc_weight,
            'expectation_accuracy':  self.expectation_accuracy(),
            'confidence_mean':       float(np.mean(self._confidence_history))
                                     if self._confidence_history else 0.0,
        }

    def reset(self):
        """Reset all state — use between test conditions."""
        self._confidence_ema            = float(CONFIDENCE_EMA_INIT)
        self._last_prediction_bias      = np.full(N_NEURONS, 1.0 / N_NEURONS,
                                                  dtype=np.float32)
        self._last_confidence_delta     = 0.0
        self._last_expected_bmu         = 0
        self._prev_expected_bmu         = -1
        self._expectation_error_history.clear()
        self._n_predictions             = 0
        self._n_close                   = 0
        self._confidence_history.clear()
        self._last_confidence           = 0.0
        self._last_expectation_error    = 0.0
        self._last_focus_entropy        = 1.0
        self._last_assoc_weight         = 0.0
        self.t                          = 0

    def summary(self):
        """Human-readable state summary."""
        s = self.get_state()
        print(f"  Thought — step {s['t']}")
        print(f"  Confidence:       {s['thought_confidence']:.4f}  "
              f"(ema={s['confidence_ema']:.4f}  delta={s['confidence_delta']:.4f})")
        print(f"  Expected BMU:     {s['expected_bmu']}  "
              f"focus_entropy={s['focus_entropy']:.4f}  (0=certain, 1=diffuse)")
        print(f"  Expectation err:  {s['expectation_error']:.4f}  "
              f"accuracy={s['expectation_accuracy']*100:.1f}%  "
              f"({self._n_close}/{self._n_predictions} close)")
        print(f"  Assoc weight:     {s['assoc_weight']:.3f}  "
              f"(M55 fraction of bias — 0=L2-only, {W_ASSOC_M55:.2f}=full blend)")
        print(f"  Mean confidence (history): {s['confidence_mean']:.4f}")