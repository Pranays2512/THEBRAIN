import numpy as np
import math
from collections import deque


# ============================================================
# FROM attention.py
# ============================================================
"""
ATTENTION — SALIENCE-GATED THALAMIC FILTER
===========================================

WHAT THIS IS
------------
Attention sits above Brain. It reads Brain.step() outputs each step
and produces a single salience score [0,1] representing "how much
should the system attend to this moment?"

It also produces an attention_gate vector (64,) — one weight per
cortical BMU — for future use by Thought or higher layers that need
to know which neurons are currently salient.

BIOLOGICAL BASIS
----------------
Models the thalamus + anterior cingulate cortex (ACC):
  - Thalamus: gates what reaches cortex (attention_gate vector)
  - ACC:      monitors conflict and surprise, controls salience signal

Attention is BOTTOM-UP only in this version:
  - Driven entirely by incoming signals (surprise, novelty, familiarity)
  - No voluntary/top-down component yet (that belongs to Thought)

Biologically: bottom-up attention is mediated by the superior colliculus
and pulvinar nucleus of the thalamus — automatic, fast, stimulus-driven.
Top-down attention (prefrontal → thalamus) is a later addition.

HOW IT WORKS
------------
Every step, four signals arrive from Brain:

  qe_norm        — is the INPUT perceptually novel to M54?
  familiarity    — is this BMU well-known to M55?
  surprise_signal — did prediction error spike above baseline?
  curiosity_delta — is L2 entering novel sequence territory?

These are combined into a raw salience:

  raw_salience = clip(
      W_SURPRISE   * surprise_signal   +   ← sequence surprise (strongest driver)
      W_QE         * qe_norm           +   ← perceptual novelty
      W_CURIOSITY  * curiosity_delta   +   ← entering new territory
      W_FAMILIARITY * (1 - familiarity)    ← unfamiliar = attend more
  , 0, 1)

Then smoothed by an EMA (tau ~5 steps) so salience doesn't jitter
step-to-step but still responds quickly to genuine transitions.

DELTA RULE (from guide Rule 1)
-------------------------------
Any feedback from Attention back into lower modules MUST follow the
delta rule: feed max(0, signal - ema_of_signal), not the raw value.

Attention's salience is itself already a delta-sensitive signal
(it's driven partly by surprise_signal and curiosity_delta which are
already deltas from Brain). So salience feeding back DOWN is safe —
but if you ever add a new feedback path, re-read guide Rule 1 first.

ATTENTION GATE VECTOR
----------------------
The attention_gate (64,) is a soft mask over the cortical BMU space.
Construction:
  1. Start from a uniform baseline (1/64 each)
  2. Boost the current BMU and its grid neighbours proportionally
     to salience
  3. Smooth across the grid with a Gaussian (sigma=1.5 grid cells)
  4. Normalize so the gate sums to 1.0

This means: high salience → the firing region is strongly boosted.
Low salience → nearly uniform (everything passes equally weakly).

The gate is informational in this version — it is not yet wired back
into M54 or M55. It exists so Thought can read it and focus on the
attended region rather than the whole 64-neuron space.

CALL ORDER (per step)
----------------------
Brain.step() is called first. Then:
  attention.step(brain_result)

Brain is unchanged. Attention only READS Brain output.

OUTPUTS
-------
salience          float [0,1]   — how much to attend this step
salience_ema      float [0,1]   — smoothed salience (5-step EMA)
salience_delta    float [0,1]   — spike above EMA (follows delta rule)
attention_gate    ndarray(64,)  — soft spatial mask over BMU space
attended_bmu      int           — BMU with highest gate weight
gate_entropy      float [0,1]   — how focused is the gate? (0=all one BMU)
t                 int           — step counter

INTERFACE
---------
  from attention import Attention

  attn = Attention()

  # standalone (no Brain needed):
  result = attn.step(
      bmu_idx         = 20,
      qe_norm         = 0.3,
      familiarity     = 0.6,
      surprise_signal = 0.0,
      curiosity_delta = 0.0,
  )

  # with Brain:
  brain_out = brain.step(...)
  attn_out  = attn.step(
      bmu_idx         = brain_out['bmu_idx'],
      qe_norm         = brain_out['qe_norm'],
      familiarity     = brain_out['familiarity'],
      surprise_signal = brain_out['surprise_signal'],
      curiosity_delta = brain_out['curiosity_delta'],
  )

BACKWARD COMPATIBILITY
-----------------------
All Brain-fed parameters default to 0.0.
Calling attn.step(bmu_idx=k, qe_norm=q, familiarity=f) with no
surprise/curiosity inputs reproduces a pure perceptual-attention
baseline — exactly how you'd test standalone.
"""






# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Default SOM size — overridden per-instance via n_neurons param
SOM_SIDE  = 8          # side length of the default 8×8 SOM
N_NEURONS = SOM_SIDE * SOM_SIDE   # 64 (default; Brain v13 uses 100)

# ── Salience weights ─────────────────────────────────────────
# How strongly each input drives raw salience.
# Sum intentionally > 1.0 — clip(0,1) at the end handles ceiling.
# Weights reflect biological priority:
#   surprise_signal:          phasic — strongest driver (dopamine PE signal)
#   qe_norm:                  perceptual novelty (immediate)
#   curiosity_delta:          sequence novelty (slower, sustained)
#   familiarity:              suppressive — known contexts reduce salience
#   thought_confidence_delta: suppressive — predicted events are less salient
W_SURPRISE    = 0.50   # surprise_signal contribution
W_QE          = 0.30   # qe_norm contribution
W_CURIOSITY   = 0.25   # curiosity_delta contribution
W_FAMILIARITY = 0.20   # (1 - familiarity) contribution
W_THOUGHT     = 0.15   # thought_confidence_delta suppression (top-down)
                       # Mild: top-down prediction reduces but never eliminates
                       # salience. Unexpected events still register even when
                       # Thought is confident. Keep ≤ W_FAMILIARITY.

# ── Salience EMA ─────────────────────────────────────────────
# tau = 1/alpha steps.
# At 0.20: tau ~5 steps — fast enough to track transitions,
# slow enough to suppress single-step noise.
# DO NOT go above 0.30 — salience becomes indistinguishable from raw.
# DO NOT go below 0.05 — salience becomes too sluggish to gate properly.
SALIENCE_EMA_ALPHA = 0.20

# Cold-start EMA value.
# 0.5 = moderate uncertainty at start — no spurious spike on step 1.
SALIENCE_EMA_INIT  = 0.5

# ── Attention gate ───────────────────────────────────────────
# Gaussian smoothing sigma over the 8×8 grid.
# At sigma=1.5: neighbours within ~1.5 cells are included.
# Matches M54's SIGMA_MIN=1.5 — gate covers same neighbourhood
# as the SOM's learning radius.
GATE_SIGMA = 1.5

# Baseline gate value before salience boost.
# At 1/N_NEURONS each neuron starts equal.
GATE_BASELINE = 1.0 / N_NEURONS   # ~0.0156

# How strongly salience boosts the winning BMU region.
# At salience=1.0: BMU activation = GATE_BASELINE + GATE_BOOST = ~1.016
# (then Gaussian-spread and renormalized). At salience=0.0: uniform gate.
GATE_BOOST = 1.0


# ═══════════════════════════════════════════════════════════════
# PRECOMPUTED GRID DISTANCES
# ═══════════════════════════════════════════════════════════════

def _build_grid_dist_sq(n: int = N_NEURONS, w: int = SOM_SIDE) -> np.ndarray:
    """Precompute pairwise squared distances for an n-neuron square SOM."""
    dist_sq = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        ri, ci = divmod(i, w)
        for j in range(n):
            rj, cj = divmod(j, w)
            dist_sq[i, j] = (ri - rj)**2 + (ci - cj)**2
    return dist_sq

_GRID_DIST_SQ = _build_grid_dist_sq()   # computed once at import (64 neurons, default)


# ═══════════════════════════════════════════════════════════════
# ATTENTION
# ═══════════════════════════════════════════════════════════════

class Attention:
    """
    Salience-gated thalamic filter for the Brain cognitive stack.

    Reads Brain.step() outputs. Produces salience score and spatial
    attention gate. Does NOT modify any lower module.

    All Brain-fed parameters default to 0.0 for standalone operation.
    """

    def __init__(self, n_neurons: int = N_NEURONS):
        self._n        = n_neurons
        self._grid_w   = int(round(n_neurons ** 0.5))  # assume square SOM
        self._gate_bl  = 1.0 / n_neurons
        self._dist_sq  = _build_grid_dist_sq(n_neurons, self._grid_w)
        self._log_n    = math.log(n_neurons)

        # Salience EMA state
        self._salience_ema = float(SALIENCE_EMA_INIT)

        # Last computed outputs (for diagnostics)
        self._last_salience       = 0.0
        self._last_salience_delta = 0.0
        self._last_gate           = np.full(n_neurons, self._gate_bl,
                                            dtype=np.float32)
        self._last_attended_bmu   = 0

        # Diagnostics
        self._salience_history = deque(maxlen=200)
        self._gate_history     = deque(maxlen=200)   # attended_bmu per step
        self.t = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             bmu_idx:                  int,
             qe_norm:                  float,
             familiarity:              float,
             surprise_signal:          float = 0.0,
             curiosity_delta:          float = 0.0,
             thought_confidence_delta: float = 0.0) -> dict:
        """
        One attention step.

        Parameters
        ----------
        bmu_idx                  : int   — which cortical neuron fired (from M54)
        qe_norm                  : float — perceptual novelty [0,1] (from M54)
        familiarity              : float — recognition signal [0,1] (from M55)
        surprise_signal          : float — delta prediction error [0,1] (from Brain)
        curiosity_delta          : float — delta curiosity [0,1] (from Brain)
        thought_confidence_delta : float — Thought confidence spike [0,1] (from Thought)
                                           When Thought suddenly gains a strong
                                           prediction, salience is dampened slightly:
                                           "I expected this — less surprising."

        Returns
        -------
        dict with keys:
            salience        float [0,1]    — raw salience this step
            salience_ema    float [0,1]    — smoothed salience
            salience_delta  float [0,1]    — spike above EMA (delta rule)
            attention_gate  ndarray (64,)  — spatial soft mask
            attended_bmu    int            — most attended neuron
            gate_entropy    float [0,1]    — gate focus (0=focused,1=diffuse)
            t               int            — step count
        """
        # ── 1. Compute raw salience ───────────────────────────
        # Five drives: four bottom-up + one top-down suppressive.
        # familiarity is SUPPRESSIVE — familiar = less need to attend.
        # thought_confidence_delta is SUPPRESSIVE — if Thought predicted
        # this event, it carries less bottom-up surprise weight.
        # The suppression is mild (W_THOUGHT=0.15) — top-down prediction
        # reduces but never eliminates salience. Unexpected events still
        # get through even when Thought is confident.
        raw = float(np.clip(
            W_SURPRISE    * float(surprise_signal)                  +
            W_QE          * float(qe_norm)                          +
            W_CURIOSITY   * float(curiosity_delta)                  +
            W_FAMILIARITY * (1.0 - float(familiarity))              -
            W_THOUGHT     * float(thought_confidence_delta),
            0.0, 1.0
        ))

        # ── 2. Delta salience (follows guide Rule 1) ──────────
        # Only upward spikes above the running EMA propagate.
        # Near-zero when salience is stable (even stably high).
        salience_delta = float(np.clip(
            raw - self._salience_ema, 0.0, 1.0
        ))

        # ── 3. Update EMA AFTER computing delta ───────────────
        self._salience_ema = ((1.0 - SALIENCE_EMA_ALPHA) * self._salience_ema
                              + SALIENCE_EMA_ALPHA * raw)

        # ── 4. Build attention gate ───────────────────────────
        # Start: uniform baseline
        gate = np.full(self._n, self._gate_bl, dtype=np.float32)

        # Gaussian boost scaled by salience.
        if raw > 1e-6:
            sigma_sq_2 = 2.0 * GATE_SIGMA * GATE_SIGMA
            gauss = np.exp(-self._dist_sq[bmu_idx] / sigma_sq_2)
            gate  = gate + (GATE_BOOST * raw * gauss).astype(np.float32)

        # Normalize to sum=1 (probability simplex)
        gate_sum = gate.sum()
        if gate_sum > 1e-9:
            gate = gate / gate_sum
        else:
            gate = np.full(self._n, self._gate_bl, dtype=np.float32)

        gate = gate.astype(np.float32)

        # ── 5. Gate diagnostics ───────────────────────────────
        attended_bmu = int(np.argmax(gate))

        # Entropy of gate distribution — 0=maximally focused, 1=uniform
        log_gate     = np.log(gate + 1e-9)
        entropy      = float(-np.sum(gate * log_gate))
        gate_entropy = float(np.clip(entropy / self._log_n, 0.0, 1.0))

        # ── 6. Store state ────────────────────────────────────
        self._last_salience       = raw
        self._last_salience_delta = salience_delta
        self._last_gate           = gate
        self._last_attended_bmu   = attended_bmu

        self._salience_history.append(raw)
        self._gate_history.append(attended_bmu)
        self.t += 1

        return {
            'salience':        raw,
            'salience_ema':    self._salience_ema,
            'salience_delta':  salience_delta,
            'attention_gate':  gate,
            'attended_bmu':    attended_bmu,
            'gate_entropy':    gate_entropy,
            't':               self.t,
        }

    # ── Convenience accessors ─────────────────────────────────

    def get_state(self) -> dict:
        """Full diagnostic snapshot."""
        return {
            't':               self.t,
            'salience':        self._last_salience,
            'salience_ema':    self._salience_ema,
            'salience_delta':  self._last_salience_delta,
            'attended_bmu':    self._last_attended_bmu,
            'gate_peak':       float(self._last_gate.max()),
            'gate_entropy':    float(np.clip(
                                   -np.sum(self._last_gate *
                                           np.log(self._last_gate + 1e-9))
                                   / self._log_n, 0.0, 1.0)),
            'salience_mean':   float(np.mean(self._salience_history))
                               if self._salience_history else 0.0,
        }

    def reset(self):
        """Reset all state — use between test conditions."""
        self._salience_ema        = float(SALIENCE_EMA_INIT)
        self._last_salience       = 0.0
        self._last_salience_delta = 0.0
        self._last_gate           = np.full(self._n, self._gate_bl,
                                            dtype=np.float32)
        self._last_attended_bmu   = 0
        self._salience_history.clear()
        self._gate_history.clear()
        self.t = 0

    def summary(self):
        """Human-readable state summary."""
        s = self.get_state()
        print(f"  Attention — step {s['t']}")
        print(f"  Salience:      {s['salience']:.4f}  "
              f"(ema={s['salience_ema']:.4f}  delta={s['salience_delta']:.4f})")
        print(f"  Attended BMU:  {s['attended_bmu']}  "
              f"gate_peak={s['gate_peak']:.4f}  "
              f"entropy={s['gate_entropy']:.4f}  "
              f"(0=focused, 1=diffuse)")
        print(f"  Mean salience (history): {s['salience_mean']:.4f}")

# ============================================================
# FROM thought.py
# ============================================================
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

# Thought section reuses the same _build_grid_dist_sq defined at top of file.


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

    def __init__(self, n_neurons: int = N_NEURONS):
        self._n     = n_neurons
        self._log_n = math.log(n_neurons)

        # ── Confidence EMA ────────────────────────────────────
        self._confidence_ema = float(CONFIDENCE_EMA_INIT)

        # ── One-step-delayed outputs for Brain to pass downward ──
        # Stored after each step, fed into L2 and Attention on the NEXT step.
        self._last_prediction_bias      = np.full(n_neurons, 1.0 / n_neurons,
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
             attended_bmu:  int,
             bmu_idx:       int,
             pred,                        # SequencePredictor instance or None
             salience:      float = 0.0,
             memory         = None,       # AssociativeMemory instance or None
             simulated_bmu: int   = -1,   # M61: simulated next BMU from M57
             sim_weight:    float = 0.0,  # M61: how much to blend simulated into attended
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

        # ── M61: blend simulated BMU into attended_bmu ───────────
        # When the thought loop is active (sim_weight > 0), Thought
        # partially shifts its attention from the real attended_bmu
        # to the simulated next BMU from M57.
        # This is the internal perception — the brain "sees" where
        # it thinks it's going before it goes there.
        # sim_weight=0.0 → pure real perception (default, unchanged)
        # sim_weight=1.0 → pure simulation (full internal focus)
        # Intermediate values blend both sources.
        # Biologically: PFC active maintenance of anticipated state
        # alongside current sensory input — prospective coding.
        if simulated_bmu >= 0 and sim_weight > 0.0:
            sw = float(np.clip(sim_weight, 0.0, 1.0))
            # Weighted blend: pick simulated if random draw < sim_weight
            # This is a soft probabilistic blend, not a hard switch.
            # It preserves attended_bmu's influence at low sim_weight.
            if np.random.random() < sw:
                attended_bmu = int(simulated_bmu)

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

        bias       = np.full(self._n, 0.0, dtype=np.float32)
        assoc_weight = 0.0   # fraction of final bias mass that came from M55

        if float(salience) < MIN_SALIENCE_FOR_BIAS:
            # Salience too low — uniform prior, no prediction
            bias = np.full(self._n, 1.0 / self._n, dtype=np.float32)
        else:
            # ── L2 bias ──────────────────────────────────────────
            l2_bias = np.zeros(self._n, dtype=np.float32)
            if pred is not None:
                top = pred.top_predictions(attended_bmu, k=TOP_K_PREDICTIONS)
                for idx, score in top:
                    l2_bias[idx] += float(score)
            l2_sum = float(l2_bias.sum())
            if l2_sum > 1e-9:
                l2_bias = l2_bias / l2_sum
            else:
                l2_bias = np.full(self._n, 1.0 / self._n, dtype=np.float32)
                l2_sum  = 0.0   # signal absent

            # ── M55 bias ─────────────────────────────────────────
            m55_bias = np.zeros(self._n, dtype=np.float32)
            if memory is not None:
                row = memory._W[attended_bmu].copy()
                row[attended_bmu] = 0.0   # exclude self-association
                # Gate on minimum strength — noise floor from early training
                row[row < MIN_ASSOC_STRENGTH] = 0.0
                m55_sum = float(row.sum())
                if m55_sum > 1e-9:
                    m55_bias = (row / m55_sum).astype(np.float32)
                else:
                    m55_bias = np.full(self._n, 1.0 / self._n, dtype=np.float32)
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
                bias = np.full(self._n, 1.0 / self._n, dtype=np.float32)
                w_l2 = w_m55 = 0.0

            if w_l2 > 0.0 or w_m55 > 0.0:
                bias = (w_l2 * l2_bias + w_m55 * m55_bias).astype(np.float32)
                b_sum = float(bias.sum())
                if b_sum > 1e-9:
                    bias = bias / b_sum
                else:
                    bias = np.full(self._n, 1.0 / self._n, dtype=np.float32)

            # Fraction of mass contributed by M55 (diagnostic)
            if has_m55 and (w_l2 > 0.0 or w_m55 > 0.0):
                assoc_weight = float(w_m55)   # = W_ASSOC_M55 when both present

        bias = bias.astype(np.float32)

        # ── 3. Compute thought_confidence ────────────────────
        # How concentrated is the prediction?
        # max(bias) - 1/N is the deviation above uniform baseline.
        # 0 = no idea (uniform). ~0.35+ = clear expectation.
        raw_confidence = float(np.clip(
            bias.max() - (1.0 / self._n),
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
        focus_entropy = float(np.clip(entropy / self._log_n, 0.0, 1.0))

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

# ============================================================
# FROM valence.py
# ============================================================
"""
VALENCE — REWARD PREDICTION ERROR (V1)
=======================================

WHAT THIS IS
------------
Valence is the dopaminergic layer of the brain stack. It models the
ventral tegmental area (VTA) and substantia nigra — the brain's reward
prediction error signal that teaches every other system what is good,
bad, expected, or surprising in a motivational sense.

Every existing module handles EPISTEMIC surprise (was the input or
sequence unexpected?). Valence adds EVALUATIVE surprise:
  "Was the OUTCOME better or worse than I expected?"

This is reward prediction error (RPE):
  rpe = actual_reward − expected_reward

A key difference from all other signals in this stack:
  RPE is SIGNED. It ranges [-1, +1].
  Positive = better than expected (dopamine burst → reinforce).
  Negative = worse than expected (dopamine dip → update, recalibrate).
  Near zero = exactly what was expected (no learning signal).

This is biologically accurate: VTA dopamine neurons fire above baseline
when outcomes exceed predictions, below baseline (pause) when they
are worse than predicted, and at baseline when outcomes match.


WHY INTRINSIC REWARD FIRST
---------------------------
Valence works WITHOUT an external reward signal. By default, it computes
an INTRINSIC reward from what the stack already knows:

    intrinsic_reward = 1.0 - prediction_error

This means: the system intrinsically "likes" correct predictions.
When L2 predicts well, reward is high. When L2 is wrong, reward is low.

This is directly motivated by predictive coding theory — the brain
treats prediction error minimisation as its fundamental objective.
It also means Valence produces meaningful signals from the moment the
stack is running, without needing external reward labels.

When an external reward is provided (reward > 0.0), it is blended with
the intrinsic signal:
    total_reward = W_EXTERNAL * reward + W_INTRINSIC * intrinsic_reward

When reward=0.0 (default), the system runs on pure intrinsic reward.


THE RPE SIGNAL AND THE DELTA RULE
-----------------------------------
RPE is already a delta signal by construction (actual − expected).
It does NOT need an additional EMA baseline subtracted (see Guide Rule 1
and Rule 13). The expected_reward EMA IS the baseline — RPE is the
deviation from it.

This is different from signals like prediction_error (which has a
structural floor of ~0.35 that requires the delta rule). Intrinsic
reward = 1 − prediction_error has a corresponding CEILING of ~0.65 at
steady state. The reward EMA converges to this ceiling, and RPE becomes
near-zero during stable familiar operation. No extra delta needed.

HOWEVER: do not feed raw RPE magnitude to M54 — it has high step-to-step
variance (~0.30 std) even during stable operation, which would inflate
eta permanently. M54 is already well-served by surprise_signal. V1 only
feeds M55.


WHAT V1 FEEDS DOWNSTREAM
--------------------------
pos_rpe → M55 (positive RPE only, clipped to [0,1]):
  "This outcome was better than expected — consolidate this memory."
  Boosts M55's Hebbian write rate on top of the existing curiosity boost.
  High pos_rpe → the current BMU's associations are strengthened more.
  Biologically: dopamine burst → hippocampal LTP (long-term potentiation).
  Formula: eta_effective = ETA_HEBB × (1 + curiosity_boost + rpe_boost)
  where rpe_boost = RPE_M55_BOOST × pos_rpe

V1 does NOT feed M54 (existing surprise_signal handles plasticity),
L2 (already has its own error signal), or Attention (already gated).
V1's signed rpe is available as an output key for M56 (action layer).


CALL ORDER
----------
V1 runs at step 6 — after L2 (needs prediction_error from step 5),
alongside the Brain delta computations, before Attention (step 7).

    6a. compute surprise_signal, curiosity_delta (existing)
    6b. valence.step(prediction_error, reward)         ← NEW
    6c. store rpe_positive for NEXT step's M55 call

V1's output rpe_positive is stored at step t and fed to M55 at step t+1,
exactly like curiosity_delta. Feeding it same-step would mean M55 learns

a temporal loop. Next-step feeding is correct.


BIOLOGICAL BASIS
----------------
VTA/SNc dopamine system:
  - Fires above baseline when reward exceeds prediction (RPE > 0)
  - Pauses below baseline when reward is less than predicted (RPE < 0)
  - Fires at baseline when reward matches prediction (RPE ≈ 0)

Downstream effects modelled:
  - RPE > 0 → hippocampal LTP (M55 write boost) — "remember this"
  - RPE < 0 → cortical plasticity increase (BUT: handled by surprise_signal
    already; we avoid double-counting by not feeding RPE to M54)
  - Signed RPE output → readable by M56 (action layer) for Q-learning


OUTPUTS
-------
rpe              float [-1, 1] — signed reward prediction error (dopamine signal)
pos_rpe          float [0, 1]  — positive RPE only (better than expected)
neg_rpe          float [0, 1]  — |negative RPE| (worse than expected, magnitude)
reward_ema       float [0, 1]  — running expected reward (baseline)
total_reward     float [0, 1]  — blended reward this step
intrinsic_reward float [0, 1]  — 1 - prediction_error this step
t                int           — step counter


INTERFACE
---------
  from valence import Valence

  v1 = Valence()

  # Standalone — every step, after L2:
  result = v1.step(
      prediction_error = l2_out['prediction_error'],
      reward           = 0.0,    # optional external reward [0,1]
  )

  result['rpe']          # signed RPE  [-1, +1]
  result['pos_rpe']      # positive only [0, 1] — feed to M55
  result['neg_rpe']      # magnitude of negative [0, 1] — informational
  result['reward_ema']   # expected reward baseline [0, 1]

  # With Brain (called inside Brain.step() at step 6b):
  v1_out = v1.step(
      prediction_error = raw_error,
      reward           = reward_arg,   # passed into brain.step()
  )
  # Brain stores v1_out['pos_rpe'] → passed to memory.step() next step
"""





# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# ── Reward EMA ───────────────────────────────────────────────
# Running baseline for expected reward.
# tau = 1/alpha steps.
# At 0.05: tau ~20 steps — same timescale as L2 curiosity EMA.
# Tracks reward baseline slowly enough that genuine improvements
# register as positive RPE, not immediately absorbed into baseline.
# Do not go above 0.15 (EMA chases reward too fast, RPE collapses).
# Do not go below 0.02 (EMA too slow, RPE inflates permanently).
RPE_EMA_ALPHA = 0.05

# Cold-start EMA. Set to 0.5 (middle of [0,1]) so RPE on step 1
# is neither strongly positive nor negative.
# At 0.5, cold-start intrinsic_reward ~0.65 → RPE ~+0.15 (small positive).
RPE_EMA_INIT = 0.5

# ── Reward blending ──────────────────────────────────────────
# When external reward is provided (reward > 0.0), blend with intrinsic.
# W_EXTERNAL + W_INTRINSIC should sum to 1.0.
#
# Default 50/50. Caller can tune at instantiation if needed.
# If you set W_EXTERNAL=1.0, W_INTRINSIC=0.0, you get pure RL mode —
# the system only cares about externally labelled rewards.
# If you set W_EXTERNAL=0.0, W_INTRINSIC=1.0, pure predictive coding —
# the system only cares about prediction accuracy.
# Navigation reward and intrinsic reward serve different purposes:
#   External reward = food/wall signal from the environment (navigation RL)
#   Intrinsic reward = 1 - prediction_error (curiosity / predictive coding)
# When external reward is nonzero, use it as the SOLE reward signal for RPE.
# Diagnostic confirmed: blending 0.5*(-0.05_wall) + 0.5*(1.0_intrinsic) = 0.475
# means 49.8% of wall hits generate POSITIVE RPE. The brain was being rewarded
# for wall-bashing because correct prediction of staying put is intrinsically
# rewarding. Separating the channels fixes this completely.
# When reward=0, intrinsic + novelty signal runs normally (exploration drive).
W_EXTERNAL  = 1.0   # weight on external reward when reward != 0
W_INTRINSIC = 0.0   # intrinsic suppressed when external reward is present

# ── M55 RPE boost ────────────────────────────────────────────
# How strongly positive RPE boosts M55's Hebbian write rate.
# formula in M55.step():
#   eta_effective = ETA_HEBB * (1 + curiosity_boost + RPE_M55_BOOST * pos_rpe)
# At pos_rpe=1.0: adds RPE_M55_BOOST to the multiplier.
# At 1.0: maximum triple write rate (curiosity + RPE both maxed).
# At 0.5: modest boost, keeps curiosity_delta as primary driver.
# Keep ≤ 1.0 — above that, RPE dominates over curiosity.
RPE_M55_BOOST = 1.0

# ── Novelty bonus ─────────────────────────────────────────────
# Intrinsic exploration reward proportional to unfamiliarity.
# Biologically: VTA novelty-driven dopamine burst (separate from food DA).
# Formula: novelty_bonus = NOVELTY_BONUS_WEIGHT * (1 - familiarity)
# Added to total_reward before RPE is computed, so it influences the EMA
# and generates a persistent pos_rpe at underexplored nodes.
#
# 0.15 keeps novelty bonus well below food reward (1.0) but above the
# intrinsic reward noise floor (~0.05 step variance). Large enough to
# break the C→E attractor; small enough not to dominate food reward.
NOVELTY_BONUS_WEIGHT = 0.30   # raised from 0.08 — must be large enough to compete
                               # with the C→E food attractor (+1.0 every ~4 steps).
                               # At 0.08 unvisited nodes generate only 0.08 bonus,
                               # completely swamped by frequent food reward. At 0.30
                               # a completely unfamiliar node (fam≈0) generates 0.30
                               # bonus — still 3× below food reward but enough to
                               # create genuine pull toward unvisited regions.

# ── Diagnostics ──────────────────────────────────────────────
HISTORY_LEN = 200


# ═══════════════════════════════════════════════════════════════
# VALENCE
# ═══════════════════════════════════════════════════════════════

class Valence:
    """
    Reward prediction error (dopaminergic) module for the Brain stack.

    Computes RPE from intrinsic reward (1 - prediction_error) and an
    optional external reward signal. Produces pos_rpe for M55 write
    rate modulation and signed rpe for M56 action learning.

    Works standalone — all Brain-fed inputs default to safe values.
    """

    def __init__(self):
        # ── Running baseline ──────────────────────────────────
        self._reward_ema = float(RPE_EMA_INIT)

        # ── One-step-delayed output for Brain to pass to M55 ──
        self._last_pos_rpe = 0.0

        # ── Diagnostics ───────────────────────────────────────
        self._rpe_history            = deque(maxlen=HISTORY_LEN)
        self._reward_history         = deque(maxlen=HISTORY_LEN)
        self._intrinsic_history      = deque(maxlen=HISTORY_LEN)
        self._last_rpe               = 0.0
        self._last_pos_rpe_out       = 0.0
        self._last_neg_rpe           = 0.0
        self._last_total_reward      = 0.0
        self._last_intrinsic_reward  = 0.0
        self.t                       = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             prediction_error: float = 0.0,
             reward:           float = 0.0,
             familiarity:      float = 1.0,
             ) -> dict:
        """
        One Valence step.

        Parameters
        ----------
        prediction_error : float [0, 1]
            L2's prediction error this step. Used to compute intrinsic reward.
            Default 0.0 (perfect prediction — gives max intrinsic reward).
        reward : float [0, 1]
            Optional external reward signal. 0.0 = no external reward (default).
            Caller defines what reward means. Should be normalised to [0, 1].
        familiarity : float [0, 1]
            M55 familiarity for the current BMU. Used to compute novelty bonus.
            Low familiarity → novelty bonus → exploration pressure at sparse nodes.
            Default 1.0 (fully familiar) → no novelty bonus added.

        Returns
        -------
        dict with keys:
            rpe              float [-1, 1] — signed reward prediction error
            pos_rpe          float [0, 1]  — positive RPE (better than expected)
            neg_rpe          float [0, 1]  — |negative RPE| (worse than expected)
            reward_ema       float [0, 1]  — running expected reward baseline
            total_reward     float [0, 1]  — blended reward this step
            intrinsic_reward float [0, 1]  — 1 - prediction_error
            novelty_bonus    float [0, 1]  — unfamiliarity exploration bonus
            t                int           — step counter
        """
        # ── 1. Compute intrinsic reward ───────────────────────
        # "How well did the stack predict this step?"
        intrinsic_reward = float(np.clip(1.0 - prediction_error, 0.0, 1.0))

        # ── 2. Novelty bonus — exploration pressure ───────────
        # Fires whenever familiarity is low, independent of food reward.
        # Fades naturally as the brain visits and learns a node.
        # Biologically: VTA novelty-DA, distinct from food-reward-DA pathway.
        novelty_bonus = float(NOVELTY_BONUS_WEIGHT * (1.0 - float(familiarity)))

        # ── 3. Build total_reward ─────────────────────────────
        # When external reward is nonzero (food or wall penalty), it is the
        # sole signal — intrinsic reward is suppressed entirely.
        # Reason: blending intrinsic (up to 1.0) with wall penalty (-0.05)
        # made 49.8% of wall hits generate positive RPE because the brain
        # correctly predicts it stays put on wall hits (low prediction error →
        # high intrinsic). Walls were being rewarded. Separating the channels
        # makes wall RPE = wall_penalty - reward_ema (strongly negative) and
        # food RPE = food_reward - reward_ema (strongly positive).
        # When reward=0, intrinsic + novelty runs normally (exploration drive).
        if abs(float(reward)) > 1e-9:
            # External navigation signal — use it alone, no intrinsic blending
            total_reward = float(np.clip(float(reward), 0.0, 1.0))
        else:
            # No external signal — pure intrinsic + novelty drive
            total_reward = float(np.clip(
                intrinsic_reward + novelty_bonus,
                0.0, 1.0
            ))
        # ── 4. Compute RPE ────────────────────────────────────
        # RPE = actual − expected.
        # Signed: positive = better than expected, negative = worse.
        # No additional delta rule needed — RPE is already a deviation.
        rpe = float(np.clip(total_reward - self._reward_ema, -1.0, 1.0))

        # ── 4. Split into positive and negative components ────
        pos_rpe = float(max(0.0, rpe))    # better than expected [0, 1]
        neg_rpe = float(max(0.0, -rpe))   # worse than expected, magnitude [0, 1]

        # ── 5. Update reward EMA ──────────────────────────────
        # Updated AFTER computing RPE (same pattern as all EMA-delta pairs).
        # EMA tracks the running expected reward — what the system "knows"
        # it tends to get. RPE is the deviation above/below this baseline.
        self._reward_ema = float(np.clip(
            (1.0 - RPE_EMA_ALPHA) * self._reward_ema
            + RPE_EMA_ALPHA * total_reward,
            0.0, 1.0
        ))

        # ── 6. Store state ────────────────────────────────────
        self._last_pos_rpe          = pos_rpe
        self._last_rpe              = rpe
        self._last_pos_rpe_out      = pos_rpe
        self._last_neg_rpe          = neg_rpe
        self._last_total_reward     = total_reward
        self._last_intrinsic_reward = intrinsic_reward
        self._last_novelty_bonus    = novelty_bonus

        self._rpe_history.append(rpe)
        self._reward_history.append(total_reward)
        self._intrinsic_history.append(intrinsic_reward)

        self.t += 1

        return {
            'rpe':              rpe,
            'pos_rpe':          pos_rpe,
            'neg_rpe':          neg_rpe,
            'reward_ema':       self._reward_ema,
            'total_reward':     total_reward,
            'intrinsic_reward': intrinsic_reward,
            'novelty_bonus':    novelty_bonus,
            't':                self.t,
        }

    # ── Convenience accessors ─────────────────────────────────

    def get_state(self) -> dict:
        """Full diagnostic snapshot."""
        rpe_hist = list(self._rpe_history)
        return {
            't':                self.t,
            'rpe':              self._last_rpe,
            'pos_rpe':          self._last_pos_rpe_out,
            'neg_rpe':          self._last_neg_rpe,
            'reward_ema':       self._reward_ema,
            'total_reward':     self._last_total_reward,
            'intrinsic_reward': self._last_intrinsic_reward,
            'rpe_mean':         float(np.mean(rpe_hist)) if rpe_hist else 0.0,
            'rpe_std':          float(np.std(rpe_hist))  if rpe_hist else 0.0,
        }

    def reset(self):
        """Reset all state — use between test conditions."""
        self._reward_ema            = float(RPE_EMA_INIT)
        self._last_pos_rpe          = 0.0
        self._last_rpe              = 0.0
        self._last_pos_rpe_out      = 0.0
        self._last_neg_rpe          = 0.0
        self._last_total_reward     = 0.0
        self._last_intrinsic_reward = 0.0
        self._rpe_history.clear()
        self._reward_history.clear()
        self._intrinsic_history.clear()
        self.t = 0

    def summary(self):
        """Human-readable state summary."""
        s = self.get_state()
        print(f"  Valence — step {s['t']}")
        print(f"  RPE:          {s['rpe']:+.4f}  "
              f"(pos={s['pos_rpe']:.4f}  neg={s['neg_rpe']:.4f})")
        print(f"  Reward:       total={s['total_reward']:.4f}  "
              f"intrinsic={s['intrinsic_reward']:.4f}  "
              f"ema={s['reward_ema']:.4f}")
        print(f"  RPE history:  mean={s['rpe_mean']:+.4f}  std={s['rpe_std']:.4f}")
