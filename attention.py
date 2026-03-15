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

import numpy as np
import math
from collections import deque


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Grid (must match M54 / M55 / L2)
GRID_H    = 8
GRID_W    = 8
N_NEURONS = GRID_H * GRID_W   # 64

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

def _build_grid_dist_sq() -> np.ndarray:
    """Precompute pairwise squared grid distances for the 8×8 map."""
    dist_sq = np.zeros((N_NEURONS, N_NEURONS), dtype=np.float32)
    for i in range(N_NEURONS):
        ri, ci = divmod(i, GRID_W)
        for j in range(N_NEURONS):
            rj, cj = divmod(j, GRID_W)
            dist_sq[i, j] = (ri - rj)**2 + (ci - cj)**2
    return dist_sq

_GRID_DIST_SQ = _build_grid_dist_sq()   # computed once at import


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

    def __init__(self):
        # Salience EMA state
        self._salience_ema = float(SALIENCE_EMA_INIT)

        # Last computed outputs (for diagnostics)
        self._last_salience       = 0.0
        self._last_salience_delta = 0.0
        self._last_gate           = np.full(N_NEURONS, GATE_BASELINE,
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
        gate = np.full(N_NEURONS, GATE_BASELINE, dtype=np.float32)

        # Gaussian boost scaled by salience.
        # At salience=0: boost=0, gate stays uniform (truly diffuse).
        # At salience=1: full Gaussian peak at BMU neighbourhood.
        # This ensures zero-salience → maximum entropy gate.
        if raw > 1e-6:
            sigma_sq_2 = 2.0 * GATE_SIGMA * GATE_SIGMA
            gauss = np.exp(-_GRID_DIST_SQ[bmu_idx] / sigma_sq_2)  # (64,)
            gate  = gate + (GATE_BOOST * raw * gauss).astype(np.float32)

        # Normalize to sum=1 (probability simplex)
        gate_sum = gate.sum()
        if gate_sum > 1e-9:
            gate = gate / gate_sum
        else:
            gate = np.full(N_NEURONS, GATE_BASELINE, dtype=np.float32)

        gate = gate.astype(np.float32)

        # ── 5. Gate diagnostics ───────────────────────────────
        attended_bmu = int(np.argmax(gate))

        # Entropy of gate distribution — 0=maximally focused, 1=uniform
        # H(p) / H(uniform) = -sum(p log p) / log(N)
        log_gate   = np.log(gate + 1e-9)
        entropy    = float(-np.sum(gate * log_gate))
        max_entropy = math.log(N_NEURONS)
        gate_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))

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
                                   / math.log(N_NEURONS), 0.0, 1.0)),
            'salience_mean':   float(np.mean(self._salience_history))
                               if self._salience_history else 0.0,
        }

    def reset(self):
        """Reset all state — use between test conditions."""
        self._salience_ema        = float(SALIENCE_EMA_INIT)
        self._last_salience       = 0.0
        self._last_salience_delta = 0.0
        self._last_gate           = np.full(N_NEURONS, GATE_BASELINE,
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