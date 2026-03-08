"""
BRAIN — Integrated Cognitive Stack with Feedback Loops
=======================================================

This file owns the three cognitive modules and wires their feedback loops.
M50 (the ear) stays separate — it feeds INTO Brain.step(), not inside it.

ARCHITECTURE
------------
                    ┌─────────────────────────────────┐
  M50 (ear)  ──────▶│            Brain                │
                    │                                 │
                    │  CortexM54  (M54)               │
                    │      │ bmu_idx, qe_norm          │
                    │      ▼                           │
                    │  AssociativeMemory (M55)         │
                    │      │ familiarity               │
                    │      ▼                           │
                    │  SequencePredictor (L2)          │
                    │      │                           │
                    │   prediction_error ──────────────┼──▶ M54.step()  (next step)
                    │   curiosity ─────────────────────┼──▶ M55.step()  (next step)
                    └─────────────────────────────────┘

FEEDBACK LOOPS
--------------
Loop 1 — L2 → M54 (prediction error → cortical plasticity)
  When L2 was wrong about what BMU would fire next, the cortex
  learns faster on the actual BMU. Unexpected events trigger
  heightened synaptic plasticity.
  Signal: prediction_error [0,1] → boosts M54's eta
  Delay: one step (previous step's error modulates current step's learning)

Loop 2 — L2 → M55 (curiosity → memory consolidation)
  When L2 is in unfamiliar sequence territory (high sustained error),
  memories are written more strongly. Novel experiences get
  better consolidation.
  Signal: curiosity [0,1] → scales M55's Hebbian write rate
  Delay: one step (same rationale as above)

CALL ORDER (per step)
---------------------
1. pred.predict()                              ← prediction BEFORE cortex fires
2. cortex.step(..., prediction_error=last_err) ← M54 learns (boosted if L2 was wrong)
3. memory.step(..., curiosity=last_curiosity)  ← M55 writes (boosted if L2 is curious)
4. memory.recall(bmu_idx)                      ← get familiarity for L2
5. pred.step(bmu_idx, ..., familiarity)        ← L2 learns, outputs new err/curiosity

Both feedback signals use the PREVIOUS step's values — correct causal order,
because the current BMU hasn't fired yet when we call cortex.step().

USAGE
-----
  from brain import Brain

  brain = Brain(seed=42)

  # In your loop (after M50 gives you decoded signals):
  result = brain.step(
      decoded_freq = fused,
      stability_w  = w,
      novelty_flag = float(nov),
      plv_vector   = plv_slow,
  )

  # result contains everything from all three modules:
  result['bmu_idx']          # M54 — which neuron fired
  result['qe_norm']          # M54 — perceptual surprise
  result['familiarity']      # M55 — recognition signal
  result['prediction_error'] # L2  — sequence surprise
  result['curiosity']        # L2  — sustained novelty signal
  result['correct']          # L2  — was prediction right?

BACKWARD COMPATIBILITY
----------------------
The individual module files (m54_cortex.py, m55_memory.py, l2_predictor.py)
still work identically if used standalone — the new feedback parameters
default to 0.0, which exactly reproduces the old behaviour.

The debug inspector (m54_experience.py) attaches to Brain via:
  buf.push(t=t, cortex_out=result, ...)
Nothing changes in its interface.
"""

import numpy as np

from m54_cortex import CortexM54
from m55_memory import AssociativeMemory
from l2_predictor import SequencePredictor


# ═══════════════════════════════════════════════════════════════
# BRAIN
# ═══════════════════════════════════════════════════════════════

class Brain:
    """
    Integrated M54 + M55 + L2 cognitive stack with bidirectional
    feedback loops between L2 and the layers below it.

    Parameters
    ----------
    seed : int
        Random seed passed to all three modules for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.cortex  = CortexM54(seed=seed)
        self.memory  = AssociativeMemory(seed=seed)
        self.pred    = SequencePredictor()

        # One-step-delayed feedback signals.
        # Initialised to zero — no feedback on the very first step.
        # These get updated at the END of every step() call.
        self._last_prediction_error = 0.0
        self._last_curiosity        = 0.0

        # Step counter
        self.t = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             decoded_freq: float,
             stability_w:  float,
             novelty_flag: float,
             plv_vector:   np.ndarray) -> dict:
        """
        One full cognitive step: perception → memory → prediction → feedback.

        Parameters
        ----------
        decoded_freq : float
            Fused frequency estimate from M50 decoder (Hz)
        stability_w : float [0,1]
            Signal stability weight from M50 (1 = stable, 0 = transitioning)
        novelty_flag : float
            CUSUM novelty flag from M50 (1 = regime change detected)
        plv_vector : ndarray
            Raw PLV components from M50 oscillator bank

        Returns
        -------
        dict with all signals from all three modules, plus feedback state:
            From M54:
                'bmu_idx'          int    — winning neuron index (0–63)
                'bmu_pos'          tuple  — (row, col) on 8×8 grid
                'qe'               float  — raw quantisation error
                'qe_norm'          float  — normalised surprise [0,1]
                'sigma'            float  — neighbourhood width
                'eta'              float  — actual learning rate used
                'is_novel'         bool   — above surprise threshold?
            From M55:
                'familiarity'      float  — recognition score [0,1]
                'top_associations' list   — top-5 associated neurons
                'wrote'            bool   — Hebbian update happened?
            From L2:
                'prediction_error' float  — sequence surprise [0,1]
                'correct'          bool   — was prediction right?
                'predicted_bmu'    int    — what L2 expected
                'curiosity'        float  — sustained novelty EMA [0,1]
                'confidence'       float  — prediction certainty [0,1]
            Feedback state:
                'fed_prediction_error' float — error fed into M54 this step
                'fed_curiosity'        float — curiosity fed into M55 this step
        """
        # ── 1. Predict BEFORE cortex fires ────────────────────
        # L2 makes its prediction using the context built from
        # all previous BMUs. This must happen before cortex.step()
        # so the prediction is genuinely prospective.
        pred_out = self.pred.predict()

        # ── 2. Cortex fires (M54) — with L2 feedback ──────────
        # prediction_error from the PREVIOUS step modulates eta.
        # If L2 was wrong last step, learn faster on this one.
        cortex_out = self.cortex.step(
            decoded_freq     = decoded_freq,
            stability_w      = stability_w,
            novelty_flag     = novelty_flag,
            plv_vector       = plv_vector,
            prediction_error = self._last_prediction_error,   # ← L2→M54 feedback
        )

        bmu_idx  = cortex_out['bmu_idx']
        qe_norm  = cortex_out['qe_norm']

        # ── 3. Memory update (M55) — with L2 feedback ─────────
        # curiosity from the PREVIOUS step scales Hebbian write strength.
        # If L2 is in novel sequence territory, consolidate memory harder.
        mem_out = self.memory.step(
            bmu_idx   = bmu_idx,
            qe_norm   = qe_norm,
            curiosity = self._last_curiosity,                 # ← L2→M55 feedback
        )

        # ── 4. Recall — get familiarity for L2 ────────────────
        recall_out = self.memory.recall(bmu_idx)

        # ── 5. L2 learns and outputs new feedback signals ─────
        l2_out = self.pred.step(
            bmu_idx     = bmu_idx,
            qe_norm     = qe_norm,
            familiarity = recall_out['familiarity'],
        )

        # ── 6. Store feedback signals for NEXT step ───────────
        self._last_prediction_error = l2_out['prediction_error']
        self._last_curiosity        = l2_out['curiosity']

        self.t += 1

        # ── 7. Return unified output ──────────────────────────
        return {
            # M54 signals
            'bmu_idx':              bmu_idx,
            'bmu_pos':              cortex_out['bmu_pos'],
            'qe':                   cortex_out['qe'],
            'qe_norm':              qe_norm,
            'sigma':                cortex_out['sigma'],
            'eta':                  cortex_out['eta'],
            'is_novel':             cortex_out['is_novel'],
            # M55 signals
            'familiarity':          recall_out['familiarity'],
            'top_associations':     recall_out['top_associations'],
            'wrote':                mem_out['wrote'],
            # L2 signals
            'prediction_error':     l2_out['prediction_error'],
            'correct':              l2_out['correct'],
            'predicted_bmu':        pred_out['predicted_bmu'],
            'confidence':           pred_out['confidence'],
            'curiosity':            l2_out['curiosity'],
            # Feedback state (what was fed INTO the modules this step)
            'fed_prediction_error': self._last_prediction_error,
            'fed_curiosity':        self._last_curiosity,
        }

    # ── Convenience accessors ─────────────────────────────────

    def reset_feedback(self):
        """Zero out feedback signals — use when restarting a session
        without rebuilding the whole Brain (e.g. between test conditions)."""
        self._last_prediction_error = 0.0
        self._last_curiosity        = 0.0

    def get_feedback_state(self) -> dict:
        """Current one-step-delayed feedback values."""
        return {
            'prediction_error': self._last_prediction_error,
            'curiosity':        self._last_curiosity,
        }

    def summary(self):
        """Human-readable state summary of all three modules."""
        print(f"\n  Brain — step {self.t}")
        print(f"  Feedback state:")
        print(f"    last prediction_error → M54 : {self._last_prediction_error:.4f}")
        print(f"    last curiosity        → M55 : {self._last_curiosity:.4f}")
        print()
        self.cortex.get_surprise_stats()
        self.memory.summary()
        self.pred.summary()