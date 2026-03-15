"""
BRAIN — Integrated Cognitive Stack with Feedback Loops  (v10)
=============================================================

This file owns the cognitive modules, wires their feedback loops,
and hosts Attention, Thought, Valence, M56 (ActionLayer), and
M57 (Planner — mental simulation / look-ahead).

M50 (the ear) stays separate — it feeds INTO Brain.step(), not inside it.

ARCHITECTURE
------------
                    ┌──────────────────────────────────────┐
  M50 (ear)  ──────▶│              Brain                   │
                    │                                      │
                    │  CortexM54  (M54)                    │
                    │      │ bmu_idx, qe_norm               │
                    │      ▼                                │
                    │  AssociativeMemory (M55)              │
                    │      │ familiarity                    │
                    │      ▼                                │
                    │  SequencePredictor (L2)               │
                    │      │                                │
                    │   surprise_signal ────────────────────┼──▶ M54 (next step)
                    │   curiosity_delta ────────────────────┼──▶ M55 (next step)
                    │      │                                │
                    │      ▼                                │
                    │  Attention                            │
                    │      │ salience, salience_delta,      │
                    │      │ attention_gate, attended_bmu   │
                    └──────────────────────────────────────┘

FEEDBACK LOOPS
--------------
Loop 1 — L2 → M54 (sequence surprise → cortical plasticity)
  Signal: surprise_signal = max(0, prediction_error − error_ema)
  Delta of prediction_error above its own running baseline.
  ~0 when stable, spikes when error suddenly increases above normal.

Loop 2 — L2 → M55 (curiosity → memory consolidation)
  Signal: curiosity_delta = max(0, curiosity − curiosity_ema)
  Same delta principle applied to L2's curiosity EMA.

ATTENTION
---------
Attention sits above all three modules. It reads Brain's outputs each
step and produces:
  salience        — how much to attend this moment [0,1]
  salience_delta  — spike above EMA (delta rule, safe for downstream)
  attention_gate  — spatial soft mask over the 64-neuron BMU space (64,)
  attended_bmu    — which BMU has the highest gate weight
  gate_entropy    — how focused the gate is (0=focused, 1=diffuse)

Attention does NOT feed back into M54/M55/L2 in this version.
It is informational — available for Thought or other higher modules.

CALL ORDER (per step)
---------------------
1. pred.predict()                                    ← prediction BEFORE cortex fires
2. cortex.step(..., prediction_error=surprise_sig)   ← M54 learns (delta-boosted)
3. memory.step(..., curiosity=curiosity_delta)        ← M55 writes (delta-boosted)
4. memory.recall(bmu_idx)                            ← get familiarity for L2
5. pred.step(bmu_idx, ..., prediction_bias)          ← L2 learns + Thought bias applied
6. update EMAs, compute deltas for NEXT step         ← store feedback state
7. attention.step(..., thought_confidence_delta)     ← Attention reads all + Thought suppression
8. thought.step(attended_bmu, bmu_idx, pred, ...)    ← Thought reads Attention, stores for next step
9. action.step(bmu_idx, rpe, focus_entropy, ...)     ← M56 updates Q + selects next action

BACKWARD COMPATIBILITY
----------------------
All module files work identically standalone. Attention is instantiated
inside Brain — callers do not need to manage it separately.

Old code reading any existing Brain key continues to work unchanged.
New Attention keys are additive.

USAGE
-----
  from brain import Brain

  brain = Brain(seed=42)

  result = brain.step(
      decoded_freq = fused,
      stability_w  = w,
      novelty_flag = float(nov),
      plv_vector   = plv_slow,
  )

  # Existing keys (unchanged)
  result['bmu_idx']          # M54 — which neuron fired
  result['qe_norm']          # M54 — perceptual surprise
  result['familiarity']      # M55 — recognition signal
  result['prediction_error'] # L2  — raw sequence error (informational)
  result['curiosity']        # L2  — raw curiosity EMA (informational)
  result['surprise_signal']  # Brain — delta fed into M54
  result['curiosity_delta']  # Brain — delta fed into M55
  result['error_ema']        # Brain — prediction error baseline
  result['curiosity_ema']    # Brain — curiosity baseline

  # Attention keys (unchanged)
  result['salience']         # Attention — how much to attend this step
  result['salience_ema']     # Attention — smoothed salience
  result['salience_delta']   # Attention — spike above EMA
  result['attention_gate']   # Attention — (64,) spatial soft mask
  result['attended_bmu']     # Attention — most attended neuron
  result['gate_entropy']     # Attention — gate focus (0=focused, 1=diffuse)

  # Thought keys (new)
  result['expected_bmu']        # Thought — BMU Thought expects to fire next
  result['prediction_bias']     # Thought — (64,) soft distribution for L2 next step
  result['thought_confidence']  # Thought — how concentrated the prediction is
  result['confidence_ema']      # Thought — smoothed confidence
  result['confidence_delta']    # Thought — spike above EMA (fed to Attention next step)
  result['expectation_error']   # Thought — was last prediction close to actual?
  result['focus_entropy']       # Thought — prediction spread (0=certain, 1=uniform)
"""

import numpy as np

from m56_cortex import CortexM56
from m55_memory import AssociativeMemory
from l2_predictor import SequencePredictor
from attention import Attention
from thought import Thought
from valence import Valence
from m56_action import ActionLayer
from m57_planner import Planner
from l3_concepts import ConceptLayer


# ═══════════════════════════════════════════════════════════════
# FEEDBACK PARAMETERS
# ═══════════════════════════════════════════════════════════════

# EMA alpha for the running baseline of prediction_error and curiosity.
# tau = 1/alpha steps. At 0.10: tau ~10 steps (~0.5s at dt=0.05).
# Fast enough to track transitions, slow enough not to chase step noise.
FEEDBACK_EMA_ALPHA = 0.10

# Initial EMA value. Set to 1.0 (ceiling) so cold-start deltas are ~0.
FEEDBACK_EMA_INIT = 1.0


# ═══════════════════════════════════════════════════════════════
# BRAIN
# ═══════════════════════════════════════════════════════════════

class Brain:
    """
    Integrated M54 + M55 + L2 + Attention cognitive stack.

    Feedback signals fed into M54 and M55 are the DELTA of L2's outputs
    above their running EMA baseline — not the raw values. This prevents
    the SOM from being permanently destabilised by L2's structurally-high
    baseline error in a many-BMU environment.

    Attention is instantiated and run inside Brain every step.
    It reads Brain outputs and produces salience + gate signals.
    It does NOT modify M54, M55, or L2.

    Parameters
    ----------
    seed : int
        Random seed passed to all modules that accept one.
    """

    def __init__(self, seed: int = 42):
        self.cortex    = CortexM56(seed=seed)
        self.memory    = AssociativeMemory(seed=seed)
        self.pred      = SequencePredictor()
        self.attention = Attention()
        self.thought   = Thought()
        self.valence   = Valence()
        self.action    = ActionLayer(seed=seed)
        self.planner   = Planner(n_actions=self.action._n_actions)
        self.l3        = ConceptLayer(n_zones=8)

        # freq_bmu_counters[fi][bmu] = visit count — fed to L3.assign_zones periodically
        from collections import Counter
        self._freq_bmu_counters = [Counter() for _ in range(8)]
        self._l3_zone_interval  = 2000   # reassign zones every N steps

        # Running EMAs for computing delta signals.
        self._error_ema     = float(FEEDBACK_EMA_INIT)
        self._curiosity_ema = float(FEEDBACK_EMA_INIT)

        # One-step-delayed delta signals fed into modules (start at 0).
        self._last_surprise_signal  = 0.0
        self._last_curiosity_delta  = 0.0
        self._last_rpe_positive     = 0.0   # V1 → M55 next step
        self._last_familiarity      = 0.0   # M55 → M54 next step (LTD suppression)

        # Raw L2 outputs from last step (diagnostics).
        self._last_prediction_error = 0.0
        self._last_curiosity        = 0.0

        self.t = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             decoded_freq: float,
             stability_w:  float,
             novelty_flag: float,
             plv_vector:   np.ndarray,
             reward:       float = 0.0,
             freq_idx:     int   = -1,
             world_moved:  bool  = True) -> dict:
        """
        One full cognitive step: perception → memory → prediction →
        feedback → attention.

        Parameters
        ----------
        decoded_freq : float    — fused frequency from M50 (Hz)
        stability_w  : float    — signal stability weight [0,1]
        novelty_flag : float    — CUSUM regime-change flag from M50
        plv_vector   : ndarray  — raw PLV components from M50

        Returns
        -------
        dict with all signals from all modules.

        M54 keys:      bmu_idx, bmu_pos, qe, qe_norm, sigma, eta, is_novel
        M55 keys:      familiarity, top_associations, wrote
        L2 raw keys:   prediction_error, correct, predicted_bmu,
                       confidence, curiosity
        Feedback keys: surprise_signal, curiosity_delta, error_ema, curiosity_ema
        Attention keys: salience, salience_ema, salience_delta,
                        attention_gate, attended_bmu, gate_entropy
        """
        # ── 1. Predict BEFORE cortex fires ────────────────────
        pred_out = self.pred.predict()

        # ── 2. Cortex fires (M54) — with delta feedback ───────
        cortex_out = self.cortex.step(
            decoded_freq     = decoded_freq,
            stability_w      = stability_w,
            novelty_flag     = novelty_flag,
            plv_vector       = plv_vector,
            prediction_error = self._last_surprise_signal,   # delta, not raw
            familiarity      = self._last_familiarity,       # M55 → M54 LTD
        )

        bmu_idx = cortex_out['bmu_idx']
        qe_norm = cortex_out['qe_norm']

        # ── 3. Memory update (M55) — with delta feedback ──────
        mem_out = self.memory.step(
            bmu_idx      = bmu_idx,
            qe_norm      = qe_norm,
            curiosity    = self._last_curiosity_delta,          # delta, not raw
            rpe_positive = self._last_rpe_positive,             # V1 → M55
        )

        # ── 4. Recall — get familiarity for L2 ────────────────
        recall_out  = self.memory.recall(bmu_idx)
        familiarity = recall_out['familiarity']

        # Store familiarity for M54 on the NEXT step (LTD suppression).
        # Fed next-step so M54's suppression is based on how well-known
        # this BMU region WAS before it fired, not during firing.
        self._last_familiarity = familiarity

        # ── 5. L2 learns and outputs raw signals ──────────────
        l2_out = self.pred.step(
            bmu_idx          = bmu_idx,
            qe_norm          = qe_norm,
            familiarity      = familiarity,
            prediction_bias  = self.thought._last_prediction_bias,
        )

        raw_error     = l2_out['prediction_error']
        raw_curiosity = l2_out['curiosity']

        # ── 5b. L3 Concept Layer — zone tracking ──────────────
        # Update freq_bmu_counters if ground-truth freq_idx supplied.
        if freq_idx >= 0:
            self._freq_bmu_counters[freq_idx][bmu_idx] += 1
            # Periodic zone reassignment after warmup
            if (self.t >= 5000 and self.t % self._l3_zone_interval == 0):
                self.l3.assign_zones_from_counters(self._freq_bmu_counters)

        l3_scores = self.pred._P[bmu_idx] if hasattr(self.pred, '_P') else None
        l3_out = self.l3.step(
            bmu_idx   = bmu_idx,
            l2_scores = l3_scores,
            freq_idx  = freq_idx,
        )
        if reward != 0.0:
            zone_for_reward = freq_idx if freq_idx >= 0 else l3_out['zone_idx']
            self.l3.update_zone_reward(zone_for_reward, reward)

        # ── 6. Compute delta signals, update EMAs ─────────────
        # Delta = how much the signal ROSE above its own running average.
        # Clipped to [0, 1]: only upward spikes propagate.
        # EMA updated AFTER delta so we compare against pre-existing baseline.
        surprise_signal = float(np.clip(raw_error     - self._error_ema,     0.0, 1.0))
        curiosity_delta = float(np.clip(raw_curiosity  - self._curiosity_ema, 0.0, 1.0))

        self._error_ema     = ((1.0 - FEEDBACK_EMA_ALPHA) * self._error_ema
                               + FEEDBACK_EMA_ALPHA * raw_error)
        self._curiosity_ema = ((1.0 - FEEDBACK_EMA_ALPHA) * self._curiosity_ema
                               + FEEDBACK_EMA_ALPHA * raw_curiosity)

        # Store for NEXT step
        self._last_surprise_signal  = surprise_signal
        self._last_curiosity_delta  = curiosity_delta
        self._last_prediction_error = raw_error
        self._last_curiosity        = raw_curiosity

        # ── 6b. Valence (V1) — reward prediction error ────────
        # Runs after L2 (needs prediction_error) and before Attention.
        # Computes RPE from intrinsic reward (1 - prediction_error) and
        # optional external reward. Produces pos_rpe for M55 next step.
        # Call order: V1 at t → pos_rpe stored → fed to M55 at t+1.
        # Same next-step pattern as surprise_signal and curiosity_delta.
        v1_out = self.valence.step(
            prediction_error = raw_error,
            reward           = float(reward),
            familiarity      = familiarity,
        )
        self._last_rpe_positive = v1_out['pos_rpe']

        # ── 7. Attention reads all outputs ────────────────────
        attn_out = self.attention.step(
            bmu_idx                  = bmu_idx,
            qe_norm                  = qe_norm,
            familiarity              = familiarity,
            surprise_signal          = surprise_signal,
            curiosity_delta          = curiosity_delta,
            thought_confidence_delta = self.thought._last_confidence_delta,
        )

        # ── 8. Thought reads Attention output ─────────────────
        # Thought queries L2's sequence memory from attended_bmu and
        # builds a prediction_bias for the NEXT step's L2 context,
        # and a confidence_delta for the NEXT step's Attention salience.
        # v6: also passes memory so Thought can blend M55 associations.
        thought_out = self.thought.step(
            attended_bmu = attn_out['attended_bmu'],
            bmu_idx      = bmu_idx,
            pred         = self.pred,
            salience     = attn_out['salience'],
            memory       = self.memory,
        )

        # ── 9. Action layer (M56) ─────────────────────────────
        # Runs last: reads rpe from V1 and focus signals from Thought.
        # Selects the action for the NEXT step AND updates Q from this
        # step's outcome. Call order:
        #   action.step(bmu_idx, rpe, focus_entropy, thought_confidence)
        #     ├─ update()   : Q[prev_bmu, prev_action] += ETA_Q * rpe * e
        #     └─ select()   : epsilon-greedy → action for next step
        # The 'action' key in the output is what to take NEXT step.
        m56_out = self.action.step(
            bmu_idx            = bmu_idx,
            rpe                = v1_out['rpe'],
            focus_entropy      = thought_out['focus_entropy'],
            thought_confidence = thought_out['thought_confidence'],
            freq_idx           = freq_idx,     # ground-truth node index
            world_moved        = world_moved,  # False on wall hits — gates replay/trace
        )

        # ── 10. Planner (M57) — mental simulation / look-ahead ──
        # Runs last. Reads L2, Valence, M55 READ-ONLY to simulate
        # PLANNING_DEPTH steps forward for each candidate action.
        # Overrides M56's habit action when planning_weight >
        # PLANNING_GATE_THRESH (confidence × focus × salience).
        # When planning is inactive it returns M56's action unchanged.
        m57_out = self.planner.step(
            bmu_idx            = bmu_idx,
            pred               = self.pred,
            valence            = self.valence,
            memory             = self.memory,
            m56_action         = m56_out['action'],
            m56_q_values       = m56_out['q_values'],
            thought_confidence = thought_out['thought_confidence'],
            focus_entropy      = thought_out['focus_entropy'],
            salience           = attn_out['salience'],
            l3                 = self.l3,
        )

        self.t += 1

        # ── 9. Return unified output ──────────────────────────
        return {
            # M54
            'bmu_idx':          bmu_idx,
            'bmu_pos':          cortex_out['bmu_pos'],
            'qe':               cortex_out['qe'],
            'qe_norm':          qe_norm,
            'sigma':            cortex_out['sigma'],
            'eta':              cortex_out['eta'],
            'is_novel':         cortex_out['is_novel'],
            # M55
            'familiarity':      familiarity,
            'top_associations': recall_out['top_associations'],
            'wrote':            mem_out['wrote'],
            # L2 raw (informational — do not feed these directly back)
            'prediction_error': raw_error,
            'correct':          l2_out['correct'],
            'predicted_bmu':    pred_out['predicted_bmu'],
            'confidence':       pred_out['confidence'],
            'curiosity':        raw_curiosity,
            # Feedback signals fed into modules this step
            'surprise_signal':  surprise_signal,
            'curiosity_delta':  curiosity_delta,
            'error_ema':        self._error_ema,
            'curiosity_ema':    self._curiosity_ema,
            # Attention
            'salience':         attn_out['salience'],
            'salience_ema':     attn_out['salience_ema'],
            'salience_delta':   attn_out['salience_delta'],
            'attention_gate':   attn_out['attention_gate'],
            'attended_bmu':     attn_out['attended_bmu'],
            'gate_entropy':     attn_out['gate_entropy'],
            # Thought
            'expected_bmu':          thought_out['expected_bmu'],
            'prediction_bias':       thought_out['prediction_bias'],
            'thought_confidence':    thought_out['thought_confidence'],
            'confidence_ema':        thought_out['confidence_ema'],
            'confidence_delta':      thought_out['confidence_delta'],
            'expectation_error':     thought_out['expectation_error'],
            'focus_entropy':         thought_out['focus_entropy'],
            'assoc_weight':          thought_out['assoc_weight'],
            # Valence (V1)
            'rpe':                   v1_out['rpe'],
            'pos_rpe':               v1_out['pos_rpe'],
            'neg_rpe':               v1_out['neg_rpe'],
            'reward_ema':            v1_out['reward_ema'],
            'total_reward':          v1_out['total_reward'],
            'intrinsic_reward':      v1_out['intrinsic_reward'],
            'novelty_bonus':         v1_out['novelty_bonus'],
            # Action (M56 — habit)
            'action':                m57_out['action'],      # M57 final (= M56 when planning inactive)
            'q_values':              m56_out['q_values'],
            'q_max':                 m56_out['q_max'],
            'action_epsilon':        m56_out['epsilon'],
            'action_explore':        m56_out['explore'],
            'q_mean':                m56_out['q_mean'],
            'q_nonzero_frac':        m56_out['q_nonzero_frac'],
            'habit_action':          m56_out['action'],      # M56 raw habit (for diagnostics)
            # Planner (M57 — deliberation)
            'planned_action':        m57_out['planned_action'],
            'planning_weight':       m57_out['planning_weight'],
            'planning_active':       m57_out['planning_active'],
            'sim_values':            m57_out['sim_values'],
            'sim_depth':             m57_out['sim_depth'],
            'plan_vs_habit_delta':   m57_out['plan_vs_habit_delta'],
            # L3 Concept Layer
            'zone_idx':              l3_out['zone_idx'],
            'zone_confidence':       l3_out['zone_confidence'],
            'zone_probs':            l3_out['zone_probs'],
            'top_zone_pred':         l3_out['top_zone_pred'],
            'zone_pred_conf':        l3_out['zone_pred_conf'],
            'zones_stable':          l3_out['zones_stable'],
        }

    # ── Convenience accessors ─────────────────────────────────

    def reset_feedback(self):
        """Reset feedback state — use between test conditions."""
        self._error_ema             = float(FEEDBACK_EMA_INIT)
        self._curiosity_ema         = float(FEEDBACK_EMA_INIT)
        self._last_surprise_signal  = 0.0
        self._last_curiosity_delta  = 0.0
        self._last_rpe_positive     = 0.0
        self._last_familiarity      = 0.0
        self._last_prediction_error = 0.0
        self._last_curiosity        = 0.0

    def get_feedback_state(self) -> dict:
        """Current feedback state — raw signals and their EMA baselines."""
        return {
            'prediction_error': self._last_prediction_error,
            'curiosity':        self._last_curiosity,
            'surprise_signal':  self._last_surprise_signal,
            'curiosity_delta':  self._last_curiosity_delta,
            'error_ema':        self._error_ema,
            'curiosity_ema':    self._curiosity_ema,
        }

    def summary(self):
        """Human-readable state summary."""
        print(f"\n  Brain — step {self.t}")
        print(f"  Feedback (delta-based):")
        print(f"    error_ema:       {self._error_ema:.4f}  "
              f"(last raw: {self._last_prediction_error:.4f})")
        print(f"    curiosity_ema:   {self._curiosity_ema:.4f}  "
              f"(last raw: {self._last_curiosity:.4f})")
        print(f"    → surprise_signal → M54: {self._last_surprise_signal:.4f}")
        print(f"    → curiosity_delta → M55: {self._last_curiosity_delta:.4f}")
        print()
        self.cortex.get_surprise_stats()
        self.memory.summary()
        self.pred.summary()
        self.attention.summary()
        self.thought.summary()
        self.action.summary()