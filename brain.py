"""
BRAIN — Integrated Cognitive Stack with Global Workspace  (v11)
=============================================================

v11: Global Workspace (GWS) integration layer added.

All module signals now converge into a single unified internal state
before action selection. This is the integration that turns a pipeline
of modules into something with a coherent "moment" — arousal, valence
tone, and curiosity pull all exist simultaneously and shape behavior
together, not sequentially.

New module: gws.py — GlobalWorkspace
  Reads: qe_norm, familiarity, prediction_error, thought_confidence,
         rpe, intrinsic_rwd, corridor_boredom, steps_since_reward,
         salience, l4_top_prob
  Broadcasts:
    arousal        — global activation (noradrenaline tone)
    valence_tone   — motivational direction (dopamine baseline)
    curiosity_pull — directed pull toward unresolved zones
    epsilon_boost  — additive exploration from global state

New behavior: curiosity is now a PULL not just epsilon noise.
  When L2's prediction error is high at a zone, GWS accumulates
  surprise_debt for that zone. The debt vector becomes curiosity_pull
  — a directed bias toward zones the brain doesn't understand yet.
  The brain is drawn back toward what confused it, not just toward
  random novelty.

Call order (per step):
  1-8b. All existing modules unchanged.
  8c.   gws.step() — integrates all signals simultaneously.
  9.    action.step(epsilon_floor=max(wm_floor, gws_boost)) — combined floor.

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
from evaluators import Attention, Thought, Valence
from m56_action import ActionLayer
from m57_planner import Planner
from l3_concepts import ConceptLayer
from l4_position import PositionBelief
from m58_workingmemory import WorkingMemory
from global_workspace import GlobalWorkspace
from m59_selfmodel import SelfModel
from m60_questions import QuestionMemory
from m62_consistency import ConsistencyChecker
from m63_episodic import EpisodicTrace
from m64_language import LanguageSeed


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

    def __init__(self, seed: int = 42, node_fi: dict = None):
        self.cortex    = CortexM56(seed=seed)
        self.memory    = AssociativeMemory(seed=seed)
        self.pred      = SequencePredictor()
        self.attention = Attention()
        self.thought   = Thought()
        self.valence   = Valence()
        self.action    = ActionLayer(seed=seed)
        self.planner   = Planner(n_actions=self.action._n_actions)
        self.l3        = ConceptLayer(n_zones=8)
        self.wm        = WorkingMemory(n_zones=8, seed=seed)
        self.gws       = GlobalWorkspace(n_zones=8)
        self.selfmodel  = SelfModel(seed=seed)
        self.questions  = QuestionMemory(seed=seed)
        self.consistency = ConsistencyChecker(n_actions=self.action._n_actions, seed=seed)
        self.episodic    = EpisodicTrace(n_actions=self.action._n_actions, seed=seed)
        self.language    = LanguageSeed(n_actions=self.action._n_actions, seed=seed)

        # L4: position belief module.
        if node_fi is not None:
            self.l4 = PositionBelief(node_fi=node_fi)
            # Tell ActionLayer about node frequency uniqueness for Q_n gating.
            # Unique nodes (fi shared by no other node) use standard threshold.
            # Aliased nodes (fi shared with another node) use higher threshold —
            # L4 must be more certain before Q_n is trusted for these nodes.
            from collections import Counter as _Counter
            fi_counts = _Counter(node_fi.values())
            import m56_action as _m56
            _m56.L4_Q_N_UNIQUE_NODES = {
                n for n, fi in node_fi.items() if fi_counts[fi] == 1
            }
            _m56.L4_Q_N_ALIASED_NODES = {
                n for n, fi in node_fi.items() if fi_counts[fi] > 1
            }
            # Large-world scaling: slower L4 belief decay and lower CTM warmup
            # for worlds with >12 nodes (e.g. World 5's 16-node grid).
            if len(node_fi) > 12:
                import l4_position as _l4
                _l4.L4_BELIEF_DECAY = _l4.L4_BELIEF_DECAY_LARGE_WORLD
                _l4.L4_CTM_WARMUP   = _l4.L4_CTM_WARMUP_LARGE_WORLD
        else:
            self.l4 = None   # L4 inactive — backward compatible

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
        self._prev_zone_for_T      = -1
        self._prev_action_for_T    = -1
        self._prev_prev_zone_for_T = -1   # two steps back — for TC context disambiguation

        # M63 prior cache — previous step's episodic output fed into next M57 call.
        # Causal order: M63 runs after M57 (records what happened), so its action_prior
        # feeds into the NEXT step's M57 (biases planning with remembered episode).
        # Initialised to uniform prior (no episode influence at step 0).
        _n_act = self.action._n_actions
        self._last_m63_recognition = 0.0
        self._last_m63_action_prior = np.ones(_n_act, dtype=np.float64) / _n_act

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
        # ── 0. Update action-conditioned zone transition model ─
        # Also record whether the transition was correctly predicted (TPE).
        # Both calls are gated on world_moved — wall hits don't update the
        # transition model and don't count as prediction opportunities.
        if world_moved and self._prev_zone_for_T >= 0 and self._prev_action_for_T >= 0 and freq_idx >= 0:
            self.l3.update_action_transition(
                prev_zone      = self._prev_zone_for_T,
                action         = self._prev_action_for_T,
                curr_zone      = freq_idx,
                prev_prev_zone = self._prev_prev_zone_for_T,
            )
            self.l3.record_transition_outcome(
                prev_zone      = self._prev_zone_for_T,
                action         = self._prev_action_for_T,
                actual_zone    = freq_idx,
                prev_prev_zone = self._prev_prev_zone_for_T,
            )

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
        # Pass last_action and world_moved so L2's PA matrix learns
        # action-conditioned transitions (v2 feature).
        # _prev_action_for_T is the action taken before this step.
        # world_moved is whether that action caused a real transition.
        l2_out = self.pred.step(
            bmu_idx          = bmu_idx,
            qe_norm          = qe_norm,
            familiarity      = familiarity,
            prediction_bias  = self.thought._last_prediction_bias,
            last_action      = self._prev_action_for_T,
            world_moved      = world_moved,
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
        # Update zone visit EMA every step — drives curiosity bonus in zone_value.
        # Uses bucketed freq_idx (sound-derived) so the brain tracks which zones
        # it has been spending time in, without needing ground-truth labels.
        visit_zone = freq_idx if freq_idx >= 0 else l3_out['zone_idx']
        if visit_zone >= 0:
            self.l3.update_zone_visit(visit_zone)

        if reward != 0.0:
            # Credit the zone where food IS (current freq_idx) — this tells
            # L3/M57 "zone 4 contains food", which is correct for planning.
            # TD credit (Q learning) correctly uses the previous action via
            # M56's eligibility trace — that's separate from zone valuation.
            zone_for_reward = freq_idx if freq_idx >= 0 else l3_out['zone_idx']
            self.l3.update_zone_reward(zone_for_reward, reward)

        # ── 5c. L4 Position Belief — Bayesian location tracking ──
        # L4 updates its belief distribution over all nodes using the
        # observed frequency and the action taken last step.
        # It learns the transition model from experience — no map given.
        # When confident (top_prob > L4_CONFIDENCE_THRESH), its top_node
        # estimate grounds M57's planning in a specific location.
        if self.l4 is not None:
            l4_out = self.l4.step(
                curr_fi     = freq_idx if freq_idx >= 0 else -1,
                action      = self._prev_action_for_T,
                world_moved = world_moved,
            )
        else:
            l4_out = {
                'top_node':       None,
                'top_prob':       0.0,
                'belief_entropy': 1.0,
                'confident':      False,
                'belief_vector':  None,
            }

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

        # ── 8b. Working Memory (M58) — trajectory buffer ──────
        # Runs before M56 so epsilon_floor is available for action selection.
        # Records (freq_idx, action, reward) for this step.
        # _prev_action_for_T is the action taken to reach the current state.
        wm_out = self.wm.step(
            freq_idx = freq_idx if freq_idx >= 0 else -1,
            action   = self._prev_action_for_T,
            reward   = float(reward),
        )

        # ── 8c. Global Workspace — unified integration ─────────
        # All signals from all modules exist here simultaneously.
        # Broadcasts: arousal (global activation), valence_tone (motivational
        # direction), curiosity_pull (directed pull toward uncertain zones),
        # epsilon_boost (additive exploration from global state).
        gws_out = self.gws.step(
            qe_norm            = qe_norm,
            familiarity        = familiarity,
            freq_idx           = freq_idx if freq_idx >= 0 else -1,
            prediction_error   = raw_error,
            thought_confidence = thought_out['thought_confidence'],
            rpe                = v1_out['rpe'],
            intrinsic_rwd      = v1_out['intrinsic_reward'],
            corridor_boredom   = wm_out['corridor_boredom'],
            steps_since_reward = wm_out['steps_since_reward'],
            salience           = attn_out['salience'],
            l4_top_prob        = l4_out['top_prob'],
        )

        # ── 8d. Self-Model (M59) — the brain's mirror ──────────
        # Runs after GWS so it can read the unified internal state.
        # Builds a characterisation of what kind of thing the brain
        # is right now — not what the world looks like, but what the
        # brain itself looks like from the inside.
        sm_out = self.selfmodel.step(
            arousal           = gws_out['arousal'],
            coherence         = gws_out['coherence'],
            valence_tone      = gws_out['valence_tone'],
            curiosity_pull    = gws_out['curiosity_pull'],
            tension           = gws_out['tension'],
            surprise_debt     = gws_out['surprise_debt'],
            thought_confidence= thought_out['thought_confidence'],
            familiarity       = familiarity,
            salience          = attn_out['salience'],
            l4_top_prob       = l4_out['top_prob'],
            reward            = float(reward),
        )

        # ── 8e. Question Memory (M60) — store and track confusion ──
        # Runs after M59. Creates questions when L2 wrong + L4 uncertain
        # + self-model says confused/curious. Returns zone_pull for M57.
        q60_out = self.questions.step(
            prediction_error = raw_error,
            freq_idx         = freq_idx if freq_idx >= 0 else -1,
            bmu_idx          = bmu_idx,
            predicted_bmu    = thought_out['expected_bmu'],
            l4_top_prob      = l4_out['top_prob'],
            sm_state_vec     = sm_out['self_state_vec'],
            sm_state_label   = sm_out['state_label'],
            reward           = float(reward),
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
            l4_top_node        = l4_out['top_node'],
            l4_top_prob        = l4_out['top_prob'],
            epsilon_floor      = max(wm_out['epsilon_floor'],
                                     gws_out['epsilon_boost']),  # best of M58 + GWS
            node_fi_override   = getattr(self, '_node_fi_override_hint', None),
        )

        # ── 10. Planner (M57) — mental simulation / look-ahead ──
        # Runs last. Reads L2, Valence, M55 READ-ONLY to simulate
        # PLANNING_DEPTH steps forward for each candidate action.
        # Overrides M56's habit action when planning_weight >
        # PLANNING_GATE_THRESH (confidence × focus × salience).
        # When planning is inactive it returns M56's action unchanged.
        # Get TPE accuracy for current zone — passed to M57 to gate planning.
        # Use context-aware TC table when available (disambiguates aliased nodes).
        # Falls back to freq_idx-only T table when TC has insufficient data.
        _curr_zone = freq_idx if freq_idx >= 0 else self.l3._last_zone_idx
        _tpe_acc   = self.l3.get_tpe_accuracy_ctx(self._prev_zone_for_T, _curr_zone)
        if _tpe_acc == 0.0:
            _tpe_acc = self.l3.get_tpe_accuracy(_curr_zone)

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
            tpe_accuracy       = _tpe_acc,
            prev_zone          = self._prev_zone_for_T,
            l4_top_node        = l4_out['top_node'],
            l4_top_prob        = l4_out['top_prob'],
            l4_confident       = l4_out['confident'],
            gws_curiosity_pull  = gws_out['curiosity_pull'],
            gws_tension         = gws_out['tension'],
            self_state_label    = sm_out['state_label'],
            q60_zone_pull       = q60_out['zone_pull'],
            m63_action_prior    = self._last_m63_action_prior,
            m63_recognition     = self._last_m63_recognition,
        )

        # ── M61: THOUGHT LOOP — internal dialogue ────────────────
        # When M60 has open questions AND the brain is in a state that
        # benefits from deliberation (confused, curious, stuck, hunting),
        # run up to M61_MAX_LOOPS additional M57→Thought cycles.
        #
        # Each cycle:
        #   1. M57's best_sim_bmu feeds into Thought as simulated_bmu
        #   2. Thought updates its prediction_bias toward the simulated future
        #   3. M57 re-simulates with the updated Thought bias
        #
        # This lets the brain dwell on a question — running the simulation
        # forward in imagination before committing to action.
        # The final m57_out (last loop iteration) is used for the action.
        #
        # Gate conditions:
        #   - M60 has at least one open question (something to think about)
        #   - Self-model is in a deliberative state (not 'satisfied'/'drifting')
        #   - M57 actually simulated this step (sim_depth > 0)
        #   - Brain is past M57 warmup (planner is active)
        #
        # Biological basis: hippocampal-prefrontal theta coupling during
        # deliberation. When an animal pauses at a choice point, place cells
        # sweep forward along candidate paths ("vicarious trial and error").
        # Each sweep is a thought loop cycle — the brain simulates the path
        # before committing to it. Observed in rats at T-maze decision points.
        #
        # M61 parameters
        M61_MAX_LOOPS        = 2      # max extra simulation cycles per step
        M61_SIM_WEIGHT_BASE  = 0.40   # how strongly simulated BMU blends into Thought
        M61_DELIBERATE_LABELS = {'confused', 'curious', 'stuck', 'hunting', 'alert'}

        m61_loops_run     = 0
        m61_thought_iters = []   # record of each loop's thought_confidence
        m61_sim_history   = []   # record of each loop's sim_values (for M62)

        if (q60_out['open_count'] > 0
                and sm_out['state_label'] in M61_DELIBERATE_LABELS
                and m57_out['sim_depth'] > 0
                and self.planner.t > 1):   # planner has run at least once

            for _loop in range(M61_MAX_LOOPS):
                # Step 1: feed M57's simulated next BMU into Thought
                # sim_weight scales with open question urgency — more questions
                # = brain is more focused on simulation vs real perception
                urgency_scale = float(np.clip(q60_out['open_count'] / 4.0, 0.0, 1.0))
                sim_w = M61_SIM_WEIGHT_BASE * (0.5 + 0.5 * urgency_scale)

                loop_thought_out = self.thought.step(
                    attended_bmu  = attn_out['attended_bmu'],
                    bmu_idx       = bmu_idx,
                    pred          = self.pred,
                    salience      = attn_out['salience'],
                    memory        = self.memory,
                    simulated_bmu = m57_out['best_sim_bmu'],
                    sim_weight    = sim_w,
                )
                m61_thought_iters.append(loop_thought_out['thought_confidence'])

                # Step 2: re-simulate with updated thought bias
                # The updated prediction_bias from loop_thought_out is now
                # stored in self.thought._last_prediction_bias and will be
                # used by pred.step() on the NEXT real step.
                # For the loop itself, we re-run M57 with updated confidence.
                loop_m57_out = self.planner.step(
                    bmu_idx            = bmu_idx,
                    pred               = self.pred,
                    valence            = self.valence,
                    memory             = self.memory,
                    m56_action         = m56_out['action'],
                    m56_q_values       = m56_out['q_values'],
                    thought_confidence = loop_thought_out['thought_confidence'],
                    focus_entropy      = loop_thought_out['focus_entropy'],
                    salience           = attn_out['salience'],
                    l3                 = self.l3,
                    tpe_accuracy       = _tpe_acc,
                    prev_zone          = self._prev_zone_for_T,
                    l4_top_node        = l4_out['top_node'],
                    l4_top_prob        = l4_out['top_prob'],
                    l4_confident       = l4_out['confident'],
                    gws_curiosity_pull  = gws_out['curiosity_pull'],
                    gws_tension         = gws_out['tension'],
                    self_state_label    = sm_out['state_label'],
                    q60_zone_pull       = q60_out['zone_pull'],
                    m63_action_prior    = self._last_m63_action_prior,
                    m63_recognition     = self._last_m63_recognition,
                )
                m61_loops_run += 1
                m61_sim_history.append(loop_m57_out['sim_values'].copy())

                # Step 3: if the loop converged (action unchanged, confidence
                # improved), stop early — no point continuing
                if (loop_m57_out['action'] == m57_out['action']
                        and loop_thought_out['thought_confidence']
                            >= thought_out['thought_confidence']):
                    m57_out      = loop_m57_out
                    thought_out  = loop_thought_out
                    break

                m57_out     = loop_m57_out
                thought_out = loop_thought_out

        self.t += 1
        self._prev_prev_zone_for_T = self._prev_zone_for_T
        self._prev_zone_for_T      = freq_idx
        self._prev_action_for_T    = int(m57_out['action'])

        # ── M62: CONSISTENCY CHECKER — internal critic ─────────────
        # Runs after M61. Checks whether thought loop iterations agreed.
        # Produces: consistency score, contradiction zones, plan_modifier.
        # plan_modifier adjusts planning_weight retroactively for logging.
        m62_out = self.consistency.step(
            sim_history      = m61_sim_history,
            thought_iters    = m61_thought_iters,
            q60_open_count   = q60_out['open_count'],
            q60_zone_pull    = q60_out['zone_pull'],
            sm_state_label   = sm_out['state_label'],
            planning_active  = m57_out['planning_active'],
        )

        # ── M63: EPISODIC TRACE — before/after memory ──────────────
        # Runs after M62. Opens an episode when M60 creates a question,
        # records actions taken, closes and stores when M60 resolves.
        # Provides recognition signal + action prior for familiar confusion.
        m63_out = self.episodic.step(
            q60_just_created  = q60_out['just_created'],
            q60_just_resolved = q60_out['just_resolved'],
            q60_zone_pull     = q60_out['zone_pull'],
            current_action    = int(m57_out['action']),
            freq_idx          = freq_idx if freq_idx >= 0 else -1,
            sm_state_vec      = sm_out['self_state_vec'],
            sm_state_label    = sm_out['state_label'],
            prediction_error  = raw_error,
        )

        # Update M63 cache — fed into next step's M57 as action_prior.
        self._last_m63_recognition  = float(m63_out['recognition'])
        self._last_m63_action_prior = m63_out['action_prior'].copy()

        # ── M64: LANGUAGE SEED — situation naming ───────────────────
        # Runs after M63. Matches current (state_label × zone) to a named
        # situation tag. Accumulates per-tag action outcome statistics from
        # M63 episodes. Returns tag_action_prior for M57 and a narrative
        # string for M61 thought loops.
        m64_out = self.language.step(
            sm_state_label    = sm_out['state_label'],
            sm_state_vec      = sm_out['self_state_vec'],
            freq_idx          = freq_idx if freq_idx >= 0 else -1,
            current_action    = int(m57_out['action']),
            prediction_error  = raw_error,
            m63_just_closed   = len(m63_out['just_closed']) > 0,
            m63_episode       = m63_out['just_closed'][0] if m63_out['just_closed'] else None,
            q60_just_resolved = q60_out['just_resolved'],
        )

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
            'tpe_accuracy':          _tpe_acc,
            'zone_idx':              l3_out['zone_idx'],
            'zone_confidence':       l3_out['zone_confidence'],
            'zone_probs':            l3_out['zone_probs'],
            'top_zone_pred':         l3_out['top_zone_pred'],
            'zone_pred_conf':        l3_out['zone_pred_conf'],
            'zones_stable':          l3_out['zones_stable'],
            # L4 Position Belief
            'l4_top_node':           l4_out['top_node'],
            'l4_top_prob':           l4_out['top_prob'],
            'l4_belief_entropy':     l4_out['belief_entropy'],
            'l4_confident':          l4_out['confident'],
            # Working Memory (M58)
            'wm_zone_recency':       wm_out['zone_recency'],
            'wm_corridor_boredom':   wm_out['corridor_boredom'],
            'wm_steps_since_reward': wm_out['steps_since_reward'],
            'wm_hunger_norm':        wm_out['steps_since_reward_norm'],
            'wm_epsilon_floor':      wm_out['epsilon_floor'],
            # Global Workspace (GWS) — unified integration
            'gws_arousal':           gws_out['arousal'],
            'gws_arousal_raw':       gws_out['arousal_raw'],
            'gws_valence_tone':      gws_out['valence_tone'],
            'gws_valence_raw':       gws_out['valence_raw'],
            'gws_curiosity_pull':    gws_out['curiosity_pull'],
            'gws_surprise_debt':     gws_out['surprise_debt'],
            'gws_epsilon_boost':     gws_out['epsilon_boost'],
            'gws_coherence':         gws_out['coherence'],
            'gws_tension':           gws_out['tension'],
            'gws_readiness':         gws_out['readiness'],
            'gws_ignited':           gws_out['ignited'],
            'gws_ignition_rate':     gws_out['ignition_rate'],
            # Self-Model (M59) — the brain's mirror
            'sm_state_label':        sm_out['state_label'],
            'sm_state_vec':          sm_out['self_state_vec'],
            'sm_state_changed':      sm_out['state_changed'],
            'sm_steps_in_state':     sm_out['steps_in_state'],
            'sm_past_similarity':    sm_out['past_similarity'],
            'sm_been_here_before':   sm_out['been_here_before'],
            'sm_steps_since_similar':sm_out['steps_since_similar'],
            'sm_urgency':            sm_out['self_state_dict']['urgency'],
            'sm_clarity':            sm_out['self_state_dict']['clarity'],
            'sm_drive':              sm_out['self_state_dict']['drive'],
            'sm_novelty':            sm_out['self_state_dict']['novelty'],
            'sm_stability':          sm_out['self_state_dict']['stability'],
            'sm_confidence':         sm_out['self_state_dict']['confidence'],
            'sm_frustration':        sm_out['self_state_dict']['frustration'],
            'sm_engagement':         sm_out['self_state_dict']['engagement'],
            # Question Memory (M60) — directed curiosity
            'q60_open_count':        q60_out['open_count'],
            'q60_zone_pull':         q60_out['zone_pull'],
            'q60_active_question':   q60_out['active_question'],
            'q60_just_resolved':     q60_out['just_resolved'],
            'q60_total_resolved':    q60_out['total_resolved'],
            'q60_just_created':      q60_out['just_created'],
            # Planner — best simulated next BMU (for diagnostics)
            'best_sim_bmu':          m57_out['best_sim_bmu'],
            # Thought loop (M61) — internal dialogue
            'm61_loops_run':         m61_loops_run,
            'm61_thought_iters':     m61_thought_iters,
            # Consistency Checker (M62) — internal critic
            'm62_consistency':       m62_out['consistency'],
            'm62_consistency_ema':   m62_out['consistency_ema'],
            'm62_contradiction_zones': m62_out['contradiction_zones'],
            'm62_plan_modifier':     m62_out['plan_modifier'],
            'm62_active':            m62_out['active'],
            # Episodic Trace (M63) — before/after memory
            'm63_recognition':       m63_out['recognition'],
            'm63_action_prior':      m63_out['action_prior'],
            'm63_episode_open':      m63_out['episode_open'],
            'm63_episodes_total':    m63_out['episodes_total'],
            'm63_best_zone':         m63_out['best_zone'],
            'm63_just_closed':       m63_out['just_closed'],
            # M63 → M57 prior feedback (did episodic memory bias planning this step?)
            'm63_prior_applied':     m57_out['m63_prior_applied'],
            # Language Seed (M64) — situation naming and composed narrative
            'm64_situation_tag':     m64_out['situation_tag'],
            'm64_tag_action_prior':  m64_out['tag_action_prior'],
            'm64_tag_confidence':    m64_out['tag_confidence'],
            'm64_narrative':         m64_out['narrative'],
            'm64_tags_total':        m64_out['tags_total'],
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