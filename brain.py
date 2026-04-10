"""
BRAIN — Integrated Cognitive Stack with Global Workspace  (v12)
=============================================================

v12: MULTIMODAL SENSING — second sensory modality (texture) added.

New modules:
  M51  (m51_texture.py)         — TextureSense: converts raw texture value
                                   to a 16-dim input vector for M54b.
  M54b (m54b_texture_cortex.py) — TextureCortex: 4×4 SOM that maps texture
                                   inputs to 16 BMU indices (bmu_texture).
  M65  (m65_fusion.py)          — MultimodalFusion: combines sound + texture
                                   observations into a fused likelihood vector
                                   for L4's Bayesian belief update.

HOW IT PLUGS IN
---------------
Two new parameters on Brain.step():
  texture_val   : float  — raw texture value from world (world.current_texture
                           or info['texture'] from world.step()). Default 0.5
                           so old harnesses work unchanged (no texture signal).
  texture_active: bool   — whether to use texture this step. Default True
                           when texture_val is provided, False otherwise.
                           Allows graceful degradation if sensor missing.

New output keys (all prefixed m51_, m54b_, m65_):
  m51_texture_val       — raw texture value received
  m51_texture_noisy     — noisy version M54b saw
  m54b_bmu_texture      — which texture neuron fired (0-15)
  m54b_qe_norm          — texture cortex perceptual surprise
  m65_fused_likelihood  — per-node fused observation (n_nodes,)
  m65_sound_weight      — effective sound weight in fusion
  m65_texture_weight    — effective texture weight in fusion
  m65_texture_conf      — how discriminating the texture signal was
  m65_fusion_active     — whether fusion has passed warmup

BACKWARD COMPATIBILITY
----------------------
All existing output keys are unchanged.
Brain.step() signature extended with optional texture_val=0.5 and
texture_active=True — existing harnesses need no changes.

If Brain is constructed without multimodal=True (default), M51/M54b/M65
are not instantiated and the new output keys are filled with zero stubs.
This keeps the module set identical to v11 for non-texture worlds.

Call this with multimodal=True for World5:
  brain = Brain(seed=42, node_fi=node_fi_dict, multimodal=True,
                node_tex=node_tex_dict)

CALL ORDER (per step, new steps shown with ★)
---------------------------------------------
1. pred.predict()
2. cortex.step()                      ← M54 (sound)
★2b. texture_sense.step()             ← M51 (texture input prep)
★2c. texture_cortex.step()            ← M54b (texture SOM)
★2d. fusion.step()                    ← M65 (sound × texture → L4)
3. memory.step()
4. memory.recall()
5. pred.step()
5b. L3 concept layer
5c. L4.step() with fused_likelihood   ← L4 now uses both signals ★
6. feedback deltas
6b. Valence
7. Attention
8. Thought
8b. WorkingMemory
8c. GlobalWorkspace
8d. SelfModel
8e. QuestionMemory
9. M56 action
10. M57 planner + M61 + M62 + M63 + M64
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

# ── NEW v12 imports ──────────────────────────────────────────
from m51_texture import TextureSense
from m54b_texture_cortex import TextureCortex
from m65_fusion import MultimodalFusion


# ═══════════════════════════════════════════════════════════════
# FEEDBACK PARAMETERS (unchanged from v11)
# ═══════════════════════════════════════════════════════════════

FEEDBACK_EMA_ALPHA = 0.10
FEEDBACK_EMA_INIT  = 1.0


# ═══════════════════════════════════════════════════════════════
# BRAIN
# ═══════════════════════════════════════════════════════════════

class Brain:
    """
    Integrated M54 + M55 + L2 + Attention cognitive stack.

    v12: Multimodal sensing added (M51 + M54b + M65).

    Parameters
    ----------
    seed       : int  — random seed
    node_fi    : dict — node name → freq_index (for L4)
    multimodal : bool — enable M51/M54b/M65 (default False for backward compat)
    node_tex   : dict — node name → texture value (required if multimodal=True)
    """

    def __init__(self, seed: int = 42, node_fi: dict = None,
                 multimodal: bool = False, node_tex: dict = None):
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

        # ── L4: position belief ───────────────────────────────
        if node_fi is not None:
            self.l4 = PositionBelief(node_fi=node_fi)
            from collections import Counter as _Counter
            fi_counts = _Counter(node_fi.values())
            import m56_action as _m56
            _m56.L4_Q_N_UNIQUE_NODES = {
                n for n, fi in node_fi.items() if fi_counts[fi] == 1
            }
            _m56.L4_Q_N_ALIASED_NODES = {
                n for n, fi in node_fi.items() if fi_counts[fi] > 1
            }
            if len(node_fi) > 12:
                import l4_position as _l4
                _l4.L4_BELIEF_DECAY = _l4.L4_BELIEF_DECAY_LARGE_WORLD
                _l4.L4_CTM_WARMUP   = _l4.L4_CTM_WARMUP_LARGE_WORLD
        else:
            self.l4 = None

        # ── NEW v12: Multimodal modules ───────────────────────
        self._multimodal = multimodal
        if multimodal:
            if node_tex is None:
                raise ValueError(
                    "Brain(multimodal=True) requires node_tex dict. "
                    "Pass node_tex=world5.NODE_TEXTURES."
                )
            if node_fi is None:
                raise ValueError(
                    "Brain(multimodal=True) requires node_fi dict."
                )
            self.texture_sense  = TextureSense(seed=seed)
            self.texture_cortex = TextureCortex(seed=seed)
            self.fusion         = MultimodalFusion(
                node_fi   = node_fi,
                node_tex  = node_tex,
                n_tex_bins = 8,
                seed       = seed,
            )
        else:
            self.texture_sense  = None
            self.texture_cortex = None
            self.fusion         = None

        # ── freq_bmu_counters ─────────────────────────────────
        from collections import Counter
        self._freq_bmu_counters = [Counter() for _ in range(8)]
        self._l3_zone_interval  = 2000

        # ── Feedback state ────────────────────────────────────
        self._error_ema     = float(FEEDBACK_EMA_INIT)
        self._curiosity_ema = float(FEEDBACK_EMA_INIT)
        self._last_surprise_signal  = 0.0
        self._last_curiosity_delta  = 0.0
        self._last_rpe_positive     = 0.0
        self._last_familiarity      = 0.0
        self._last_prediction_error = 0.0
        self._last_curiosity        = 0.0

        self.t = 0
        self._prev_zone_for_T      = -1
        self._prev_action_for_T    = -1
        self._prev_prev_zone_for_T = -1

        _n_act = self.action._n_actions
        self._last_m63_recognition  = 0.0
        self._last_m63_action_prior = np.ones(_n_act, dtype=np.float64) / _n_act

        # ── Food trajectory store (for hippocampal replay) ────
        # Snapshots of M56's replay buffer at each food event.
        # Each entry: {'trajectory': [(prev_bmu, curr_bmu, action, fi, l4_node, explore),...],
        #              'food_node': str, 'reward': float}
        from collections import deque as _dq
        self._food_trajectories = _dq(maxlen=10)

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             decoded_freq:   float,
             stability_w:    float,
             novelty_flag:   float,
             plv_vector:     np.ndarray,
             reward:         float = 0.0,
             freq_idx:       int   = -1,
             world_moved:    bool  = True,
             texture_val:    float = 0.5,    # NEW v12 — texture from world
             texture_active: bool  = False,  # NEW v12 — False = no texture signal
             ) -> dict:
        """
        One full cognitive step. All v11 parameters unchanged.

        New parameters
        --------------
        texture_val    : float — raw texture value from world.current_texture
                                 or info['texture'] from world.step().
                                 Ignored if texture_active=False.
        texture_active : bool  — True when a real texture reading is available.
                                 Set to False in open-loop or non-texture steps.
        """
        # ── 0. Zone transition update ─────────────────────────
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

        # ── 2. Cortex fires (M54 — sound) ────────────────────
        cortex_out = self.cortex.step(
            decoded_freq     = decoded_freq,
            stability_w      = stability_w,
            novelty_flag     = novelty_flag,
            plv_vector       = plv_vector,
            prediction_error = self._last_surprise_signal,
            familiarity      = self._last_familiarity,
        )
        bmu_idx = cortex_out['bmu_idx']
        qe_norm = cortex_out['qe_norm']

        # ── 2b-d. Texture pipeline (M51 → M54b → M65) — NEW v12 ──
        if self._multimodal and texture_active:
            m51_out  = self.texture_sense.step(texture_val)
            m54b_out = self.texture_cortex.step(
                texture_vec = m51_out['texture_vec'],
                texture_val = float(texture_val),
            )
            m65_out = self.fusion.step(
                curr_fi            = freq_idx if freq_idx >= 0 else -1,
                texture_likelihood = m54b_out['texture_likelihood'],
                tex_bin            = m54b_out['tex_bin'],
                sound_qe_norm      = qe_norm,
                tex_qe_norm        = m54b_out['qe_norm'],
            )
        else:
            # Stubs — no texture signal
            m51_out  = {'texture_val': texture_val, 'texture_noisy': texture_val}
            m54b_out = {'bmu_texture': 0, 'qe_norm': 0.0}
            m65_out  = {
                'fused_likelihood':  None,
                'sound_weight':      1.0,
                'texture_weight':    0.0,
                'texture_confidence': 0.0,
                'fusion_active':     False,
                'n_tex_matching':    self.fusion.n_nodes if self.fusion else 0,
            }

        # ── 3. Memory update (M55) ────────────────────────────
        mem_out = self.memory.step(
            bmu_idx      = bmu_idx,
            qe_norm      = qe_norm,
            curiosity    = self._last_curiosity_delta,
            rpe_positive = self._last_rpe_positive,
        )

        # ── 4. Recall — familiarity ───────────────────────────
        recall_out  = self.memory.recall(bmu_idx)
        familiarity = recall_out['familiarity']
        self._last_familiarity = familiarity

        # ── 5. L2 learns ──────────────────────────────────────
        l2_out = self.pred.step(
            bmu_idx         = bmu_idx,
            qe_norm         = qe_norm,
            familiarity     = familiarity,
            prediction_bias = self.thought._last_prediction_bias,
            last_action     = self._prev_action_for_T,
            world_moved     = world_moved,
        )
        raw_error     = l2_out['prediction_error']
        raw_curiosity = l2_out['curiosity']

        # ── 5b. L3 concept layer ──────────────────────────────
        if freq_idx >= 0:
            self._freq_bmu_counters[freq_idx][bmu_idx] += 1
            if (self.t >= 5000 and self.t % self._l3_zone_interval == 0):
                self.l3.assign_zones_from_counters(self._freq_bmu_counters)

        l3_scores = self.pred._P[bmu_idx] if hasattr(self.pred, '_P') else None
        l3_out = self.l3.step(
            bmu_idx   = bmu_idx,
            l2_scores = l3_scores,
            freq_idx  = freq_idx,
        )
        visit_zone = freq_idx if freq_idx >= 0 else l3_out['zone_idx']
        if visit_zone >= 0:
            self.l3.update_zone_visit(visit_zone)

        if reward != 0.0:
            zone_for_reward = freq_idx if freq_idx >= 0 else l3_out['zone_idx']
            self.l3.update_zone_reward(zone_for_reward, reward)

        # ── 5c. L4 Position Belief — with optional fusion ─────
        # v12: if fusion is active, apply fused_likelihood as an
        # additional observation update BEFORE L4's own sound update.
        # L4's internal sound_likelihood handles the sound part;
        # we apply the texture component here as a belief multiplier.
        if self.l4 is not None:
            l4_out = self.l4.step(
                curr_fi     = freq_idx if freq_idx >= 0 else -1,
                action      = self._prev_action_for_T,
                world_moved = world_moved,
            )
            # ── NEW v12: Apply texture likelihood to L4 belief ──
            # After L4's standard sound update, multiply belief by
            # the texture component from M65. This is "late fusion":
            # L4 does its sound update first, then texture refines it.
            # We access L4's internal belief and update it directly.
            if (self._multimodal and texture_active
                    and m65_out['fusion_active']
                    and m65_out['fused_likelihood'] is not None):
                fused = m65_out['fused_likelihood']
                # Texture-only component = fused / sound component
                # (to avoid double-counting sound which L4 already applied)
                # Instead: multiply belief by tex_obs (node tex bin match)
                tex_w = m65_out['texture_weight']
                if tex_w > 0.01:
                    # Build texture-only likelihood per node
                    tex_bin = m54b_out['tex_bin']
                    tex_like = np.array([
                        1.0 if self.fusion._node_tex_bin[i] == tex_bin
                        else 0.10
                        for i in range(self.fusion.n_nodes)
                    ], dtype=np.float64)
                    # Apply with texture weight as exponent (soft update)
                    tex_update = tex_like ** tex_w
                    self.l4._belief *= tex_update
                    s = self.l4._belief.sum()
                    if s > 1e-12:
                        self.l4._belief /= s
                    # Refresh L4 output with updated belief
                    l4_out = self.l4._make_output()
        else:
            l4_out = {
                'top_node':       None,
                'top_prob':       0.0,
                'belief_entropy': 1.0,
                'confident':      False,
                'belief_vector':  None,
            }

        # ── 6. Feedback deltas ────────────────────────────────
        surprise_signal = float(np.clip(raw_error     - self._error_ema,     0.0, 1.0))
        curiosity_delta = float(np.clip(raw_curiosity  - self._curiosity_ema, 0.0, 1.0))
        self._error_ema     = ((1.0 - FEEDBACK_EMA_ALPHA) * self._error_ema
                               + FEEDBACK_EMA_ALPHA * raw_error)
        self._curiosity_ema = ((1.0 - FEEDBACK_EMA_ALPHA) * self._curiosity_ema
                               + FEEDBACK_EMA_ALPHA * raw_curiosity)
        self._last_surprise_signal  = surprise_signal
        self._last_curiosity_delta  = curiosity_delta
        self._last_prediction_error = raw_error
        self._last_curiosity        = raw_curiosity

        # ── 6b. Valence ───────────────────────────────────────
        v1_out = self.valence.step(
            prediction_error = raw_error,
            reward           = float(reward),
            familiarity      = familiarity,
        )
        self._last_rpe_positive = v1_out['pos_rpe']

        # ── 7. Attention ──────────────────────────────────────
        attn_out = self.attention.step(
            bmu_idx                  = bmu_idx,
            qe_norm                  = qe_norm,
            familiarity              = familiarity,
            surprise_signal          = surprise_signal,
            curiosity_delta          = curiosity_delta,
            thought_confidence_delta = self.thought._last_confidence_delta,
        )

        # ── 8. Thought ────────────────────────────────────────
        thought_out = self.thought.step(
            attended_bmu = attn_out['attended_bmu'],
            bmu_idx      = bmu_idx,
            pred         = self.pred,
            salience     = attn_out['salience'],
            memory       = self.memory,
        )

        # ── 8b. Working Memory ────────────────────────────────
        wm_out = self.wm.step(
            freq_idx = freq_idx if freq_idx >= 0 else -1,
            action   = self._prev_action_for_T,
            reward   = float(reward),
        )

        # ── 8c. Global Workspace ─────────────────────────────
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

        # ── 8d. Self-Model ────────────────────────────────────
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

        # ── 8e. Question Memory ───────────────────────────────
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

        # ── 9. Action (M56) ───────────────────────────────────
        m56_out = self.action.step(
            bmu_idx            = bmu_idx,
            rpe                = v1_out['rpe'],
            focus_entropy      = thought_out['focus_entropy'],
            thought_confidence = thought_out['thought_confidence'],
            freq_idx           = freq_idx,
            world_moved        = world_moved,
            l4_top_node        = l4_out['top_node'],
            l4_top_prob        = l4_out['top_prob'],
            epsilon_floor      = max(wm_out['epsilon_floor'],
                                     gws_out['epsilon_boost']),
            node_fi_override   = getattr(self, '_node_fi_override_hint', None),
            raw_reward         = float(reward),
        )

        # ── 10. Planner (M57) ─────────────────────────────────
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

        # ── M61: Thought loop ─────────────────────────────────
        M61_MAX_LOOPS         = 2
        M61_SIM_WEIGHT_BASE   = 0.40
        M61_DELIBERATE_LABELS = {'confused', 'curious', 'stuck', 'hunting', 'alert'}

        m61_loops_run     = 0
        m61_thought_iters = []
        m61_sim_history   = []

        if (q60_out['open_count'] > 0
                and sm_out['state_label'] in M61_DELIBERATE_LABELS
                and m57_out['sim_depth'] > 0
                and self.planner.t > 1):

            for _loop in range(M61_MAX_LOOPS):
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

                if (loop_m57_out['action'] == m57_out['action']
                        and loop_thought_out['thought_confidence']
                            >= thought_out['thought_confidence']):
                    m57_out     = loop_m57_out
                    thought_out = loop_thought_out
                    break

                m57_out     = loop_m57_out
                thought_out = loop_thought_out

        self.t += 1
        self._prev_prev_zone_for_T = self._prev_zone_for_T
        self._prev_zone_for_T      = freq_idx
        self._prev_action_for_T    = int(m57_out['action'])

        # ── M62: Consistency Checker ──────────────────────────
        m62_out = self.consistency.step(
            sim_history      = m61_sim_history,
            thought_iters    = m61_thought_iters,
            q60_open_count   = q60_out['open_count'],
            q60_zone_pull    = q60_out['zone_pull'],
            sm_state_label   = sm_out['state_label'],
            planning_active  = m57_out['planning_active'],
        )

        # ── M63: Episodic Trace ───────────────────────────────
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
        self._last_m63_recognition  = float(m63_out['recognition'])
        self._last_m63_action_prior = m63_out['action_prior'].copy()

        # ── M64: Language Seed ────────────────────────────────
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

        # ── Return unified output ─────────────────────────────
        return {
            # M54 (sound cortex)
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
            # L2
            'prediction_error': raw_error,
            'correct':          l2_out['correct'],
            'predicted_bmu':    pred_out['predicted_bmu'],
            'confidence':       pred_out['confidence'],
            'curiosity':        raw_curiosity,
            # Feedback
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
            # Valence
            'rpe':                   v1_out['rpe'],
            'pos_rpe':               v1_out['pos_rpe'],
            'neg_rpe':               v1_out['neg_rpe'],
            'reward_ema':            v1_out['reward_ema'],
            'total_reward':          v1_out['total_reward'],
            'intrinsic_reward':      v1_out['intrinsic_reward'],
            'novelty_bonus':         v1_out['novelty_bonus'],
            # Action (M56/M57)
            'action':                m57_out['action'],
            'q_values':              m56_out['q_values'],
            'q_max':                 m56_out['q_max'],
            'action_epsilon':        m56_out['epsilon'],
            'action_explore':        m56_out['explore'],
            'q_mean':                m56_out['q_mean'],
            'q_nonzero_frac':        m56_out['q_nonzero_frac'],
            'habit_action':          m56_out['action'],
            # Planner (M57)
            'planned_action':        m57_out['planned_action'],
            'planning_weight':       m57_out['planning_weight'],
            'planning_active':       m57_out['planning_active'],
            'sim_values':            m57_out['sim_values'],
            'sim_depth':             m57_out['sim_depth'],
            'plan_vs_habit_delta':   m57_out['plan_vs_habit_delta'],
            # L3
            'tpe_accuracy':          _tpe_acc,
            'zone_idx':              l3_out['zone_idx'],
            'zone_confidence':       l3_out['zone_confidence'],
            'zone_probs':            l3_out['zone_probs'],
            'top_zone_pred':         l3_out['top_zone_pred'],
            'zone_pred_conf':        l3_out['zone_pred_conf'],
            'zones_stable':          l3_out['zones_stable'],
            # L4
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
            # Global Workspace (GWS)
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
            # Self-Model (M59)
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
            # Question Memory (M60)
            'q60_open_count':        q60_out['open_count'],
            'q60_zone_pull':         q60_out['zone_pull'],
            'q60_active_question':   q60_out['active_question'],
            'q60_just_resolved':     q60_out['just_resolved'],
            'q60_total_resolved':    q60_out['total_resolved'],
            'q60_just_created':      q60_out['just_created'],
            # M57 best sim
            'best_sim_bmu':          m57_out['best_sim_bmu'],
            # M61 thought loop
            'm61_loops_run':         m61_loops_run,
            'm61_thought_iters':     m61_thought_iters,
            # M62 consistency
            'm62_consistency':       m62_out['consistency'],
            'm62_consistency_ema':   m62_out['consistency_ema'],
            'm62_contradiction_zones': m62_out['contradiction_zones'],
            'm62_plan_modifier':     m62_out['plan_modifier'],
            'm62_active':            m62_out['active'],
            # M63 episodic
            'm63_recognition':       m63_out['recognition'],
            'm63_action_prior':      m63_out['action_prior'],
            'm63_episode_open':      m63_out['episode_open'],
            'm63_episodes_total':    m63_out['episodes_total'],
            'm63_best_zone':         m63_out['best_zone'],
            'm63_just_closed':       m63_out['just_closed'],
            'm63_prior_applied':     m57_out['m63_prior_applied'],
            # M64 language
            'm64_situation_tag':     m64_out['situation_tag'],
            'm64_tag_action_prior':  m64_out['tag_action_prior'],
            'm64_tag_confidence':    m64_out['tag_confidence'],
            'm64_narrative':         m64_out['narrative'],
            'm64_tags_total':        m64_out['tags_total'],
            # ── NEW v12: Multimodal keys ──────────────────────
            'm51_texture_val':       m51_out['texture_val'],
            'm51_texture_noisy':     m51_out['texture_noisy'],
            'm54b_bmu_texture':      m54b_out['bmu_texture'],
            'm54b_qe_norm':          m54b_out['qe_norm'],
            'm65_fused_likelihood':  m65_out['fused_likelihood'],
            'm65_sound_weight':      m65_out['sound_weight'],
            'm65_texture_weight':    m65_out['texture_weight'],
            'm65_texture_conf':      m65_out['texture_confidence'],
            'm65_fusion_active':     m65_out['fusion_active'],
        }

    # ── Convenience accessors (unchanged from v11) ───────────

    def reset_feedback(self):
        self._error_ema             = float(FEEDBACK_EMA_INIT)
        self._curiosity_ema         = float(FEEDBACK_EMA_INIT)
        self._last_surprise_signal  = 0.0
        self._last_curiosity_delta  = 0.0
        self._last_rpe_positive     = 0.0
        self._last_familiarity      = 0.0
        self._last_prediction_error = 0.0
        self._last_curiosity        = 0.0

    def get_feedback_state(self) -> dict:
        return {
            'prediction_error': self._last_prediction_error,
            'curiosity':        self._last_curiosity,
            'surprise_signal':  self._last_surprise_signal,
            'curiosity_delta':  self._last_curiosity_delta,
            'error_ema':        self._error_ema,
            'curiosity_ema':    self._curiosity_ema,
        }

    def summary(self):
        print(f"\n  Brain v12 — step {self.t}  multimodal={self._multimodal}")
        print(f"  Feedback:")
        print(f"    error_ema:     {self._error_ema:.4f}")
        print(f"    curiosity_ema: {self._curiosity_ema:.4f}")
        if self._multimodal and self.texture_sense:
            self.texture_sense.summary()
            self.texture_cortex.summary()
            self.fusion.summary()
        self.cortex.get_surprise_stats()
        self.memory.summary()
        self.pred.summary()
        self.attention.summary()
        self.thought.summary()
        self.action.summary()

    # ══════════════════════════════════════════════════════════════
    # HIPPOCAMPAL REPLAY — food trajectory replay
    # ══════════════════════════════════════════════════════════════

    def record_food_trajectory(self, food_node: str, reward: float):
        """
        Snapshot the M56 replay buffer when food/door reward is collected.

        Call from the harness immediately after a food event. The replay
        buffer contains the actual (prev_bmu, curr_bmu, action, freq_idx,
        l4_node, was_explore) tuples from real navigation — this is the
        ground-truth path that led to food.
        """
        trajectory = list(self.action._replay_buffer)
        if trajectory:
            self._food_trajectories.append({
                'trajectory': trajectory,
                'food_node':  food_node,
                'reward':     float(reward),
            })

    def idle_step(self) -> int:
        """
        Replay stored food trajectories forward through Q_n.

        This is the hippocampal replay that solves temporal credit
        assignment for button→door sequences. It replays the actual
        navigation path (from M56's replay buffer snapshot) that led
        to food, applying geometric-decay credit to Q_n for every
        node visited along the way.

        Unlike M63 episode replay which:
          - Contained confusion arcs (not food paths)
          - Credited Q/Q_f at a single zone (too coarse for 64 nodes)
          - Used self-transitions (replay_e[bmu,bmu,...] — never queried)

        This version:
          - Replays actual food trajectories with real (node, action) pairs
          - Credits Q_n directly (the table that drives aliased navigation)
          - Uses forward replay so eligibility trace accumulates correctly

        Returns
        -------
        int — total replay steps executed across all trajectories
        """
        if not self._food_trajectories:
            return 0

        from m56_action import REPLAY_GAMMA, REPLAY_ALPHA, Q_MIN, Q_MAX

        total_replayed = 0

        # Replay last 3 food trajectories
        recent = list(self._food_trajectories)[-3:]
        for ft in recent:
            trajectory = ft['trajectory']
            reward     = ft['reward']
            if not trajectory or reward <= 0.0:
                continue

            # Forward replay with geometric discount from LAST entry.
            # Entry[-1] is the step that arrived at food. Entry[-2] is
            # one step before, etc. Discount increases as we go back.
            n = len(trajectory)
            for i, entry in enumerate(trajectory):
                # Distance from food: last entry = 0, first = n-1
                dist_from_food = n - 1 - i
                discount = REPLAY_GAMMA ** dist_from_food
                q_delta  = REPLAY_ALPHA * reward * discount

                prev_bmu = entry[0]
                curr_bmu = entry[1]
                action   = entry[2]
                fi       = entry[3] if len(entry) > 3 else -1
                l4_node  = entry[4] if len(entry) > 4 else None
                was_expl = entry[5] if len(entry) > 5 else False
                explore_discount = 0.30 if was_expl else 1.0

                # Credit Q_n (node-keyed — the table that matters)
                # Gate on L4 confidence: only credit Q_n when L4 was
                # confident about WHERE the brain was at that step.
                # Without this gate, ~24% of steps credit the wrong node
                # (L4 accuracy = 76%), creating contradictory Q_n entries.
                l4_conf_in_buf = entry[6] if len(entry) > 6 else 1.0
                if l4_node is not None and l4_conf_in_buf >= 0.70:
                    if l4_node not in self.action._Q_n:
                        self.action._Q_n[l4_node] = np.zeros(
                            self.action._n_actions, dtype=np.float32)
                    self.action._Q_n[l4_node][action] = float(np.clip(
                        self.action._Q_n[l4_node][action]
                        + q_delta * 0.50 * explore_discount,
                        Q_MIN, Q_MAX
                    ))
                    self.action._Q_n_count[l4_node] = (
                        self.action._Q_n_count.get(l4_node, 0) + 1)

                # Credit Q_f (zone-keyed — secondary)
                if 0 <= fi < self.action._Q_f.shape[0]:
                    self.action._Q_f[fi, action] = float(np.clip(
                        self.action._Q_f[fi, action]
                        + q_delta * 0.25 * explore_discount,
                        Q_MIN, Q_MAX
                    ))

                total_replayed += 1

        return total_replayed