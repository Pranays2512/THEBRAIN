"""
M56: ACTION LAYER — ACTOR-CRITIC
=================================

Replaces Q-learning (3 Q-tables, threshold zoo, override rows) with
Actor-Critic: a policy (actor) updated by V1's dopamine signal (critic).

BIOLOGY
-------
Basal ganglia = actor  (direct/indirect pathway competition selects action)
Dopamine (V1/Valence) = critic  (TD error δ modulates synaptic plasticity)
Hippocampus (idle replay) = consolidates food-path credit during rest

The actor learns a preference π(state, action) for each action at each
cortical state. V1 already computes the TD error (rpe). M56 uses it directly
to update π via dopamine-gated eligibility traces — biologically, this is
LTP/LTD in the striatum modulated by dopamine burst/dip.

WHY ACTOR-CRITIC OVER Q-LEARNING
---------------------------------
Old Q-learning required three separate tables (Q_bmu, Q_f, Q_n) because
aliased nodes contaminated each other through the shared frequency row.
The fix was a hard L4 confidence gate (0.55), creating a threshold zoo
that broke whenever the gate was inconsistently applied.

Actor-Critic fixes this structurally:

  Soft L4 integration: instead of "if L4 > 0.55 use Q_n else Q_f",
    π_eff = (1 - blend) × π_bmu + blend × π_node[top_node]
    blend = clip(l4_prob × trust_factor, 0, 0.8)
  At l4_prob=0.25: blend=0.15  (small influence, correct direction)
  At l4_prob=0.70: blend=0.50  (node dominates, confident)
  No hard cutoff. Wrong nodes receive near-zero weight automatically.

  Per-node update weighted by L4 probability:
    π_node[top_node][a] += l4_prob × ALPHA_NODE × rpe
  At l4_prob=0.25: update × 0.25  (weak, uncertain — doesn't corrupt)
  At l4_prob=0.70: update × 0.70  (strong, confident — dominates)

  Temperature replaces epsilon-greedy:
    action = sample(softmax(π_eff / temperature))
  High temperature = near-uniform = exploration
  Low temperature = peaked = exploitation
  Driven by focus_entropy — diffuse attention → explore naturally.

WHAT WAS REMOVED
-----------------
  Q_f  — frequency-indexed table. Contamination source; gone entirely.
  Q_n  — node-indexed table with hard gates. Replaced by π_node (soft).
  Override rows — ad-hoc aliasing fix. Replaced by soft L4 weighting.
  Threshold zoo — 0.2 / 0.55 / 0.65 / 0.40. One parameter: NODE_L4_MIN.
  Epsilon-greedy — replaced by softmax temperature.
  boost_exploration — temperature naturally rises with curiosity/entropy.

WHAT WAS KEPT
--------------
  Eligibility traces  — dopamine-gated LTP/LTD, biologically grounded.
  Hippocampal replay  — back-propagate food credit through recent path.
  Harness interface   — step(), replay_on_reward(), set_node_fi_map(),
                        decay_node_qn(), boost_exploration() all present.
"""

import numpy as np
from collections import deque
import m56_fast as _fast

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

N_NEURONS = 64
N_ACTIONS = 4

# ── Actor learning ────────────────────────────────────────────
ALPHA_ACTOR = 0.04    # policy gradient step size
                      # Critic is V1 (Valence) — M56 is actor only.
                      # Smaller than old ETA_Q (0.05) to compensate for
                      # more frequent updates (every step, not just food/wall).

ALPHA_NODE  = 0.06    # per-node policy update rate (L4-weighted)
                      # Slightly higher — sparse visits need faster learning.

# ── Eligibility trace ─────────────────────────────────────────
TRACE_DECAY = 0.20    # e *= (1 - TRACE_DECAY) each step → τ ≈ 5 steps
                      # Identical to old Q-learning trace — same credit window.

# ── Policy bounds ─────────────────────────────────────────────
# Actor preferences are logits. Kept tight (1.5) so softmax stays
# discriminative. At PI_MAX=3.0, saturation ([3,3,3,3]) → all actions
# 25% each — completely random despite "learning". At 1.5, the brain
# must earn directional preference rather than flooding everything.
PI_MAX =  1.5
PI_MIN = -1.5
# Backward-compat aliases — brain.py idle_step() imports these
Q_MAX = PI_MAX
Q_MIN = PI_MIN

# ── Softmax temperature ───────────────────────────────────────
# Replaces epsilon-greedy. High = uniform (explore). Low = greedy (exploit).
TEMP_INIT  = 2.5      # start: lots of exploration
TEMP_MIN   = 0.40     # floor: always some stochasticity (biological noise)
TEMP_DECAY = 1.0 - 2e-5   # decay → halves in ~35k steps, reaches floor ~130k steps.
                           # At 8e-6 (prev): floor hit at ~300k — 300k steps of random
                           # exploration means every direction gets replayed, saturating
                           # policy to ±1.5 in all directions. At 2e-5: floor by 130k,
                           # so the brain commits to a policy while it can still be refined.
TEMP_WARMUP_STEPS = 25_000 # keep TEMP_INIT during warmup (equivalent to old warmup)

# How much focus_entropy scales temperature:
#   temp_eff = base_temp × (TEMP_ENTROPY_MIN + (1-TEMP_ENTROPY_MIN) × entropy)
# At entropy=1.0 (diffuse, confused): temp_eff = base_temp      (full exploration)
# At entropy=0.0 (focused, confident): temp_eff = 0.4 × base_temp (exploit)
TEMP_ENTROPY_MIN = 0.40

# ── Per-node policy ───────────────────────────────────────────
NODE_L4_MIN   = 0.30  # minimum L4 probability to accumulate per-node stats
                      # Low — we accumulate softly, weighted by probability.
                      # No hard gate. Wrong nodes get tiny updates and don't corrupt.

NODE_MIN_VISITS = 10  # visits before per-node policy blends into action selection.
                      # Below this, BMU policy is used unchanged — cold π_node is zeros.

# ── Replay ────────────────────────────────────────────────────
REPLAY_BUFFER_LEN = 100    # recent transitions for hippocampal replay
                           # Doubled from 50: button→door paths require pressing a button
                           # then reaching a door (can be 15-40 steps). Credit must reach
                           # the decision point that started the path — 50 was too short.
REPLAY_GAMMA      = 0.92   # credit discount per step backward (raised from 0.90)
                           # At len=100, 0.90^100=2.7e-5 (essentially zero). 0.92^100=2.5e-3
                           # gives meaningful credit at 100 steps back.
REPLAY_ALPHA      = 0.20   # base replay amplitude
                           # 0.30 saturated at PI_MAX=1.5 (all directions +1.5).
                           # 0.12 was too weak — policy values near zero (+0.05 best).
                           # 0.20 is the middle: enough signal without flooding.

# ── RPE dead zone (FIX: asymmetric drain) ─────────────────────
# Skip actor updates when |rpe| < RPE_DEAD_ZONE.
# Rationale: when V1 perfectly predicts reward, RPE → 0. Positive
# reinforcement stops. But forced exploration (TEMP_MIN=0.40) generates
# many tiny negative RPE events that erode all logits toward PI_MIN
# over 500k steps. Filtering out this noise prevents the death spiral.
RPE_DEAD_ZONE = 0.015

# ── Positive RPE amplification (FIX: asymmetric drain) ────────
# Dopamine burst (reward surprise) has stronger synaptic effect than
# dopamine dip (disappointment) in biology. Scale positive RPE by this
# factor to counteract the natural negative bias from exploration noise.
POS_RPE_SCALE = 1.5

# ── L2 weight decay (FIX: dead logit recovery) ───────────────
# Slowly pull all logits toward 0.0 so ancient punishments fade.
# Biologically: homeostatic synaptic scaling — inactive synapses
# decay toward baseline. At 0.002 per 500 steps, logits at -1.5
# recover to -1.0 in ~100k steps. Active reinforcement far outpaces
# this decay, so well-learned policies are barely affected.
WEIGHT_DECAY          = 0.002   # fraction pulled toward 0 per interval
WEIGHT_DECAY_INTERVAL = 500     # steps between decay applications

# ── Module-level globals (harness compat) ─────────────────────
# Used by test harnesses for policy audits. Set by set_node_fi_map().
L4_Q_N_ALIASED_NODES: set = set()
L4_Q_N_UNIQUE_NODES:  set = set()
L4_Q_N_WARMUP: int = NODE_MIN_VISITS   # alias for old harness code


# ═══════════════════════════════════════════════════════════════
# ACTOR-CRITIC LAYER
# ═══════════════════════════════════════════════════════════════

class ActionLayer:
    """
    Actor-Critic action selection for the Brain stack.

    One policy table (actor) + V1's rpe (critic).
    Soft L4 integration — no hard confidence gates.
    Softmax temperature — no epsilon-greedy.
    """

    def __init__(self, n_actions: int = N_ACTIONS, seed: int = 0):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)

        # ── Actor: BMU-level policy ───────────────────────────
        # π_bmu[bmu_idx, action]: preference for each action at each cortical state.
        # Updated by: π_bmu += ALPHA_ACTOR × rpe × e_trace
        # Biologically: synaptic weights in striatum (direct pathway).
        self._pi_bmu = np.zeros((N_NEURONS, n_actions), dtype=np.float32)

        # ── Actor: per-node policy (L4-gated, soft) ───────────
        # π_node[node_name] = float32[n_actions]
        # Updated weighted by L4 probability — uncertain updates contribute less.
        # Blended into action selection proportionally to L4 confidence and trust.
        self._pi_node   = {}   # dict[str, np.ndarray(n_actions)]
        self._node_visits = {} # dict[str, int] — visit count for trust scaling

        # ── Eligibility trace ─────────────────────────────────
        # e[bmu_idx, action]: recency-weighted responsibility.
        # Decays each step; stamped at 1.0 on real transitions.
        self._e = np.zeros((N_NEURONS, n_actions), dtype=np.float32)

        # ── State tracking ────────────────────────────────────
        self._prev_bmu    = 0
        self._prev_action = 0
        self._prev_l4_node = None   # L4 top node from PREVIOUS step (for update)
        self._prev_l4_prob = 0.0    # L4 probability from previous step
        self._prev_valid  = False   # False on first step
        self._prev_probs  = np.full(n_actions, 1.0/n_actions, dtype=np.float32)

        # ── Temperature (replaces epsilon) ────────────────────
        self._base_temp = float(TEMP_INIT)
        self._current_temp = float(TEMP_INIT)
        self._boost_steps  = 0    # countdown for post-food-move exploration boost

        # ── Replay buffer ─────────────────────────────────────
        # Format: (prev_bmu, curr_bmu, action, fi, l4_node, was_explore, l4_prob)
        # Matches brain.py idle_step() expectations (7-element tuples).
        self._replay_buffer = deque(maxlen=REPLAY_BUFFER_LEN)
        self._current_was_explore = False

        # ── Node info (stored by harness) ─────────────────────
        self._node_fi = {}     # node → freq_idx (set by set_node_fi_map)
        self._current_freq_idx = -1   # freq_idx at current step (for replay)

        # ── Backward-compat aliases for brain.py idle_step() ──
        # idle_step() writes to _Q_n[node][action] which is exactly
        # what we want — it updates the per-node actor policy.
        self._Q_n       = self._pi_node    # same object: dict[str, np.array]
        self._Q_n_count = self._node_visits # same object: dict[str, int]
        # _Q_f is written by idle_step() but unused in Actor-Critic — dummy array
        self._Q_f = np.zeros((12, n_actions), dtype=np.float32)

        # ── Diagnostics ───────────────────────────────────────
        self.t = 0
        self._n_explorations   = 0
        self._n_exploitations  = 0

    # ── Harness setup ─────────────────────────────────────────

    def set_node_fi_map(self, node_fi: dict) -> None:
        """Store node→freq_idx mapping. Sets module-level aliasing globals."""
        self._node_fi = dict(node_fi)
        from collections import Counter
        fi_counts = Counter(node_fi.values())
        global L4_Q_N_ALIASED_NODES, L4_Q_N_UNIQUE_NODES
        L4_Q_N_ALIASED_NODES = {n for n, fi in node_fi.items() if fi_counts[fi] > 1}
        L4_Q_N_UNIQUE_NODES  = {n for n, fi in node_fi.items() if fi_counts[fi] == 1}

    def set_node_fi_override(self, nodes: list) -> None:
        """Compatibility stub — override rows not needed in Actor-Critic."""
        pass

    def boost_exploration(self, steps: int = 500) -> None:
        """
        Add a temporary exploration bump after a food move.
        Uses a separate additive boost that decays independently from _base_temp,
        so repeated calls (every 2000 steps) don't fight the long-term decay.
        The bump adds +0.8 to effective temperature and halves every 200 steps.
        """
        self._boost_temp  = getattr(self, '_boost_temp', 0.0)
        self._boost_temp  = max(self._boost_temp, 0.8)   # additive, not multiplicative
        self._boost_steps = max(getattr(self, '_boost_steps', 0), int(steps))

    def decay_node_qn(self, node: str, factor: float = 0.0) -> None:
        """
        Decay per-node policy when food moves away from that node.
        factor=0.0: fully clear. factor=0.3: retain 30% of credit.
        """
        if node in self._pi_node:
            self._pi_node[node] = self._pi_node[node] * factor
        if node in self._node_visits and factor < 0.5:
            self._node_visits[node] = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             bmu_idx:            int,
             rpe:                float,
             focus_entropy:      float = 0.5,
             thought_confidence: float = 0.0,
             freq_idx:           int   = -1,     # kept for harness compat, not used
             world_moved:        bool  = True,
             l4_top_node:        str   = None,
             l4_top_prob:        float = 0.0,
             epsilon_floor:      float = 0.0,
             node_fi_override:   str   = None,   # kept for harness compat, not used
             raw_reward:         float = 0.0,
             alpha_scale:        float = 1.0,    # M66 ACh: scales learning rate
             ne_temp_add:        float = 0.0,    # M66 NE:  additive temperature boost
             **kwargs) -> dict:
        """
        One Actor-Critic step: update policy from outcome, then select action.

        rpe (V1's dopamine signal) acts as the TD error δ.
        Actor updates: π += ALPHA_ACTOR × δ × eligibility_trace
        Action selection: softmax(π_eff / temperature)
        """
        rpe = float(np.clip(rpe, -1.0, 1.0))

        # ── 1+2. Trace decay + policy gradient + actor update ───
        # Proper score function: ∇ log π(a|s) = one_hot(a) - π(a|s)
        # Self-normalizing: update → 0 as π(a) → 1 → no saturation.
        # Other actions pushed DOWN: eliminates all-direction accumulation.
        #
        # FIX: RPE dead zone — skip update when |rpe| < RPE_DEAD_ZONE.
        # This filters the tiny negative RPE noise from forced exploration
        # that accumulates over 500k steps into policy destruction.
        #
        # FIX: Positive RPE amplification — scale positive RPE by POS_RPE_SCALE.
        # Dopamine burst > dip in biology. Counteracts negative exploration bias.
        rpe_for_update = rpe
        if rpe > 0.0:
            rpe_for_update = rpe * POS_RPE_SCALE  # amplify positive signal

        if self._prev_valid and world_moved:
            _fast.decay_and_pg_trace(
                self._e, self._prev_bmu, self._prev_action,
                self._prev_probs, TRACE_DECAY)

            # RPE dead zone: only update actor when signal is meaningful
            if abs(rpe_for_update) >= RPE_DEAD_ZONE:
                # M66 ACh modulates both BMU-level and per-node learning rates.
                _fast.update_all_pi_bmu(
                    self._pi_bmu, self._e, rpe_for_update,
                    ALPHA_ACTOR * alpha_scale, PI_MIN, PI_MAX)

                # Per-node: same PG gradient, weighted by L4 confidence
                if (self._prev_l4_node is not None
                        and self._prev_l4_prob >= NODE_L4_MIN):
                    node = self._prev_l4_node
                    prob = float(self._prev_l4_prob)
                    if node not in self._pi_node:
                        self._pi_node[node] = np.zeros(self._n_actions, dtype=np.float32)
                    pg = -self._prev_probs.copy()
                    pg[self._prev_action] += 1.0
                    self._pi_node[node] = np.clip(
                        self._pi_node[node] + prob * ALPHA_NODE * alpha_scale * rpe_for_update * pg,
                        PI_MIN, PI_MAX)
                    self._node_visits[node] = self._node_visits.get(node, 0) + 1
        else:
            # Wall hit or first step — just decay, no gradient stamp
            self._e *= (1.0 - TRACE_DECAY)

        # ── 3. Wall penalty: PG-based per-node policy update ─────
        # FIX: Old code used raw additive subtraction which drained logits
        # one-way toward PI_MIN without lifting alternatives. Now uses
        # pg_replay_update which applies proper PG gradient: pushes the
        # wall action DOWN and proportionally lifts other actions UP,
        # maintaining the probability simplex.
        if (not world_moved and rpe < 0.0
                and l4_top_node is not None
                and l4_top_prob >= 0.55):
            node = l4_top_node
            if node not in self._pi_node:
                self._pi_node[node] = np.zeros(self._n_actions, dtype=np.float32)
            curr_action = getattr(self, '_current_action', 0)
            # Use PG-based update: penalty pushes wall action down,
            # lifts alternatives proportionally via softmax gradient.
            wall_credit = float(l4_top_prob * ALPHA_NODE * abs(rpe))
            _fast.pg_replay_update(
                self._pi_node[node], curr_action,
                -wall_credit,   # negative credit = penalise this action
                float(max(self._current_temp, TEMP_MIN)), PI_MIN, PI_MAX)
            self._node_visits[node] = self._node_visits.get(node, 0) + 1

        # ── 4. Trace stamp replaced by PG gradient in step 1+2 ──

        # ── 5. Direct food credit on per-node policy ─────────
        # When food reward arrives (raw_reward > 0), immediately credit the
        # action that led to food at the current L4 node — no trace needed.
        if (raw_reward > 0.1 and world_moved
                and l4_top_node is not None
                and l4_top_prob >= 0.55):
            node = l4_top_node
            if node not in self._pi_node:
                self._pi_node[node] = np.zeros(self._n_actions, dtype=np.float32)
            credit_rpe = max(float(rpe), 0.20)
            _fast.pg_replay_update(
                self._pi_node[node], self._prev_action,
                float(l4_top_prob * ALPHA_NODE * credit_rpe),
                float(max(self._current_temp, TEMP_MIN)), PI_MIN, PI_MAX)
            self._node_visits[node] = self._node_visits.get(node, 0) + 1

        # ── 6. Temperature ────────────────────────────────────
        # _base_temp decays continuously after warmup — this is the long-term
        # exploitation drive. Never reset by boost_exploration.
        if self.t >= TEMP_WARMUP_STEPS:
            self._base_temp = max(TEMP_MIN, self._base_temp * TEMP_DECAY)

        # _boost_temp is a separate additive bump set by boost_exploration.
        # Decays fast (halves every 200 steps) so it clears between food moves.
        self._boost_temp = getattr(self, '_boost_temp', 0.0)
        if self._boost_temp > 0.01:
            self._boost_temp *= 0.9966   # halves in ~200 steps
        else:
            self._boost_temp = 0.0

        # Scale base by focus_entropy + thought_confidence, then add boost.
        entropy_scale = TEMP_ENTROPY_MIN + (1.0 - TEMP_ENTROPY_MIN) * float(focus_entropy)
        confidence_suppression = 1.0 - 0.25 * float(thought_confidence)
        # M66 NE adds directly on top — arousal can push temp above base decay.
        temp_eff = (self._base_temp * entropy_scale * confidence_suppression
                    + self._boost_temp + ne_temp_add)

        # M58 epsilon_floor → temperature floor (rescaled)
        if epsilon_floor > 0.0:
            temp_floor = TEMP_MIN + float(epsilon_floor) * (TEMP_INIT - TEMP_MIN)
            temp_eff = max(temp_eff, temp_floor)

        temp_eff = float(np.clip(temp_eff, TEMP_MIN, TEMP_INIT + 0.60))  # +0.60 = M66 NE_TEMP_MAX
        self._current_temp = temp_eff

        # ── 6b. L2 weight decay (homeostatic plasticity) ──────
        # FIX: pull all logits slowly toward 0.0 so ancient punishments
        # fade. Without this, logits trapped at PI_MIN (-1.5) never
        # recover — the brain permanently "forgets" that an action
        # exists. Biologically: inactive synapses decay toward baseline.
        # Applied every WEIGHT_DECAY_INTERVAL steps to amortise cost.
        if self.t > 0 and self.t % WEIGHT_DECAY_INTERVAL == 0:
            self._pi_bmu *= (1.0 - WEIGHT_DECAY)
            for _node_key in self._pi_node:
                self._pi_node[_node_key] *= (1.0 - WEIGHT_DECAY)

        # ── 6. Effective policy (soft L4 blend) ───────────────
        pi_eff = self._pi_bmu[bmu_idx].copy()

        if l4_top_node is not None and l4_top_node in self._pi_node:
            visits = self._node_visits.get(l4_top_node, 0)
            if visits >= NODE_MIN_VISITS:
                # Trust factor: ramps from 0 → 1 between NODE_MIN_VISITS and 3×
                trust = float(np.clip(
                    (visits - NODE_MIN_VISITS) / (2.0 * NODE_MIN_VISITS + 1e-9),
                    0.0, 1.0))
                # Blend weight: proportional to L4 probability × trust
                blend = float(np.clip(l4_top_prob * trust * 0.85, 0.0, 0.85))
                pi_eff = (1.0 - blend) * pi_eff + blend * self._pi_node[l4_top_node]

        # ── 7. Softmax action selection (numba-compiled) ──────
        probs  = _fast.compute_softmax(pi_eff.astype(np.float32), float(temp_eff))
        action = int(_fast.sample_action(probs, float(self._rng.random_sample())))

        # Greedy action for diagnostics
        greedy_action = int(np.argmax(pi_eff))
        is_explore = (action != greedy_action)
        if is_explore:
            self._n_explorations += 1
        else:
            self._n_exploitations += 1

        # ── 8. Replay buffer ──────────────────────────────────
        # Format matches brain.py idle_step(): 7-element tuple
        # (prev_bmu, curr_bmu, action, fi, l4_node, was_explore, l4_prob)
        #
        # IMPORTANT: store the PREVIOUS step's action/L4 fields, not current.
        # This tuple represents the transition prev_bmu → bmu_idx, so we need
        # the action that CAUSED that move (self._prev_action), the L4 node
        # at prev_bmu (self._prev_l4_node), and the explore flag from the
        # previous selection (self._current_was_explore, not yet overwritten).
        if world_moved and self._prev_valid:
            self._replay_buffer.append((
                self._prev_bmu, bmu_idx, self._prev_action,
                self._current_freq_idx,        # fi (unused in replay, kept for compat)
                self._prev_l4_node,            # L4 node at prev_bmu
                self._current_was_explore,     # explore flag from previous step
                self._prev_l4_prob,            # L4 confidence at prev_bmu
            ))

        # ── 9. Advance state ──────────────────────────────────
        self._prev_bmu         = bmu_idx
        self._prev_action      = action
        self._prev_l4_node     = l4_top_node if l4_top_prob >= NODE_L4_MIN else None
        self._prev_l4_prob     = l4_top_prob
        self._prev_valid       = True
        self._prev_probs       = probs.copy()   # store for next step's PG gradient
        self._current_action   = action
        self._current_freq_idx = freq_idx
        self._current_was_explore = is_explore
        self.t += 1

        # Pseudo-epsilon for backward compatibility reporting
        # (temperature normalized to [0,1] range)
        pseudo_eps = float(np.clip(
            (temp_eff - TEMP_MIN) / (TEMP_INIT - TEMP_MIN), 0.0, 1.0))

        return {
            'action':         action,
            'q_values':       pi_eff,          # actor preferences (for M57 compat)
            'q_max':          float(pi_eff.max()),
            'epsilon':        pseudo_eps,       # brain.py maps this to 'action_epsilon'
            'explore':        is_explore,
            'td_error':       rpe,
            'q_mean':         float(self._pi_bmu.mean()),
            'q_nonzero_frac': float((np.abs(self._pi_bmu) > 1e-4).mean()),
            'temperature':    temp_eff,
        }

    # ── Hippocampal replay ────────────────────────────────────

    def replay_on_reward(self,
                         reward:        float,
                         familiarity:   float,
                         food_freq_idx: int   = -1,   # kept for harness compat
                         food_node:     str   = None) -> int:
        """
        Back-propagate food reward through recent transitions (hippocampal replay).
        Called immediately after a food event from the harness.

        Credit starts at reward × REPLAY_ALPHA × (1 - familiarity×0.5)
        and decays by REPLAY_GAMMA per step backward.
        Familiar food nodes get less amplification (already well-learned).
        food_node receives extra direct credit at full strength (no decay).
        """
        if not self._replay_buffer:
            return 0

        credit = float(reward) * REPLAY_ALPHA * (1.0 - float(familiarity) * 0.5)
        n_replayed = 0

        temp = float(max(self._current_temp, TEMP_MIN))
        for pb, cb, act, fi, l4_node, was_explore, l4_prob in reversed(self._replay_buffer):
            if credit < 0.005:
                break
            explore_scale = 0.70 if was_explore else 1.0
            # PG-based replay: scales by actual current softmax probs.
            # Self-normalizing: when π[act] already dominant, update → 0.
            # Prevents PI_MIN saturation of non-chosen directions.
            _fast.pg_replay_update(
                self._pi_bmu[pb], act,
                float(credit * explore_scale), temp, PI_MIN, PI_MAX)
            if l4_prob >= 0.30 and l4_node is not None:
                node_to_credit = food_node if (l4_node == food_node and food_node) else l4_node
                if node_to_credit in self._pi_node:
                    scale = (0.8 if node_to_credit == food_node else 0.5) * explore_scale
                    _fast.pg_replay_update(
                        self._pi_node[node_to_credit], act,
                        float(credit * scale), temp, PI_MIN, PI_MAX)
            credit *= REPLAY_GAMMA
            n_replayed += 1

        return n_replayed

    # ── Policy audit helpers ──────────────────────────────────

    def get_node_policy(self, node: str):
        """Return per-node policy preferences and visit count. None if no data."""
        pi = self._pi_node.get(node)
        v  = self._node_visits.get(node, 0)
        return pi, v

    def get_bmu_policy(self, bmu_idx: int):
        """Return BMU-level policy preferences."""
        return self._pi_bmu[bmu_idx].copy()

    def planning_rate(self) -> float:
        """Compatibility stub — planning rate is tracked by the Planner."""
        return 0.0
