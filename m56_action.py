"""
M56: ACTION LAYER — CORTICO-STRIATAL Q-LEARNING
================================================

WHAT THIS IS
------------
M56 is the output layer of the stack. It learns action-values and
selects actions. Every layer below it perceives, predicts, attends, and
evaluates — M56 is the first layer that DOES something.

It answers one question each step: given where I am (bmu_idx) and what
I expect (Thought), what should I do next?

HOW IT WORKS
------------
Two data structures. That's the entire system.

Q TABLE  (shape: 64 × 64 × N_ACTIONS, float32)
  State = (prev_bmu, curr_bmu) — the PAIR of where the brain came from
  and where it is now. This resolves the partial-observability aliasing:
  the same current frequency/BMU can be reached from different prior nodes
  requiring different optimal actions (e.g. C reached from A vs from E★).
  Learned via TD with eligibility traces:
    Q[s_prev, s_curr, a_prev] += ETA_Q * rpe * e[s_prev, s_curr, a_prev]
  rpe = V1's signed reward prediction error — already the TD signal.
  No separate reward bookkeeping needed; V1 computes it.
  Biologically: context-dependent cortico-striatal synaptic weights
  (dorsal striatum with hippocampal context input)

ELIGIBILITY TRACE  (shape: 64 × 64 × N_ACTIONS, float32)
  e[s_prev, s_curr, a] = recency-weighted responsibility for outcome
  Updated each step: e *= TRACE_DECAY, then e[prev, curr, action] += 1.0
  Q update: ΔQ = ETA_Q * rpe * e
  Biologically: dopamine-gated eligibility traces with context gating

ACTION SELECTION
  epsilon-greedy with adaptive epsilon:
    base_epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * focus_entropy
    epsilon      = base_epsilon * (1 - CONFIDENCE_GATE * thought_confidence)

  High focus_entropy (diffuse attention)  → high epsilon → explore
  Low focus_entropy  (focused attention)  → low epsilon  → exploit
  High thought_confidence                 → further suppresses epsilon
    (Thought is confident about context → less reason to explore)

  At each step:
    if random() < epsilon: action = random choice (explore)
    else:                  action = argmax Q[bmu_idx, :] (exploit)

  Biologically:
    epsilon  = noradrenaline locus coeruleus arousal signal
    argmax Q = basal ganglia direct pathway disinhibition of motor cortex

SIGNALS CONSUMED (from Brain.step() output)
  bmu_idx           (M54) — current cortical state
  rpe               (V1)  — TD error: better/worse than expected
  focus_entropy     (Thought) — attention diffuseness → exploration rate
  thought_confidence (Thought) — prediction certainty → exploitation rate

SIGNALS OUTPUT each step
  'action'          int   — chosen action index [0, N_ACTIONS)
  'q_values'        array — (N_ACTIONS,) current Q estimates for bmu_idx
  'q_max'           float — max Q value in current state
  'epsilon'         float — effective exploration probability this step
  'explore'         bool  — True if this step was exploratory
  'td_error'        float — rpe used to update Q (= V1's rpe)
  'q_mean'          float — mean of entire Q table (learning progress)
  'q_nonzero_frac'  float — fraction of Q entries with |Q| > 1e-4

CALL ORDER
----------
Each step in Brain:

  # BEFORE brain.step() — choose action for this step
  action = m56.select_action(bmu_idx=r_prev['bmu_idx'],
                             focus_entropy=r_prev['focus_entropy'],
                             thought_confidence=r_prev['thought_confidence'])

  # Execute action (external environment)
  reward = environment.step(action)

  # brain.step() with the reward
  r = brain.step(..., reward=reward)

  # AFTER brain.step() — learn from outcome
  m56_out = m56.update(bmu_idx=r['bmu_idx'],
                       action=action,
                       rpe=r['rpe'])

Alternatively, Brain can own M56 internally and call it automatically.
The standalone interface above is for environments that control the loop.

SPACE COST
----------
Q table:            64 × 64 × N_ACTIONS × float32
Eligibility traces: 64 × 64 × N_ACTIONS × float32
At N_ACTIONS=4:     2 × 64 × 64 × 4 × 4 = 128 KB RAM  (trivial)

BIOLOGICAL GROUNDING
--------------------
1. TD(λ) learning
   ΔQ[s,a] = ETA_Q × rpe × e[s,a]
   Biologically: dopamine burst/dip modulates synaptic LTP/LTD in
   striatum proportionally to the eligibility trace (recent activity).

2. Decay of unused action values
   Q *= (1 - Q_DECAY) each step
   Unused action-state associations fade — prevents stale values from
   dominating exploration in rarely-visited states.

3. Exploration via noradrenergic signal
   epsilon scales with attentional diffuseness (focus_entropy).
   High arousal / uncertain attention → explore.
   Biologically: locus coeruleus NE modulates basal ganglia gate threshold.

4. Confidence-gated exploitation
   Thought's confidence_delta suppresses exploration.
   When the system knows what's coming next (high Thought confidence),
   there is less benefit to trying random actions.
   Biologically: PFC top-down control over striatal action gating.
"""

import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Grid (must match M54, M55, L2, Thought)
N_NEURONS  = 64

# Action space
N_ACTIONS  = 4   # number of discrete output actions (configurable)

# Zone count — must match ConceptLayer n_zones (one per map node)
# Used for the freq_idx-keyed Q table that bypasses BMU aliasing.
N_ZONES    = 8

# ── Q learning ───────────────────────────────────────────────
# Learning rate for TD update
# Matches L2's ETA_BASE — same timescale as sequence learning
ETA_Q      = 0.05   # conservative learning rate. High values (0.15) cause
                    # Q oscillation — food and wall-penalty RPE overwrite each
                    # other on successive visits rather than accumulating.

# Synaptic decay on Q.
# Set to 0 — the map is static, rewards don't change, Q should accumulate
# permanently. Diagnostic confirmed: at 0.0002, Q halves in 3500 steps and
# reaches zero by ~15k steps. The brain was relearning the same policy from
# scratch every 15k cycle (food rate oscillated but never improved). With
# no decay, Q values compound across the full 200k run.
Q_DECAY    = 0.0

# Eligibility trace decay
# τ ≈ 1/TRACE_DECAY = 5 steps — recent actions get most credit
TRACE_DECAY = 0.20

# Q value bounds — rpe is in [-1, +1] so Q represents expected rpe
# and should stay in the same range
Q_MAX  =  1.0
Q_MIN  = -1.0

# ── Exploration ──────────────────────────────────────────────
# epsilon-greedy with adaptive epsilon:
#   base_epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * focus_entropy
#   epsilon      = base_epsilon * (1 - CONFIDENCE_GATE * thought_confidence)
#
# At focus_entropy=1.0 (uniform gate, no attention): epsilon=EPSILON_MAX=0.50
# At focus_entropy=0.0 (perfectly focused): epsilon=EPSILON_MIN=0.05
# thought_confidence further suppresses by up to CONFIDENCE_GATE fraction
EPSILON_MIN          = 0.15   # raised from 0.10: 16-node world needs a higher
                               # exploration floor — at 0.10 the brain commits too
                               # early to unstable Q_f patterns post-warmup.
EPSILON_MAX          = 0.35   # lowered from 0.50: at 0.50 the brain permanently
                               # thrashes (50% random) even after Q_f is well-formed.
                               # At 0.35: 65% exploitation post-warmup — enough to
                               # build a stable policy while still exploring.
EPSILON_WARMUP_STEPS = 25_000  # shortened from 40k: Q_f forms good signal by step 20k
                                # in a 16-node world. Warmup past that point just delays
                                # exploitation. 25k gives ~13k closed-loop warmup steps
                                # (open loop consumed 0 after the action.t reset).
EPSILON_WARMUP_VAL   = 0.45    # epsilon during warmup — guarantees food found early
CONFIDENCE_GATE      = 0.30
Q_SPREAD_GATE        = 0.30   # lowered from 0.60: in a fully-aliased 16-node world,
                               # BMU Q spread is unreliable — aliased nodes share BMUs
                               # so a "confident" BMU Q spread doesn't mean the brain
                               # knows where it is. High Q_SPREAD_GATE was crushing
                               # epsilon to 0.10 and locking the brain into bad policies.

# ── Replay buffer ─────────────────────────────────────────────
# Short experience replay for long-path credit assignment.
# Biologically: hippocampal sharp-wave ripple replay during reward events.
# On food arrival, back-propagate discounted reward through recent transitions.
# Replay bonus scales with (1 - familiarity_at_food_node) so frequent food
# nodes (high familiarity, e.g. E★) get little amplification while rare nodes
# (H★, near-zero familiarity) get strong back-propagation.
REPLAY_BUFFER_LEN   = 12
REPLAY_GAMMA        = 0.90
REPLAY_ALPHA        = 0.20  # raised from 0.10 — K★ is 6 steps away; credit at step 1
                             # = 0.20×0.90^5 = 0.118, meaningful after 3-4 food events.

# ── Conflict-driven epsilon floor ────────────────────────────
# When the Q-margin (best - second-best action) is small, the brain is
# genuinely undecided between actions. In this case, lock epsilon to a
# minimum floor so the brain keeps exploring rather than committing to
# whichever signal happens to be slightly ahead.
#
# Only fires when Q[best] > CONFLICT_Q_MIN — i.e. the brain has actually
# learned something here. Prevents the floor from being a no-op on cold
# Q entries (where all values are near zero and margin≈0 trivially).
# Biologically: anterior cingulate cortex conflict monitoring — uncertainty
# about the best action keeps the system from premature commitment.
CONFLICT_THRESHOLD  = 0.10   # margin below which state is "conflicted"
CONFLICT_FLOOR      = 0.18   # minimum epsilon when conflicted
CONFLICT_Q_MIN      = 0.02   # must have learned something (Q[best] > this)
                          # spread = (max_Q - min_Q) / (|mean_Q| + 0.1); clipped [0,1]
                          # Biologically: basal ganglia exploit when action values differ


# ── Node-keyed Q table (L4-gated) ────────────────────────────
# When L4's position belief is confident (top_prob >= threshold),
# use a node-specific Q entry that cannot be contaminated by aliased nodes.
# This is the core fix for D vs L and E vs K disambiguation:
#   Q_f[3] is shared by D and L — D→South should have high value (path to K★)
#   but L→South is a dead end. Without node-keying, these contaminate each other.
#   Q_n['D'][South] and Q_n['L'][South] are fully independent.
#
# L4_Q_BLEND_THRESH: minimum L4 top_prob before Q_n is blended in.
#   At 0.60: L4 must be quite confident before overriding Q_f.
#   Below this: falls back to Q_f entirely (safe, stable).
#
# L4_Q_BLEND_MAX: maximum blend weight from Q_n even at full confidence.
#   Kept at 0.20: Q_f stays dominant (80%). Q_n is a gentle nudge,
#   not an override. Higher values (0.50) caused regression — cold Q_n
#   zeros diluted working Q_f policy and doubled wall rate.
#
# L4_Q_N_WARMUP: minimum real transitions before Q_n influences selection.
#   Below this per-node count, Q_f is used unchanged even if L4 is
#   confident. Prevents cold Q_n from degrading a working Q_f policy.
#   At 30: ~3× the minimum needed for Q_f to show meaningful signal.
L4_Q_BLEND_THRESH         = 0.35  # lowered: need Q_n to activate in World 5
L4_Q_BLEND_THRESH_ALIASED = 0.35  # same lower threshold — World 5 is fully aliased
L4_Q_BLEND_MAX    = 0.70   # raised: let Q_n fully override Q_f when L4 is confident
                           # (Q_f[7] is useless at food nodes — both H and P write to it)
L4_Q_N_WARMUP     = 5      # FIXED: lowered from 15. With anchor resets giving clean L4
                           # beliefs at food nodes, Q_n accumulates reliable samples fast.
                           # 10 anchored observations is enough to trust Q_n['H'/'P'].

# Q_n amplification at food nodes (for fully-aliased food nodes where Q_f is useless)
# When L4 is confident the brain is at a registered food node, Q_n update gets this
# multiplier. Rationale: Q_f[7] averages H and P signals and is structurally noisy;
# Q_n[H] and Q_n[P] are clean — they just need to learn faster to compensate.
L4_Q_N_FOOD_AMP   = 6.0   # 4× normal ETA_Q at food nodes. After 10 visits: Q_n[H,N]≈+0.8
                           # ~15 confident hits per food node in 200k steps.
                           # 50 was effectively disabling Q_n for the entire run.

# Set by Brain.__init__ when L4 is active.
# Maps each node to whether its freq_index is shared with other nodes.
# Used to select the appropriate confidence threshold — aliased nodes need higher confidence.
# Format: set of aliased node names (those sharing fi with another node).
L4_Q_N_UNIQUE_NODES: set = set()   # nodes with unique fi — use lower threshold
L4_Q_N_ALIASED_NODES: set = set()  # nodes with shared fi — use higher threshold

class ActionLayer:
    """
    Cortico-striatal Q-learning action selection module.

    State: BMU index from M54 (which cortical pattern is active)
    Action: discrete index in [0, N_ACTIONS)
    Learning signal: V1's rpe (signed TD error, already computed)

    Standalone or wired into Brain — see module docstring.
    """

    def __init__(self, n_actions: int = N_ACTIONS, seed: int = 0):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)

        # ── Q table (BMU-keyed, context-dependent) ────────────
        # Q[prev_bmu, curr_bmu, a] = value of action a given context (prev→curr)
        self._Q = np.zeros((N_NEURONS, N_NEURONS, n_actions), dtype=np.float32)

        # ── Q table (freq_idx-keyed) ───────────────────────────
        # Q_f[freq_idx, a] — flat 8×4 table keyed by ground-truth node index.
        # Diagnostic confirmed: G (1.7Hz) and H (2.0Hz) share modal BMU=4.
        # The BMU-keyed table cannot store different policies for G vs H.
        # This table uses freq_idx directly (passed through from world.step),
        # bypassing the SOM entirely. When freq_idx >= 0 it is authoritative.
        # The BMU table is retained for open-loop / unfamiliar-signal steps
        # where freq_idx = -1.
        #
        # EXTENDED to 10 rows: rows 0-7 are standard fi-keyed Q values.
        # Rows 8+ are GROUND-TRUTH NODE overrides for fully aliased food nodes
        # where fi-sharing causes write-collisions (H and P both write to fi=7).
        # The harness passes node_fi_override to replay_on_reward() and
        # the wall-penalty path to direct credit to the correct row.
        # _node_fi_override_map: node_str -> Q_f row index (8, 9, 10, ...)
        self._Q_f = np.zeros((N_ZONES + 4, N_ACTIONS), dtype=np.float32)
        self._node_fi_override_map: dict = {}   # set by harness via set_node_fi_override()
        self._node_fi_override_row: dict = {}   # node -> Q_f row (8, 9, ...)
        self._next_override_row = N_ZONES       # allocate from row 8 upward

        # Eligibility trace for Q_f — separate from BMU Q trace.
        # Shape: (N_ZONES, N_ACTIONS). Gated by world_moved (real transitions only).
        # This prevents wall-hit actions from receiving food credit via trace,
        # which was causing all Q_f entries to saturate equally.
        self._e_f = np.zeros((N_ZONES + 4, n_actions), dtype=np.float32)

        # prev freq_idx for the freq-keyed update (mirrors _prev_bmu)
        self._prev_freq_idx    = -1
        self._current_freq_idx = -1   # set by select_action each step
        self._reward_freq_idx  = -1   # set by replay_on_reward each call
        # The action that caused the PREVIOUS real transition (for Q_f update).
        self._transition_action = -1  # action that caused prev_fi→curr_fi move

        # ── Node-keyed Q table (L4-gated) ────────────────────────
        # _Q_n[node_str] = float32[n_actions] — per-node Q values.
        # Created lazily on first confident L4 observation of each node.
        # Independent of _Q_f: D and L have separate entries even though
        # both have freq_idx=3.
        self._Q_n = {}   # dict[str, np.ndarray]

        # Per-node observation count — Q_n only blends into action selection
        # once _Q_n_count[node] >= L4_Q_N_WARMUP. Before warmup, Q_f is used
        # unchanged. This prevents cold zero Q_n entries from diluting Q_f.
        self._Q_n_count = {}   # dict[str, int]

        # L4 state tracking — stored from select_action for use in update()
        self._l4_node_at_select  = None   # L4 top_node when action was chosen
        self._l4_prob_at_select  = 0.0    # L4 top_prob when action was chosen
        self._l4_confident_now   = False  # whether L4 was confident this step
        self._current_node_override = None  # ground-truth node override from harness
        self._prev_node_override    = None  # previous step's node override (for e_f trace)

        # ── Eligibility traces ────────────────────────────────
        # e[prev_bmu, curr_bmu, a] — same shape as Q
        self._e = np.zeros((N_NEURONS, N_NEURONS, n_actions), dtype=np.float32)

        # ── State tracking ────────────────────────────────────
        # Previous BMU — needed to form the (prev, curr) state key
        self._prev_bmu    = 0
        self._prev_action = 0
        self._prev_valid  = False   # False on first step (no prev context yet)

        # Current action selected this step (for post-step update)
        self._current_action  = 0
        self._current_bmu     = 0
        self._current_epsilon = EPSILON_MAX

        # External epsilon override — harness sets this to force exploration.
        # Consumed (reset to -1) immediately after each select_action call.
        self.epsilon_override = -1.0   # -1 = inactive

        # ── Diagnostics ───────────────────────────────────────
        self.t             = 0
        self._n_explorations = 0
        self._n_exploitations = 0
        self._td_history   = deque(maxlen=200)
        self._reward_history = deque(maxlen=200)
        self._action_counts  = np.zeros(n_actions, dtype=np.int32)

        # ── Replay buffer ─────────────────────────────────────
        # Stores (prev_bmu, curr_bmu, action) tuples for recent transitions.
        # On food arrival, iterates backward applying discounted Q updates.
        self._replay_buffer = deque(maxlen=REPLAY_BUFFER_LEN)

    # ── Node-fi override registration ────────────────────────

    def set_node_fi_override(self, nodes: list) -> None:
        """
        Register ground-truth node strings that need their own Q_f rows
        because they share fi with another node (fi write-collision).

        Call once from the harness before closed-loop training:
            brain.action.set_node_fi_override(['H', 'P'])

        After this, replay_on_reward(..., food_node='H') and the direct
        wall-penalty path will write to Q_f[8] (H) and Q_f[9] (P)
        instead of the shared Q_f[7].  select_action() reads these
        override rows when the harness passes node_fi_override=node_str.
        """
        for node in nodes:
            if node not in self._node_fi_override_row:
                self._node_fi_override_row[node] = self._next_override_row
                self._next_override_row += 1

    def _node_override_row(self, node: str) -> int:
        """Return the dedicated Q_f row for node, or -1 if not registered."""
        return self._node_fi_override_row.get(node, -1)

    # ── Action selection ──────────────────────────────────────

    def select_action(self,
                      bmu_idx:          int,
                      focus_entropy:    float = 0.5,
                      thought_confidence: float = 0.0,
                      freq_idx:         int   = -1,
                      l4_top_node:      str   = None,
                      l4_top_prob:      float = 0.0,
                      epsilon_floor:    float = 0.0,
                      node_fi_override: str   = None) -> int:
        """
        Choose an action for the current step.

        Parameters
        ----------
        bmu_idx            : int   — current cortical state (from M54)
        focus_entropy      : float — Thought's attention diffuseness [0,1]
        thought_confidence : float — Thought's prediction certainty [0,1]
        freq_idx           : int   — ground-truth freq index [0,7] or -1
        l4_top_node        : str   — L4's most-likely node ('A','B',...) or None
        l4_top_prob        : float — L4's confidence in top_node [0,1]
        epsilon_floor      : float — minimum epsilon from M58 working memory [0,1]
                                     Applied AFTER all other epsilon computation.
                                     Does not override epsilon_override or warmup.

        Returns
        -------
        int — chosen action index in [0, N_ACTIONS)
        """
        # ── Epsilon: override → warmup → adaptive ────────────
        if self.epsilon_override >= 0.0:
            epsilon = float(self.epsilon_override)
            self.epsilon_override = -1.0   # consume — one-step only
        elif self.t < EPSILON_WARMUP_STEPS:
            epsilon = EPSILON_WARMUP_VAL
        else:
            base_epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * float(focus_entropy)
            epsilon      = float(base_epsilon * (1.0 - CONFIDENCE_GATE * float(thought_confidence)))

        # Marginal BMU Q for action selection
        q_row = self._Q[:, bmu_idx, :].max(axis=0)

        q_range  = float(q_row.max() - q_row.min())
        q_denom  = float(abs(q_row.mean()) + 0.1)
        q_spread = float(np.clip(q_range / q_denom, 0.0, 1.0))
        q_max    = float(q_row.max())
        pos_gate = float(np.clip((q_max - 0.15) / 0.15, 0.0, 1.0))
        epsilon  = float(epsilon * (1.0 - Q_SPREAD_GATE * q_spread * pos_gate))
        epsilon  = float(np.clip(epsilon, EPSILON_MIN, EPSILON_MAX))

        # Conflict floor
        sorted_q = np.sort(q_row)[::-1]
        q_margin = float(sorted_q[0] - sorted_q[1]) if len(sorted_q) > 1 else 1.0
        if q_max > CONFLICT_Q_MIN and q_margin < CONFLICT_THRESHOLD:
            epsilon = float(max(epsilon, CONFLICT_FLOOR))

        # ── M58 working memory floor ──────────────────────────
        # Applied after conflict floor — after all other epsilon logic.
        # Only active post-warmup: during warmup epsilon is already high
        # (EPSILON_WARMUP_VAL=0.45) so the floor would never bind anyway.
        if self.t >= EPSILON_WARMUP_STEPS and float(epsilon_floor) > 0.0:
            epsilon = float(max(epsilon, float(epsilon_floor)))
        epsilon = float(np.clip(epsilon, EPSILON_MIN, EPSILON_MAX))

        # ── Q_f base (freq_idx-keyed) ─────────────────────────
        # If the harness has identified the exact node (e.g. at a food node
        # after reset_to_node), use the dedicated override row instead of
        # the shared fi row — prevents H/P write-collision contamination.
        if node_fi_override is not None:
            ov_row = self._node_override_row(node_fi_override)
            if ov_row >= 0:
                q_effective = self._Q_f[ov_row].copy()
                self._current_node_override = node_fi_override
            elif freq_idx >= 0:
                q_effective = self._Q_f[freq_idx].copy()
                self._current_node_override = None
            else:
                q_effective = q_row.copy()
                self._current_node_override = None
        elif freq_idx >= 0:
            # Special case: shared food frequency — use override rows directly
            if freq_idx == 7 and self._node_fi_override_row:
                # Blend override rows by L4 belief when available, else use equal weight
                q_effective = np.zeros(self._n_actions, dtype=np.float32)
                total_weight = 0.0
                for node, ov_row in self._node_fi_override_row.items():
                    # Weight by L4 belief for this node if available
                    if l4_top_node == node and l4_top_prob > 0.2:
                        w = float(l4_top_prob)
                    else:
                        w = 0.5  # equal fallback
                    q_effective += w * self._Q_f[ov_row]
                    total_weight += w
                if total_weight > 1e-6:
                    q_effective /= total_weight
                self._current_node_override = None
            else:
                q_effective = self._Q_f[freq_idx].copy()
                self._current_node_override = None
        else:
            q_effective = q_row.copy()
            self._current_node_override = None

        # ── Q_n blend (L4 node-keyed) — ALIASED NODES ONLY ──────
        # Q_n is blended into action selection ONLY for aliased nodes
        # (those sharing fi with another node: A/I, B/J, D/L, E/K).
        # For unique-fi nodes (C, E, F, G, H in World 3), Q_f[fi] is
        # already node-specific — blending Q_n only adds corruption.
        #
        # Why unique-node Q_n corrupts: at C (fi=2, unique), the brain
        # visits C 30%+ of the time but usually bounces back (West/North)
        # via K-path credit. Q_n[C,West] and Q_n[C,North] accumulate
        # large positive values from K-path reward. At blend=0.35,
        # Q_eff[C,West] > Q_eff[C,South] even though Q_f[2,South]=1.0.
        # The correct signal (Q_f) is overridden by the noisy one (Q_n).
        #
        # Aliased nodes still need Q_n: A and I share fi=0, so Q_f[0]
        # cannot distinguish them. Q_n['A'] and Q_n['I'] are the only
        # per-node signals available.
        node_is_aliased = (l4_top_node in L4_Q_N_ALIASED_NODES) if L4_Q_N_ALIASED_NODES else False
        thresh = L4_Q_BLEND_THRESH_ALIASED
        l4_confident = (l4_top_node is not None and
                        node_is_aliased and
                        l4_top_prob >= thresh and
                        self._Q_n_count.get(l4_top_node, 0) >= L4_Q_N_WARMUP)
        if l4_confident:
            if l4_top_node not in self._Q_n:
                self._Q_n[l4_top_node] = np.zeros(self._n_actions, dtype=np.float32)
            blend = float(np.clip(
                (l4_top_prob - thresh) /
                max(1.0 - thresh, 1e-9) * L4_Q_BLEND_MAX,
                0.0, L4_Q_BLEND_MAX))
            q_effective = ((1.0 - blend) * q_effective
                           + blend * self._Q_n[l4_top_node])

        # Store L4 state for update(). Record only for aliased nodes —
        # unique nodes no longer write to Q_n (see above).
        l4_above_thresh = (l4_top_node is not None and
                           node_is_aliased and
                           l4_top_prob >= thresh)
        self._l4_node_at_select = l4_top_node if l4_above_thresh else None
        self._l4_prob_at_select = l4_top_prob
        self._l4_confident_now  = l4_confident

        # Epsilon-greedy selection on effective Q
        if self._rng.random() < epsilon:
            action  = int(self._rng.randint(0, self._n_actions))
            self._n_explorations += 1
        else:
            noise  = self._rng.uniform(0, 1e-6, size=q_effective.shape)
            action = int(np.argmax(q_effective + noise))
            self._n_exploitations += 1

        self._current_action   = action
        self._current_bmu      = bmu_idx
        self._current_freq_idx = freq_idx
        self._current_epsilon  = epsilon
        self._action_counts[action] += 1

        return action

    # ── Update ────────────────────────────────────────────────

    def update(self,
               bmu_idx:     int,
               action:      int,
               rpe:         float,
               world_moved: bool = True) -> dict:
        """
        Update Q table from outcome. Call AFTER brain.step().

        Parameters
        ----------
        bmu_idx     : int  — state that resulted from action (current bmu_idx)
        action      : int  — action that was taken (from select_action)
        rpe         : float — signed reward prediction error from V1 [-1, +1]
        world_moved : bool — True if the world actually transitioned to a new node.
                             False on wall hits. Used to gate replay buffer writes
                             and eligibility trace sets. Without this, BMU drift
                             within a single node's SOM cluster (different BMU fired
                             even though the agent didn't move) looks like a real
                             transition, causing wall-hit actions to enter the replay
                             buffer and receive food credit via replay.

        Returns
        -------
        dict with 'action', 'q_values', 'q_max', 'epsilon', 'explore',
                  'td_error', 'q_mean', 'q_nonzero_frac'
        """
        rpe = float(np.clip(rpe, -1.0, 1.0))

        # ── 1. Decay eligibility traces ────────────────────────
        self._e   *= (1.0 - TRACE_DECAY)
        self._e_f *= (1.0 - TRACE_DECAY)   # decay Q_f trace in lockstep

        # ── 2. Set trace for BMU-keyed Q ───────────────────────
        bmu_moved = world_moved and (self._prev_bmu != self._current_bmu)
        if bmu_moved:
            self._e[self._prev_bmu, self._current_bmu, self._current_action] = 1.0

        # ── 2b. Set trace for Q_f ──────────────────────────────
        # Gate on world_moved only (real transitions), same as BMU trace.
        # If the PREVIOUS node's fi has a registered override row (i.e. it's a
        # food node like H or P), credit the override row directly and leave
        # the shared fi row untouched — prevents e_f trace from leaking food
        # credit into the shared fi=7 row via decay residuals.
        if world_moved and self._prev_freq_idx >= 0 and self._current_action >= 0:
            ov_row = -1
            # Check if prev fi maps to an override node (via current_node_override set
            # when we were AT that node last step)
            if (hasattr(self, '_prev_node_override') and
                    self._prev_node_override is not None):
                ov_row = self._node_override_row(self._prev_node_override)
            if ov_row >= 0:
                self._e_f[ov_row, self._current_action] = 1.0
            elif self._prev_freq_idx == 7 and self._node_fi_override_row:
                # Distribute trace to override rows instead of shared fi=7
                l4_top_node = getattr(self, '_l4_node_at_select', None)
                l4_top_prob = getattr(self, '_l4_prob_at_select', 0.0)
                for node, orow in self._node_fi_override_row.items():
                    if l4_top_node == node and l4_top_prob > 0.2:
                        w = l4_top_prob
                    else:
                        w = 0.5
                    self._e_f[orow, self._current_action] = max(
                        self._e_f[orow, self._current_action], float(w)
                    )
            else:
                self._e_f[self._prev_freq_idx, self._current_action] = 1.0

        # ── 3. Update BMU Q via trace ──────────────────────────
        self._Q += ETA_Q * rpe * self._e
        np.clip(self._Q, Q_MIN, Q_MAX, out=self._Q)

        # ── 3b. Update Q_f via eligibility trace ─────────────
        if rpe >= 0.0:
            # Row 7 is permanently silenced for positive credit.
            # Override rows 8/9 (H, P) receive credit normally via _e_f.
            for row in range(len(self._Q_f)):
                if row == 7:
                    continue  # shared fi=7 — never gets positive credit
                if self._e_f[row].max() < 1e-6:
                    continue  # skip unvisited rows
                self._Q_f[row] += ETA_Q * rpe * self._e_f[row]
            np.clip(self._Q_f, Q_MIN, Q_MAX, out=self._Q_f)

        # ── 3b-wall. Direct wall penalty on Q_f ──────────────
        # On wall hits (world_moved=False), penalise the wall action
        # directly. The trace decays to ~0 during sustained bashing
        # (10 hits → e_f≈0.11) so trace-based learning is useless.
        # Direct write: Q_f[fi,a] += ETA_Q * rpe each hit.
        # After 20 hits: Q_f[fi,a] ≈ -0.55. Brain stops trying it.
        # Also write to the node override row if the harness registered one —
        # this is critical for H and P which share fi=7.
        if (not world_moved and self._current_freq_idx >= 0 and rpe < 0.0):
            fi = self._current_freq_idx
            a  = self._current_action
            if self._current_node_override is not None:
                ov_row = self._node_override_row(self._current_node_override)
                if ov_row >= 0:
                    self._Q_f[ov_row, a] = float(np.clip(
                        self._Q_f[ov_row, a] + 2.0 * ETA_Q * rpe, Q_MIN, Q_MAX))
            elif fi == 7 and self._node_fi_override_row:
                # Distribute direct wall penalty to override rows based on belief
                l4_top_node = getattr(self, '_l4_node_at_select', None)
                l4_top_prob = getattr(self, '_l4_prob_at_select', 0.0)
                for node, orow in self._node_fi_override_row.items():
                    if l4_top_node == node and l4_top_prob > 0.2:
                        w = float(l4_top_prob)
                    else:
                        w = 0.5
                    self._Q_f[orow, a] = float(np.clip(
                        self._Q_f[orow, a] + 2.0 * ETA_Q * rpe * w, Q_MIN, Q_MAX))
            else:
                if fi != 7: # extra safety just to be sure we never write to 7
                    self._Q_f[fi, a] = float(np.clip(
                        self._Q_f[fi, a] + 2.0 * ETA_Q * rpe, Q_MIN, Q_MAX))

        # ── 3b-wall-n. Direct wall penalty on Q_n (L4-gated) ─
        # Without this, Q_n learns food-approach actions but never learns
        # "action X hits a wall from node Y". In World 5 there is no harness
        # surgery for wall penalties (unlike World 3), so Q_n must learn
        # walls directly. Only writes when L4 is confident about the node.
        if (not world_moved and rpe < 0.0 and
                self._l4_node_at_select is not None):
            node = self._l4_node_at_select
            a    = self._current_action
            if node not in self._Q_n:
                self._Q_n[node] = np.zeros(self._n_actions, dtype=np.float32)
            self._Q_n[node][a] = float(np.clip(
                self._Q_n[node][a] + ETA_Q * rpe, Q_MIN, Q_MAX))

        # ── 3c. Update Q_n (node-keyed, L4-gated) ─────────────
        # Updated on all RPE including negative (real moves only —
        # wall-hit Q_n penalties handled by harness surgery in brain_in_world3).
        # For aliased food nodes (H, P in World 5): Q_f[7] averages both nodes
        # and is structurally noisy. Q_n['H'] and Q_n['P'] are clean per-node
        # signals but need to learn faster to compensate. Apply L4_Q_N_FOOD_AMP
        # when the brain is confirmed at a food-override node.
        if (world_moved and
                self._l4_node_at_select is not None and
                self._transition_action >= 0):
            node = self._l4_node_at_select
            pa   = self._transition_action
            if node not in self._Q_n:
                self._Q_n[node] = np.zeros(self._n_actions, dtype=np.float32)
            # Amplify Q_n learning rate when brain is at a registered food-override
            # node — those are the nodes where Q_f is structurally contaminated.
            amp = (L4_Q_N_FOOD_AMP
                   if node in self._node_fi_override_row
                   else 1.0)
            self._Q_n[node][pa] = float(np.clip(
                self._Q_n[node][pa] + amp * ETA_Q * rpe, Q_MIN, Q_MAX))
        # Always count toward warmup when L4 is confident and world moved,
        # regardless of RPE sign — the count is about L4 observation quality,
        # not about whether this step earned reward.
        if (world_moved and
                self._l4_node_at_select is not None and
                self._transition_action >= 0):
            self._Q_n_count[self._l4_node_at_select] = (
                self._Q_n_count.get(self._l4_node_at_select, 0) + 1)

        # ── 4. Record transition BEFORE advancing context ─────
        # Gate replay buffer on world_moved (not bmu_moved).
        # BMU drift within a node's cluster makes bmu_moved=True even on wall
        # hits — that was the original replay corruption bug.
        # world_moved is ground truth from the environment.
        if world_moved:
            self._replay_buffer.append(
                (self._prev_bmu, self._current_bmu,
                 self._current_action, self._current_freq_idx,
                 self._l4_node_at_select)   # L4 node at time of action
            )

        # ── 5. Advance context (AFTER recording) ───────────────
        self._prev_bmu      = self._current_bmu
        self._prev_freq_idx = self._current_freq_idx
        self._prev_node_override = getattr(self, '_current_node_override', None)

        # ── 5b. Record the action that caused this transition ──
        # Only update _transition_action when a real world move happened.
        # This is the action we want to credit in Q_f next step.
        # On wall hits: keep the previous _transition_action intact so
        # Q_f still gets updated with the correct action if this wall hit
        # occurs between two real moves.
        if world_moved:
            self._transition_action = self._current_action

        # ── 6. Diagnostics ─────────────────────────────────────
        td_error = float(rpe)
        self._td_history.append(abs(td_error))
        self._reward_history.append(float(max(0, rpe)))
        self.t += 1

        return {
            'action':          action,
            'q_values':        self._Q_f[self._current_freq_idx].copy()
                               if self._current_freq_idx >= 0
                               else self._Q[self._prev_bmu, bmu_idx].copy(),
            'q_max':           float(self._Q_f[self._current_freq_idx].max())
                               if self._current_freq_idx >= 0
                               else float(self._Q[self._prev_bmu, bmu_idx].max()),
            'epsilon':         self._current_epsilon,
            'explore':         self._n_explorations > 0,
            'td_error':        td_error,
            'q_mean':          float(np.abs(self._Q_f).mean()),
            'q_nonzero_frac':  float((np.abs(self._Q_f) > 1e-4).mean()),
        }

    def replay_on_reward(self, reward: float, familiarity: float = 0.0,
                         food_freq_idx: int = -1,
                         food_node: str = None) -> int:
        """
        Back-propagate a reward through the recent transition buffer.
        Call immediately after a food/significant reward event.

        food_freq_idx: the freq_idx of the node where food was collected.
        Q_f updates in replay are restricted to buffer entries with this
        freq_idx, preventing cross-node credit assignment (e.g. E★ food
        crediting K★ transitions that happen to be in the buffer).

        food_node: the ground-truth node string where food was collected
        (e.g. 'K' or 'E'). When provided, all Q_n replay credits go to
        this specific node, regardless of what L4 believed at the time.
        This is the key fix for aliased food nodes: E and K share fi=4 so
        Q_f cannot distinguish them, but the harness knows ground-truth which
        node was visited and can direct Q_n['K'] credit correctly.
        When None, falls back to using _l4_node_at_select from the buffer.

        The replay bonus scales with (1 - familiarity_at_reward_node):
          - High familiarity node (E★, visited often): small amplification
          - Low familiarity node (K★, rare): full back-propagation strength

        Returns the number of transitions replayed.
        """
        if not self._replay_buffer or reward <= 0.0:
            return 0

        self._reward_freq_idx = food_freq_idx

        # Scale replay strength by novelty of the reward node.
        # Novelty bonus: unfamiliar food nodes get stronger replay (more to learn).
        # Minimum floor of 0.15 ensures even frequently-visited food nodes (high
        # familiarity) still get enough replay to build Q_n. Without the floor,
        # K at 31% visit rate has familiarity≈1.0 → replay_strength≈0 → Q_n never builds.
        novelty = 1.0 - float(np.clip(familiarity, 0.0, 1.0))
        replay_strength = float(reward) * max(novelty, 0.15)
        if replay_strength < 1e-4:
            return 0

        transitions = list(self._replay_buffer)
        replayed = 0
        for i, entry in enumerate(reversed(transitions)):
            prev_bmu, curr_bmu, action = entry[0], entry[1], entry[2]
            fi       = entry[3] if len(entry) > 3 else -1
            discount = REPLAY_GAMMA ** i
            q_delta  = REPLAY_ALPHA * replay_strength * discount
            self._Q[prev_bmu, curr_bmu, action] = float(np.clip(
                self._Q[prev_bmu, curr_bmu, action] + q_delta, Q_MIN, Q_MAX
            ))
            if fi >= 0:
                # If the food node has a dedicated override row, use that
                # instead of the shared fi row — prevents H/P write-collision.
                ov_row = self._node_override_row(food_node) if food_node else -1
                target_row = ov_row if ov_row >= 0 else fi
                self._Q_f[target_row, action] = float(np.clip(
                    self._Q_f[target_row, action] + q_delta, Q_MIN, Q_MAX
                ))
            # Q_n is NOT updated from replay — only from direct step RPE in update().
            # Replay buffer entries are steps before food (the path leading to it).
            # Crediting Q_n from replay saturates all actions to +1.0 because
            # every prior step's action gets credit, not just the food-exit action.
            # The direct update in update() handles Q_n correctly.
            replayed += 1
            if discount < 0.01:
                break

        return replayed

    # ── Combined step (convenience) ───────────────────────────
    def step(self,
             bmu_idx:           int,
             rpe:               float,
             focus_entropy:     float = 0.5,
             thought_confidence: float = 0.0,
             freq_idx:          int   = -1,
             world_moved:       bool  = True,
             l4_top_node:       str   = None,
             l4_top_prob:       float = 0.0,
             epsilon_floor:     float = 0.0,
             node_fi_override:  str   = None) -> dict:
        """
        Select action AND update in one call (for Brain-internal use).
        """
        out = self.update(
            bmu_idx=bmu_idx,
            action=self._current_action,
            rpe=rpe,
            world_moved=world_moved,
        )

        next_action = self.select_action(
            bmu_idx=bmu_idx,
            focus_entropy=focus_entropy,
            thought_confidence=thought_confidence,
            freq_idx=freq_idx,
            l4_top_node=l4_top_node,
            l4_top_prob=l4_top_prob,
            epsilon_floor=epsilon_floor,
            node_fi_override=node_fi_override,
        )

        out['action'] = next_action
        return out

    # ── Diagnostics ───────────────────────────────────────────

    def best_action(self, bmu_idx: int, prev_bmu: int = None) -> tuple:
        """
        Returns (best_action, q_value) for state (prev_bmu, bmu_idx).
        If prev_bmu is None, uses the marginal Q (max over all prev contexts).
        Use for policy inspection — does not advance t or select.
        """
        if prev_bmu is not None:
            q = self._Q[prev_bmu, bmu_idx]
        else:
            # Marginal: take the max Q across all previous contexts
            q = self._Q[:, bmu_idx, :].max(axis=0)
        a = int(np.argmax(q))
        return a, float(q[a])

    def policy_table(self, prev_bmu: int = None) -> np.ndarray:
        """
        (64,) array — greedy action for each current BMU state.
        If prev_bmu given, uses that context. Otherwise uses marginal max.
        """
        if prev_bmu is not None:
            return np.argmax(self._Q[prev_bmu], axis=1).astype(np.int32)
        return np.argmax(self._Q.max(axis=0), axis=1).astype(np.int32)

    def q_map(self, prev_bmu: int = None) -> np.ndarray:
        """
        (8, 8) array — max Q value per current BMU, reshaped to cortical grid.
        """
        if prev_bmu is not None:
            return self._Q[prev_bmu].max(axis=1).reshape(8, 8)
        return self._Q.max(axis=0).max(axis=1).reshape(8, 8)

    def exploration_rate(self) -> float:
        """Fraction of steps that were exploratory."""
        total = self._n_explorations + self._n_exploitations
        if total == 0:
            return 1.0
        return float(self._n_explorations / total)

    def get_state(self) -> dict:
        """Full diagnostic snapshot."""
        return {
            't':                 self.t,
            'n_explorations':    self._n_explorations,
            'n_exploitations':   self._n_exploitations,
            'exploration_rate':  self.exploration_rate(),
            'mean_td_error':     float(np.mean(self._td_history)) if self._td_history else 0.0,
            'q_mean':            float(np.abs(self._Q).mean()),
            'q_max':             float(self._Q.max()),
            'q_nonzero_frac':    float((np.abs(self._Q) > 1e-4).mean()),
            'action_counts':     self._action_counts.copy(),
            'Q_snapshot':        self._Q.copy(),
            'e_snapshot':        self._e.copy(),
        }

    def summary(self):
        """Human-readable state summary."""
        s = self.get_state()
        print(f"  ActionLayer (M56) — step {s['t']}")
        print(f"  Explorations:   {s['n_explorations']}  exploitations={s['n_exploitations']}  "
              f"rate={s['exploration_rate']*100:.1f}%")
        print(f"  Mean |TD error|: {s['mean_td_error']:.4f}")
        print(f"  Q mean/max:     {s['q_mean']:.5f} / {s['q_max']:.4f}")
        print(f"  Q nonzero:      {s['q_nonzero_frac']*100:.1f}%")
        print(f"  Action counts:  {dict(enumerate(s['action_counts']))}")
        top3 = np.argsort(self._Q.max(axis=1))[::-1][:3]
        print(f"  Highest-value states: " +
              "  ".join(f"BMU{i}(Q={self._Q[i].max():.3f})" for i in top3))