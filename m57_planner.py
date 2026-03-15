"""
M57: PLANNER — MENTAL SIMULATION AND LOOK-AHEAD PLANNING
=========================================================

WHAT THIS IS
------------
M57 is the deliberative layer of the brain stack. Every layer below it
perceives, predicts, remembers, attends, and reacts. M57 is the first
layer that THINKS BEFORE ACTING.

Instead of asking "what action has worked in the past?" (M56, Q-learning),
M57 asks: "if I take each candidate action, where do I end up, and how
good is that future state?"

This is the difference between reflexive and deliberate behaviour.
A reflex selects the action with the highest historical Q-value.
Planning simulates forward, scores each imagined trajectory, and selects
the action whose simulated future is best.

Biologically this is the prefrontal cortex (PFC) using hippocampal replay
for prospective planning — the "mental time travel" that distinguishes
mammalian cognition from purely reactive systems.


HOW MENTAL SIMULATION WORKS
----------------------------
At each step, M57 runs a 3-step look-ahead tree over all N_ACTIONS:

    Current BMU (s₀)
         │
    ┌────┴────┬────────┬────────┐
   a=0      a=1      a=2      a=3
    │         │         │         │
   s₁[0]    s₁[1]    s₁[2]    s₁[3]     ← L2 predicts next BMU per action
    │         │         │         │
   s₂[0]    s₂[1]    s₂[2]    s₂[3]     ← L2 predicts one step deeper
    │         │         │         │
   s₃[0]    s₃[1]    s₃[2]    s₃[3]     ← final horizon

At each simulated state, M57 scores it using the Valence module's reward
signal (intrinsic: how predictable is this state? → how much does the
system "like" being there?).

Value propagates back up with discount factor γ (GAMMA=0.85):
    V(s₀, a) = r₁ + γ·r₂ + γ²·r₃

The action with the highest discounted value is the PLANNED action.


THE CRITICAL CONSTRAINT: READ-ONLY SIMULATION
----------------------------------------------
The simulation MUST NOT modify any module's internal state.

This is the most important rule for M57. Mental rehearsal is not experience.
If the simulation wrote to M55's W matrix, M54's weights, or L2's P matrix,
the brain would learn from imagined events as strongly as real ones — which
produces hallucination-driven learning and rapid divergence from reality.

M57 reads from:
  - L2's P matrix (sequence predictions) — via top_predictions(), read-only
  - Valence's reward_ema (expected reward baseline) — read-only scalar
  - M55's W matrix (associative weights) — read-only direct lookup

M57 does NOT call:
  - cortex.step()  — would rewrite SOM weights
  - memory.step()  — would write Hebbian associations
  - pred.step()    — would update sequence probabilities
  - valence.step() — would shift the reward baseline

For scoring simulated states, M57 uses a lightweight _score_state() method
that computes an approximation of intrinsic reward from L2's prediction
confidence — entirely from existing read-only data.


HOW M57 INTERACTS WITH M56
---------------------------
M57 does not REPLACE M56. It OVERRIDES M56's action selection when it has
enough confidence to plan.

When planning is reliable (focus_entropy LOW, thought_confidence HIGH):
    M57's planned_action overrides M56's Q-selected action.
    "I know what's coming — deliberate choice."

When planning is unreliable (focus_entropy HIGH, salience spikes):
    M57 defers to M56's greedy Q selection.
    "I don't know where I am — fall back to habit."

This mirrors the biological relationship between PFC deliberation and
basal ganglia habit learning. When the PFC has a clear model of the
current situation, it overrides the habitual (Q-table) response.
When the situation is novel or confusing, habits take over.

The blend weight:
    planning_weight = PLANNING_WEIGHT_BASE
                      * thought_confidence
                      * (1 - focus_entropy)
                      * salience          ← only plan when attending

    final_action = planned_action  if planning_weight > PLANNING_GATE_THRESH
                   else m56_action


SIGNAL MEANINGS (new outputs)
------------------------------
planned_action       int          — best action from look-ahead tree
planning_weight      float [0,1]  — how much planning is trusted this step
sim_values           ndarray(N_ACTIONS,) — simulated value per action
sim_depth            int          — actual depth reached (≤ PLANNING_DEPTH)
planning_active      bool         — True if M57 overrode M56 this step
plan_vs_habit_delta  float        — sim_values[planned] - Q[m56_action]
                                    positive = planning found something better
                                    negative = habit was already optimal


CALL ORDER
----------
M57 runs at step 10 — after M56 (step 9), as the final decision layer.

    9.  action.step(bmu_idx, rpe, …)          ← M56: Q update + habit action
    10. planner.step(bmu_idx, pred, valence,  ← M57: simulate + maybe override
                     m56_out, thought_out,
                     attention_out)

M57 returns a final 'action' key. Brain uses M57's action if planning_active,
otherwise uses M56's action. The environment always consumes Brain's output.


PARAMETERS
----------
PLANNING_DEPTH      = 3      — look-ahead steps (beyond 3: uncertainty compounds)
GAMMA               = 0.85   — future discount (each step worth 85% of previous)
PLANNING_WEIGHT_BASE= 1.0    — scales planning_weight formula
PLANNING_GATE_THRESH= 0.35   — planning_weight must exceed this to override M56
SIM_TOP_K           = 3      — top-K L2 predictions used per simulated step
                               (not full 64-neuron distribution — efficient)
ACTION_BMU_INFLUENCE= 0.20   — how much each action shifts the predicted BMU
                               (actions modulate the trajectory, not determine it)


BIOLOGICAL BASIS
----------------
Hippocampal prospective coding:
  Place cells in hippocampus "pre-play" future trajectories during planning.
  M57's simulated BMU sequences are the cortical analogue of this.

PFC deliberation vs. BG habit:
  PFC (M57 planning) overrides striatal Q-values (M56) when the situation
  is well-understood. Novel/uncertain situations revert to habit (M56).
  The planning_weight gate implements this switching.

Orbitofrontal value signals:
  OFC maintains a model of expected outcomes for each action.
  M57's sim_values are the computational equivalent.

Mental time travel (Tulving, 1985):
  The ability to project oneself forward in time and evaluate future states.
  The defining cognitive capacity M57 implements.


INTERFACE
---------
  from m57_planner import Planner

  planner = Planner(n_actions=4)

  # Standalone (no Brain):
  out = planner.step(
      bmu_idx          = 20,
      pred             = None,    # L2 instance — if None, random simulation
      valence          = None,    # Valence instance — if None, uniform scoring
      m56_action       = 0,       # M56's habit action
      m56_q_values     = None,    # M56's Q values for current state
      thought_confidence = 0.0,
      focus_entropy    = 1.0,
      salience         = 0.0,
  )

  # With Brain (step 10, after M56):
  planner_out = planner.step(
      bmu_idx            = r['bmu_idx'],
      pred               = self.pred,
      valence            = self.valence,
      m56_action         = m56_out['action'],
      m56_q_values       = m56_out['q_values'],
      thought_confidence = r['thought_confidence'],
      focus_entropy      = r['focus_entropy'],
      salience           = r['salience'],
  )

  final_action = planner_out['action']   # use this for environment.step()
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

# ── Planning horizon ──────────────────────────────────────────
# How many steps forward to simulate.
# At 3: tree has 4³ = 64 leaves — trivially cheap.
# Beyond 3: L2 prediction uncertainty compounds fast enough that
# simulated BMUs become unreliable. Keep at 3.
PLANNING_DEPTH = 3

# ── Future discount ───────────────────────────────────────────
# How much to discount rewards further into the future.
# V(s, a) = r₁ + γ·r₂ + γ²·r₃
# At 0.85: 3 steps out is worth 0.85² = 0.72 of immediate reward.
# Too low (< 0.5): only cares about the next step — myopic.
# Too high (> 0.95): overweights uncertain far-future states.
GAMMA = 0.85

# ── Planning gate ─────────────────────────────────────────────
# Minimum planning_weight to override M56's habit action.
# planning_weight = BASE * thought_confidence * (1-focus_entropy) * salience
#
# CALIBRATED TO REAL BRAIN SIGNALS (measured, do not guess):
#   thought_confidence: peaks ~0.18–0.20 in trained Brain
#   focus_entropy:      sits ~0.63–0.67 during stable operation
#   salience:           ~0.10–0.26 during stable operation
#
# At typical trained values (tc=0.18, fe=0.65, sal=0.20):
#   planning_weight ≈ 1.0 × 0.18 × 0.35 × 0.20 = 0.013
# At good conditions (tc=0.20, fe=0.60, sal=0.30):
#   planning_weight ≈ 1.0 × 0.20 × 0.40 × 0.30 = 0.024
# At transitions/high-salience (tc=0.20, fe=0.60, sal=0.50):
#   planning_weight ≈ 1.0 × 0.20 × 0.40 × 0.50 = 0.040
#
# Threshold 0.005: planning engages when all three signals are
# meaningfully above floor simultaneously. Too low = always planning.
# Too high = never planning given real signal magnitudes.
# Rule: threshold should sit at ~30% of typical peak weight.
PLANNING_WEIGHT_BASE  = 1.0
PLANNING_GATE_THRESH  = 0.005

# ── Simulation breadth ────────────────────────────────────────
# Top-K predictions from L2 used at each simulated step.
# K=3 keeps simulation fast while covering the likely next cluster.
# Using all 64 neurons per step would be exact but 64³ = 262,144 paths.
# With K=3: only 3 paths per step, 9 total — practical.
SIM_TOP_K = 3

# ── Action→BMU influence ──────────────────────────────────────
# Actions don't directly select BMUs — the SOM does that.
# But actions modulate which part of the trajectory is taken.
# This parameter controls how strongly an action biases the simulated
# next-step BMU away from L2's pure sequence prediction.
# At 0.20: action contributes 20% weight; L2 contributes 80%.
# Keep ≤ 0.40 — above that, actions dominate over learned sequence
# structure and simulation loses contact with what L2 actually knows.
ACTION_BMU_INFLUENCE = 0.20

# ── Scoring ───────────────────────────────────────────────────
# How to score a simulated state.
# We use L2's prediction confidence as a proxy for "how good is this state?"
# High confidence = the brain knows what's coming = low surprise = high reward.
# This matches the intrinsic reward definition: 1 - prediction_error.
# SCORE_FAMILIARITY_WEIGHT: bonus for landing on a familiar (well-known) BMU.
# Familiar states tend to have lower prediction error → higher intrinsic reward.
SCORE_FAMILIARITY_WEIGHT = 0.20   # small bonus for familiarity
SCORE_ZONE_WEIGHT        = 0.30   # bonus for zones with high historical reward (L3)

# ── Diagnostics ───────────────────────────────────────────────
HISTORY_LEN = 200


# ═══════════════════════════════════════════════════════════════
# GRID UTILITIES
# ═══════════════════════════════════════════════════════════════

def _build_grid_dist_sq():
    dist_sq = np.zeros((N_NEURONS, N_NEURONS), dtype=np.float32)
    for i in range(N_NEURONS):
        ri, ci = divmod(i, GRID_W)
        for j in range(N_NEURONS):
            rj, cj = divmod(j, GRID_W)
            dist_sq[i, j] = (ri - rj)**2 + (ci - cj)**2
    return dist_sq

_GRID_DIST_SQ = _build_grid_dist_sq()

# Action offset table: each action nudges the BMU in a grid direction.
# Default 4-action layout: up, right, down, left (grid offsets).
# This is a DEFAULT — caller can override by subclassing or replacing
# _ACTION_OFFSETS before instantiation.
# The offsets are in (row, col) units on the 8×8 SOM grid.
_DEFAULT_ACTION_OFFSETS = [
    (-1,  0),  # action 0: up    (toward lower-index rows)
    ( 0, +1),  # action 1: right
    (+1,  0),  # action 2: down
    ( 0, -1),  # action 3: left
]


# ═══════════════════════════════════════════════════════════════
# PLANNER
# ═══════════════════════════════════════════════════════════════

class Planner:
    """
    Mental simulation and look-ahead planning for the Brain stack.

    Runs a 3-step look-ahead tree over all candidate actions each step.
    Uses L2's sequence predictions (read-only) to simulate future BMU states.
    Scores simulated trajectories using Valence's reward model.
    Overrides M56's habit action when planning confidence is high enough.

    READ-ONLY with respect to all other modules. Does not write to M54,
    M55, L2, or Valence state. Mental simulation only.

    All Brain-fed parameters default to safe standalone values.
    """

    def __init__(self, n_actions: int = 4,
                 action_offsets: list = None):
        """
        Parameters
        ----------
        n_actions : int
            Number of discrete actions. Must match M56's N_ACTIONS.
        action_offsets : list of (int, int) or None
            (row, col) grid offsets per action — how each action nudges
            the simulated BMU trajectory. If None, uses default 4-direction
            layout (up/right/down/left). Must have len == n_actions.
        """
        self.n_actions = n_actions

        # Action→BMU offset table
        if action_offsets is not None:
            assert len(action_offsets) == n_actions, \
                f"action_offsets length {len(action_offsets)} != n_actions {n_actions}"
            self._action_offsets = list(action_offsets)
        else:
            # Pad or trim defaults to match n_actions
            offsets = list(_DEFAULT_ACTION_OFFSETS)
            while len(offsets) < n_actions:
                # Generate additional offsets as diagonal directions
                row_off = (len(offsets) % 3) - 1
                col_off = (len(offsets) // 3) % 3 - 1
                offsets.append((row_off, col_off))
            self._action_offsets = offsets[:n_actions]

        # ── Planning state ────────────────────────────────────
        self._last_sim_values        = np.zeros(n_actions, dtype=np.float32)
        self._last_planned_action    = 0
        self._last_planning_weight   = 0.0
        self._last_planning_active   = False
        self._last_plan_vs_habit     = 0.0
        self._last_sim_depth         = 0
        self._last_final_action      = 0

        # ── Diagnostics ───────────────────────────────────────
        self._planning_active_history = deque(maxlen=HISTORY_LEN)
        self._planning_weight_history = deque(maxlen=HISTORY_LEN)
        self._sim_value_history       = deque(maxlen=HISTORY_LEN)
        self._plan_vs_habit_history   = deque(maxlen=HISTORY_LEN)
        self._n_planning_overrides    = 0
        self._n_steps                 = 0
        self.t                        = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             bmu_idx:            int,
             pred                     = None,   # L2 SequencePredictor
             valence                  = None,   # Valence instance
             memory                   = None,   # M55 AssociativeMemory (for familiarity scoring)
             m56_action:         int   = 0,
             m56_q_values              = None,   # (N_ACTIONS,) or None
             thought_confidence: float = 0.0,
             focus_entropy:      float = 1.0,
             salience:           float = 0.0,
             l3                       = None,   # L3 ConceptLayer (read-only zone scores)
             ) -> dict:
        """
        One Planner step.

        Parameters
        ----------
        bmu_idx : int
            Current BMU from M54 — starting state for simulation.
        pred : SequencePredictor or None
            L2 instance. Used READ-ONLY (top_predictions only).
            If None, simulation returns uniform scores → defers to M56.
        valence : Valence or None
            V1 instance. Used READ-ONLY (reward_ema scalar only).
            If None, uses flat reward baseline of 0.5.
        memory : AssociativeMemory or None
            M55 instance. Used READ-ONLY (_W matrix only).
            Adds familiarity-weighted scoring bonus.
        m56_action : int
            M56's habit-selected action this step.
        m56_q_values : ndarray (N_ACTIONS,) or None
            M56's Q values for current state — used to compute plan_vs_habit.
        thought_confidence : float [0,1]
            From Thought — how confident is the predictive model.
        focus_entropy : float [0,1]
            From Thought/Attention — 0=focused, 1=diffuse.
        salience : float [0,1]
            From Attention — how salient is the current step.

        Returns
        -------
        dict with keys:
            action               int          — final chosen action (M57 or M56)
            planned_action       int          — M57's preferred action
            planning_weight      float [0,1]  — confidence in planning this step
            planning_active      bool         — True if M57 overrode M56
            sim_values           ndarray      — (N_ACTIONS,) simulated values
            sim_depth            int          — look-ahead depth used
            plan_vs_habit_delta  float        — planned value minus habit value
            t                    int          — step counter
        """
        # ── 1. Compute planning weight ────────────────────────
        # How much should we trust planning vs habit right now?
        # High when: Thought is confident AND attention is focused AND salient.
        planning_weight = float(np.clip(
            PLANNING_WEIGHT_BASE
            * float(thought_confidence)
            * max(0.0, 1.0 - float(focus_entropy))
            * float(salience),
            0.0, 1.0
        ))

        # ── 2. Simulate — only if planning might be used ──────
        # Skip simulation entirely if planning_weight is clearly too low.
        # This avoids wasting cycles when the brain is confused/exploring.
        if planning_weight > (PLANNING_GATE_THRESH * 0.5) and pred is not None:
            sim_values, sim_depth = self._simulate(
                start_bmu = bmu_idx,
                pred      = pred,
                valence   = valence,
                memory    = memory,
                l3        = l3,
            )
        else:
            # No simulation — fall back to M56 entirely
            sim_values = np.zeros(self.n_actions, dtype=np.float32)
            sim_depth  = 0

        # ── 3. Choose planned action ──────────────────────────
        planned_action = int(np.argmax(sim_values)) if sim_depth > 0 else m56_action

        # ── 4. Gate: override M56 or defer? ───────────────────
        planning_active = (planning_weight > PLANNING_GATE_THRESH) and (sim_depth > 0)
        final_action    = planned_action if planning_active else m56_action

        # ── 5. Plan vs habit delta ─────────────────────────────
        # How much better (or worse) is the planned action vs M56's habit?
        # Positive = planning found a better path.
        # Negative = habit was already at the best option.
        if m56_q_values is not None and sim_depth > 0:
            q_arr = np.asarray(m56_q_values, dtype=np.float32)
            habit_q  = float(q_arr[m56_action]) if len(q_arr) > m56_action else 0.0
            planned_v = float(sim_values[planned_action])
            plan_vs_habit = planned_v - habit_q
        else:
            plan_vs_habit = 0.0

        # ── 6. Store state ─────────────────────────────────────
        self._last_sim_values      = sim_values
        self._last_planned_action  = planned_action
        self._last_planning_weight = planning_weight
        self._last_planning_active = planning_active
        self._last_plan_vs_habit   = plan_vs_habit
        self._last_sim_depth       = sim_depth
        self._last_final_action    = final_action

        if planning_active:
            self._n_planning_overrides += 1

        self._planning_active_history.append(int(planning_active))
        self._planning_weight_history.append(planning_weight)
        self._sim_value_history.append(float(np.max(sim_values)))
        self._plan_vs_habit_history.append(plan_vs_habit)

        self._n_steps += 1
        self.t += 1

        return {
            'action':              final_action,
            'planned_action':      planned_action,
            'planning_weight':     planning_weight,
            'planning_active':     planning_active,
            'sim_values':          sim_values,
            'sim_depth':           sim_depth,
            'plan_vs_habit_delta': plan_vs_habit,
            't':                   self.t,
        }

    # ── Simulation engine ─────────────────────────────────────

    def _simulate(self, start_bmu: int, pred, valence, memory, l3=None) -> tuple:
        """
        Run the look-ahead tree. Returns (sim_values, depth_reached).

        For each action, simulates PLANNING_DEPTH steps forward using L2's
        sequence predictions. Scores each trajectory with discounted reward.

        READ-ONLY. Does not modify pred, valence, or memory state.
        """
        reward_baseline = float(valence._reward_ema) if valence is not None else 0.5
        sim_values = np.zeros(self.n_actions, dtype=np.float32)

        for action in range(self.n_actions):
            # Simulate one trajectory for this action
            value       = 0.0
            discount    = 1.0
            current_bmu = start_bmu

            for depth in range(PLANNING_DEPTH):
                # Pass action only on first step; deeper steps follow pure L2
                next_bmu, confidence = self._predict_next_bmu(
                    current_bmu, action if depth == 0 else -1, pred
                )

                # Score this simulated state
                reward = self._score_state(next_bmu, confidence,
                                           reward_baseline, memory, l3)

                # Accumulate discounted value
                value       += discount * reward
                discount    *= GAMMA
                current_bmu  = next_bmu

            sim_values[action] = float(value)

        return sim_values.astype(np.float32), PLANNING_DEPTH

    def _predict_next_bmu(self, bmu: int, action: int, pred) -> tuple:
        """
        Predict the most likely next BMU given current BMU and action.
        Returns (predicted_bmu, confidence).

        If action >= 0 (first step): blend L2's prediction with action offset.
        If action < 0 (deeper steps): use pure L2 prediction.

        READ-ONLY on pred._P.
        """
        if pred is None:
            # No L2 — return a nearby BMU deterministically
            row, col = divmod(bmu, GRID_W)
            if action >= 0 and action < len(self._action_offsets):
                dr, dc = self._action_offsets[action]
                row = int(np.clip(row + dr, 0, GRID_W - 1))
                col = int(np.clip(col + dc, 0, GRID_W - 1))
            return row * GRID_W + col, 0.0

        # Get L2's top predictions for this BMU (read-only)
        top = pred.top_predictions(bmu, k=SIM_TOP_K)

        if not top:
            # No predictions — stay at current BMU
            return bmu, 0.0

        # Build probability distribution over top-K predictions
        top_indices = [idx for idx, _ in top]
        top_scores  = np.array([score for _, score in top], dtype=np.float32)
        top_scores  = top_scores / (top_scores.sum() + 1e-9)

        if action >= 0 and action < len(self._action_offsets):
            # First simulation step: each action selects a different trajectory.
            # Strategy: action i picks the i-th top prediction from L2,
            # biased toward its grid direction.
            # This guarantees different actions → different simulated next BMUs,
            # while still respecting what L2 has actually learned.
            #
            # If we have enough top predictions, pick the one that best aligns
            # with the action's grid direction. Fall back to index-based if needed.
            dr, dc = self._action_offsets[action]
            row, col = divmod(bmu, GRID_W)

            if len(top_indices) >= 2:
                # Score each top prediction by alignment with action direction
                best_idx   = top_indices[0]
                best_score = -999.0
                for cand_bmu in top_indices:
                    cr, cc = divmod(cand_bmu, GRID_W)
                    # Dot product of (candidate direction) with (action direction)
                    diff_r = cr - row
                    diff_c = cc - col
                    alignment = diff_r * dr + diff_c * dc
                    # Tie-break with L2 score (higher is better)
                    l2_score = float(top_scores[top_indices.index(cand_bmu)])
                    combined = alignment * 0.5 + l2_score
                    if combined > best_score:
                        best_score = combined
                        best_idx   = cand_bmu
                predicted_bmu = best_idx
                confidence    = float(top_scores[top_indices.index(best_idx)])
            else:
                # Only one prediction — pick it regardless of action
                predicted_bmu = top_indices[0]
                confidence    = float(top_scores[0])
        else:
            # Deeper steps: pure L2 sequence prediction
            predicted_bmu = top_indices[0]  # most likely next BMU
            confidence    = float(top_scores[0])

        return predicted_bmu, confidence

    def _score_state(self, bmu: int, l2_confidence: float,
                     reward_baseline: float, memory, l3=None) -> float:
        """
        Score a simulated BMU state.

        Uses L2 confidence as a proxy for intrinsic reward.
        Adds familiarity bonus from M55 if available.
        Adds zone reward history bonus from L3 if available — this is the
        spatial map signal: zones with historically higher reward get a
        planning bonus, steering M57 toward food-adjacent zones.

        Returns a value in approximately [0, 1].
        READ-ONLY.
        """
        base_score = float(np.clip(l2_confidence, 0.0, 1.0))

        # Familiarity bonus
        fam_bonus = 0.0
        if memory is not None:
            try:
                row_sum = float(memory._W[bmu].sum())
                fam_bonus = float(np.clip(row_sum / 20.0, 0.0, 1.0))
            except (AttributeError, IndexError):
                fam_bonus = 0.0

        # Zone reward bonus — L3 tells us which zone this BMU belongs to,
        # and the Z matrix tells us which zones historically followed rewarded
        # zones. We use the zone's mean reward history as a bonus signal.
        zone_bonus = 0.0
        if l3 is not None:
            try:
                zone_idx = int(l3._bmu_to_zone[bmu])
                if zone_idx >= 0 and hasattr(l3, '_zone_reward_ema'):
                    zone_bonus = float(np.clip(l3._zone_reward_ema[zone_idx], 0.0, 1.0))
            except (AttributeError, IndexError):
                zone_bonus = 0.0

        score = float(np.clip(
            base_score
            + SCORE_FAMILIARITY_WEIGHT * fam_bonus
            + SCORE_ZONE_WEIGHT        * zone_bonus,
            0.0, 1.0
        ))

        return score

    # ── Convenience accessors ─────────────────────────────────

    def planning_rate(self) -> float:
        """Fraction of steps where M57 overrode M56."""
        if self._n_steps == 0:
            return 0.0
        return float(self._n_planning_overrides / self._n_steps)

    def get_state(self) -> dict:
        """Full diagnostic snapshot."""
        pw_hist  = list(self._planning_weight_history)
        pvh_hist = list(self._plan_vs_habit_history)
        return {
            't':                   self.t,
            'planned_action':      self._last_planned_action,
            'final_action':        self._last_final_action,
            'planning_weight':     self._last_planning_weight,
            'planning_active':     self._last_planning_active,
            'sim_values':          self._last_sim_values.tolist(),
            'sim_depth':           self._last_sim_depth,
            'plan_vs_habit_delta': self._last_plan_vs_habit,
            'planning_rate':       self.planning_rate(),
            'n_overrides':         self._n_planning_overrides,
            'weight_mean':         float(np.mean(pw_hist)) if pw_hist else 0.0,
            'pvh_mean':            float(np.mean(pvh_hist)) if pvh_hist else 0.0,
        }

    def reset(self):
        """Reset all state — use between test conditions."""
        self._last_sim_values        = np.zeros(self.n_actions, dtype=np.float32)
        self._last_planned_action    = 0
        self._last_planning_weight   = 0.0
        self._last_planning_active   = False
        self._last_plan_vs_habit     = 0.0
        self._last_sim_depth         = 0
        self._last_final_action      = 0
        self._planning_active_history.clear()
        self._planning_weight_history.clear()
        self._sim_value_history.clear()
        self._plan_vs_habit_history.clear()
        self._n_planning_overrides   = 0
        self._n_steps                = 0
        self.t                       = 0

    def summary(self):
        """Human-readable state summary."""
        s = self.get_state()
        print(f"  Planner — step {s['t']}")
        print(f"  Planning:   weight={s['planning_weight']:.3f}  "
              f"active={s['planning_active']}  "
              f"rate={s['planning_rate']*100:.1f}%  "
              f"({s['n_overrides']}/{s['t']} overrides)")
        vals = [f"{v:.3f}" for v in s['sim_values']]
        print(f"  Sim values: [{', '.join(vals)}]  depth={s['sim_depth']}")
        print(f"  Action:     planned={s['planned_action']}  "
              f"final={s['final_action']}  "
              f"plan_vs_habit={s['plan_vs_habit_delta']:+.3f}")
        print(f"  History:    weight_mean={s['weight_mean']:.3f}  "
              f"pvh_mean={s['pvh_mean']:+.3f}")