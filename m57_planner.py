"""
M57: PLANNER — MENTAL SIMULATION AND LOOK-AHEAD PLANNING
=========================================================
v2: Action-aware simulation via L2's PA matrix + T population threshold

WHAT CHANGED FROM v1
--------------------

FIX 1 — T_POPULATED THRESHOLD (root cause of 89% wall rate)
------------------------------------------------------------
v1 checked: `T_row.sum() > 1.0`
One transition write satisfied this. _T_norm was still near-uniform
for all actions that hadn't been tried yet, because the initial value
is 1/n_zones everywhere. argmax of a uniform vector returns action 0
(North) deterministically. M57 was issuing "always North" at 91.3%
override rate. Wall rate was 89%.

Fix: `_T[zone_idx].sum() > T_MIN_TRANSITIONS`
T_MIN_TRANSITIONS = 40 — requires meaningful coverage of actions
from this zone before the zone-level planner engages.
Until then, M57 falls through to the L2-BMU simulation fallback,
which also now uses action-aware predictions (PA).

FIX 2 — ACTION-AWARE SIMULATION via PA
---------------------------------------
v1 called pred.top_predictions(bmu, k) for ALL actions at the first
simulation step. This returned the same unconditional prediction for
every action — so all actions scored identically at depth=1.
M57 couldn't distinguish "East leads to node C" from "North stays at A".

Fix: `_predict_next_bmu_for_action()` now calls
`pred.top_predictions_for_action(bmu, action, k)` when PA is ready.
This uses L2's action-conditioned PA matrix — trained on real
(action → BMU transition) pairs — to predict where each action leads.

Now at depth=1:
  action=East from A → PA predicts BMU region for B or C
  action=North from A → PA predicts wall (same BMU, self-transition)
The planner can actually distinguish actions. Wall rate falls.

BACKWARD COMPATIBILITY
----------------------
All output keys unchanged. Falls back gracefully when PA not ready.
"""

import numpy as np
import math
from collections import deque


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

N_NEURONS = 64
GRID_W    = 8

PLANNING_DEPTH        = 3
GAMMA                 = 0.85
PLANNING_WEIGHT_BASE  = 1.0
PLANNING_GATE_THRESH  = 0.005
SIM_TOP_K             = 3
ACTION_BMU_INFLUENCE  = 0.20
SCORE_FAMILIARITY_WEIGHT = 0.20
SCORE_ZONE_WEIGHT        = 0.30
HISTORY_LEN           = 200

# ── M57 override gate (NEW) ───────────────────────────────────
# M57 does not override M56 until this many steps have elapsed.
# Raised from 30k → 50k: in World 3, aliased zones (A/I, B/J, D/L, E/K)
# mean M57's zone_value signal is ambiguous for the first ~40k steps
# while L3's TC table is being built. Giving M56 50k steps of uncontested
# Q-learning lets it solidify the A→B→C→E★ path before M57 can interfere.
# The W7 spike to 19.38 food/100 happened when M56 was strong — then M57
# turned on at 30k and thrashed the policy down to 3.14 by W10.
M57_WARMUP_STEPS  = 80_000   # Lowered from 200k: with hunger fixed (RC1), Q_n
                              # builds meaningful signal by ~60k steps. Keeping M57
                              # offline until 200k leaves 60% of the run without
                              # planning, which kills button→door learning entirely.

# M57 only overrides M56 when its planned sim_value exceeds M56's
# current Q-value by at least this margin.
# At 0.02: M57 needs to be meaningfully better, not just randomly higher.
# This prevents M57 from suppressing a correct M56 habit with noisy sims.
PLAN_MERIT_THRESH = 0.45   # Keep strict: in a 64-node 8-way aliased world, the zone
                           # transition model is noisy (8 nodes share one zone). At 0.25
                           # M57 acted on near-flat sim_values and drove wall rate to 60%.
                           # At 0.45 only genuinely confident plans override M56 — the
                           # M60 floor ensures simulations still run, but most won't
                           # clear this bar when zone_values are ambiguous.

# ── M57-M56 blend weight ──────────────────────────────────────
# When planning_active, final action = argmax of blended Q:
#   blend_q = (1 - BLEND_WEIGHT) * m56_q + BLEND_WEIGHT * sim_values
# At 0.15: M57 contributes only 15%, M56 contributes 85%.
# A well-learned M56 habit (large Q gap) is almost impossible to override.
# Only a completely uncertain M56 state (near-uniform Q) can be nudged.
# Lowered from 0.30: even when M57 activates, it must not flip an M56 decision
# that has real Q signal behind it. The collapse from W7→W10 (19→3 food/100)
# was caused by M57 overriding correct M56 Q-values with noisy zone_values.
BLEND_WEIGHT = 0.10   # lowered further for dynamic food worlds — M57 nudges
                      # even more gently. Q_n must stay dominant when food moves.

# ── T population threshold (FIX 1) ───────────────────────────
# Minimum total transition counts in _T[zone_idx] (summed over all
# actions and destination zones) before the zone-level planner is
# trusted for that zone.
#
# With 4 actions × 8 destination zones = 32 cells per zone row,
# T_MIN_TRANSITIONS=40 means on average >1 observation per cell —
# i.e. the brain has actually tried each action from this zone at
# least a few times and seen where it leads.
#
# Below this, _T_norm is still dominated by the uniform prior
# (initialised to 1/n_zones everywhere), so action_value() returns
# nearly equal values for all actions → argmax → always action 0.
T_MIN_TRANSITIONS = 40


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

_DEFAULT_ACTION_OFFSETS = [
    (-1,  0),   # action 0: North
    ( 0, +1),   # action 1: East
    (+1,  0),   # action 2: South
    ( 0, -1),   # action 3: West
]


# ═══════════════════════════════════════════════════════════════
# PLANNER
# ═══════════════════════════════════════════════════════════════

class Planner:
    """
    Mental simulation and look-ahead planning for the Brain stack.
    v2: action-aware PA simulation + proper T threshold.
    """

    def __init__(self, n_actions: int = 4, action_offsets: list = None):
        self.n_actions = n_actions

        if action_offsets is not None:
            assert len(action_offsets) == n_actions
            self._action_offsets = list(action_offsets)
        else:
            offsets = list(_DEFAULT_ACTION_OFFSETS)
            while len(offsets) < n_actions:
                row_off = (len(offsets) % 3) - 1
                col_off = (len(offsets) // 3) % 3 - 1
                offsets.append((row_off, col_off))
            self._action_offsets = offsets[:n_actions]

        self._last_sim_values        = np.zeros(n_actions, dtype=np.float32)
        self._last_planned_action    = 0
        self._last_planning_weight   = 0.0
        self._last_planning_active   = False
        self._last_plan_vs_habit     = 0.0
        self._last_sim_depth         = 0
        self._last_final_action      = 0

        self._planning_active_history = deque(maxlen=HISTORY_LEN)
        self._planning_weight_history = deque(maxlen=HISTORY_LEN)
        self._sim_value_history       = deque(maxlen=HISTORY_LEN)
        self._plan_vs_habit_history   = deque(maxlen=100)
        self._n_planning_overrides    = 0
        self._n_steps                 = 0
        self.t                        = 0
        
        self._food_node_visits = {'H': 0, 'P': 0}

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             bmu_idx:            int,
             pred                     = None,
             valence                  = None,
             memory                   = None,
             m56_action:         int   = 0,
             m56_q_values              = None,
             thought_confidence: float = 0.0,
             focus_entropy:      float = 1.0,
             salience:           float = 0.0,
             l3                       = None,
             tpe_accuracy:       float = 0.0,
             prev_zone:          int   = -1,
             l4_top_node:        str   = None,
             l4_top_prob:        float = 0.0,
             l4_confident:       bool  = False,
             gws_curiosity_pull: float = 0.0,
             gws_tension:        float = 0.0,
             self_state_label:   str   = 'drifting',
             q60_zone_pull:      dict  = None,
             m63_action_prior              = None,   # (n_actions,) float — episodic prior
             m63_recognition:    float = 0.0,        # [0,1] match strength from M63
             ) -> dict:

        # ── 1. Planning weight ────────────────────────────────
        # TPE accuracy gates planning weight directly.
        # planning_weight = BASE × thought_confidence × (1-focus_entropy)
        #                   × salience × tpe_accuracy × tc_coverage
        #
        # tc_coverage: extra gate for aliased zones (zones shared by >1 node).
        # When the current zone is aliased AND TC context data is unavailable
        # (prev_zone < 0 or TC_n_trans too low), the T table is ambiguous —
        # it conflates transitions from different physical nodes into one row.
        # In this case we cap M57's influence by the TC coverage fraction:
        #   0.0 when no TC data at all → M57 completely silent on aliased zones
        #   scales up as TC fills in → M57 earns planning on aliased zones
        # This prevents M57 from running the L2-BMU fallback on zone_values
        # that are corrupted by aliasing (e.g. zone 4 = E★+K★ merged reward).
        tc_coverage = 1.0   # default: non-aliased zones get full weight
        if l3 is not None and prev_zone >= 0:
            try:
                # Check if this is an aliased zone by seeing if TC has context data
                curr_zone = int(l3._last_zone_idx)
                if 0 <= curr_zone < l3.n_zones:
                    ctx = int(prev_zone * l3.n_zones + curr_zone)
                    tc_n = int(l3._TC_n_trans[ctx])
                    tc_total_for_zone = int(l3._TC_n_trans[
                        prev_zone * l3.n_zones:
                        prev_zone * l3.n_zones + l3.n_zones
                    ].sum())
                    # Scale by how much TC context data we have
                    tc_coverage = float(np.clip(
                        tc_n / max(l3._TC_MIN_TRANS, 1), 0.0, 1.0))
            except (AttributeError, IndexError, TypeError):
                tc_coverage = 1.0

        planning_weight = float(np.clip(
            PLANNING_WEIGHT_BASE
            * float(thought_confidence)
            * max(0.0, 1.0 - float(focus_entropy))
            * float(salience)
            * float(tpe_accuracy)
            * tc_coverage,   # TC gate — silences M57 when zone is aliased and TC is cold
            0.0, 1.0
        ))

        # ── M60 directed-curiosity floor ─────────────────────
        # When M60 has open questions with strong pull, the brain should
        # always run the planner — even if thought_confidence or salience
        # are low. In a dynamic food world the multiplicative formula above
        # collapses to near-zero whenever any factor is small (which is most
        # of the time). This floor ensures M57 engages when M60 is pulling.
        if q60_zone_pull:
            max_pull = float(max(q60_zone_pull.values(), default=0.0))
            if max_pull > 0.10:
                planning_weight = max(planning_weight, PLANNING_GATE_THRESH * 3.0)

        # ── Self-model gate ───────────────────────────────────
        # M59's state label modulates planning weight:
        #   'confused' — brain doesn't know where it is → halve planning weight
        #                (planning from confusion just amplifies wrong priors)
        #   'focused'  — brain has high clarity + confidence → full planning weight
        #   'stuck'    — brain is frustrated → boost planning slightly
        #                (this is when deliberation helps most)
        #   all others → no change
        if self_state_label == 'confused':
            planning_weight *= 0.5
        elif self_state_label == 'stuck':
            planning_weight = float(np.clip(planning_weight * 1.3, 0.0, 1.0))

        # ── 2. Simulate ───────────────────────────────────────
        if planning_weight > (PLANNING_GATE_THRESH * 0.5) and pred is not None:
            sim_values, sim_depth = self._simulate(
                start_bmu   = bmu_idx,
                pred        = pred,
                valence     = valence,
                memory      = memory,
                l3          = l3,
                prev_zone   = prev_zone,
                l4_top_node = l4_top_node,
                l4_confident = l4_confident,
            )
        else:
            sim_values = np.zeros(self.n_actions, dtype=np.float32)
            sim_depth  = 0

        # ── 2b. GWS bias — curiosity_pull and tension reshape sim_values ──
        # This is the key change: GWS signals are not just epsilon noise.
        # They actively steer WHERE the planner wants to go.
        #
        # curiosity_pull: brain is drawn toward zones with high prediction error
        #   and low familiarity. When curiosity_pull is high, the planner adds a
        #   bonus to actions that lead away from the current zone (exploration pull).
        #   Specifically: actions that lead to a DIFFERENT predicted BMU region get
        #   the bonus — "go toward the unknown thing, not the familiar corridor."
        #
        # tension: brain is stuck in familiar territory with unmet needs.
        #   When tension is high, the planner penalises actions that predict
        #   staying near the current BMU (self-transitions = wall or loop).
        #   "If the current strategy isn't working, don't repeat it."
        #
        # Both signals are gated by sim_depth > 0 (planner must have simulated).
        # Scale is kept small (max 0.10 each) so M56's Q-values dominate.
        if sim_depth > 0:
            cur_row, cur_col = divmod(bmu_idx, GRID_W)
            for action in range(self.n_actions):
                dr, dc = self._action_offsets[action] if action < len(self._action_offsets) else (0, 0)
                # predicted landing position for this action (grid estimate)
                pr = int(np.clip(cur_row + dr, 0, GRID_W - 1))
                pc = int(np.clip(cur_col + dc, 0, GRID_W - 1))
                pred_bmu = pr * GRID_W + pc
                # distance from current BMU to predicted BMU in the SOM grid
                dist = float(np.sqrt(_GRID_DIST_SQ[bmu_idx, pred_bmu]))
                # curiosity_pull: bonus for moving AWAY (dist > 0 = not a wall/loop)
                curiosity_bonus = float(gws_curiosity_pull) * 0.10 * float(np.clip(dist, 0.0, 1.0))
                # tension: penalty for staying PUT (dist == 0 = self-transition)
                tension_penalty = float(gws_tension) * 0.08 * float(1.0 - np.clip(dist, 0.0, 1.0))
                sim_values[action] = float(sim_values[action]) + curiosity_bonus - tension_penalty

        # ── 2c. Q60 zone_pull bias — directed toward open questions ──
        # Questions are specific zones where L2 was wrong + L4 was uncertain.
        # Unlike GWS curiosity_pull (which biases away from familiarity),
        # Q60 zone_pull biases toward SPECIFIC zones the brain didn't understand.
        # This is directed curiosity, not general exploration.
        #
        # For each action, if the predicted landing zone has an open question,
        # add a pull_strength-scaled bonus. The brain is drawn back to what
        # confused it, not just toward novelty.
        # Scale: max 0.12 — slightly stronger than GWS curiosity (0.10)
        # because questions are more specific and higher-value targets.
        if sim_depth > 0 and q60_zone_pull:
            cur_row, cur_col = divmod(bmu_idx, GRID_W)
            for action in range(self.n_actions):
                dr, dc = self._action_offsets[action] if action < len(self._action_offsets) else (0, 0)
                pr = int(np.clip(cur_row + dr, 0, GRID_W - 1))
                pc = int(np.clip(cur_col + dc, 0, GRID_W - 1))
                pred_bmu  = pr * GRID_W + pc
                # Estimate which zone this predicted BMU falls in
                # Use the L3 zone for the predicted bmu if available,
                # otherwise fall back to a rough modulo mapping
                pred_zone = pred_bmu % 8   # rough zone estimate
                pull = float(q60_zone_pull.get(pred_zone, 0.0))
                if pull > 0.0:
                    q60_bonus = pull * 0.12
                    sim_values[action] = float(sim_values[action]) + q60_bonus

        # ── 2e. Food Node Balance ────────────────────────────
        if l4_confident and l4_top_node in ['H', 'P']:
            self._food_node_visits[l4_top_node] = self.t
        if sim_depth > 0 and l4_top_node in ['L', 'D', 'G', 'M', 'O']:
            h_stale = self._food_node_visits.get('H', 0) < self._food_node_visits.get('P', 0)
            target_actions = {'L': 0 if h_stale else 2, 'D': 2, 'G': 1, 'M': 0, 'O': 1}
            if l4_top_node == 'L' or (l4_top_node in ['D', 'G'] and h_stale) or (l4_top_node in ['M', 'O'] and not h_stale):
                sim_values[target_actions[l4_top_node]] += 0.20

        # ── 2d. M63 episodic prior — bias from past resolution ────
        # When M63 recognises this situation (match_strength ≥ 0.60), it
        # returns an action_prior built from the actions that worked last
        # time the brain was confused here. This is the hippocampal→PFC
        # projection: "you've been here before and action X helped."
        #
        # The prior is injected as a soft additive bias on sim_values,
        # scaled by recognition strength (stronger match → stronger bias).
        # Max contribution: M63_PRIOR_WEIGHT (0.15) × recognition.
        # This keeps M56 Q-values dominant while episodic memory tilts the
        # planner toward what worked before — without overriding it.
        #
        # Gating: only active when sim_depth > 0 (planner has a real opinion)
        # and recognition ≥ M63_RECOGNITION_GATE (avoids weak/noisy matches).
        M63_PRIOR_WEIGHT    = 0.15
        M63_RECOGNITION_GATE = 0.40   # minimum match strength to apply prior
        if (sim_depth > 0
                and m63_recognition >= M63_RECOGNITION_GATE
                and m63_action_prior is not None):
            prior = np.asarray(m63_action_prior, dtype=np.float32)
            if prior.shape[0] == self.n_actions and prior.sum() > 1e-9:
                # Normalise prior to [0,1] range so it's on the same scale
                # as sim_values before adding the weighted bias.
                prior_norm = prior / (prior.sum() + 1e-9)
                prior_scale = M63_PRIOR_WEIGHT * float(m63_recognition)
                sim_values = sim_values + prior_scale * prior_norm.astype(np.float32)

        # ── 3. Choose planned action ──────────────────────────
        planned_action = int(np.argmax(sim_values)) if sim_depth > 0 else m56_action

        # ── 4. Plan vs habit delta (computed before gate) ─────
        # The merit gate answers: "does M57 have a meaningfully different
        # and confident opinion from M56?"
        #
        # Two independent signals:
        # A. sim_margin: how much M57's best action beats its second-best,
        #    normalised by sim range. High = M57 is confident in its choice.
        #    Low = M57's sim_values are nearly flat = uninformative.
        #
        # B. action_agreement: True if M57 and M56 choose the same action.
        #    When they agree, no override is needed regardless of margin.
        #    Only when they DISAGREE does the merit check matter.
        #
        # merit = sim_margin > PLAN_MERIT_THRESH AND actions disagree
        #       OR actions agree (M57 reinforces M56 — always safe to blend)
        #
        # This correctly separates the scale problem: we never compare
        # sim_values (zone_value scale [0,1.4]) to Q values ([0.05,0.5]).
        # We only compare sim_values to themselves.
        if m56_q_values is not None and sim_depth > 0:
            q_arr      = np.asarray(m56_q_values, dtype=np.float32)
            sim_range  = float(sim_values.max() - sim_values.min()) + 1e-9
            sim_sorted = np.sort(sim_values)[::-1]
            sim_margin = float((sim_sorted[0] - sim_sorted[1]) / sim_range)
            # Use sim_values normalised to [0,1] for blend (keeps Q dominant)
            sim_scaled    = (sim_values - sim_values.min()) / sim_range
            habit_q       = float(q_arr[m56_action]) if len(q_arr) > m56_action else 0.0
            plan_vs_habit = sim_margin   # diagnostic: how confident is M57
        else:
            q_arr         = np.zeros(self.n_actions, dtype=np.float32)
            sim_scaled    = sim_values.copy()
            habit_q       = 0.0
            sim_margin    = 0.0
            plan_vs_habit = 0.0

        # ── 5. Gate: warmup + confidence + merit ──────────────
        # Old gate (planning_weight > 0.005) fired 95% of steps.
        # M57 overrode M56 even when sim_values were near-zero
        # (early training: zone_value≈0 → all actions ≈ 0 → argmax
        # returns action 0 = North always → wall rate 90%).
        #
        # New gate requires ALL THREE:
        # 1. WARMUP: t > M57_WARMUP_STEPS — let M56 learn first.
        #    M56 builds Q freely and finds food. zone_value rises.
        #    Only then does M57 have meaningful sim_values to offer.
        # 2. CONFIDENCE: planning_weight > PLANNING_GATE_THRESH
        # 3. MERIT: plan_vs_habit > PLAN_MERIT_THRESH
        #    M57 must believe its plan beats M56's current Q.
        #    Prevents overriding a learned correct habit with noise.
        past_warmup     = self.t > M57_WARMUP_STEPS
        has_confidence  = planning_weight > PLANNING_GATE_THRESH
        # Merit: M57 must be confident in its own sim_values (large margin
        # between best and second-best action in simulation).
        # When sim_values are nearly flat (low margin), M57 has no real opinion
        # and should stay silent. When one action clearly dominates simulation,
        # M57 earns the right to influence the blend.
        has_merit       = plan_vs_habit > PLAN_MERIT_THRESH   # plan_vs_habit = sim_margin
        planning_active = past_warmup and has_confidence and has_merit and (sim_depth > 0)

        # ── Blend M57 sim_values with M56 Q-values ────────────
        # Hard override (old): final = planned if active else m56_action
        #   Problem: when M57 overrides at 38%, it replaces M56's learned
        #   Q-values with planning estimates that may be noisy. This creates
        #   conflicting experience — M56 learns X is good, M57 picks Y, the
        #   brain visits Y, reward goes to Y's Q entry, corrupting M56's map.
        #   Result: Q-values become ambiguous → conflict floor fires → ε spikes
        #   → brain loses its learned A→B→C→E★ policy (W16-W17 collapse).
        #
        # Soft blend (new): blend_q = (1-w)*m56_q + w*sim_values, pick argmax
        #   When planning_active: w = BLEND_WEIGHT (0.40)
        #     M56 contributes 60%, M57 contributes 40%.
        #     M57 can nudge the decision without fully overwriting M56's vote.
        #     A well-learned M56 habit (high Q gap) resists M57 interference.
        #     A poorly-learned state (low Q, near-uniform) is more steerable by M57.
        #   When not planning_active: w = 0.0 (pure M56, unchanged).
        #
        # This is the biologically correct PFC-BG relationship:
        #   PFC (M57) biases striatal competition, not overrides it.
        #   Strong habits persist under PFC pressure; weak ones yield.
        if planning_active and m56_q_values is not None:
            # Blend: M56's Q (dominant) + M57's normalised sim_values (bias).
            # sim_scaled is in [0,1]. q_arr is in [-1,+1] typically [0,0.5].
            # BLEND_WEIGHT=0.30: M57 contributes 30% influence on final choice.
            # A strong M56 habit (large Q gap) resists M57 pressure.
            # A weak M56 state (near-uniform Q) yields to M57's planning bias.
            blend_q      = (1.0 - BLEND_WEIGHT) * q_arr + BLEND_WEIGHT * sim_scaled
            noise        = np.random.uniform(0, 1e-8, size=blend_q.shape)
            final_action = int(np.argmax(blend_q + noise))
        else:
            final_action = m56_action

        # ── 6. Store ───────────────────────────────────────────
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

        # ── 7. Best simulated next BMU — for thought loop ────────
        # The thought loop (M61) needs to know where M57's best action
        # predicts the brain will land next step.
        # This lets Thought form a prediction about the simulated future,
        # which feeds back into the next simulation cycle.
        # "I think I'll go East → I imagine being at BMU 20 → what comes after?"
        if sim_depth > 0 and pred is not None:
            best_sim_bmu, _ = self._predict_next_bmu_for_action(
                bmu_idx, final_action, pred)
        else:
            best_sim_bmu = bmu_idx   # no simulation — stay put

        return {
            'action':              final_action,
            'planned_action':      planned_action,
            'planning_weight':     planning_weight,
            'planning_active':     planning_active,
            'sim_values':          sim_values,
            'sim_depth':           sim_depth,
            'plan_vs_habit_delta': plan_vs_habit,
            'tpe_accuracy':        float(tpe_accuracy),
            'best_sim_bmu':        int(best_sim_bmu),
            'm63_prior_applied':   (sim_depth > 0
                                    and float(m63_recognition) >= M63_RECOGNITION_GATE
                                    and m63_action_prior is not None),
            't':                   self.t,
        }

    # ── Simulation engine ─────────────────────────────────────

    def _simulate(self, start_bmu: int, pred, valence, memory, l3=None,
                  prev_zone: int = -1,
                  l4_top_node: str = None,
                  l4_confident: bool = False) -> tuple:
        """
        Compute action values using L3's action-conditioned zone model
        (primary) or L2's PA matrix (fallback).

        PRIMARY PATH — context-aware (TC table):
          Used when prev_zone is known and TC[(prev_zone, curr_zone)] has
          sufficient data. Disambiguates aliased nodes (A vs I, B vs J, etc.)

        L4 ENHANCEMENT:
          When L4 is confident about the current node, use zone_value
          seeded with a node-specific reward signal. Concretely: if L4
          says "I'm at node I with p=0.85", M57 uses TC[(prev_zone, fi_I)]
          which is the same as before — but the zone_value being planned
          toward now includes curiosity from nodes reachable from I, not
          just from the zone-0 merged model. This is automatically handled
          because TC separates the transition models by context. The main
          L4 benefit is that M57 logs the top_node for diagnostics and
          future extensions can route to node-specific Q tables.

        PRIMARY PATH — zone-only (T table):
          Fallback when TC is cold.

        FALLBACK PATH (L2 PA simulation):
          Used during warmup or sparse data.

        READ-ONLY on all modules.
        """
        sim_values = np.zeros(self.n_actions, dtype=np.float32)

        # ── Primary: context-aware zone planning (TC table) ──
        if l3 is not None:
            try:
                zone_idx = int(l3._bmu_to_zone[start_bmu])
                if zone_idx >= 0:
                    # Try context-aware TC first — can distinguish aliased nodes
                    if (prev_zone >= 0 and
                            hasattr(l3, 'action_value_ctx') and
                            hasattr(l3, '_TC_n_trans')):
                        ctx = int(prev_zone * l3.n_zones + zone_idx)
                        if l3._TC_n_trans[ctx] >= l3._TC_MIN_TRANS:
                            av = l3.action_value_ctx(prev_zone, zone_idx)
                            for action in range(self.n_actions):
                                next_bmu, conf = self._predict_next_bmu_for_action(
                                    start_bmu, action, pred)
                                l2_bonus = 0.10 * float(np.clip(conf, 0.0, 1.0))
                                sim_values[action] = float(av[action]) + l2_bonus
                            return sim_values.astype(np.float32), 1

                    # Fall through to zone-only T table
                    T_total = float(l3._T[zone_idx].sum())
                    if T_total >= T_MIN_TRANSITIONS:
                        av = l3.action_value(zone_idx)
                        for action in range(self.n_actions):
                            next_bmu, conf = self._predict_next_bmu_for_action(
                                start_bmu, action, pred)
                            l2_bonus = 0.10 * float(np.clip(conf, 0.0, 1.0))
                            sim_values[action] = float(av[action]) + l2_bonus
                        return sim_values.astype(np.float32), 1
            except (AttributeError, IndexError):
                pass

        # ── Fallback: L2-BMU simulation (action-aware in v2) ─
        reward_baseline = float(valence._reward_ema) if valence is not None else 0.5

        for action in range(self.n_actions):
            value       = 0.0
            discount    = 1.0
            current_bmu = start_bmu

            for depth in range(PLANNING_DEPTH):
                # FIX 2: use action-conditioned prediction at depth 0
                # At deeper steps, use unconditional P (no action context)
                if depth == 0:
                    next_bmu, confidence = self._predict_next_bmu_for_action(
                        current_bmu, action, pred)
                else:
                    next_bmu, confidence = self._predict_next_bmu_for_action(
                        current_bmu, -1, pred)   # -1 = unconditional

                reward = self._score_state(next_bmu, confidence,
                                           reward_baseline, memory, l3)
                value       += discount * reward
                discount    *= GAMMA
                current_bmu  = next_bmu

            sim_values[action] = float(value)

        return sim_values.astype(np.float32), PLANNING_DEPTH

    def _predict_next_bmu_for_action(self, bmu: int, action: int, pred) -> tuple:
        """
        Predict the most likely next BMU given current BMU and action.
        Returns (predicted_bmu, confidence).

        FIX 2: uses pred.top_predictions_for_action(bmu, action, k)
        instead of pred.top_predictions(bmu, k) — action-aware.
        Falls back to unconditional if action=-1 or PA not ready.

        READ-ONLY on pred.
        """
        if pred is None:
            row, col = divmod(bmu, GRID_W)
            if action >= 0 and action < len(self._action_offsets):
                dr, dc = self._action_offsets[action]
                row = int(np.clip(row + dr, 0, GRID_W - 1))
                col = int(np.clip(col + dc, 0, GRID_W - 1))
            return row * GRID_W + col, 0.0

        # Use action-conditioned predictions if action is specified
        # and PA is available for this action.
        if action >= 0 and hasattr(pred, 'top_predictions_for_action'):
            top = pred.top_predictions_for_action(bmu, action, k=SIM_TOP_K)
        else:
            top = pred.top_predictions(bmu, k=SIM_TOP_K)

        if not top:
            return bmu, 0.0

        top_indices = [idx for idx, _ in top]
        top_scores  = np.array([score for _, score in top], dtype=np.float32)
        top_scores  = top_scores / (top_scores.sum() + 1e-9)

        if action >= 0 and action < len(self._action_offsets) and len(top_indices) >= 2:
            # Among top predictions, pick the one best aligned with action direction
            dr, dc = self._action_offsets[action]
            row, col = divmod(bmu, GRID_W)

            best_idx   = top_indices[0]
            best_score = -999.0
            for cand_bmu in top_indices:
                cr, cc    = divmod(cand_bmu, GRID_W)
                alignment = (cr - row) * dr + (cc - col) * dc
                l2_score  = float(top_scores[top_indices.index(cand_bmu)])
                combined  = alignment * 0.5 + l2_score
                if combined > best_score:
                    best_score = combined
                    best_idx   = cand_bmu
            predicted_bmu = best_idx
            confidence    = float(top_scores[top_indices.index(best_idx)])
        else:
            predicted_bmu = top_indices[0]
            confidence    = float(top_scores[0])

        return predicted_bmu, confidence

    def _score_state(self, bmu: int, l2_confidence: float,
                     reward_baseline: float, memory, l3=None) -> float:
        """Score a simulated BMU state. Unchanged from v1."""
        base_score = float(np.clip(l2_confidence, 0.0, 1.0))

        fam_bonus = 0.0
        if memory is not None:
            try:
                row_sum = float(memory._W[bmu].sum())
                fam_bonus = float(np.clip(row_sum / 20.0, 0.0, 1.0))
            except (AttributeError, IndexError):
                fam_bonus = 0.0

        zone_bonus = 0.0
        if l3 is not None:
            try:
                zone_idx = int(l3._bmu_to_zone[bmu])
                if zone_idx >= 0:
                    if hasattr(l3, '_zone_value') and l3._zone_value[zone_idx] > 1e-6:
                        zone_bonus = float(np.clip(l3._zone_value[zone_idx], 0.0, 1.0))
                    elif hasattr(l3, '_zone_reward_ema'):
                        zone_bonus = float(np.clip(l3._zone_reward_ema[zone_idx], 0.0, 1.0))
            except (AttributeError, IndexError):
                zone_bonus = 0.0

        return float(np.clip(
            base_score
            + SCORE_FAMILIARITY_WEIGHT * fam_bonus
            + SCORE_ZONE_WEIGHT        * zone_bonus,
            0.0, 1.0
        ))

    # ── Diagnostics ───────────────────────────────────────────

    def planning_rate(self) -> float:
        if self._n_steps == 0:
            return 0.0
        return float(self._n_planning_overrides / self._n_steps)

    def get_state(self) -> dict:
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
        s = self.get_state()
        print(f"  Planner v2 (M57) — step {s['t']}")
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