"""
M62: CONSISTENCY CHECKER — THE BRAIN'S INTERNAL CRITIC
=======================================================

WHAT THIS IS
------------
M62 is the first module that judges the brain's own reasoning.

Before M62, the brain could simulate (M57), dwell on questions (M61),
and remember confusion (M60). But it never checked whether its own
simulations agreed with each other. Two thought-loop cycles could
produce opposite action preferences about the same zone — and the
brain would just act on the latest one, unaware of the contradiction.

M62 closes that gap. After the thought loop (M61) runs, M62 reads
the sequence of sim_values produced across the loop cycles and asks:
  "Do these simulations agree?"

If they agree well: confidence is boosted. The brain trusts its plan.
If they contradict: confidence is penalized. The brain knows it's uncertain.
If one zone triggers both high and low valuations across cycles: that
zone is flagged as a CONTRADICTION ZONE — the brain has identified
something it cannot yet reason about consistently.

THREE THINGS M62 PROVIDES
--------------------------

1. CONSISTENCY SCORE [0, 1]
   How much did M57's simulations agree across the thought loop?
   1.0 = all cycles preferred the same action with similar confidence
   0.0 = cycles disagreed maximally (different actions, similar magnitudes)

   Formula: based on variance of sim_values across loop iterations.
   Low variance = consistent. High variance = inconsistent.

2. CONTRADICTION ZONES — set of zone indices
   A zone is a contradiction zone if:
     - It appeared in M60's open questions
     - Across the thought loop cycles, its sim_value swung by more than
       M62_CONTRADICTION_THRESH (different cycles valued it very differently)
   Contradiction zones are added to the zone_pull in the return dict —
   the brain is drawn to revisit them deliberately, not just randomly.

3. PLANNING CONFIDENCE MODIFIER — float [-0.3, +0.2]
   A signed modifier applied to M57's planning_weight in brain.py:
     Consistent   (+) → boost planning_weight (trust the plan)
     Inconsistent (-) → reduce planning_weight (don't trust the plan)
   This is the first time the brain can modulate its own planning based
   on the quality of its thinking, not just external signals.

HOW IT PLUGS IN
---------------
M62.step() is called inside Brain.step() AFTER the M61 thought loop,
BEFORE the final action is dispatched.

It reads:
  m57_out         — the final M57 output (sim_values, planning_active)
  m61_thought_iters — list of thought_confidence values from each loop cycle
  m61_sim_history  — list of sim_values arrays from each loop cycle
  q60_open_count   — number of open questions (scales sensitivity)
  sm_state_label   — gates: only runs in deliberative states

It writes into Brain return dict:
  m62_consistency       float [0,1]    — inter-simulation agreement
  m62_contradiction_zones set          — zones where sim disagreed most
  m62_plan_modifier     float          — planning_weight modifier
  m62_active            bool           — did M62 fire this step?

MATHEMATICAL BASIS
------------------
Consistency is computed as 1 - normalized variance across the action
dimension of the sim_values history.

For each action a:
  v_a = [sim_values_cycle_i[a] for i in 0..N_cycles]
  variance_a = var(v_a)

consistency = 1 - clip(mean(variance_a) / M62_VAR_SCALE, 0, 1)

Where M62_VAR_SCALE normalizes the variance to [0,1].

For contradictions:
  swing_a = max(v_a) - min(v_a) for each action
  If max(swing_a) > M62_CONTRADICTION_THRESH:
    The best action is being valued very differently across cycles.
    Contradiction detected.

BIOLOGICAL BASIS
----------------
M62 corresponds to anterior cingulate cortex (ACC) conflict monitoring.

The ACC fires when:
  - Two competing responses are simultaneously activated (conflict)
  - An expected outcome doesn't match a predicted one (prediction conflict)

In our architecture:
  - Two thought loop cycles preferring different actions = ACC conflict
  - Contradiction zone = ACC's persistent conflict signal for a location

The planning_confidence modifier corresponds to ACC's role in
adjusting response caution when conflict is detected:
  High ACC signal → slower response, more deliberation
  Low ACC signal  → faster, confident action

The contradiction zone set corresponds to the ACC's role in directing
attention toward unresolved conflicts — the brain keeps returning to
what confused it until the conflict is resolved.

PARAMETERS
----------
M62_VAR_SCALE           = 0.10  — variance normalization.
                                   sim_values are in roughly [0, 1].
                                   Variance of 0.10 = max inconsistency.
                                   Tuned to real M57 sim_value range.

M62_CONTRADICTION_THRESH = 0.30 — swing threshold for contradiction zones.
                                   If best action's value swings by >0.30
                                   across cycles, that's a real contradiction.
                                   Below 0.30 = noise in simulation.

M62_CONFIDENCE_BOOST     = 0.20 — max planning_weight boost for consistent sims.
                                   Keeps boost modest — M56's Q-values still dominate.

M62_CONFIDENCE_PENALTY   = 0.30 — max planning_weight penalty for inconsistent sims.
                                   Stronger than boost: inconsistency should
                                   suppress planning more than consistency boosts it.

M62_MIN_CYCLES           = 2    — minimum thought loop cycles needed to check
                                   consistency. With only 1 cycle there's nothing
                                   to compare. M62 stays silent when cycles < 2.

M62_ACTIVE_LABELS        = {'confused', 'curious', 'stuck', 'hunting', 'alert'}
                                   Same deliberative states as M61.
                                   M62 only runs when M61 ran.
"""

import numpy as np
from collections import deque
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

M62_VAR_SCALE            = 0.10
M62_CONTRADICTION_THRESH = 0.30
M62_CONFIDENCE_BOOST     = 0.20
M62_CONFIDENCE_PENALTY   = 0.30
M62_MIN_CYCLES           = 2
M62_HISTORY_LEN          = 200
M62_ACTIVE_LABELS        = {'confused', 'curious', 'stuck', 'hunting', 'alert'}

# EMA for consistency score — smoothing over recent steps
M62_CONSISTENCY_EMA_ALPHA = 0.20


class ConsistencyChecker:
    """
    M62 — checks whether the brain's own thought loop agrees with itself.

    Reads the history of sim_values produced across M61 thought loop
    cycles and produces a consistency score, contradiction zones, and
    a planning confidence modifier.
    """

    def __init__(self, n_actions: int = 4, seed: int = 42):
        self.n_actions = n_actions

        # Running state
        self._consistency_ema    = 0.5    # start neutral
        self._contradiction_zones: set   = set()
        self._last_plan_modifier = 0.0
        self._last_consistency   = 0.5

        # History for diagnostics
        self._consistency_history = deque(maxlen=M62_HISTORY_LEN)
        self._contradiction_history = deque(maxlen=M62_HISTORY_LEN)

        # Step counter
        self.t = 0

        # Lifetime stats
        self._n_active     = 0
        self._n_consistent = 0   # steps where consistency > 0.7
        self._n_conflict   = 0   # steps where consistency < 0.3

    def step(self,
             sim_history:      List[np.ndarray],
             thought_iters:    List[float],
             q60_open_count:   int,
             q60_zone_pull:    dict,
             sm_state_label:   str,
             planning_active:  bool,
             ) -> dict:
        """
        One M62 step.

        Parameters
        ----------
        sim_history      : list of np.ndarray (n_actions,)
                           sim_values from each thought loop cycle.
                           Empty or length-1 → M62 stays silent.
        thought_iters    : list of float
                           thought_confidence from each loop cycle.
        q60_open_count   : int — how many open questions exist
        q60_zone_pull    : dict — {zone: pull_strength} from M60
        sm_state_label   : str — current self-model label
        planning_active  : bool — did M57 plan this step?

        Returns
        -------
        dict with keys:
          consistency          float [0,1]  — inter-simulation agreement
          consistency_ema      float [0,1]  — smoothed consistency
          contradiction_zones  set          — zones with contradictory sims
          plan_modifier        float        — planning_weight modifier
          active               bool         — did M62 fire?
          n_cycles             int          — number of loop cycles analyzed
        """
        active = False
        consistency      = self._last_consistency
        plan_modifier    = 0.0
        contradiction_zones: set = set()
        n_cycles = len(sim_history)

        # Gate: only run if there were enough thought loop cycles
        # and we're in a deliberative state
        if (n_cycles >= M62_MIN_CYCLES
                and sm_state_label in M62_ACTIVE_LABELS):
            active = True
            self._n_active += 1

            # ── 1. Compute consistency ─────────────────────────
            # Stack sim_values into (n_cycles, n_actions) matrix
            sim_matrix = np.stack(sim_history, axis=0).astype(np.float64)
            # Variance per action across cycles
            per_action_var = np.var(sim_matrix, axis=0)  # (n_actions,)
            mean_var = float(np.mean(per_action_var))
            # Normalize: 0 var = perfectly consistent, VAR_SCALE = maximally inconsistent
            consistency = float(np.clip(
                1.0 - mean_var / M62_VAR_SCALE, 0.0, 1.0))

            # ── 2. Contradiction zones ─────────────────────────
            # Find which action had the biggest swing across cycles
            per_action_swing = np.max(sim_matrix, axis=0) - np.min(sim_matrix, axis=0)
            max_swing = float(np.max(per_action_swing))
            contradicted = max_swing > M62_CONTRADICTION_THRESH

            if contradicted:
                # Add all zones that have open questions as contradiction zones
                # — the brain can't consistently reason about these zones
                for zone in q60_zone_pull:
                    contradiction_zones.add(int(zone))

            self._contradiction_zones = contradiction_zones

            # ── 3. Planning modifier ───────────────────────────
            if consistency >= 0.7:
                # Consistent simulations → boost planning confidence
                # Scale boost with how consistent: 0.7→0, 1.0→full boost
                scale = (consistency - 0.7) / 0.3
                plan_modifier = M62_CONFIDENCE_BOOST * scale
                self._n_consistent += 1
            elif consistency < 0.3:
                # Contradictory simulations → suppress planning
                # Scale penalty with how inconsistent: 0.3→0, 0.0→full penalty
                scale = (0.3 - consistency) / 0.3
                plan_modifier = -M62_CONFIDENCE_PENALTY * scale
                self._n_conflict += 1
            else:
                plan_modifier = 0.0

            # Scale penalty by open question count — more questions = more
            # reason to be cautious when simulations disagree
            if plan_modifier < 0 and q60_open_count > 1:
                question_scale = float(np.clip(q60_open_count / 4.0, 1.0, 2.0))
                plan_modifier *= question_scale

            plan_modifier = float(np.clip(plan_modifier,
                                          -M62_CONFIDENCE_PENALTY,
                                           M62_CONFIDENCE_BOOST))

        # ── EMA update ────────────────────────────────────────
        self._consistency_ema = ((1.0 - M62_CONSISTENCY_EMA_ALPHA) * self._consistency_ema
                                 + M62_CONSISTENCY_EMA_ALPHA * consistency)
        self._last_consistency  = consistency
        self._last_plan_modifier = plan_modifier

        # ── History ───────────────────────────────────────────
        self._consistency_history.append(consistency)
        self._contradiction_history.append(len(contradiction_zones))

        self.t += 1

        return {
            'consistency':         consistency,
            'consistency_ema':     float(self._consistency_ema),
            'contradiction_zones': contradiction_zones,
            'plan_modifier':       plan_modifier,
            'active':              active,
            'n_cycles':            n_cycles,
        }

    # ── Diagnostics ───────────────────────────────────────────

    def consistency_rate(self) -> float:
        """Fraction of active steps where consistency > 0.7."""
        if self._n_active == 0: return 0.0
        return float(self._n_consistent / self._n_active)

    def conflict_rate(self) -> float:
        """Fraction of active steps where consistency < 0.3."""
        if self._n_active == 0: return 0.0
        return float(self._n_conflict / self._n_active)

    def recent_consistency(self, window: int = 20) -> float:
        """Mean consistency over the last `window` steps."""
        if not self._consistency_history: return 0.5
        h = list(self._consistency_history)[-window:]
        return float(np.mean(h))

    def summary(self):
        print(f"  ConsistencyChecker (M62) — step {self.t}")
        print(f"  Active steps:       {self._n_active}")
        print(f"  Consistency EMA:    {self._consistency_ema:.3f}")
        print(f"  Consistent rate:    {self.consistency_rate():.1%}  (>0.7)")
        print(f"  Conflict rate:      {self.conflict_rate():.1%}   (<0.3)")
        print(f"  Last modifier:      {self._last_plan_modifier:+.3f}")
        print(f"  Contradiction zones:{self._contradiction_zones}")
        if self._consistency_history:
            recent = list(self._consistency_history)[-10:]
            vals   = '  '.join(f'{v:.2f}' for v in recent)
            print(f"  Recent (10):        {vals}")

    def get_state(self) -> dict:
        return {
            't':                self.t,
            'consistency_ema':  self._consistency_ema,
            'n_active':         self._n_active,
            'consistency_rate': self.consistency_rate(),
            'conflict_rate':    self.conflict_rate(),
            'last_modifier':    self._last_plan_modifier,
            'contradiction_zones': list(self._contradiction_zones),
        }