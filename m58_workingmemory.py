"""
M58: WORKING MEMORY — SHORT-TERM TRAJECTORY BUFFER
====================================================

WHAT THIS IS
------------
M58 is the brain's sense of "where have I been recently?"

Every layer below it operates on the current moment. M54 fires on the
current frequency. M55 recognises the current BMU. L2 predicts the next
BMU. M56 looks up Q[prev_bmu, curr_bmu]. None of them can answer: "I've
been in the K corridor for 30 steps — I should try somewhere different."

M58 closes that gap. It stores the last N (freq_idx, action, reward)
tuples and exposes three derived signals:

  zone_recency    — EMA-smoothed visit count per zone (8,) float array.
                    High values = zones visited often in recent steps.
                    Equivalent to a "mental heatmap" of recent territory.

  steps_since_reward — how long since the brain last got food.
                    Drives impatience: if the brain hasn't eaten in a
                    while, it should explore more aggressively.

  corridor_boredom — a [0,1] float: how concentrated is recent activity?
                    0 = diverse zone coverage (curious/exploring)
                    1 = all recent steps in one or two zones (stuck)
                    Computed as Gini coefficient of zone_recency.
                    When high, this raises M56's epsilon floor — the brain
                    gets restless in a familiar corridor.

HOW IT PLUGS IN
---------------
M58 reads (freq_idx, action, reward) from each step — all available in
Brain.step() — and produces corridor_boredom and steps_since_reward.

Brain.step() passes these to M56.select_action() as two new keyword args:
  corridor_boredom    → raises epsilon floor when brain is stuck
  steps_since_reward  → raises epsilon floor when hungry

M56 applies them in select_action() via a simple additive floor:
  boredom_floor  = BOREDOM_EPSILON_SCALE * corridor_boredom
  hunger_floor   = HUNGER_EPSILON_SCALE  * steps_since_reward_norm
  epsilon = max(epsilon, boredom_floor + hunger_floor)

This is the ONLY change to M56 — the working memory signal is injected
via epsilon without touching Q-values, replay, or traces. Safe, reversible.

BIOLOGICAL GROUNDING
--------------------
Working memory in real brains lives in prefrontal cortex (PFC), layer 3
pyramidal neurons with strong recurrent connections and NMDA-dependent
sustained firing. The buffer here is a simplified version: no recurrent
dynamics, just a ring buffer with EMA smoothing.

  zone_recency         — sustained PFC firing for recently visited contexts
  corridor_boredom     — anterior cingulate cortex (ACC) conflict signal
                         (same ACC that drives conflict-floor epsilon in M56,
                         but now driven by trajectory diversity not Q-margin)
  steps_since_reward   — ventral striatum / nucleus accumbens hunger signal:
                         dopamine baseline falls when reward hasn't arrived,
                         increasing sensitivity to novelty and driving
                         exploration of new routes

PARAMETERS
----------
WM_BUFFER_LEN      = 16   — ring buffer length (steps)
                            16 at dt≈1 step covers ~2 round trips on K path.
                            Shorter: too reactive to transient noise.
                            Longer: too slow to detect fresh corridor lock-in.

WM_ZONE_EMA_ALPHA  = 0.15 — smoothing for zone_recency.
                            tau ≈ 6.5 steps. Fast enough to shift between
                            paths, slow enough to ignore single-step noise.

WM_MAX_HUNGER      = 40   — steps_since_reward is normalised to [0,1]
                            by dividing by this value and clipping.
                            K★ path is 6 steps × replay = effective ~8 steps.
                            At 40: brain is "hungry" after ~5 full K cycles
                            with no reward — realistic threshold for route change.

BOREDOM_EPSILON_SCALE = 0.20  — max epsilon boost from corridor_boredom.
                                When fully bored (Gini=1): +0.20 to epsilon.
                                K path's base epsilon ≈ 0.10 → rises to 0.30.
                                Enough to guarantee occasional departure
                                without destroying the K policy.

HUNGER_EPSILON_SCALE  = 0.10  — max epsilon boost from steps_since_reward.
                                Adds on top of boredom — but ONLY when boredom
                                is already above BOREDOM_GATE_THRESH. A hungry
                                brain that is still exploring broadly is not
                                stuck — hunger should not disrupt healthy
                                exploration. Both signals must agree the brain
                                is in a rut before hunger boosts epsilon.
                                Total max boost: +0.30.

BOREDOM_GATE_THRESH   = 0.50  — corridor_boredom must exceed this before
                                boredom floor activates. Prevents noise from
                                spuriously raising epsilon during healthy
                                K-path exploitation. At Gini=0.50 the brain
                                has meaningfully concentrated activity.
"""

import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

WM_BUFFER_LEN         = 24   # raised from 16: 4×4 grid has longer paths (6+ steps)
WM_ZONE_EMA_ALPHA     = 0.15
WM_MAX_HUNGER         = 40
BOREDOM_EPSILON_SCALE = 0.10  # restored slightly: W04-W07 dead-zone needs a real kick
                              # when boredom fires, epsilon should move meaningfully
HUNGER_EPSILON_SCALE  = 0.10
BOREDOM_GATE_THRESH   = 0.52  # FIXED: was 0.80, never fired (boredom sits at 0.56-0.77).
                              # 0.52 activates during dead-zone episodes (boredom>0.56)
                              # but stays silent at 0.38 (healthy broad exploration W08).
                              # Calibrated from the run trace: W04-W07 boredom was 0.72-0.76
                              # — exactly the regime where the brain needed a push out.
N_ZONES               = 8     # must match ConceptLayer / M56 N_ZONES


# ═══════════════════════════════════════════════════════════════
# WORKING MEMORY
# ═══════════════════════════════════════════════════════════════

class WorkingMemory:
    """
    Short-term trajectory buffer — M58.

    Maintains a ring buffer of recent (freq_idx, action, reward) tuples
    and exposes derived signals for M56's epsilon computation.

    Parameters
    ----------
    n_zones   : int  — number of distinct frequency zones (default 8)
    seed      : int  — random seed (not used yet, reserved for future noise)
    """

    def __init__(self, n_zones: int = N_ZONES, seed: int = 42):
        self.n_zones = n_zones

        # Ring buffer — stores (freq_idx, action, reward) per step
        self._buffer = deque(maxlen=WM_BUFFER_LEN)

        # Smoothed zone visit counts — exponential moving average
        # Initialised to uniform so cold-start Gini ≈ 0 (not bored yet)
        self._zone_recency = np.ones(n_zones, dtype=np.float64) / n_zones

        # Steps since last reward
        self._steps_since_reward = 0

        # Step counter
        self.t = 0

        # Last outputs (for diagnostics)
        self._last_corridor_boredom  = 0.0
        self._last_steps_since_reward_norm = 0.0
        self._last_epsilon_floor     = 0.0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             freq_idx : int,
             action   : int,
             reward   : float) -> dict:
        """
        Record one step and compute working-memory signals.

        Parameters
        ----------
        freq_idx : int   — current zone index [0, n_zones) or -1 if unknown
        action   : int   — action taken this step
        reward   : float — reward received this step (0 or positive)

        Returns
        -------
        dict with keys:
          zone_recency         (n_zones,) float — smoothed zone visit EMA
          corridor_boredom     float [0,1] — Gini of recent zone coverage
          steps_since_reward   int — raw step count since last food
          steps_since_reward_norm float [0,1] — normalised hunger signal
          epsilon_floor        float — recommended epsilon floor for M56
        """
        # ── 1. Store in buffer ─────────────────────────────────
        self._buffer.append((freq_idx, action, float(reward)))

        # ── 2. Update zone_recency EMA ─────────────────────────
        # Build a one-hot visit vector for this step.
        # Unknown freq_idx (-1): no zone gets credited — recency decays evenly.
        visit = np.zeros(self.n_zones, dtype=np.float64)
        if 0 <= freq_idx < self.n_zones:
            visit[freq_idx] = 1.0

        self._zone_recency = ((1.0 - WM_ZONE_EMA_ALPHA) * self._zone_recency
                              + WM_ZONE_EMA_ALPHA * visit)

        # Normalise to sum=1 for Gini computation
        z_sum = self._zone_recency.sum()
        if z_sum > 1e-9:
            z_norm = self._zone_recency / z_sum
        else:
            z_norm = np.ones(self.n_zones, dtype=np.float64) / self.n_zones

        # ── 3. Corridor boredom — Gini coefficient ─────────────
        # Gini(x) = (2 * sum_i i*x_sorted_i) / (N * sum_i x_i) - (N+1)/N
        # 0 = perfectly uniform (diverse zone coverage)
        # 1 = all mass in one zone (total corridor lock-in)
        corridor_boredom = float(_gini(z_norm))
        self._last_corridor_boredom = corridor_boredom

        # ── 4. Hunger — steps since last reward ────────────────
        if float(reward) > 0.0:
            self._steps_since_reward = 0
        else:
            self._steps_since_reward += 1

        hunger_norm = float(np.clip(self._steps_since_reward / WM_MAX_HUNGER,
                                    0.0, 1.0))
        self._last_steps_since_reward_norm = hunger_norm

        # ── 5. Compute epsilon floor ───────────────────────────
        # Both signals are gated on boredom > BOREDOM_GATE_THRESH.
        # Rationale: if the brain is exploring broadly (low boredom),
        # it is NOT stuck — it just hasn't found food recently in a
        # large world. Hunger alone should not raise epsilon in that
        # case, which would disrupt healthy exploration with noise.
        # Hunger only amplifies when the brain is already corridor-locked.
        boredom_contribution = 0.0
        hunger_contribution  = 0.0

        if corridor_boredom > BOREDOM_GATE_THRESH:
            # Scale linearly from 0 at threshold to max at 1.0
            scaled = ((corridor_boredom - BOREDOM_GATE_THRESH)
                      / (1.0 - BOREDOM_GATE_THRESH))
            boredom_contribution = BOREDOM_EPSILON_SCALE * float(scaled)
            # Hunger amplifies only when already bored (corridor-locked)
            hunger_contribution  = HUNGER_EPSILON_SCALE * hunger_norm

        epsilon_floor = float(np.clip(boredom_contribution + hunger_contribution,
                                      0.0, BOREDOM_EPSILON_SCALE + HUNGER_EPSILON_SCALE))
        self._last_epsilon_floor = epsilon_floor

        self.t += 1

        return {
            'zone_recency':              self._zone_recency.copy(),
            'corridor_boredom':          corridor_boredom,
            'steps_since_reward':        self._steps_since_reward,
            'steps_since_reward_norm':   hunger_norm,
            'epsilon_floor':             epsilon_floor,
        }

    # ── Diagnostics ───────────────────────────────────────────

    def top_zone(self) -> int:
        """Zone with highest recent visit count."""
        return int(np.argmax(self._zone_recency))

    def zone_diversity(self) -> float:
        """
        Shannon entropy of zone_recency, normalised to [0,1].
        1 = uniform (diverse), 0 = concentrated in one zone.
        Complement of corridor_boredom — different measure, same concept.
        """
        z = self._zone_recency
        z_sum = z.sum()
        if z_sum < 1e-9:
            return 1.0
        p = z / z_sum
        p_pos = p[p > 1e-12]
        H = float(-np.sum(p_pos * np.log(p_pos)))
        H_max = float(np.log(self.n_zones))
        return float(H / H_max) if H_max > 0 else 1.0

    def buffer_snapshot(self) -> list:
        """Return a copy of the recent trajectory buffer."""
        return list(self._buffer)

    def summary(self):
        """Human-readable state summary."""
        print(f"  WorkingMemory (M58) — step {self.t}")
        print(f"  Buffer length:    {len(self._buffer)}/{WM_BUFFER_LEN}")
        print(f"  Top zone:         {self.top_zone()}  "
              f"(recency={self._zone_recency[self.top_zone()]:.3f})")
        print(f"  Zone diversity:   {self.zone_diversity():.3f}  "
              f"(1=diverse, 0=stuck)")
        print(f"  Corridor boredom: {self._last_corridor_boredom:.3f}  "
              f"(Gini coefficient)")
        print(f"  Steps since food: {self._steps_since_reward}  "
              f"(norm={self._last_steps_since_reward_norm:.3f})")
        print(f"  Epsilon floor:    {self._last_epsilon_floor:.3f}")
        zone_str = "  ".join(
            f"z{i}={self._zone_recency[i]:.2f}" for i in range(self.n_zones))
        print(f"  Zone recency:     {zone_str}")


# ═══════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════

def _gini(x: np.ndarray) -> float:
    """
    Gini coefficient of a non-negative array normalised to sum=1.
    Returns 0 for uniform, approaches 1 for fully concentrated.
    """
    if x.sum() < 1e-9:
        return 0.0
    x = np.sort(x)         # ascending
    n = len(x)
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(idx, x)) / (n * x.sum()) - (n + 1.0) / n)