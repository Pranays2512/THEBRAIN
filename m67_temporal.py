"""
M67: TEMPORAL CONTEXT — MULTI-SCALE TIME AWARENESS
====================================================

Gives the brain a genuine sense of time at multiple scales.

BIOLOGICAL BASIS
----------------
Real brains have multiple timescale mechanisms:

  Circadian rhythm — slow oscillation (~24h). Biologically: SCN (supra-
    chiasmatic nucleus). Here: a slow sinusoidal phase tied to step count.
    Creates predictable temporal structure the brain can learn to anticipate.

  Inter-reward interval tracking — how long between food events? Rats learn
    to expect food on fixed-interval schedules (FI behavior). Here: EMA of
    gaps between rewards → "expected_interval". Over time the brain learns
    roughly how often food appears and can detect when it's overdue.

  Temporal urgency — if food hasn't arrived in 2× the expected interval, the
    brain is overdue. Maps to dopamine dip / hunger drive.
    Output: overdue_score [0,1].

  Reward rate trend — is the environment getting richer or poorer?
    Compares a short EMA of reward to a long EMA of reward.
    Output: reward_trend [-1, +1].
    +1 = rapidly improving, -1 = deteriorating, 0 = stable.

WHAT THIS IS NOT
----------------
M63 already gives episodic time context (duration of episodes, when events
happened as absolute step numbers). M67 is different: it tracks CONTINUOUS
temporal structure — rhythms, intervals, and temporal expectations — not
discrete episode timing. Think of M63 as the brain's diary vs M67 as its
internal clock.

CALL ORDER
----------
Called after M58 (WorkingMemory) in Brain.step() between steps 8b and 8c.
Inputs: global step counter, steps_since_reward (raw steps from WM),
        reward_ema (from Valence), got_reward (reward > 0 this step).

Outputs → included in brain.step() return dict:
  m67_temporal_phase    : float [0,1]  — slow oscillation phase (0=trough, 1=peak)
  m67_expected_interval : float        — learned expected steps between food events
  m67_overdue_score     : float [0,1]  — how overdue is the next food event?
  m67_temporal_urgency  : float [0,1]  — combined urgency (overdue × phase)
  m67_reward_trend      : float [-1,1] — is reward rate rising (+) or falling (-)?
"""

import numpy as np
import math

# ── Circadian / phase parameters ──────────────────────────────
# Period of 500 steps creates a predictable temporal rhythm.
# At ~1 step/decision, 500 steps ≈ a "session phase" oscillation.
# This gives the brain a sense of "early session" vs "late session"
# even without explicit time labels.
TEMPORAL_PERIOD = 500   # steps per full cycle

# ── Inter-reward interval tracking ───────────────────────────
# EMA alpha for interval estimate. tau ≈ 1/alpha = 7 food events.
# So after ~7 food events the estimate reflects recent experience.
# Slow enough to ignore one-off lucky finds, fast enough to adapt
# if world structure changes.
INTERVAL_EMA_ALPHA = 0.15
INTERVAL_INIT      = 80    # prior: expect food every 80 steps

# When overdue_raw = steps_since_reward / (OVERDUE_SCALE × expected),
# it saturates at 1.0 when we're OVERDUE_SCALE× overdue.
OVERDUE_SCALE = 2.0

# ── Reward trend tracking ─────────────────────────────────────
# Short EMA: tau ≈ 50 steps  → reacts to recent reward density
# Long EMA:  tau ≈ 200 steps → baseline reward rate over session
TREND_SHORT_ALPHA = 0.020
TREND_LONG_ALPHA  = 0.005


class TemporalContext:
    """
    M67: Multi-scale temporal tracking.

    Tracks four temporal signals:
      1. Phase      — slow sinusoidal oscillation (biological clock analog)
      2. Interval   — learned expected gap between food events
      3. Overdue    — how late is the next food, relative to expectation?
      4. Trend      — is the reward rate improving or declining?

    All outputs are [0,1] or [-1,1] — dimensionless, ready to
    feed downstream modules without additional normalization.
    """

    def __init__(self):
        self._expected_interval = float(INTERVAL_INIT)
        self._reward_short_ema  = 0.0
        self._reward_long_ema   = 0.0
        self.t = 0

    def step(self,
             global_step:        int,
             steps_since_reward: float,
             reward_ema:         float,
             got_reward:         bool,
             ) -> dict:
        """
        Update temporal state and emit signals.

        Parameters
        ----------
        global_step        : brain.t (integer step counter, not normalized)
        steps_since_reward : raw steps since last food event (from WM)
        reward_ema         : rolling average reward from Valence
        got_reward         : True if reward > 0 this step
        """
        # ── 1. Circadian-like phase ───────────────────────────
        # Sinusoidal oscillation over TEMPORAL_PERIOD steps.
        # Maps smoothly from 0 (trough) to 1 (peak) and back.
        # Can be learned by downstream modules as a temporal cue:
        # "food tends to appear more often near phase=0.8" etc.
        phase = 0.5 + 0.5 * math.sin(
            2.0 * math.pi * global_step / TEMPORAL_PERIOD
        )

        # ── 2. Inter-reward interval update ──────────────────
        # On food events, update expected interval using the actual gap.
        # Only update if the gap is meaningful (>1 step) to avoid
        # double-counting reward re-triggers in the same area.
        if got_reward and steps_since_reward > 1.0:
            self._expected_interval = (
                (1.0 - INTERVAL_EMA_ALPHA) * self._expected_interval
                + INTERVAL_EMA_ALPHA * float(steps_since_reward)
            )

        # ── 3. Overdue score ─────────────────────────────────
        # Ratio: how many expected intervals have passed without food?
        # Saturates at 1.0 when we're OVERDUE_SCALE times overdue.
        overdue_raw = float(steps_since_reward) / (
            OVERDUE_SCALE * max(self._expected_interval, 1.0)
        )
        overdue_score = float(np.clip(overdue_raw, 0.0, 1.0))

        # ── 4. Temporal urgency ───────────────────────────────
        # Combines "how overdue" with "what phase are we in".
        # Phase = 1 (peak): urgency amplified.
        # Phase = 0 (trough): urgency dampened.
        # Biology: hunger interacts with arousal phase — animals
        # are more motivated to seek food during alert/active phases.
        temporal_urgency = float(np.clip(
            overdue_score * (0.5 + 0.5 * phase),
            0.0, 1.0
        ))

        # ── 5. Reward trend ───────────────────────────────────
        # Short EMA = recent reward density.
        # Long EMA  = baseline reward density.
        # Trend = (short - long) / max → rises when recent > baseline.
        self._reward_short_ema = (
            (1.0 - TREND_SHORT_ALPHA) * self._reward_short_ema
            + TREND_SHORT_ALPHA * float(reward_ema)
        )
        self._reward_long_ema = (
            (1.0 - TREND_LONG_ALPHA) * self._reward_long_ema
            + TREND_LONG_ALPHA * float(reward_ema)
        )
        denom = max(self._reward_short_ema, self._reward_long_ema, 1e-9)
        reward_trend = float(np.clip(
            (self._reward_short_ema - self._reward_long_ema) / denom,
            -1.0, 1.0
        ))

        self.t += 1
        return {
            'm67_temporal_phase':    float(phase),
            'm67_expected_interval': float(self._expected_interval),
            'm67_overdue_score':     overdue_score,
            'm67_temporal_urgency':  temporal_urgency,
            'm67_reward_trend':      reward_trend,
        }
