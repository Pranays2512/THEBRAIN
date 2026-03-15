"""
VALENCE — REWARD PREDICTION ERROR (V1)
=======================================

WHAT THIS IS
------------
Valence is the dopaminergic layer of the brain stack. It models the
ventral tegmental area (VTA) and substantia nigra — the brain's reward
prediction error signal that teaches every other system what is good,
bad, expected, or surprising in a motivational sense.

Every existing module handles EPISTEMIC surprise (was the input or
sequence unexpected?). Valence adds EVALUATIVE surprise:
  "Was the OUTCOME better or worse than I expected?"

This is reward prediction error (RPE):
  rpe = actual_reward − expected_reward

A key difference from all other signals in this stack:
  RPE is SIGNED. It ranges [-1, +1].
  Positive = better than expected (dopamine burst → reinforce).
  Negative = worse than expected (dopamine dip → update, recalibrate).
  Near zero = exactly what was expected (no learning signal).

This is biologically accurate: VTA dopamine neurons fire above baseline
when outcomes exceed predictions, below baseline (pause) when they
are worse than predicted, and at baseline when outcomes match.


WHY INTRINSIC REWARD FIRST
---------------------------
Valence works WITHOUT an external reward signal. By default, it computes
an INTRINSIC reward from what the stack already knows:

    intrinsic_reward = 1.0 - prediction_error

This means: the system intrinsically "likes" correct predictions.
When L2 predicts well, reward is high. When L2 is wrong, reward is low.

This is directly motivated by predictive coding theory — the brain
treats prediction error minimisation as its fundamental objective.
It also means Valence produces meaningful signals from the moment the
stack is running, without needing external reward labels.

When an external reward is provided (reward > 0.0), it is blended with
the intrinsic signal:
    total_reward = W_EXTERNAL * reward + W_INTRINSIC * intrinsic_reward

When reward=0.0 (default), the system runs on pure intrinsic reward.


THE RPE SIGNAL AND THE DELTA RULE
-----------------------------------
RPE is already a delta signal by construction (actual − expected).
It does NOT need an additional EMA baseline subtracted (see Guide Rule 1
and Rule 13). The expected_reward EMA IS the baseline — RPE is the
deviation from it.

This is different from signals like prediction_error (which has a
structural floor of ~0.35 that requires the delta rule). Intrinsic
reward = 1 − prediction_error has a corresponding CEILING of ~0.65 at
steady state. The reward EMA converges to this ceiling, and RPE becomes
near-zero during stable familiar operation. No extra delta needed.

HOWEVER: do not feed raw RPE magnitude to M54 — it has high step-to-step
variance (~0.30 std) even during stable operation, which would inflate
eta permanently. M54 is already well-served by surprise_signal. V1 only
feeds M55.


WHAT V1 FEEDS DOWNSTREAM
--------------------------
pos_rpe → M55 (positive RPE only, clipped to [0,1]):
  "This outcome was better than expected — consolidate this memory."
  Boosts M55's Hebbian write rate on top of the existing curiosity boost.
  High pos_rpe → the current BMU's associations are strengthened more.
  Biologically: dopamine burst → hippocampal LTP (long-term potentiation).
  Formula: eta_effective = ETA_HEBB × (1 + curiosity_boost + rpe_boost)
  where rpe_boost = RPE_M55_BOOST × pos_rpe

V1 does NOT feed M54 (existing surprise_signal handles plasticity),
L2 (already has its own error signal), or Attention (already gated).
V1's signed rpe is available as an output key for M56 (action layer).


CALL ORDER
----------
V1 runs at step 6 — after L2 (needs prediction_error from step 5),
alongside the Brain delta computations, before Attention (step 7).

    6a. compute surprise_signal, curiosity_delta (existing)
    6b. valence.step(prediction_error, reward)         ← NEW
    6c. store rpe_positive for NEXT step's M55 call

V1's output rpe_positive is stored at step t and fed to M55 at step t+1,
exactly like curiosity_delta. Feeding it same-step would mean M55 learns
from a reward signal computed from the outcome it is currently recording —
a temporal loop. Next-step feeding is correct.


BIOLOGICAL BASIS
----------------
VTA/SNc dopamine system:
  - Fires above baseline when reward exceeds prediction (RPE > 0)
  - Pauses below baseline when reward is less than predicted (RPE < 0)
  - Fires at baseline when reward matches prediction (RPE ≈ 0)

Downstream effects modelled:
  - RPE > 0 → hippocampal LTP (M55 write boost) — "remember this"
  - RPE < 0 → cortical plasticity increase (BUT: handled by surprise_signal
    already; we avoid double-counting by not feeding RPE to M54)
  - Signed RPE output → readable by M56 (action layer) for Q-learning


OUTPUTS
-------
rpe              float [-1, 1] — signed reward prediction error (dopamine signal)
pos_rpe          float [0, 1]  — positive RPE only (better than expected)
neg_rpe          float [0, 1]  — |negative RPE| (worse than expected, magnitude)
reward_ema       float [0, 1]  — running expected reward (baseline)
total_reward     float [0, 1]  — blended reward this step
intrinsic_reward float [0, 1]  — 1 - prediction_error this step
t                int           — step counter


INTERFACE
---------
  from valence import Valence

  v1 = Valence()

  # Standalone — every step, after L2:
  result = v1.step(
      prediction_error = l2_out['prediction_error'],
      reward           = 0.0,    # optional external reward [0,1]
  )

  result['rpe']          # signed RPE  [-1, +1]
  result['pos_rpe']      # positive only [0, 1] — feed to M55
  result['neg_rpe']      # magnitude of negative [0, 1] — informational
  result['reward_ema']   # expected reward baseline [0, 1]

  # With Brain (called inside Brain.step() at step 6b):
  v1_out = v1.step(
      prediction_error = raw_error,
      reward           = reward_arg,   # passed into brain.step()
  )
  # Brain stores v1_out['pos_rpe'] → passed to memory.step() next step
"""

import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# ── Reward EMA ───────────────────────────────────────────────
# Running baseline for expected reward.
# tau = 1/alpha steps.
# At 0.05: tau ~20 steps — same timescale as L2 curiosity EMA.
# Tracks reward baseline slowly enough that genuine improvements
# register as positive RPE, not immediately absorbed into baseline.
# Do not go above 0.15 (EMA chases reward too fast, RPE collapses).
# Do not go below 0.02 (EMA too slow, RPE inflates permanently).
RPE_EMA_ALPHA = 0.05

# Cold-start EMA. Set to 0.5 (middle of [0,1]) so RPE on step 1
# is neither strongly positive nor negative.
# At 0.5, cold-start intrinsic_reward ~0.65 → RPE ~+0.15 (small positive).
RPE_EMA_INIT = 0.5

# ── Reward blending ──────────────────────────────────────────
# When external reward is provided (reward > 0.0), blend with intrinsic.
# W_EXTERNAL + W_INTRINSIC should sum to 1.0.
#
# Default 50/50. Caller can tune at instantiation if needed.
# If you set W_EXTERNAL=1.0, W_INTRINSIC=0.0, you get pure RL mode —
# the system only cares about externally labelled rewards.
# If you set W_EXTERNAL=0.0, W_INTRINSIC=1.0, pure predictive coding —
# the system only cares about prediction accuracy.
# Navigation reward and intrinsic reward serve different purposes:
#   External reward = food/wall signal from the environment (navigation RL)
#   Intrinsic reward = 1 - prediction_error (curiosity / predictive coding)
# When external reward is nonzero, use it as the SOLE reward signal for RPE.
# Diagnostic confirmed: blending 0.5*(-0.05_wall) + 0.5*(1.0_intrinsic) = 0.475
# means 49.8% of wall hits generate POSITIVE RPE. The brain was being rewarded
# for wall-bashing because correct prediction of staying put is intrinsically
# rewarding. Separating the channels fixes this completely.
# When reward=0, intrinsic + novelty signal runs normally (exploration drive).
W_EXTERNAL  = 1.0   # weight on external reward when reward != 0
W_INTRINSIC = 0.0   # intrinsic suppressed when external reward is present

# ── M55 RPE boost ────────────────────────────────────────────
# How strongly positive RPE boosts M55's Hebbian write rate.
# formula in M55.step():
#   eta_effective = ETA_HEBB * (1 + curiosity_boost + RPE_M55_BOOST * pos_rpe)
# At pos_rpe=1.0: adds RPE_M55_BOOST to the multiplier.
# At 1.0: maximum triple write rate (curiosity + RPE both maxed).
# At 0.5: modest boost, keeps curiosity_delta as primary driver.
# Keep ≤ 1.0 — above that, RPE dominates over curiosity.
RPE_M55_BOOST = 1.0

# ── Novelty bonus ─────────────────────────────────────────────
# Intrinsic exploration reward proportional to unfamiliarity.
# Biologically: VTA novelty-driven dopamine burst (separate from food DA).
# Formula: novelty_bonus = NOVELTY_BONUS_WEIGHT * (1 - familiarity)
# Added to total_reward before RPE is computed, so it influences the EMA
# and generates a persistent pos_rpe at underexplored nodes.
#
# 0.15 keeps novelty bonus well below food reward (1.0) but above the
# intrinsic reward noise floor (~0.05 step variance). Large enough to
# break the C→E attractor; small enough not to dominate food reward.
NOVELTY_BONUS_WEIGHT = 0.30   # raised from 0.08 — must be large enough to compete
                               # with the C→E food attractor (+1.0 every ~4 steps).
                               # At 0.08 unvisited nodes generate only 0.08 bonus,
                               # completely swamped by frequent food reward. At 0.30
                               # a completely unfamiliar node (fam≈0) generates 0.30
                               # bonus — still 3× below food reward but enough to
                               # create genuine pull toward unvisited regions.

# ── Diagnostics ──────────────────────────────────────────────
HISTORY_LEN = 200


# ═══════════════════════════════════════════════════════════════
# VALENCE
# ═══════════════════════════════════════════════════════════════

class Valence:
    """
    Reward prediction error (dopaminergic) module for the Brain stack.

    Computes RPE from intrinsic reward (1 - prediction_error) and an
    optional external reward signal. Produces pos_rpe for M55 write
    rate modulation and signed rpe for M56 action learning.

    Works standalone — all Brain-fed inputs default to safe values.
    """

    def __init__(self):
        # ── Running baseline ──────────────────────────────────
        self._reward_ema = float(RPE_EMA_INIT)

        # ── One-step-delayed output for Brain to pass to M55 ──
        self._last_pos_rpe = 0.0

        # ── Diagnostics ───────────────────────────────────────
        self._rpe_history            = deque(maxlen=HISTORY_LEN)
        self._reward_history         = deque(maxlen=HISTORY_LEN)
        self._intrinsic_history      = deque(maxlen=HISTORY_LEN)
        self._last_rpe               = 0.0
        self._last_pos_rpe_out       = 0.0
        self._last_neg_rpe           = 0.0
        self._last_total_reward      = 0.0
        self._last_intrinsic_reward  = 0.0
        self.t                       = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             prediction_error: float = 0.0,
             reward:           float = 0.0,
             familiarity:      float = 1.0,
             ) -> dict:
        """
        One Valence step.

        Parameters
        ----------
        prediction_error : float [0, 1]
            L2's prediction error this step. Used to compute intrinsic reward.
            Default 0.0 (perfect prediction — gives max intrinsic reward).
        reward : float [0, 1]
            Optional external reward signal. 0.0 = no external reward (default).
            Caller defines what reward means. Should be normalised to [0, 1].
        familiarity : float [0, 1]
            M55 familiarity for the current BMU. Used to compute novelty bonus.
            Low familiarity → novelty bonus → exploration pressure at sparse nodes.
            Default 1.0 (fully familiar) → no novelty bonus added.

        Returns
        -------
        dict with keys:
            rpe              float [-1, 1] — signed reward prediction error
            pos_rpe          float [0, 1]  — positive RPE (better than expected)
            neg_rpe          float [0, 1]  — |negative RPE| (worse than expected)
            reward_ema       float [0, 1]  — running expected reward baseline
            total_reward     float [0, 1]  — blended reward this step
            intrinsic_reward float [0, 1]  — 1 - prediction_error
            novelty_bonus    float [0, 1]  — unfamiliarity exploration bonus
            t                int           — step counter
        """
        # ── 1. Compute intrinsic reward ───────────────────────
        # "How well did the stack predict this step?"
        intrinsic_reward = float(np.clip(1.0 - prediction_error, 0.0, 1.0))

        # ── 2. Novelty bonus — exploration pressure ───────────
        # Fires whenever familiarity is low, independent of food reward.
        # Fades naturally as the brain visits and learns a node.
        # Biologically: VTA novelty-DA, distinct from food-reward-DA pathway.
        novelty_bonus = float(NOVELTY_BONUS_WEIGHT * (1.0 - float(familiarity)))

        # ── 3. Build total_reward ─────────────────────────────
        # When external reward is nonzero (food or wall penalty), it is the
        # sole signal — intrinsic reward is suppressed entirely.
        # Reason: blending intrinsic (up to 1.0) with wall penalty (-0.05)
        # made 49.8% of wall hits generate positive RPE because the brain
        # correctly predicts it stays put on wall hits (low prediction error →
        # high intrinsic). Walls were being rewarded. Separating the channels
        # makes wall RPE = wall_penalty - reward_ema (strongly negative) and
        # food RPE = food_reward - reward_ema (strongly positive).
        # When reward=0, intrinsic + novelty runs normally (exploration drive).
        if abs(float(reward)) > 1e-9:
            # External navigation signal — use it alone, no intrinsic blending
            total_reward = float(np.clip(float(reward), 0.0, 1.0))
        else:
            # No external signal — pure intrinsic + novelty drive
            total_reward = float(np.clip(
                intrinsic_reward + novelty_bonus,
                0.0, 1.0
            ))
        # ── 4. Compute RPE ────────────────────────────────────
        # RPE = actual − expected.
        # Signed: positive = better than expected, negative = worse.
        # No additional delta rule needed — RPE is already a deviation.
        rpe = float(np.clip(total_reward - self._reward_ema, -1.0, 1.0))

        # ── 4. Split into positive and negative components ────
        pos_rpe = float(max(0.0, rpe))    # better than expected [0, 1]
        neg_rpe = float(max(0.0, -rpe))   # worse than expected, magnitude [0, 1]

        # ── 5. Update reward EMA ──────────────────────────────
        # Updated AFTER computing RPE (same pattern as all EMA-delta pairs).
        # EMA tracks the running expected reward — what the system "knows"
        # it tends to get. RPE is the deviation above/below this baseline.
        self._reward_ema = float(np.clip(
            (1.0 - RPE_EMA_ALPHA) * self._reward_ema
            + RPE_EMA_ALPHA * total_reward,
            0.0, 1.0
        ))

        # ── 6. Store state ────────────────────────────────────
        self._last_pos_rpe          = pos_rpe
        self._last_rpe              = rpe
        self._last_pos_rpe_out      = pos_rpe
        self._last_neg_rpe          = neg_rpe
        self._last_total_reward     = total_reward
        self._last_intrinsic_reward = intrinsic_reward
        self._last_novelty_bonus    = novelty_bonus

        self._rpe_history.append(rpe)
        self._reward_history.append(total_reward)
        self._intrinsic_history.append(intrinsic_reward)

        self.t += 1

        return {
            'rpe':              rpe,
            'pos_rpe':          pos_rpe,
            'neg_rpe':          neg_rpe,
            'reward_ema':       self._reward_ema,
            'total_reward':     total_reward,
            'intrinsic_reward': intrinsic_reward,
            'novelty_bonus':    novelty_bonus,
            't':                self.t,
        }

    # ── Convenience accessors ─────────────────────────────────

    def get_state(self) -> dict:
        """Full diagnostic snapshot."""
        rpe_hist = list(self._rpe_history)
        return {
            't':                self.t,
            'rpe':              self._last_rpe,
            'pos_rpe':          self._last_pos_rpe_out,
            'neg_rpe':          self._last_neg_rpe,
            'reward_ema':       self._reward_ema,
            'total_reward':     self._last_total_reward,
            'intrinsic_reward': self._last_intrinsic_reward,
            'rpe_mean':         float(np.mean(rpe_hist)) if rpe_hist else 0.0,
            'rpe_std':          float(np.std(rpe_hist))  if rpe_hist else 0.0,
        }

    def reset(self):
        """Reset all state — use between test conditions."""
        self._reward_ema            = float(RPE_EMA_INIT)
        self._last_pos_rpe          = 0.0
        self._last_rpe              = 0.0
        self._last_pos_rpe_out      = 0.0
        self._last_neg_rpe          = 0.0
        self._last_total_reward     = 0.0
        self._last_intrinsic_reward = 0.0
        self._rpe_history.clear()
        self._reward_history.clear()
        self._intrinsic_history.clear()
        self.t = 0

    def summary(self):
        """Human-readable state summary."""
        s = self.get_state()
        print(f"  Valence — step {s['t']}")
        print(f"  RPE:          {s['rpe']:+.4f}  "
              f"(pos={s['pos_rpe']:.4f}  neg={s['neg_rpe']:.4f})")
        print(f"  Reward:       total={s['total_reward']:.4f}  "
              f"intrinsic={s['intrinsic_reward']:.4f}  "
              f"ema={s['reward_ema']:.4f}")
        print(f"  RPE history:  mean={s['rpe_mean']:+.4f}  std={s['rpe_std']:.4f}")