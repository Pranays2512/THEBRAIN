"""
M66: NEUROMODULATOR SYSTEM
===========================

Models three neuromodulatory systems that tune the brain's own learning
and exploration rates based on moment-to-moment cognitive state.

BIOLOGICAL BASIS
----------------
Real brains don't learn at fixed rates. Four neuromodulators turn the
dials continuously:

  Dopamine  (DA)  — already modeled by V1/Valence as TD error (RPE).
                     Not duplicated here.

  Acetylcholine (ACh) — released by basal forebrain nucleus when
    sensory input is uncertain or novel. Tells cortex: "something
    important just happened — update weights more aggressively."
    Driven by: L4 belief entropy (confused about position?) + qe_norm
    (is current sound unfamiliar?).
    Output → alpha_scale: multiplies ALPHA_ACTOR and ALPHA_NODE.

  Norepinephrine (NE) — released by locus coeruleus on surprise or
    detected threat/opportunity. Causes broad cortical arousal and
    widens action search (more exploration, faster reaction).
    Driven by: L2 sequence surprise + absolute RPE magnitude.
    Output → ne_temp_add: additive boost to softmax temperature.

  Serotonin (5-HT) — released by dorsal raphe nucleus. Tracks tonic
    reward rate. High 5-HT = rich environment = patient (high γ).
    Low 5-HT = scarce environment = impulsive (low γ, grab now).
    Driven by: Valence reward_ema (rolling average reward).
    Output → gamma_scale: scales REPLAY_GAMMA dynamically.

HOW IT HOOKS INTO THE BRAIN
----------------------------
brain.py calls neuromod.step() between Valence (step 6b) and Attention
(step 7). By then all three inputs are available:
  - L4 belief_entropy   → from l4_out (step 5c)
  - surprise_signal     → from feedback deltas (step 6)
  - reward_ema, rpe     → from Valence (step 6b)

The three output scales are then forwarded:
  alpha_scale  → m56_action.step(alpha_scale=...)  [learning rate]
  ne_temp_add  → m56_action.step(ne_temp_add=...)  [exploration]
  gamma_scale  → brain stores; used in replay_on_reward()           [patience]

CALL ORDER (per step, relative to brain.py)
-------------------------------------------
  ... step 6b Valence ...
  6c. neuromod.step()        ← THIS MODULE
  7.  Attention
  8.  Thought
  9.  M56 Action (alpha_scale, ne_temp_add injected)
  10. Planner ...
  [on food event] replay_on_reward(gamma_scale=...)
"""

import numpy as np

# ── ACh parameters ─────────────────────────────────────────────
# Slow EMA: ACh stays elevated through an entire confusing patch
# (~20 steps), not just the one surprise step. Biologically, ACh
# release from the basal forebrain is sustained while the animal
# is engaged in a novel or uncertain situation.
ACH_EMA_ALPHA = 0.05    # τ ≈ 20 steps

# Learning rate multiplier range.
# At ACh=0 (completely familiar, certain): alpha × 0.5  (exploit, slow)
# At ACh=1 (maximally confused):           alpha × 2.0  (rapid plasticity)
ACH_ALPHA_MIN = 0.5
ACH_ALPHA_MAX = 2.0

# Input weights: L4 confusion (where am I?) is primary, perceptual
# novelty (what am I hearing?) is secondary.
ACH_L4_W  = 0.65
ACH_QE_W  = 0.35

# ── NE parameters ──────────────────────────────────────────────
# Fast EMA: NE spikes and recovers within seconds in biology.
# Here: τ ≈ 7 steps → a surprise causes ~7 steps of elevated arousal.
NE_EMA_ALPHA = 0.15

# Maximum additive temperature contribution from NE.
# At NE=1: adds 0.60 to temperature. At base_temp=0.40 (floor),
# this can double effective temperature → much more exploration.
# At NE=0: adds 0 → no effect.
NE_TEMP_MAX = 0.60

# Input weights: sequence surprise (L2 delta) and RPE magnitude
# both signal "something unexpected happened."
NE_SURP_W = 0.60
NE_RPE_W  = 0.40

# ── 5-HT parameters ────────────────────────────────────────────
# Very slow EMA: serotonin tone reflects the environment over hundreds
# of steps, not individual events. τ ≈ 50 steps.
SHT_EMA_ALPHA = 0.02

# Replay gamma (discount) range controlled by serotonin:
# Low 5-HT (scarce env):  REPLAY_GAMMA → SHT_GAMMA_MIN = 0.86  (myopic)
# High 5-HT (rich env):   REPLAY_GAMMA → SHT_GAMMA_MAX = 0.95  (patient)
SHT_GAMMA_MIN = 0.86
SHT_GAMMA_MAX = 0.95

# Reward rate at which 5-HT saturates (normalized to [0,1]).
# reward_ema ≈ 0.08 means ~8 food events per 100 steps — a "rich" world.
SHT_REWARD_SCALE = 0.08


class NeuromodState:
    """
    Tracks three neuromodulatory levels and emits corresponding scale factors.

    All internal levels are EMAs in [0, 1].
    Scale factors are dimensionless multipliers over existing M56 parameters.
    """

    def __init__(self):
        # Start mid-level: brain hasn't seen the world yet.
        # ACh starts moderate (world is new → slightly uncertain).
        # NE starts calm (no surprise yet).
        # 5-HT starts lean (no reward yet → slight impulsivity).
        self.ach_level = 0.50
        self.ne_level  = 0.00
        self.sht_level = 0.20
        self.t         = 0

    def step(self,
             l4_belief_entropy: float,
             qe_norm:           float,
             surprise_signal:   float,
             reward_ema:        float,
             abs_rpe:           float,
             ) -> dict:
        """
        Update all neuromodulator levels and return scale factors.

        Parameters
        ----------
        l4_belief_entropy : float [0, 1]
            L4 position belief entropy. 1.0 = completely uncertain where
            the brain is. 0.0 = perfectly certain of one node.
        qe_norm : float [0, 1]
            M54 quantization error, normalized. High = current sound is
            perceptually unfamiliar to the cortex.
        surprise_signal : float [0, 1]
            Sequence surprise from brain.py feedback deltas — how much
            the current L2 prediction error exceeds its own EMA.
        reward_ema : float >= 0
            Rolling average reward from Valence. Grows with frequent food.
        abs_rpe : float >= 0
            Absolute TD error this step (magnitude of dopamine burst/dip).
        """
        # ── ACh: uncertainty × perceptual novelty ─────────────
        ach_raw = (ACH_L4_W * float(l4_belief_entropy)
                   + ACH_QE_W * float(qe_norm))
        ach_raw = float(np.clip(ach_raw, 0.0, 1.0))
        self.ach_level = ((1.0 - ACH_EMA_ALPHA) * self.ach_level
                          + ACH_EMA_ALPHA * ach_raw)

        # ── NE: sequence surprise + RPE magnitude ─────────────
        ne_raw = (NE_SURP_W * float(surprise_signal)
                  + NE_RPE_W  * float(np.clip(abs_rpe, 0.0, 1.0)))
        ne_raw = float(np.clip(ne_raw, 0.0, 1.0))
        self.ne_level = ((1.0 - NE_EMA_ALPHA) * self.ne_level
                         + NE_EMA_ALPHA * ne_raw)

        # ── 5-HT: tonic reward rate ───────────────────────────
        # reward_ema grows as food events accumulate. Normalize by the
        # "rich world" ceiling so 5-HT saturates at a healthy food rate.
        sht_raw = float(np.clip(
            float(reward_ema) / SHT_REWARD_SCALE,
            0.0, 1.0
        ))
        self.sht_level = ((1.0 - SHT_EMA_ALPHA) * self.sht_level
                          + SHT_EMA_ALPHA * sht_raw)

        # ── Convert levels to scale factors ───────────────────
        # alpha_scale: linear interpolation over [ACH_ALPHA_MIN, ACH_ALPHA_MAX]
        alpha_scale = (ACH_ALPHA_MIN
                       + (ACH_ALPHA_MAX - ACH_ALPHA_MIN) * self.ach_level)

        # ne_temp_add: scales proportionally to NE level
        ne_temp_add = NE_TEMP_MAX * self.ne_level

        # gamma_scale: serotonin slides replay discount between patience bounds
        gamma_scale = (SHT_GAMMA_MIN
                       + (SHT_GAMMA_MAX - SHT_GAMMA_MIN) * self.sht_level)

        self.t += 1
        return {
            'ach_level':   float(self.ach_level),
            'ne_level':    float(self.ne_level),
            'sht_level':   float(self.sht_level),
            'alpha_scale': float(alpha_scale),
            'ne_temp_add': float(ne_temp_add),
            'gamma_scale': float(gamma_scale),
        }
