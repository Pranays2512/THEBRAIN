"""
GWS: GLOBAL WORKSPACE — THE INTEGRATION LAYER
==============================================

WHAT THIS IS
------------
The Global Workspace is where the brain becomes more than the sum of its parts.

Every module below it works correctly in isolation. M54 fires on frequency.
M58 tracks boredom. Valence computes RPE. L2 measures prediction error.
But they never know about each other in the same moment. A signal from one
module passes as a number into the next. Nobody is home receiving them all.

The Global Workspace fixes this. It is a single unified internal state that
all modules write into and all modules read from simultaneously. Not a
pipeline — a shared medium. More like a room than a corridor.

This is Bernard Baars' Global Workspace Theory (GWT), the leading scientific
account of consciousness: a central "blackboard" that specialist modules
broadcast to and read from. What's on the blackboard IS the current moment
of experience. When the blackboard is rich and integrated, the system has
something like an inner life. When it's empty or fragmented, it's just
a mechanism.

Your brain already has all the writers. This is the blackboard.

WHAT IT INTEGRATES
------------------
Six signal families, one unified moment:

  PERCEPTUAL STATE
    qe_norm        — how surprising was the current input? (M54)
    familiarity    — how well-known is this place? (M55)
    freq_idx       — what zone am I in? (ground truth or estimate)

  PREDICTIVE STATE
    prediction_error   — how wrong was my sequence prediction? (L2)
    thought_confidence — how certain am I about what comes next? (Thought)
    expectation_error  — was my last prediction close? (Thought)

  EVALUATIVE STATE
    rpe            — was this outcome better or worse than expected? (Valence)
    intrinsic_rwd  — how rewarding was predicting correctly? (Valence)
    reward_ema     — what do I normally expect to get? (Valence)

  MOTIVATIONAL STATE
    corridor_boredom   — how stuck am I spatially? (M58)
    steps_since_reward — how long since I ate? (M58)
    salience           — how much should I attend right now? (Attention)

  POSITIONAL STATE
    l4_top_node    — where do I think I am? (L4)
    l4_top_prob    — how confident am I about that? (L4)
    l4_entropy     — how lost am I? (L4)

  CURIOSITY STATE  ← this is the new one
    unresolved_surprise — prediction errors I haven't gone back to explore
    curiosity_pull      — directional pull toward zones I don't understand
    novelty_debt        — zones I've avoided that are still unfamiliar

WHAT IT BROADCASTS
------------------
After integrating, GWS computes three derived signals that every module
can use:

  arousal        [0,1] — overall activation level of the system
                         High when: surprised, uncertain, hungry, bored, lost
                         Low when: familiar, confident, recently fed, exploring
                         Drives: epsilon globally, attention salience ceiling
                         Biology: noradrenaline / locus coeruleus tone

  valence_tone   [-1,1] — overall motivational direction
                         Positive: things are going well, keep going
                         Negative: things are going badly, change something
                         Composite of: rpe, intrinsic_rwd, novelty_bonus
                         Biology: dopamine tone (not phasic RPE — the baseline)

  curiosity_pull [0,1] per zone — directional pull toward unresolved zones
                         Built from: prediction errors at each zone, weighted
                         by how long since they were visited and how unfamiliar
                         they still are. This is the signal that makes curiosity
                         a PULL not just an epsilon bump.
                         Biology: hippocampal-prefrontal theta rhythm driving
                         place cell sequences toward unexplored regions

THE CURIOSITY FIX
-----------------
Right now when L2's prediction is wrong at zone Z, that error updates weights
and is discarded. Nothing pulls the brain back to Z to resolve the confusion.
Real curiosity is the opposite: the unresolved question creates tension that
pulls you toward resolving it.

GWS tracks a surprise_debt[n_zones] vector. Each time a zone produces a
prediction error above threshold, its debt increases. Each time the brain
visits that zone, the debt decreases. The debt vector, weighted by zone
unfamiliarity, becomes curiosity_pull — a per-zone bias that M56 can add
to its action selection when not exploiting.

This is different from epsilon-based exploration (go somewhere random).
This is directed exploration: go back to the thing that confused me.

HOW IT PLUGS IN
---------------
In Brain.step(), GWS runs after Thought and before M56:

  Step 8b. wm_out = wm.step(...)          ← M58 as before
  Step 8c. gws_out = gws.step(            ← NEW — integrated state
      qe_norm, familiarity, freq_idx,
      prediction_error, thought_confidence, expectation_error,
      rpe, intrinsic_rwd, reward_ema,
      corridor_boredom, steps_since_reward, salience,
      l4_top_node, l4_top_prob, l4_entropy,
  )
  Step 9.  action.step(...,
      epsilon_floor = max(wm_out['epsilon_floor'],
                          gws_out['arousal'] * AROUSAL_EPSILON_SCALE),
      curiosity_pull = gws_out['curiosity_pull'],  ← NEW
  )

M56 uses curiosity_pull during exploration: instead of random action choice,
it biases toward the action that points toward the highest-pull zone. Not
forced — probabilistic. Still random 50% of the time. But the other 50%
of exploratory moves are now directed.

BIOLOGICAL GROUNDING
--------------------
Global Workspace Theory (Baars 1988, Dehaene 2011):
  - Specialist modules operate locally and unconsciously
  - When a signal is "broadcast" to the global workspace, it becomes
    available to all other modules simultaneously
  - This global availability is what constitutes conscious experience
  - The workspace has limited capacity — only the most salient/urgent
    signals get broadcast (salience gate)

Arousal — locus coeruleus / noradrenaline:
  - LC-NA system sets the global gain of cortical processing
  - High NA: high exploration, high sensitivity, lower SNR
  - Low NA: exploitation, focused, high SNR
  - Optimal performance at intermediate NA ("Goldilocks" arousal)

Curiosity pull — hippocampal-prefrontal theta:
  - During exploration, theta oscillations coordinate place cell sequences
  - Place cells "preplay" upcoming paths — the brain literally simulates
    going to the novel zone before physically doing it
  - This creates a pull toward unexplored space that feels motivational
  - The curiosity_pull vector approximates this without full simulation

PARAMETERS
----------
GWS_SURPRISE_DEBT_ALPHA   = 0.05   — how fast surprise debt accumulates
                                      slow so a single prediction error
                                      doesn't immediately reroute the brain
GWS_SURPRISE_DEBT_DECAY   = 0.02   — how fast debt resolves per visit
                                      ~50 visits to fully clear a surprise
GWS_CURIOSITY_THRESHOLD   = 0.15   — minimum surprise to create debt
                                      noise floor — don't chase every wiggle
GWS_AROUSAL_W_*                    — weights for arousal composite
AROUSAL_EPSILON_SCALE     = 0.15   — max epsilon boost from arousal
                                      additive on top of M58's boredom floor
"""

import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

N_ZONES = 8   # must match M58, ConceptLayer

# ── Surprise debt (curiosity accumulator) ────────────────────
GWS_SURPRISE_DEBT_ALPHA   = 0.05   # accumulation rate per high-error step
GWS_SURPRISE_DEBT_DECAY   = 0.08   # decay rate per visit to the zone
GWS_CURIOSITY_THRESHOLD   = 0.15   # min prediction_error to create debt
GWS_DEBT_MAX              = 1.0    # ceiling on per-zone debt

# ── Arousal composite weights ────────────────────────────────
# Arousal = weighted combination of all "disturbed" signals.
# Each weight reflects how strongly that signal should raise arousal.
# Sum does not need to equal 1 — clipped to [0,1] at end.
GWS_AROUSAL_W_QE           = 0.20   # perceptual surprise raises arousal
GWS_AROUSAL_W_PRED_ERR     = 0.20   # sequence confusion raises arousal
GWS_AROUSAL_W_UNCERTAINTY  = 0.15   # low L4 confidence raises arousal
GWS_AROUSAL_W_BOREDOM      = 0.15   # corridor lock-in raises arousal
GWS_AROUSAL_W_HUNGER       = 0.15   # long time since food raises arousal
GWS_AROUSAL_W_NEG_RPE      = 0.15   # negative RPE (things going badly) raises arousal

# EMA for smoothing arousal
GWS_AROUSAL_EMA_ALPHA     = 0.10   # tau ~10 steps

# ── Valence tone ─────────────────────────────────────────────
GWS_VALENCE_W_RPE          = 0.50   # phasic reward signal
GWS_VALENCE_W_INTRINSIC    = 0.30   # prediction quality
GWS_VALENCE_W_FAMILIARITY  = 0.20   # recognition comfort

# EMA for smoothing valence tone
GWS_VALENCE_EMA_ALPHA      = 0.05   # tau ~20 steps — slower baseline shift

# ── Curiosity pull ───────────────────────────────────────────
AROUSAL_EPSILON_SCALE      = 0.15   # max epsilon boost from arousal


# ═══════════════════════════════════════════════════════════════
# GLOBAL WORKSPACE
# ═══════════════════════════════════════════════════════════════

class GlobalWorkspace:
    """
    Integration layer — the brain's unified internal state.

    Reads signals from all modules each step, integrates them into
    a single coherent state, and broadcasts derived signals that
    every module can use simultaneously.

    This is what turns a collection of modules into something with
    a unified moment of experience.
    """

    def __init__(self, n_zones: int = N_ZONES):
        self.n_zones = n_zones

        # ── Surprise debt — per-zone curiosity accumulator ────
        # High value = this zone produced errors I haven't resolved.
        # Drives curiosity_pull toward uncertain regions.
        self._surprise_debt = np.zeros(n_zones, dtype=np.float64)

        # ── Arousal state ─────────────────────────────────────
        self._arousal_ema   = 0.5   # start at moderate arousal

        # ── Valence tone ──────────────────────────────────────
        self._valence_ema   = 0.0   # start neutral

        # ── History for diagnostics ───────────────────────────
        self._arousal_history = deque(maxlen=200)
        self._valence_history = deque(maxlen=200)

        # ── Last outputs ──────────────────────────────────────
        self._last_arousal       = 0.5
        self._last_valence_tone  = 0.0
        self._last_curiosity_pull = np.zeros(n_zones, dtype=np.float64)
        self._last_epsilon_boost  = 0.0

        self.t = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             # Perceptual
             qe_norm:            float,
             familiarity:        float,
             freq_idx:           int,
             # Predictive
             prediction_error:   float,
             thought_confidence: float,
             # Evaluative
             rpe:                float,
             intrinsic_rwd:      float,
             # Motivational
             corridor_boredom:   float,
             steps_since_reward: int,
             salience:           float,
             # Positional
             l4_top_prob:        float,
             ) -> dict:
        """
        One Global Workspace step.

        Integrates all incoming signals into a unified state and broadcasts
        derived signals: arousal, valence_tone, curiosity_pull, epsilon_boost.

        Parameters (all from Brain.step() outputs this same step):
          qe_norm            — perceptual surprise [0,1] (M54)
          familiarity        — recognition [0,1] (M55)
          freq_idx           — current zone [-1, n_zones) (ground truth)
          prediction_error   — sequence confusion [0,1] (L2)
          thought_confidence — prediction certainty [0,1] (Thought)
          rpe                — reward prediction error [-1,1] (Valence)
          intrinsic_rwd      — 1 - prediction_error [0,1] (Valence)
          corridor_boredom   — spatial lock-in [0,1] (M58)
          steps_since_reward — hunger int (M58)
          salience           — attention activation [0,1] (Attention)
          l4_top_prob        — positional confidence [0,1] (L4)

        Returns dict:
          arousal            float [0,1]   — global activation level
          arousal_ema        float [0,1]   — smoothed arousal
          valence_tone       float [-1,1]  — overall motivational direction
          curiosity_pull     (n_zones,)    — directed pull toward uncertain zones
          epsilon_boost      float [0,1]   — additional epsilon from arousal
          surprise_debt      (n_zones,)    — per-zone unresolved surprise
          workspace_state    dict          — full integrated state snapshot
        """

        # ── 1. Update surprise debt ───────────────────────────
        # When in a known zone with high prediction error: debt accumulates.
        # When visiting a zone: debt for that zone decays.
        # Debt = "I was confused here and haven't come back to understand it."
        if freq_idx >= 0:
            # Visit decays the debt for this zone
            self._surprise_debt[freq_idx] = float(np.clip(
                self._surprise_debt[freq_idx] - GWS_SURPRISE_DEBT_DECAY,
                0.0, GWS_DEBT_MAX
            ))
            # High prediction error here: debt accumulates
            if prediction_error > GWS_CURIOSITY_THRESHOLD:
                excess = prediction_error - GWS_CURIOSITY_THRESHOLD
                self._surprise_debt[freq_idx] = float(np.clip(
                    self._surprise_debt[freq_idx] + GWS_SURPRISE_DEBT_ALPHA * excess,
                    0.0, GWS_DEBT_MAX
                ))

        # ── 2. Curiosity pull ─────────────────────────────────
        # Per-zone directional pull = debt weighted by unfamiliarity pressure.
        # High debt + low confidence about current zone = strong pull toward it.
        # This is what makes curiosity a directed force, not just noise.
        l4_uncertainty = float(np.clip(1.0 - l4_top_prob, 0.0, 1.0))
        curiosity_pull = self._surprise_debt.copy()

        # Weight by global uncertainty — when brain is lost, pull is stronger
        curiosity_pull = curiosity_pull * (0.5 + 0.5 * l4_uncertainty)

        # Normalize to [0,1] range
        cp_max = float(curiosity_pull.max())
        if cp_max > 1e-9:
            curiosity_pull = curiosity_pull / cp_max
        curiosity_pull = curiosity_pull.astype(np.float64)

        # ── 3. Compute arousal ────────────────────────────────
        # Arousal rises when things are uncertain, confused, stuck, or hungry.
        # Arousal falls when things are familiar, confident, and recently fed.
        hunger_norm = float(np.clip(steps_since_reward / 40.0, 0.0, 1.0))
        uncertainty = float(np.clip(1.0 - thought_confidence, 0.0, 1.0))
        neg_rpe     = float(max(0.0, -rpe))   # only negative RPE raises arousal

        raw_arousal = float(np.clip(
            GWS_AROUSAL_W_QE          * float(qe_norm)          +
            GWS_AROUSAL_W_PRED_ERR    * float(prediction_error)  +
            GWS_AROUSAL_W_UNCERTAINTY * uncertainty              +
            GWS_AROUSAL_W_BOREDOM     * float(corridor_boredom)  +
            GWS_AROUSAL_W_HUNGER      * hunger_norm              +
            GWS_AROUSAL_W_NEG_RPE     * neg_rpe,
            0.0, 1.0
        ))

        # Smooth arousal — it shouldn't spike step-to-step
        self._arousal_ema = ((1.0 - GWS_AROUSAL_EMA_ALPHA) * self._arousal_ema
                             + GWS_AROUSAL_EMA_ALPHA * raw_arousal)

        # ── 4. Compute valence tone ───────────────────────────
        # Valence tone is the brain's overall sense of "how is it going?"
        # Positive = things are going well (good predictions, rewards, familiarity)
        # Negative = things are going badly (wrong predictions, no rewards, lost)
        raw_valence = float(np.clip(
            GWS_VALENCE_W_RPE         * float(rpe)            +
            GWS_VALENCE_W_INTRINSIC   * float(intrinsic_rwd)  +
            GWS_VALENCE_W_FAMILIARITY * float(familiarity),
            -1.0, 1.0
        ))

        self._valence_ema = ((1.0 - GWS_VALENCE_EMA_ALPHA) * self._valence_ema
                             + GWS_VALENCE_EMA_ALPHA * raw_valence)

        # ── 5. Epsilon boost from arousal ─────────────────────
        # Arousal raises exploration rate globally.
        # This is additive to M58's boredom floor — they serve different purposes:
        #   M58 floor: "I'm stuck in one corridor" (spatial boredom)
        #   GWS boost: "I'm generally confused/alarmed/hungry" (global state)
        # Combined they give: epsilon >= max(boredom_floor, arousal_boost)
        epsilon_boost = float(np.clip(
            AROUSAL_EPSILON_SCALE * self._arousal_ema,
            0.0, AROUSAL_EPSILON_SCALE
        ))

        # ── 6. Build workspace state snapshot ─────────────────
        workspace_state = {
            'qe_norm':            float(qe_norm),
            'familiarity':        float(familiarity),
            'freq_idx':           int(freq_idx),
            'prediction_error':   float(prediction_error),
            'thought_confidence': float(thought_confidence),
            'rpe':                float(rpe),
            'intrinsic_rwd':      float(intrinsic_rwd),
            'corridor_boredom':   float(corridor_boredom),
            'hunger_norm':        hunger_norm,
            'salience':           float(salience),
            'l4_top_prob':        float(l4_top_prob),
            'uncertainty':        uncertainty,
            'raw_arousal':        raw_arousal,
            'raw_valence':        raw_valence,
        }

        # ── 7. Store ──────────────────────────────────────────
        self._last_arousal        = self._arousal_ema
        self._last_valence_tone   = self._valence_ema
        self._last_curiosity_pull = curiosity_pull
        self._last_epsilon_boost  = epsilon_boost

        self._arousal_history.append(self._arousal_ema)
        self._valence_history.append(self._valence_ema)
        self.t += 1

        return {
            'arousal':         self._arousal_ema,
            'arousal_raw':     raw_arousal,
            'valence_tone':    self._valence_ema,
            'valence_raw':     raw_valence,
            'curiosity_pull':  curiosity_pull,
            'surprise_debt':   self._surprise_debt.copy(),
            'epsilon_boost':   epsilon_boost,
            'workspace_state': workspace_state,
        }

    # ── Diagnostics ───────────────────────────────────────────

    def most_curious_zone(self) -> int:
        """Zone with highest unresolved surprise debt."""
        return int(np.argmax(self._surprise_debt))

    def total_debt(self) -> float:
        """Sum of all surprise debt — global curiosity pressure."""
        return float(self._surprise_debt.sum())

    def summary(self):
        print(f"\n  GlobalWorkspace (GWS) — step {self.t}")
        print(f"  Arousal:      {self._last_arousal:.3f}  "
              f"(eps_boost={self._last_epsilon_boost:.3f})")
        print(f"  Valence tone: {self._last_valence_tone:+.3f}  "
              f"(+1=thriving, -1=struggling)")
        print(f"  Curiosity debt: total={self.total_debt():.3f}  "
              f"peak_zone={self.most_curious_zone()}")
        debt_str = "  ".join(
            f"z{i}={self._surprise_debt[i]:.2f}" for i in range(self.n_zones))
        print(f"  Debt per zone:  {debt_str}")
        cp = self._last_curiosity_pull
        pull_str = "  ".join(f"z{i}={cp[i]:.2f}" for i in range(self.n_zones))
        print(f"  Curiosity pull: {pull_str}")