"""
GLOBAL WORKSPACE — UNIFIED BRAIN STATE BROADCASTER  (GW1)
==========================================================

The integration layer that was missing.

Every module works in isolation: M58 knows boredom, Valence knows hunger,
L4 knows uncertainty, L2 knows prediction error. But nothing knows all of
this simultaneously. There is no unified "now."

The Global Workspace closes that gap.

Each step it:
  1. COLLECTS signals from every module
  2. INTEGRATES them into five broadcast signals no single module can compute
  3. BROADCASTS epsilon_boost (unified exploration pressure) into M56

FIVE BROADCAST SIGNALS
----------------------
  arousal        [0,1]  — global urgency (boredom + hunger + novelty)
  valence_tone  [-1,1]  — motivational direction (smoothed RPE)
  curiosity_pull [0,1]  — pull toward genuine unknowns (pe × unfamiliarity)
  surprise_debt  [0,1]  — accumulated unresolved prediction error
  epsilon_boost  [0,1]  — unified exploration pressure fed to M56

DERIVED INTEGRATION SIGNALS
----------------------------
  coherence   [0,1]  — are the modules agreeing? (1 - std of knowledge signals)
  tension     [0,1]  — arousal × familiarity (stuck in known place = change strategy)
  readiness   [0,1]  — coherence × confidence × familiarity (safe to plan?)

BIOLOGICAL BASIS — GLOBAL WORKSPACE THEORY (Baars 1988)
---------------------------------------------------------
Conscious experience arises when information is broadcast to a global
workspace all specialist modules access simultaneously.
  - Ignition: salience exceeds threshold → whole workspace lights up
  - Integration: siloed signals combine into one broadcast
Below ignition: modules work in isolation (current state of the brain)
Above ignition: integrated broadcast (what GW1 provides)
"""

import numpy as np
from collections import deque

# ── Parameters ──────────────────────────────────────────────────
GW_AROUSAL_DECAY       = 0.85   # EMA for arousal (tau ~7 steps)
GW_SURPRISE_DEBT_DECAY = 0.90   # EMA for surprise_debt (tau ~10 steps)
GW_VALENCE_DECAY       = 0.80   # EMA for valence_tone (tau ~5 steps)
GW_EPSILON_SCALE       = 0.20   # max epsilon contribution from GW
GW_IGNITION_THRESHOLD  = 0.40   # salience floor for full broadcast
GW_COHERENCE_ALPHA     = 0.15   # EMA smoothing for coherence
GW_MAX_HUNGER_STEPS    = 40     # normalise steps_since_reward to [0,1]
HISTORY_LEN            = 200


class GlobalWorkspace:
    """
    Unified brain state broadcaster — GW1.

    Interface matches brain.py's gws.step() call exactly.
    Parameters match what Brain already passes; outputs match what
    Brain already reads from gws_out.
    """

    def __init__(self, n_zones: int = 8, seed: int = 42):
        self.n_zones = n_zones

        self._arousal       = 0.0
        self._valence_tone  = 0.0
        self._surprise_debt = 0.0
        self._coherence_ema = 0.5

        self._ignited        = False
        self._ignition_count = 0

        self._history = deque(maxlen=HISTORY_LEN)
        self._last    = {}
        self.t        = 0

    def step(self,
             qe_norm:            float,
             familiarity:        float,
             freq_idx:           int,
             prediction_error:   float,
             thought_confidence: float,
             rpe:                float,
             intrinsic_rwd:      float,
             corridor_boredom:   float,
             steps_since_reward: int,
             salience:           float,
             l4_top_prob:        float,
             ) -> dict:
        """
        One GW step. Collects all module signals → five broadcast signals.

        Returns
        -------
        dict: arousal, arousal_raw, valence_tone, valence_raw,
              curiosity_pull, surprise_debt, epsilon_boost,
              coherence, tension, readiness, ignited, ignition_rate
        """
        # Clip all inputs
        fam    = float(np.clip(familiarity,        0.0, 1.0))
        qe     = float(np.clip(qe_norm,            0.0, 1.0))
        pe     = float(np.clip(prediction_error,   0.0, 1.0))
        conf   = float(np.clip(thought_confidence, 0.0, 1.0))
        rpe_   = float(np.clip(rpe,               -1.0, 1.0))
        bored  = float(np.clip(corridor_boredom,   0.0, 1.0))
        sal    = float(np.clip(salience,           0.0, 1.0))
        l4p    = float(np.clip(l4_top_prob,        0.0, 1.0))
        hunger = float(np.clip(
            steps_since_reward / GW_MAX_HUNGER_STEPS, 0.0, 1.0))

        # ── Ignition ──────────────────────────────────────────
        ignited = sal >= GW_IGNITION_THRESHOLD
        ignition_factor = float(np.clip(
            sal / max(GW_IGNITION_THRESHOLD, 1e-6), 0.0, 1.0))
        if ignited:
            self._ignition_count += 1

        # ── 1. Arousal — global urgency ───────────────────────
        # Three "something needs to change" signals combined.
        # All three must be high for peak arousal — no single one dominates.
        arousal_raw = float(np.clip(
            0.40 * bored + 0.35 * hunger + 0.25 * qe, 0.0, 1.0))
        self._arousal = (GW_AROUSAL_DECAY * self._arousal
                         + (1.0 - GW_AROUSAL_DECAY) * arousal_raw)
        arousal = float(self._arousal)

        # ── 2. Valence tone — motivational direction ──────────
        # Smoothed signed RPE: positive = exploit more, negative = escape
        valence_raw = rpe_
        self._valence_tone = (GW_VALENCE_DECAY * self._valence_tone
                              + (1.0 - GW_VALENCE_DECAY) * valence_raw)
        valence_tone = float(self._valence_tone)

        # ── 3. Curiosity pull — genuine unknowns ──────────────
        # High prediction error in unfamiliar territory.
        # Different from boredom (repetition) — this is about the truly unknown.
        curiosity_pull = float(np.clip(pe * (1.0 - fam), 0.0, 1.0))

        # ── 4. Surprise debt — unresolved prediction error ────
        # Accumulates when errors keep occurring; falls as the brain learns.
        self._surprise_debt = (GW_SURPRISE_DEBT_DECAY * self._surprise_debt
                               + (1.0 - GW_SURPRISE_DEBT_DECAY) * pe)
        surprise_debt = float(self._surprise_debt)

        # ── 5. Coherence — inter-module agreement ─────────────
        # 1 - 2*std of the four "positive knowledge" signals.
        # All high and similar → high coherence → exploit.
        # Modules disagree → low coherence → explore.
        knowledge = np.array([fam, conf, l4p, 1.0 - bored], dtype=np.float64)
        coherence_raw = float(np.clip(1.0 - 2.0 * float(np.std(knowledge)),
                                      0.0, 1.0))
        self._coherence_ema = ((1.0 - GW_COHERENCE_ALPHA) * self._coherence_ema
                               + GW_COHERENCE_ALPHA * coherence_raw)
        coherence = float(self._coherence_ema)

        # ── 6. Tension — stuck in known place ─────────────────
        # Arousal × familiarity: high urgency in familiar territory means
        # the current strategy isn't working — change it.
        tension = float(np.clip(arousal * fam, 0.0, 1.0))

        # ── 7. Readiness — planning gate ─────────────────────
        # Brain must know where it is (coherence), what to expect (confidence),
        # and be in familiar ground (familiarity) to plan reliably.
        readiness = float(np.clip(coherence * conf * fam, 0.0, 1.0))

        # ── 8. Epsilon boost — unified exploration pressure ───
        # Three sources, all gated by ignition_factor (salience).
        # Low salience → urgency doesn't register → no boost.
        epsilon_raw = float(np.clip(
            0.50 * tension
            + 0.30 * curiosity_pull
            + 0.20 * surprise_debt,
            0.0, 1.0
        ))
        epsilon_boost = float(np.clip(
            GW_EPSILON_SCALE * epsilon_raw * ignition_factor,
            0.0, GW_EPSILON_SCALE
        ))

        out = {
            'arousal':        arousal,
            'arousal_raw':    arousal_raw,
            'valence_tone':   valence_tone,
            'valence_raw':    valence_raw,
            'curiosity_pull': curiosity_pull,
            'surprise_debt':  surprise_debt,
            'epsilon_boost':  epsilon_boost,
            'coherence':      coherence,
            'tension':        tension,
            'readiness':      readiness,
            'ignited':        ignited,
            'ignition_rate':  self.ignition_rate(),
        }
        self._last = out
        self._history.append({k: v for k, v in out.items()
                               if isinstance(v, (int, float, bool))})
        self.t += 1
        return out

    def ignition_rate(self) -> float:
        if self.t == 0: return 0.0
        return float(self._ignition_count / self.t)

    def dominant_drive(self) -> str:
        if not self._last: return 'none'
        candidates = {k: self._last[k] for k in
                      ('arousal', 'curiosity_pull', 'surprise_debt', 'tension')}
        return max(candidates, key=candidates.get)

    def summary(self):
        if not self._last:
            print("  GlobalWorkspace (GW1) — no steps yet"); return
        L = self._last
        print(f"  GlobalWorkspace (GW1) — step {self.t}")
        print(f"  Ignited:       {L['ignited']}  (rate={self.ignition_rate():.1%})")
        print(f"  Arousal:       {L['arousal']:.3f}  (raw={L['arousal_raw']:.3f})")
        print(f"  Valence tone:  {L['valence_tone']:+.3f}  (raw={L['valence_raw']:+.3f})")
        print(f"  Curiosity pull:{L['curiosity_pull']:.3f}")
        print(f"  Surprise debt: {L['surprise_debt']:.3f}")
        print(f"  Coherence:     {L['coherence']:.3f}")
        print(f"  Tension:       {L['tension']:.3f}")
        print(f"  Readiness:     {L['readiness']:.3f}")
        print(f"  ε boost:       {L['epsilon_boost']:.3f}")
        print(f"  Dominant:      {self.dominant_drive()}")

    def get_state(self) -> dict:
        return {'t': self.t, 'ignition_rate': self.ignition_rate(),
                'dominant': self.dominant_drive(), **self._last}