"""
M59: SELF-MODEL — THE BRAIN'S MIRROR
=====================================

WHAT THIS IS
------------
The self-model is the first module that is not about the world.

Every module below it points outward:
  M54 asks: what frequency is this?
  L2  asks: what BMU comes next?
  L4  asks: where am I?
  GWS asks: how urgent is my situation?

M59 asks: what kind of thing am I right now?

It reads the GWS broadcast each step and builds a persistent
representation of the brain's own internal state — not as raw
numbers, but as a named characterisation the brain can
compare across time.

THREE THINGS M59 PROVIDES
--------------------------

1. SELF-STATE VECTOR (8 floats)
   A compact representation of the current internal state,
   built from GWS signals. Not the raw signals — a derived
   characterisation. Updated every step via EMA smoothing.

   Dimensions:
     [0] urgency       — how much pressure is there to act differently
     [1] clarity       — how well does the brain understand its situation
     [2] drive         — net motivational direction (toward/away from reward)
     [3] novelty       — how unfamiliar is the current context
     [4] stability     — how consistent has the recent state been
     [5] confidence    — how certain is the brain about its own predictions
     [6] frustration   — high arousal + low progress (stuck signal)
     [7] engagement    — salience × coherence (am I paying attention to something real)

2. STATE MEMORY (last HISTORY_LEN states)
   Every step, the self-state vector is stored.
   M59 can answer: "what was I like N steps ago?"
   M59 can answer: "have I been in a state like this before?"
   Similarity is cosine distance between self-state vectors.

3. STATE LABEL
   A human-readable characterisation of the dominant current state.
   One of: 'confused', 'focused', 'stuck', 'curious', 'satisfied',
           'hunting', 'drifting', 'alert'
   Computed from the self-state vector each step.
   This is not for output — it is for the brain's own use.
   When the label changes, that is a transition event.

WHY THIS MATTERS
----------------
Without M59, the brain's internal state exists but is invisible to
the brain itself. Arousal happens. Tension happens. The brain
responds to them but cannot reflect on them.

With M59, the brain can ask:
  "I am currently in state X."
  "I was in state X 30 steps ago and reward followed 5 steps later."
  "I am in state X again — I should try what worked last time."

That is the first step toward learning from experience at the level
of internal state, not just external reward.

HOW IT PLUGS IN
---------------
M59.step() is called inside Brain.step() after GWS, before M57.
It reads gws_out + thought_out + l4_out.
It writes self_state_vec and state_label into the Brain return dict.

M57 can optionally read M59's state to modulate planning weight:
  - When M59 says 'confused': lower planning_weight (don't plan when lost)
  - When M59 says 'focused':  raise planning_weight (good time to plan)

BIOLOGICAL BASIS
----------------
The self-model corresponds to the Default Mode Network (DMN) in the
human brain — a set of regions (medial PFC, posterior cingulate,
angular gyrus) that are active during self-referential processing,
autobiographical memory, and prospective thinking.

The DMN is notable because it is active at REST — when no external
task is being performed. This is when the brain is thinking about
itself. M59 is the first module in this architecture that could
eventually run during "rest" — during the open-loop phase between
tasks — giving the brain something to do between inputs.

  urgency     ≈ anterior insula (interoceptive urgency signal)
  clarity     ≈ dorsolateral PFC (cognitive clarity / working memory load)
  drive       ≈ nucleus accumbens / ventral striatum (approach/avoidance)
  novelty     ≈ hippocampal CA1 (novelty detection, mismatch signal)
  stability   ≈ posterior cingulate cortex (context stability monitoring)
  confidence  ≈ anterior PFC (metacognitive confidence)
  frustration ≈ anterior cingulate cortex (conflict + effort signal)
  engagement  ≈ locus coeruleus / noradrenaline (global attentional state)

PARAMETERS
----------
SM_EMA_ALPHA    = 0.20  — smoothing for self-state vector (tau ~5 steps)
                          Fast enough to track state transitions,
                          slow enough to ignore single-step noise.
SM_SIM_THRESH   = 0.85  — cosine similarity threshold for "I've been here before"
SM_HISTORY_LEN  = 200   — steps of self-state history to retain
SM_FRUSTRATION_GATE = 0.40  — arousal must exceed this before frustration fires
"""

import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

SM_EMA_ALPHA       = 0.20
SM_SIM_THRESH      = 0.85
SM_HISTORY_LEN     = 200
SM_FRUSTRATION_GATE = 0.40
SM_VEC_DIM         = 8

# State label thresholds
_LABEL_CONF_THRESH  = 0.45   # clarity must exceed for 'focused'
_LABEL_FRUST_THRESH = 0.45   # frustration threshold for 'stuck'
_LABEL_CUR_THRESH   = 0.55   # novelty threshold for 'curious' (raised so conversation doesn't always trigger confused)
_LABEL_SAT_THRESH   = 0.55   # drive threshold for 'satisfied'
_LABEL_HUNT_THRESH  = 0.45   # urgency + drive combined for 'hunting'
_LABEL_ALERT_THRESH = 0.50   # engagement threshold for 'alert'


# ═══════════════════════════════════════════════════════════════
# SELF-MODEL
# ═══════════════════════════════════════════════════════════════

class SelfModel:
    """
    M59 — the brain's representation of itself.

    Reads GWS broadcast + Thought confidence + L4 certainty each step.
    Produces a self-state vector, a state label, and similarity to past states.
    """

    def __init__(self, seed: int = 42):
        # Smoothed self-state vector
        self._state_vec = np.full(SM_VEC_DIM, 0.5, dtype=np.float64)

        # History of raw self-state vectors (for similarity queries)
        self._history     = deque(maxlen=SM_HISTORY_LEN)
        self._label_history = deque(maxlen=SM_HISTORY_LEN)

        # Transition tracking
        self._last_label       = 'drifting'
        self._label_at_reward  = []   # (step, label) pairs when reward > 0
        self._transition_count = 0

        # Steps since last state-label change
        self._steps_in_state   = 0

        self._last_out = {}
        self.t = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             arousal:          float,
             coherence:        float,
             valence_tone:     float,
             curiosity_pull:   float,
             tension:          float,
             surprise_debt:    float,
             thought_confidence: float,
             familiarity:      float,
             salience:         float,
             l4_top_prob:      float,
             reward:           float = 0.0,
             ) -> dict:
        """
        One self-model step.

        Parameters
        ----------
        All floats from GWS broadcast + Thought + L4.

        Returns
        -------
        dict with keys:
          self_state_vec      (8,) float  — current self-state vector
          state_label         str         — dominant state characterisation
          state_changed       bool        — did the label change this step?
          steps_in_state      int         — how long in current label
          past_similarity     float       — max cosine similarity to past states
          been_here_before    bool        — past_similarity > SM_SIM_THRESH
          steps_since_similar int         — steps ago of most similar past state
          self_state_dict     dict        — named dimensions for readability
        """
        # Clip inputs
        ar  = float(np.clip(arousal,           0.0, 1.0))
        co  = float(np.clip(coherence,         0.0, 1.0))
        vt  = float(np.clip(valence_tone,     -1.0, 1.0))
        cp  = float(np.clip(curiosity_pull,    0.0, 1.0))
        tn  = float(np.clip(tension,           0.0, 1.0))
        sd  = float(np.clip(surprise_debt,     0.0, 1.0))
        tc  = float(np.clip(thought_confidence,0.0, 1.0))
        fam = float(np.clip(familiarity,       0.0, 1.0))
        sal = float(np.clip(salience,          0.0, 1.0))
        l4p = float(np.clip(l4_top_prob,       0.0, 1.0))

        # ── Build raw self-state vector ────────────────────────
        # Each dimension is a derived characterisation, not a raw signal.
        # The brain is looking at what it IS, not what the world IS.

        # [0] urgency: how much pressure to change strategy
        #     High when arousal is high AND tension is high (stuck + urgent)
        urgency = float(np.clip(0.60 * ar + 0.40 * tn, 0.0, 1.0))

        # [1] clarity: how well does the brain understand its situation
        #     High when coherence (modules agree) AND L4 knows where it is
        clarity = float(np.clip(0.55 * co + 0.45 * l4p, 0.0, 1.0))

        # [2] drive: motivational direction, normalised to [0,1]
        #     0 = strongly aversive / escaping
        #     0.5 = neutral
        #     1 = strongly appetitive / approaching
        drive = float(np.clip(0.5 + 0.5 * vt, 0.0, 1.0))

        # [3] novelty: how unfamiliar is the current context
        #     High when familiarity is LOW and surprise_debt is HIGH
        novelty = float(np.clip(0.55 * (1.0 - fam) + 0.45 * sd, 0.0, 1.0))

        # [4] stability: how consistent has the recent state been
        #     Computed as 1 - variance of recent self-state vectors
        #     On cold start: 0.5 (unknown)
        if len(self._history) >= 5:
            recent = np.array(list(self._history)[-5:], dtype=np.float64)
            stability = float(np.clip(1.0 - float(np.mean(np.std(recent, axis=0))),
                                      0.0, 1.0))
        else:
            stability = 0.5

        # [5] confidence: how certain is the brain about its own predictions
        #     Thought confidence + L4 location certainty
        confidence = float(np.clip(0.50 * tc + 0.50 * l4p, 0.0, 1.0))

        # [6] frustration: stuck without progress
        #     High arousal (something needs to change) + low drive (not getting reward)
        #     + tension (familiar territory but still failing)
        #     Gated: arousal must be above SM_FRUSTRATION_GATE to fire
        if ar > SM_FRUSTRATION_GATE:
            frust_raw = float(np.clip(
                ar * (1.0 - drive) * (0.5 + 0.5 * tn), 0.0, 1.0))
        else:
            frust_raw = 0.0
        frustration = frust_raw

        # [7] engagement: am I paying attention to something real
        #     Salience × coherence: high salience but incoherent = noise
        #     Both must be high for genuine engagement
        engagement = float(np.clip(sal * co, 0.0, 1.0))

        raw_vec = np.array([
            urgency, clarity, drive, novelty,
            stability, confidence, frustration, engagement
        ], dtype=np.float64)

        # ── Smooth via EMA ─────────────────────────────────────
        self._state_vec = ((1.0 - SM_EMA_ALPHA) * self._state_vec
                           + SM_EMA_ALPHA * raw_vec)
        sv = self._state_vec.copy()

        # ── State label ────────────────────────────────────────
        label = self._compute_label(sv)

        state_changed = (label != self._last_label)
        if state_changed:
            self._transition_count += 1
            self._steps_in_state = 0
        else:
            self._steps_in_state += 1

        self._last_label = label

        if float(reward) > 0.0:
            self._label_at_reward.append((self.t, label))
            if len(self._label_at_reward) > 50:
                self._label_at_reward.pop(0)

        # ── Similarity to past states ──────────────────────────
        past_similarity    = 0.0
        steps_since_similar = -1

        if len(self._history) >= 3:
            hist_arr = np.array(list(self._history), dtype=np.float64)
            # Cosine similarity between current state and each past state
            sv_norm   = sv / (np.linalg.norm(sv) + 1e-9)
            hist_norms = hist_arr / (np.linalg.norm(hist_arr, axis=1, keepdims=True) + 1e-9)
            sims = hist_norms @ sv_norm   # (N,)
            best_idx = int(np.argmax(sims))
            past_similarity = float(sims[best_idx])
            # steps_since_similar: how many steps ago was the most similar state
            steps_since_similar = len(self._history) - 1 - best_idx

        been_here_before = past_similarity >= SM_SIM_THRESH

        # ── Store ──────────────────────────────────────────────
        self._history.append(sv.copy())
        self._label_history.append(label)

        self_state_dict = {
            'urgency':     float(sv[0]),
            'clarity':     float(sv[1]),
            'drive':       float(sv[2]),
            'novelty':     float(sv[3]),
            'stability':   float(sv[4]),
            'confidence':  float(sv[5]),
            'frustration': float(sv[6]),
            'engagement':  float(sv[7]),
        }

        out = {
            'self_state_vec':      sv,
            'state_label':         label,
            'state_changed':       state_changed,
            'steps_in_state':      self._steps_in_state,
            'past_similarity':     past_similarity,
            'been_here_before':    been_here_before,
            'steps_since_similar': steps_since_similar,
            'self_state_dict':     self_state_dict,
        }
        self._last_out = out
        self.t += 1
        return out

    # ── Label computation ──────────────────────────────────────

    def _compute_label(self, sv: np.ndarray) -> str:
        """
        Map self-state vector to a human-readable label.

        Priority order matters — a brain can be both confused and alert,
        but the most actionable label is returned.
        """
        urgency, clarity, drive, novelty, stability, confidence, frustration, engagement = sv

        # 'stuck': high frustration — brain is trapped and knows it
        if frustration > _LABEL_FRUST_THRESH:
            return 'stuck'

        # 'confused': low clarity, high novelty — brain doesn't know where it is
        if clarity < (1.0 - _LABEL_CONF_THRESH) and novelty > _LABEL_CUR_THRESH:
            return 'confused'

        # 'focused': high clarity + high confidence — brain knows what's happening
        if clarity > _LABEL_CONF_THRESH and confidence > _LABEL_CONF_THRESH:
            return 'focused'

        # 'curious': high novelty, not frustrated — genuinely exploring
        if novelty > _LABEL_CUR_THRESH and frustration < _LABEL_FRUST_THRESH * 0.5:
            return 'curious'

        # 'satisfied': high drive, low urgency — reward just received
        if drive > _LABEL_SAT_THRESH and urgency < 0.35:
            return 'satisfied'

        # 'hunting': high urgency + high drive — knows what it wants, going for it
        if urgency > _LABEL_HUNT_THRESH and drive > 0.55:
            return 'hunting'

        # 'alert': high engagement — attending to something salient
        if engagement > _LABEL_ALERT_THRESH:
            return 'alert'

        # default
        return 'drifting'

    # ── Diagnostics ───────────────────────────────────────────

    def dominant_dimension(self) -> str:
        """Which self-state dimension is currently highest?"""
        names = ['urgency','clarity','drive','novelty',
                 'stability','confidence','frustration','engagement']
        return names[int(np.argmax(self._state_vec))]

    def label_distribution(self) -> dict:
        """Fraction of time spent in each label over recent history."""
        from collections import Counter
        if not self._label_history:
            return {}
        c = Counter(self._label_history)
        total = sum(c.values())
        return {k: v/total for k, v in c.items()}

    def reward_label_summary(self) -> dict:
        """Which state labels preceded rewards most often?"""
        from collections import Counter
        if not self._label_at_reward:
            return {}
        c = Counter(label for _, label in self._label_at_reward)
        total = sum(c.values())
        return {k: v/total for k, v in c.most_common()}

    def summary(self):
        sv = self._state_vec
        print(f"  SelfModel (M59) — step {self.t}")
        print(f"  Label:        {self._last_label}  "
              f"(in state {self._steps_in_state} steps,  "
              f"{self._transition_count} transitions)")
        print(f"  Urgency:      {sv[0]:.3f}   Clarity:     {sv[1]:.3f}")
        print(f"  Drive:        {sv[2]:.3f}   Novelty:     {sv[3]:.3f}")
        print(f"  Stability:    {sv[4]:.3f}   Confidence:  {sv[5]:.3f}")
        print(f"  Frustration:  {sv[6]:.3f}   Engagement:  {sv[7]:.3f}")
        print(f"  Dominant dim: {self.dominant_dimension()}")
        if self._last_out:
            lo = self._last_out
            print(f"  Been here before: {lo['been_here_before']}  "
                  f"(similarity={lo['past_similarity']:.3f}, "
                  f"{lo['steps_since_similar']} steps ago)")
        dist = self.label_distribution()
        if dist:
            top = sorted(dist.items(), key=lambda x: -x[1])[:3]
            print(f"  Label history: " + "  ".join(f"{k}={v:.0%}" for k,v in top))
        rls = self.reward_label_summary()
        if rls:
            top_r = sorted(rls.items(), key=lambda x: -x[1])[:3]
            print(f"  Pre-reward states: " + "  ".join(f"{k}={v:.0%}" for k,v in top_r))

    def get_state(self) -> dict:
        return {
            't':                  self.t,
            'state_label':        self._last_label,
            'steps_in_state':     self._steps_in_state,
            'transition_count':   self._transition_count,
            'dominant_dim':       self.dominant_dimension(),
            'self_state_vec':     self._state_vec.tolist(),
            'label_distribution': self.label_distribution(),
            'reward_labels':      self.reward_label_summary(),
        }