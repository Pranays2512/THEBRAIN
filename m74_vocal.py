"""
M74: VOCAL OUTPUT — PRODUCTION FROM INTERNAL STATE
====================================================

Generates phoneme sequences from the brain's internal state using
M73's binding weights. No grammar rules, no templates, no tokens.
The "utterance" emerges from which phoneme patterns are most strongly
associated with what the brain is currently experiencing.

BIOLOGICAL BASIS
----------------
Speech production in real brains follows a cascade:

  1. CONCEPTUAL PREPARATION (left PFC / temporal lobe)
     The intention to communicate arises from internal state.
     "I want to express that I am confused" — not a decision,
     but an activation threshold crossing in the self-model.

  2. LEXICAL ACCESS (left MTG, angular gyrus)
     The conceptual representation activates associated word forms.
     This is NOT a dictionary lookup — it's activation spreading
     through the semantic network until a phonological form reaches
     threshold. M73's W_state matrix encodes this spreading.

  3. PHONOLOGICAL ENCODING (left STG, IFG / Broca's area)
     The word form activates its phoneme sequence.
     M72's transition model provides this: given starting phoneme BMU,
     generate the most probable continuing sequence.

  4. ARTICULATION (motor cortex, cerebellum)
     Phoneme sequences drive articulatory movements.
     In our hardware: phoneme sequence → vocoder → speaker.

  5. SELF-MONITORING (right STG, auditory cortex)
     The brain hears itself speak. If what it hears differs from
     what it intended, prediction error arises → learning.
     M71 feeds back the brain's own voice (if microphone is on).

WHAT DETERMINES WHETHER THE BRAIN SPEAKS
-----------------------------------------
Not a random timer. Not an external command. The brain speaks when:

  1. INTERNAL PRESSURE exceeds threshold — M59 state is at a
     "communicative" level. A brain that is satisfied/calm says
     nothing. A brain that is confused, curious, or has just
     resolved something has something to say.

  2. BINDING STRENGTH is sufficient — the brain has to have
     heard words in contexts similar to what it's experiencing.
     A brain with no language exposure (W_state all zeros) cannot
     produce anything meaningful.

  3. GWS has ignited — the global workspace has reached broadcast
     threshold. This ensures speech is a global, conscious act —
     not a subconscious reflex.

  4. Sufficient time since last utterance — the brain doesn't
     produce continuous speech. A minimum inter-utterance interval
     (MIN_SILENCE_STEPS) ensures turn-taking-like rhythm.

PRODUCTION ALGORITHM
--------------------
1. Retrieve top-k phoneme BMUs most associated with current state
   from M73.state_to_phonemes(state_vec, k=3).

2. Seed the sequence with the highest-association phoneme.

3. Extend the sequence using M72's transition probability model:
   at each step, sample next phoneme from P[current_phoneme],
   biased by M73's state binding (blend TP and state similarity).

4. Stop when:
   - Reached a natural boundary (high boundary_strength in M72)
   - Sequence length > max utterance length
   - Next phoneme association drops below PRODUCTION_THRESH

5. Return the phoneme BMU sequence.

OUTPUT
------
The phoneme BMU sequence is the brain's "utterance" in its internal
representation. The hardware layer converts this to audio:
  phoneme_bmu_sequence → phoneme labels (once the SOM is mature)
                       → IPA phonemes
                       → vocoder synthesis
                       → speaker

Initially: the sequence will be noise — the brain has no learned
associations and M71's SOM is random. Over thousands of hours of
exposure, the utterances become more structured as bindings strengthen.
This is exactly what babbling → words → sentences looks like.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

N_PHONEMES = 400   # must match M71 N_NEURONS

# Minimum binding strength for the brain to attempt production.
# Below this: the brain doesn't have words for what it's experiencing.
PRODUCTION_THRESH   = 0.03

# Maximum phoneme sequence length per utterance (~500ms at 25ms/frame).
MAX_UTTERANCE_LEN   = 20

# Minimum utterance length (utterances shorter than this are suppressed —
# single-phoneme "words" are not yet meaningful).
MIN_UTTERANCE_LEN   = 3

# GWS ignition threshold for speech trigger.
# Lowered from 0.35 → 0.20 so the brain can speak during normal conversation
# (arousal sits ~0.19 at rest; 0.20 lets it cross with mild stimulation).
GWS_SPEAK_THRESH    = 0.20

# Minimum steps of silence between utterances.
# Prevents the brain from talking continuously.
# ~50 frames × 25ms/frame ≈ 1.25 seconds minimum pause.
MIN_SILENCE_STEPS   = 50

# Temperature for phoneme sampling during production.
# Higher = more varied/creative utterances.
# Lower = more consistent/predictable.
# Modulated by M66 NE level (arousal increases temperature).
PRODUCTION_TEMP_BASE = 1.2

# How much M73 state similarity biases the TP-based next-phoneme selection.
# 0.0 = pure transition probability (grammar-like continuation)
# 1.0 = pure state association (semantic coherence)
# 0.5 = blend of both (most natural)
STATE_BLEND  = 0.50

# Communicative states: only these M59 state labels trigger speech.
# A brain that is "satisfied" or "drifting" has nothing urgent to say.
# A brain that is "confused", "curious", "alert", or has "just resolved"
# something is in a communicative state.
COMMUNICATIVE_STATES = {'confused', 'curious', 'alert', 'hunting', 'stuck',
                        'focused', 'listening'}


class VocalOutput:
    """
    M74: Speech production from internal state via phoneme sequence generation.

    Uses M73 binding weights to find phonemes associated with current
    state, then extends the sequence using M72's transition model.
    """

    def __init__(self, seed: int = 42):
        self._rng             = np.random.default_rng(seed)
        self._steps_since_utterance = MIN_SILENCE_STEPS  # start ready to speak
        self._last_utterance  = []
        self.t                = 0

        # Running utterance history (last 20)
        self._utterance_history = []
        self._total_utterances  = 0

    # ── Should the brain speak? ───────────────────────────────

    def _should_speak(self,
                      state_label:      str,
                      gws_arousal:      float,
                      binding_strength: float,
                      ) -> bool:
        """
        Decide whether to produce an utterance this step.

        Conditions:
          1. Brain is in a communicative state
          2. GWS arousal is high enough (global workspace ignition)
          3. At least one phoneme has meaningful binding
          4. Enough silence since last utterance (inter-utterance interval)
        """
        if self._steps_since_utterance < MIN_SILENCE_STEPS:
            return False
        if state_label not in COMMUNICATIVE_STATES:
            return False
        if float(gws_arousal) < GWS_SPEAK_THRESH:
            return False
        if float(binding_strength) < PRODUCTION_THRESH:
            return False
        return True

    # ── Phoneme sequence generation ───────────────────────────

    def _generate_sequence(self,
                           seed_phoneme:  int,
                           state_vec:     np.ndarray,
                           binding:       'SemanticBinding',
                           phoneme_seq:   'PhonemeSequencePredictor',
                           ne_level:      float,
                           ) -> list:
        """
        Generate a phoneme sequence starting from seed_phoneme.

        Each step:
          - Get next-phoneme distribution from M72's transition model
          - Get state-similarity of each candidate from M73
          - Blend the two distributions
          - Sample from the blended distribution (temperature-scaled)
          - Stop at natural boundary or max length
        """
        sequence   = [seed_phoneme]
        curr       = seed_phoneme

        # Temperature: higher NE (arousal) → more varied speech
        temp = PRODUCTION_TEMP_BASE + float(ne_level) * 0.5

        sv = np.array(state_vec, dtype=np.float64)
        sv_norm = float(np.linalg.norm(sv))

        for _ in range(MAX_UTTERANCE_LEN - 1):
            # ── Transition distribution ────────────────────────
            tp_dist = phoneme_seq._P[curr].copy()   # (100,) TP distribution

            # ── State similarity distribution ──────────────────
            if sv_norm > 1e-9 and binding._W_state.max() > 1e-9:
                state_sims = binding._W_state @ (sv / sv_norm)
                state_sims = np.maximum(state_sims, 0.0)
                state_sum  = state_sims.sum()
                if state_sum > 1e-9:
                    state_dist = state_sims / state_sum
                else:
                    state_dist = np.ones(N_PHONEMES) / N_PHONEMES
            else:
                state_dist = np.ones(N_PHONEMES) / N_PHONEMES

            # ── Blend ──────────────────────────────────────────
            blended = ((1.0 - STATE_BLEND) * tp_dist
                       + STATE_BLEND        * state_dist)
            blended = np.maximum(blended, 1e-9)

            # ── Temperature scaling ────────────────────────────
            log_p  = np.log(blended) / temp
            log_p -= log_p.max()   # numerical stability
            probs  = np.exp(log_p)
            probs /= probs.sum()

            # ── Sample next phoneme ────────────────────────────
            next_p = int(self._rng.choice(N_PHONEMES, p=probs))

            # ── Check for natural boundary ─────────────────────
            # Low TP from current to next = word boundary = stop here
            if (len(sequence) >= MIN_UTTERANCE_LEN
                    and phoneme_seq.transition_prob(curr, next_p) < 0.05):
                break

            sequence.append(next_p)
            curr = next_p

        return sequence

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             state_vec:    np.ndarray,
             state_label:  str,
             zone_idx:     int,
             gws_arousal:  float,
             ne_level:     float,
             binding,
             phoneme_seq,
             ) -> dict:
        """
        One vocal output step. May or may not produce an utterance.

        Parameters
        ----------
        state_vec    : (8,)   — M59 self-state vector
        state_label  : str    — M59 state label
        zone_idx     : int    — current zone (L3)
        gws_arousal  : float  — GWS arousal level (ignition signal)
        ne_level     : float  — M66 NE level (arousal/temperature)
        binding      : SemanticBinding (M73) — for state→phoneme lookup
        phoneme_seq  : PhonemeSequencePredictor (M72) — for TP model

        Returns
        -------
        dict with:
          speaking           bool      — True if an utterance was produced
          utterance          list[int] — phoneme BMU sequence (empty if silent)
          utterance_len      int       — length of utterance
          utterance_strength float     — binding strength of seed phoneme
          total_utterances   int       — lifetime utterance count
          steps_since_speech int       — steps since last utterance
        """
        self.t += 1
        self._steps_since_utterance += 1

        # ── Find best phoneme for current state ───────────────
        top_phonemes = binding.state_to_phonemes(state_vec, top_k=3)
        if not top_phonemes:
            seed_phoneme   = 0
            seed_strength  = 0.0
        else:
            seed_phoneme, seed_strength = top_phonemes[0]

        # Also consider zone-associated phonemes
        zone_phonemes = binding.zone_to_phonemes(zone_idx, top_k=2)
        if zone_phonemes and zone_phonemes[0][1] > seed_strength:
            seed_phoneme, seed_strength = zone_phonemes[0]

        # ── Decide whether to speak ───────────────────────────
        if not self._should_speak(state_label, gws_arousal, seed_strength):
            return {
                'speaking':           False,
                'utterance':          [],
                'utterance_len':      0,
                'utterance_strength': float(seed_strength),
                'total_utterances':   self._total_utterances,
                'steps_since_speech': self._steps_since_utterance,
            }

        # ── Generate utterance ────────────────────────────────
        utterance = self._generate_sequence(
            seed_phoneme = seed_phoneme,
            state_vec    = state_vec,
            binding      = binding,
            phoneme_seq  = phoneme_seq,
            ne_level     = float(ne_level),
        )

        if len(utterance) < MIN_UTTERANCE_LEN:
            # Too short — suppress
            return {
                'speaking':           False,
                'utterance':          [],
                'utterance_len':      0,
                'utterance_strength': float(seed_strength),
                'total_utterances':   self._total_utterances,
                'steps_since_speech': self._steps_since_utterance,
            }

        # ── Store and reset ───────────────────────────────────
        self._last_utterance             = utterance
        self._steps_since_utterance      = 0
        self._total_utterances          += 1
        self._utterance_history.append(utterance)
        if len(self._utterance_history) > 20:
            self._utterance_history.pop(0)

        return {
            'speaking':           True,
            'utterance':          utterance,
            'utterance_len':      len(utterance),
            'utterance_strength': float(seed_strength),
            'total_utterances':   self._total_utterances,
            'steps_since_speech': 0,
        }
