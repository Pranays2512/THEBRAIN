"""
M73: SEMANTIC BINDING — CROSS-MODAL HEBBIAN WORD GROUNDING
===========================================================

Binds phoneme patterns to the brain's internal experience through
co-occurrence. This is how words get meaning — not from a dictionary,
not from explicit teaching, but from the brain noticing that certain
sounds reliably co-occur with certain internal states, zones, and events.

BIOLOGICAL BASIS
----------------
Word learning in infants is driven by cross-modal Hebbian binding:

  Angular gyrus (BA39) — the hub of cross-modal binding. Lesions here
    cause semantic aphasia: words lose their meaning. The angular gyrus
    sits at the junction of auditory, visual, and somatosensory cortex
    and receives input from all of them.

  Parahippocampal cortex — mediates association between sensory forms
    (what the word sounds like) and episodic context (when/where you
    heard it, what was happening). Critical for fast mapping in children.

  Left MTG (middle temporal gyrus) — semantic storage. When a word's
    phoneme pattern activates, MTG activates associated concepts.
    Damage causes anomia (word-finding difficulty), not forgetting the concept.

THE MECHANISM
-------------
When phoneme sequence X fires at the same time as:
  - M59 internal state Y (curious, satisfied, confused...)
  - Zone Z (the brain is at node 4 — the food zone)
  - Reward event (food just arrived)
  - Another sensory feature (M54 sound cortex BMU)

M73 writes:
  W_state[phoneme_bmu, :] += eta * state_vec     (bind to state)
  W_zone[phoneme_bmu, zone] += eta               (bind to zone)
  W_reward[phoneme_bmu] += eta * reward          (bind to reward)
  W_sound[phoneme_bmu, sound_bmu] += eta         (bind to sound)

Over many co-occurrences, these weights strengthen. When the word is
heard again later, W_state[phoneme_bmu] activates the associated
internal state — this IS the word's meaning.

Production (inverse): given an internal state, retrieve the phoneme
pattern most strongly associated with it. This is how the brain
finds the word for what it's experiencing.

IMPORTANT: WHAT THIS IS AND ISN'T
----------------------------------
This is NOT a look-up table. The brain doesn't store "food → zone 4".
The binding weights ARE the meaning. They encode statistical associations
across thousands of co-occurrences. A word heard once barely shifts the
weights. A word heard 100 times in consistent contexts has strong weights.

This explains:
  - Why children learn words gradually, not in one shot (weak individual updates)
  - Why context matters for word meaning (zone + state both bind)
  - Why emotional words are learned faster (reward binding amplifies)
  - Why nouns before verbs (nouns bind to stable states; verbs require
    binding to temporal transitions — more complex)

BINDING TARGETS
---------------
M73 binds phoneme patterns to four types of internal signals:

  1. STATE binding  — W_state[phoneme_bmu, 8]
     Associates phoneme with M59 self-model state vector.
     "Confused" sounds like [k-ə-n-f-j-u-z-d] while the brain IS confused.

  2. ZONE binding   — W_zone[phoneme_bmu, 8]
     Associates phoneme with which zone the brain is in.
     "Food" heard while at the food zone → W_zone[food_phonemes, 4] strengthens.

  3. REWARD binding — W_reward[phoneme_bmu]
     Associates phoneme with reward presence.
     Words that co-occur with food/reward bind strongly to reward valence.

  4. SOUND binding  — W_sound[phoneme_bmu, 64]
     Associates phoneme with M54 sound cortex BMU.
     Links speech sounds to environmental sounds heard simultaneously.

RETRIEVAL
---------
Given current phoneme_bmu, M73 returns:
  associated_state : (8,) — what state is this word associated with?
  associated_zone  : int  — which zone does this word evoke?
  associated_reward: float — is this a reward-valenced word?
  binding_strength : float — how confidently has this phoneme been bound?

Given current internal state (for production):
  best_phoneme_bmu : int   — which phoneme is most associated with this state?
  This is the starting point for M74 vocal production.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

N_PHONEMES  = 400   # must match M71
N_STATE_DIM = 8     # M59 self-state vector dimension
N_ZONES     = 8     # L3 zones
N_SOUND_BMU = 64    # M54 sound cortex neurons

# Hebbian learning rate — deliberately very slow.
# Each co-occurrence shifts weights by only 0.002.
# A word needs ~100 consistent co-occurrences to reach binding_strength 0.5.
# This is realistic: children hear common words thousands of times before
# they're fully learned.
BIND_ETA         = 0.002

# Reward amplification — reward events boost binding by this factor.
# A word heard during food reward binds 5× faster than in neutral context.
# Biology: dopamine release during reward strengthens synaptic potentiation.
REWARD_AMPLIFY   = 5.0

# Synaptic decay — slow forgetting of unused bindings.
# tau ~5000 updates at BIND_DECAY=0.0002.
# A word not heard in 5000 steps slowly loses its binding.
BIND_DECAY       = 0.0002

# Minimum binding strength to consider a phoneme "bound"
# (i.e., has meaningful associations rather than noise).
BOUND_THRESH     = 0.05

# Context window: binding uses an eligibility trace over recent phonemes,
# not just the current frame. This allows binding to sequences (words)
# not just single phonemes.
# At ETA_TRACE=0.80 and window of ~5 frames (~125ms) — covers ~1 syllable.
TRACE_DECAY      = 0.80


class SemanticBinding:
    """
    M73: Cross-modal Hebbian binding between phoneme patterns and internal
    state, zone, reward, and environmental sound.

    Weights are the memory. No explicit word store. Meaning is distributed
    across the weight matrices, exactly as in biological neural semantics.
    """

    def __init__(self, n_phonemes: int = N_PHONEMES,
                 n_state_dim: int = N_STATE_DIM,
                 n_zones: int = N_ZONES,
                 n_sound_bmu: int = N_SOUND_BMU,
                 seed: int = 42):
        self.n_phonemes  = n_phonemes
        self.n_state_dim = n_state_dim
        self.n_zones     = n_zones
        self.n_sound_bmu = n_sound_bmu

        # ── Binding weight matrices ───────────────────────────
        # All initialised to zero — no meaning before exposure.
        self._W_state  = np.zeros((n_phonemes, n_state_dim), dtype=np.float64)
        self._W_zone   = np.zeros((n_phonemes, n_zones),     dtype=np.float64)
        self._W_reward = np.zeros(n_phonemes,                dtype=np.float64)
        self._W_sound  = np.zeros((n_phonemes, n_sound_bmu), dtype=np.float64)

        # ── Eligibility trace over recent phoneme BMUs ────────
        # trace[bmu] = how recently/strongly did phoneme bmu fire?
        # Allows binding to sequences: the trace accumulates over a
        # phoneme sequence so the whole word gets credited, not just
        # the last phoneme.
        self._trace = np.zeros(n_phonemes, dtype=np.float64)

        # ── Binding strength per phoneme (how well bound) ─────
        # binding_strength[bmu] = norm of W_state[bmu] + W_zone[bmu]
        # Rises as weights strengthen through consistent co-occurrence.
        self._binding_strength = np.zeros(n_phonemes, dtype=np.float64)

        # ── Step counter ──────────────────────────────────────
        self.t = 0

    # ── Update trace ──────────────────────────────────────────

    def _update_trace(self, phoneme_bmu: int):
        """
        Decay eligibility trace and add current phoneme.

        The trace represents "recently active phonemes" over a ~125ms
        window, allowing co-occurrence binding to credit whole sequences
        (words) rather than individual phonemes.
        """
        self._trace *= TRACE_DECAY
        self._trace[phoneme_bmu] += 1.0
        # Normalize so max trace = 1.0
        mx = self._trace.max()
        if mx > 1.0:
            self._trace /= mx

    # ── Hebbian binding update ────────────────────────────────

    def _bind(self,
              state_vec:  np.ndarray,
              zone_idx:   int,
              reward:     float,
              sound_bmu:  int,
              ):
        """
        Apply Hebbian update: strengthen associations between
        current trace (active phonemes) and current context.

        Uses the eligibility trace so that the entire recent phoneme
        sequence (not just the last phoneme) gets bound to context.
        """
        # Reward amplifies binding — dopamine-modulated LTP
        amp = 1.0 + REWARD_AMPLIFY * float(np.clip(reward, 0.0, 1.0))
        eta = BIND_ETA * amp

        # ── State binding ─────────────────────────────────────
        # For each phoneme in trace: W_state[bmu, :] += eta * trace[bmu] * state_vec
        # Outer product over all active phonemes simultaneously
        state_norm = np.array(state_vec, dtype=np.float64)
        s = float(np.linalg.norm(state_norm))
        if s > 1e-9:
            state_norm /= s
        self._W_state += eta * np.outer(self._trace, state_norm)

        # ── Zone binding ──────────────────────────────────────
        if 0 <= zone_idx < self.n_zones:
            self._W_zone[:, zone_idx] += eta * self._trace

        # ── Reward binding ────────────────────────────────────
        if float(reward) > 0.0:
            self._W_reward += eta * self._trace * float(reward)

        # ── Sound binding ─────────────────────────────────────
        if 0 <= sound_bmu < self.n_sound_bmu:
            self._W_sound[:, sound_bmu] += eta * self._trace

    # ── Synaptic decay ────────────────────────────────────────

    def _decay(self):
        """Slow forgetting — unused bindings weaken over time."""
        self._W_state  *= (1.0 - BIND_DECAY)
        self._W_zone   *= (1.0 - BIND_DECAY)
        self._W_reward *= (1.0 - BIND_DECAY)
        self._W_sound  *= (1.0 - BIND_DECAY)

    # ── Update binding strength ───────────────────────────────

    def _update_strength(self):
        """
        Recompute binding_strength per phoneme.
        strength = mean of L2 norms across all binding targets.
        """
        s_state  = np.linalg.norm(self._W_state,  axis=1)
        s_zone   = np.linalg.norm(self._W_zone,   axis=1)
        s_reward = np.abs(self._W_reward)
        # Weighted mean: state carries most of "meaning"
        self._binding_strength = (0.5 * s_state + 0.3 * s_zone + 0.2 * s_reward)

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             phoneme_bmu:  int,
             is_voiced:    bool,
             state_vec:    np.ndarray,
             zone_idx:     int,
             reward:       float,
             sound_bmu:    int,
             ) -> dict:
        """
        One binding step: update trace, bind current context to phoneme.

        Parameters
        ----------
        phoneme_bmu : int     — current phoneme BMU from M71
        is_voiced   : bool    — True if speech frame (not silence)
        state_vec   : (8,)    — M59 self-state vector
        zone_idx    : int     — current L3 zone (freq_idx)
        reward      : float   — reward this step (0 normally, >0 on food)
        sound_bmu   : int     — M54 sound cortex BMU (environmental sound)

        Returns
        -------
        dict with:
          associated_state   : (8,) — what state does current phoneme evoke?
          associated_zone    : int  — which zone does current phoneme evoke?
          associated_reward  : float — reward association of current phoneme
          binding_strength   : float — how well bound is current phoneme?
          n_bound_phonemes   : int  — phonemes above BOUND_THRESH
          best_state_phoneme : int  — which phoneme is most associated with current state?
        """
        self.t += 1

        # ── Update trace ──────────────────────────────────────
        if is_voiced:
            self._update_trace(phoneme_bmu)

        # ── Bind current context to trace ─────────────────────
        # Binding happens even if not voiced — if a word was just heard
        # (trace is elevated) and then reward arrives, we want to bind them.
        # This handles slight temporal offsets between word and reward.
        if self._trace.max() > 0.01:
            self._bind(
                state_vec = state_vec,
                zone_idx  = zone_idx,
                reward    = float(reward),
                sound_bmu = sound_bmu,
            )

        # ── Decay ─────────────────────────────────────────────
        # Every step — even silent ones — weights decay slightly.
        self._decay()

        # ── Update binding strength ───────────────────────────
        if self.t % 10 == 0:   # only recompute every 10 steps (efficiency)
            self._update_strength()

        # ── Retrieval: what does current phoneme mean? ─────────
        bmu_strength = float(self._binding_strength[phoneme_bmu])

        # Associated state: what M59 state is this phoneme linked to?
        assoc_state = self._W_state[phoneme_bmu].copy()
        s = float(np.linalg.norm(assoc_state))
        if s > 1e-9:
            assoc_state /= s

        # Associated zone: which zone has strongest binding for this phoneme?
        zone_weights = self._W_zone[phoneme_bmu]
        assoc_zone = int(np.argmax(zone_weights)) if zone_weights.max() > 1e-9 else -1

        # Associated reward: how strongly does this phoneme predict reward?
        assoc_reward = float(np.clip(self._W_reward[phoneme_bmu], 0.0, 1.0))

        # Production hint: which phoneme is most associated with current state?
        # This is used by M74 for vocal production.
        sv = np.array(state_vec, dtype=np.float64)
        sv_norm = float(np.linalg.norm(sv))
        if sv_norm > 1e-9 and self._W_state.max() > 1e-9:
            # Cosine similarity between each phoneme's state association and current state
            state_sims = self._W_state @ (sv / sv_norm)
            best_state_phoneme = int(np.argmax(state_sims))
        else:
            best_state_phoneme = 0

        # Count bound phonemes
        n_bound = int(np.sum(self._binding_strength > BOUND_THRESH))

        return {
            'associated_state':    assoc_state,
            'associated_zone':     assoc_zone,
            'associated_reward':   assoc_reward,
            'binding_strength':    bmu_strength,
            'n_bound_phonemes':    n_bound,
            'best_state_phoneme':  best_state_phoneme,
        }

    # ── Production API ────────────────────────────────────────

    def state_to_phonemes(self,
                          state_vec: np.ndarray,
                          top_k: int = 5,
                          ) -> list[tuple[int, float]]:
        """
        Given an internal state, return the top-k phoneme BMUs most
        associated with it. Used by M74 for vocal production.

        Returns list of (phoneme_bmu, similarity) sorted by similarity.
        """
        sv = np.array(state_vec, dtype=np.float64)
        sv_norm = float(np.linalg.norm(sv))
        if sv_norm < 1e-9 or self._W_state.max() < 1e-9:
            return [(i, 0.0) for i in range(top_k)]
        sims = self._W_state @ (sv / sv_norm)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top_indices]

    def zone_to_phonemes(self,
                         zone_idx: int,
                         top_k: int = 5,
                         ) -> list[tuple[int, float]]:
        """
        Given a zone, return the top-k phoneme BMUs most associated with it.
        """
        if zone_idx < 0 or zone_idx >= self.n_zones:
            return []
        zone_col = self._W_zone[:, zone_idx]
        top_indices = np.argsort(zone_col)[::-1][:top_k]
        return [(int(i), float(zone_col[i])) for i in top_indices]
