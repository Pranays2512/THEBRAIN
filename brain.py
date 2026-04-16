"""
BRAIN v13 — Language-First Cognitive Stack
==========================================

Navigation scaffold removed. The brain's only sensory interface is sound.

  brain.hear(mfcc_vec)    — inject a 13-dim MFCC audio frame (25ms)
  brain.step(reward=0.0)  — run one cognitive cycle
  brain.dream(n=8)        — run a REM dream cycle (offline, between steps)

PRIMARY SENSORY: M71 SpeechCortex
  13-dim MFCC → 26-dim (MFCC + delta) → 10×10 acoustic SOM → phoneme_bmu.
  phoneme_bmu drives the full cognitive stack:
    M55 (associative memory of phoneme patterns)
    L2  (sequence prediction of next phoneme)
    Attention, Thought, Valence, SelfModel — all operating over phoneme space.

COGNITIVE STACK (in step order)
--------------------------------
  M71  SpeechCortex       — acoustic SOM (primary sensory)
  M72  PhonemeSequence    — transition statistics, word boundary detection
  M55  AssociativeMemory  — Hebbian familiarity over phoneme patterns
  L2   SequencePredictor  — predict next phoneme
  Valence                 — reward/novelty/familiarity integration
  M66  NeuromodState      — ACh / NE / 5-HT modulation
  Attention               — salience gating
  Thought                 — expectation, focus entropy
  M58  WorkingMemory      — time-based recency, steps since reward
  GWS  GlobalWorkspace    — global ignition / broadcast signal
  M59  SelfModel          — internal state vector and label
  M60  QuestionMemory     — curiosity / confusion tracking
  M62  ConsistencyChecker — reasoning critic
  M63  EpisodicTrace      — episodic memory
  M64  LanguageSeed       — situation tagging from internal state
  M67  TemporalContext     — circadian phase, overdue score
  M73  SemanticBinding    — Hebbian word-to-state binding
  M74  VocalOutput        — speech production

REMOVED FROM v12
----------------
  M56  ActionLayer  — navigation Q-learning (was test scaffold)
  M57  Planner      — tree-search planner (needs zone graph)
  L3   ConceptLayer — zone clustering (navigation-specific)
  L4   PositionBelief — Bayesian location (navigation-specific)
  M51  TextureSense   — texture sensor (body-specific)
  M54b TextureCortex  — texture SOM (body-specific)
  M65  MultimodalFusion — sound×texture fusion (body-specific)
  M68  InferenceEngine  — BFS over zone graph (needs L3, to be rearchitected)
  M69  ImaginationEngine — stochastic zone walks (needs L3, to be rearchitected)
  CortexM56 (M54) — frequency-domain cortex (replaced by M71 as primary sensory)
"""

import numpy as np

from m55_memory     import AssociativeMemory
from l2_predictor   import SequencePredictor
from evaluators     import Attention, Thought, Valence
from m58_workingmemory import WorkingMemory
from global_workspace  import GlobalWorkspace
from m59_selfmodel     import SelfModel
from m60_questions     import QuestionMemory
from m62_consistency   import ConsistencyChecker
from m63_episodic      import EpisodicTrace
from m64_language      import LanguageSeed
from m66_neuromod      import NeuromodState
from m67_temporal      import TemporalContext
from m70_dream         import DreamEngine

# Language stack
from m71_speech_cortex import SpeechCortex
from m72_phoneme_seq   import PhonemeSequencePredictor
from m73_binding       import SemanticBinding
from m74_vocal         import VocalOutput
from m75_semantic      import SemanticMemory


# ═══════════════════════════════════════════════════════════════
# WORD-LEVEL TRANSITION PREDICTOR
# ═══════════════════════════════════════════════════════════════

class WordTransitionPredictor:
    """
    Sparse word-to-word transition probability matrix.

    Operates on word strings directly — no SOM, no BMU collisions.
    Learns: after word X, word Y follows with probability P(Y|X).

    This is the fix for the 78-BMU bottleneck: 300 words crammed into
    78 BMU slots meant the phoneme-level TP could not distinguish words.
    The word-level TP has one slot per word — zero collisions.
    """

    _SEPARATOR = '<SEP>'   # marks input→response boundary
    _END       = '<END>'   # marks end of response

    def __init__(self):
        self._P = {}           # {word: {next_word: count}}
        self._totals = {}      # {word: total_count}
        self._prev_word = None

    def observe(self, word: str):
        """Record a word transition: prev_word → word."""
        if self._prev_word is not None:
            if self._prev_word not in self._P:
                self._P[self._prev_word] = {}
            d = self._P[self._prev_word]
            d[word] = d.get(word, 0) + 1
            self._totals[self._prev_word] = self._totals.get(self._prev_word, 0) + 1
        self._prev_word = word

    def observe_separator(self):
        """Mark the boundary between user input and brain response."""
        self.observe(self._SEPARATOR)

    def observe_end(self):
        """Mark the end of a response."""
        self.observe(self._END)
        self._prev_word = None

    def reset_context(self):
        """Reset previous word (between exchanges)."""
        self._prev_word = None

    def get_transitions(self, word: str) -> dict:
        """Get raw transition counts from a word."""
        return dict(self._P.get(word, {}))

    def generate(self, seed_words: list, max_len: int = 10,
                 rng=None) -> list:
        """
        Generate a response by walking the word TP matrix.

        1. Feed all seed_words to find the transition context
        2. From the last seed word (or <SEP>), walk the matrix
        3. Stop at <END>, max_len, or dead end
        """
        if rng is None:
            rng = np.random.default_rng()

        # Aggregate transition scores from seed words.
        # Words later in the sequence get higher weight (recency bias).
        agg = {}  # {next_word: weighted_score}
        n = len(seed_words)
        for i, word in enumerate(seed_words):
            weight = 0.5 + 0.5 * (i / max(n - 1, 1))  # 0.5 → 1.0
            transitions = self._P.get(word, {})
            total = self._totals.get(word, 1)
            for next_w, count in transitions.items():
                agg[next_w] = agg.get(next_w, 0.0) + weight * count / total

        # Also check transitions from <SEP> (what typically starts a response)
        sep_trans = self._P.get(self._SEPARATOR, {})
        sep_total = self._totals.get(self._SEPARATOR, 1)
        for next_w, count in sep_trans.items():
            agg[next_w] = agg.get(next_w, 0.0) + 1.5 * count / sep_total

        # Remove special tokens from candidates
        agg.pop(self._SEPARATOR, None)
        agg.pop(self._END, None)

        if not agg:
            return []

        # Pick first word by weighted sampling
        words_list = list(agg.keys())
        scores = np.array([agg[w] for w in words_list], dtype=np.float64)
        scores = np.maximum(scores, 0.0)
        total = scores.sum()
        if total < 1e-9:
            return []
        probs = scores / total
        first = words_list[int(rng.choice(len(words_list), p=probs))]

        # Walk the TP matrix from first word
        result = [first]
        current = first
        for _ in range(max_len - 1):
            transitions = self._P.get(current, {})
            total = self._totals.get(current, 0)
            if not transitions or total == 0:
                break

            # Build probability distribution
            candidates = []
            counts = []
            for w, c in transitions.items():
                if w == self._SEPARATOR:
                    continue
                if w == self._END:
                    # End token acts as a stop signal
                    if rng.random() < c / total:
                        break
                    continue
                if w not in result:  # avoid loops
                    candidates.append(w)
                    counts.append(c)

            if not candidates:
                break

            counts_arr = np.array(counts, dtype=np.float64)
            p = counts_arr / counts_arr.sum()
            current = candidates[int(rng.choice(len(candidates), p=p))]
            result.append(current)

        return result

    def n_words(self) -> int:
        """Number of unique words seen."""
        return len(self._P)

    def n_transitions(self) -> int:
        """Total number of unique bigram transitions."""
        return sum(len(v) for v in self._P.values())


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

FEEDBACK_EMA_ALPHA = 0.10
FEEDBACK_EMA_INIT  = 1.0

_NO_ACTION = -1   # action index placeholder (navigation removed)


# ═══════════════════════════════════════════════════════════════
# BRAIN
# ═══════════════════════════════════════════════════════════════

class Brain:
    """
    Integrated cognitive stack — language-first, no navigation.

    Parameters
    ----------
    seed : int — random seed for all stochastic modules.

    Primary interface
    -----------------
    brain.hear(mfcc_vec)   — inject 13-dim MFCC before step()
    brain.step(reward)     — one cognitive cycle
    brain.dream(n)         — offline REM dream cycle
    """

    def __init__(self, seed: int = 42):
        # ── Primary sensory: speech cortex (M71) ─────────────
        self.speech_cortex = SpeechCortex(seed=seed)

        # ── Language modules ──────────────────────────────────
        self.phoneme_seq   = PhonemeSequencePredictor()
        self.binding       = SemanticBinding(seed=seed)
        self.vocal         = VocalOutput(seed=seed)

        # ── Cognitive stack ───────────────────────────────────
        # All these now operate over phoneme BMU space (0..99)
        # instead of frequency/zone space.
        # M55 sized to match M71's 100-neuron phoneme SOM
        self.memory        = AssociativeMemory(seed=seed, n_neurons=400)
        self.pred          = SequencePredictor(n_neurons=400)
        self.attention     = Attention(n_neurons=400)
        self.thought       = Thought(n_neurons=400)
        self.valence       = Valence()
        self.wm            = WorkingMemory(n_zones=8, seed=seed)
        self.gws           = GlobalWorkspace(n_zones=8)
        self.selfmodel     = SelfModel(seed=seed)
        self.questions     = QuestionMemory(seed=seed)
        self.consistency   = ConsistencyChecker(n_actions=4, seed=seed)
        self.episodic      = EpisodicTrace(n_actions=4, seed=seed)
        self.language      = LanguageSeed(n_actions=4, seed=seed)

        # ── Neuromodulation and time ──────────────────────────
        self.neuromod   = NeuromodState()
        self.temporal   = TemporalContext()

        # ── Dreams ───────────────────────────────────────────
        self.dreamer    = DreamEngine(seed=seed)

        # ── Word trajectory memory (for language dreaming) ────
        # Each entry: {'bmus': [int,...], 'reward': float, 'state_label': str}
        # Filled by live.py after each conversation turn.
        # brain.dream_language() replays these during sleep cycles.
        from collections import deque as _deque
        self._word_trajectories: _deque = _deque(maxlen=200)

        # ── Word-level transition predictor ────────────────────
        # Operates on word strings directly — bypasses SOM BMU collisions.
        # Trained by train_dialogue.py alongside the phoneme pipeline.
        self.word_tp = WordTransitionPredictor()

        # ── Semantic memory (M75) — declarative fact store ─────
        # Stores entity-attribute facts independently of emotion.
        # This IS part of the brain (cortical semantic memory),
        # so it gets pickled with save() and survives restarts.
        self.semantic = SemanticMemory()

        # ── Feedback state ────────────────────────────────────
        self._error_ema             = float(FEEDBACK_EMA_INIT)
        self._curiosity_ema         = float(FEEDBACK_EMA_INIT)
        self._last_surprise_signal  = 0.0
        self._last_curiosity_delta  = 0.0
        self._last_rpe_positive     = 0.0
        self._last_familiarity      = 0.0
        self._last_prediction_error = 0.0
        self._last_curiosity        = 0.0

        # ── MFCC buffer (set by hear()) ───────────────────────
        self._mfcc_input = np.zeros(13, dtype=np.float32)

        self.t = 0

    # ── Sensory input ─────────────────────────────────────────

    def hear(self, mfcc_vec: np.ndarray):
        """
        Inject a 13-dim MFCC audio frame before the next step().

        Call once per audio frame (25ms) from your audio pipeline:

            mfcc = compute_mfcc(audio_frame)
            brain.hear(mfcc)
            out  = brain.step()

        The MFCC is stored as _mfcc_input and consumed by M71 inside step().
        If hear() is never called, M71 sees zeros (silence) and stays quiet.
        """
        self._mfcc_input = np.array(mfcc_vec, dtype=np.float32)

    # ── Main cognitive cycle ───────────────────────────────────

    def step(self, reward: float = 0.0) -> dict:
        """
        One full cognitive step.

        Parameters
        ----------
        reward : float — external reward this step (0 normally, >0 on food/praise)

        Returns
        -------
        dict — full state snapshot from all modules
        """
        reward = float(reward)

        # ── 1. Speech cortex (M71) — primary sensory ─────────
        # phoneme_bmu is the brain's primary sensory representation.
        # It replaces freq-domain bmu_idx from the old CortexM56.
        m71_out = self.speech_cortex.step(
            mfcc_vec    = self._mfcc_input,
            familiarity = float(self._last_familiarity),
        )
        bmu_idx = m71_out['phoneme_bmu']   # primary BMU for cognitive stack
        qe_norm = m71_out['phoneme_qe_norm']

        # Clear MFCC buffer — consume once per step
        self._mfcc_input = np.zeros(13, dtype=np.float32)

        # ── 2. Phoneme sequence / word boundary (M72) ─────────
        m72_out = self.phoneme_seq.step(
            phoneme_bmu = bmu_idx,
            is_voiced   = m71_out['is_voiced'],
        )

        # ── 3. Predict BEFORE memory writes ───────────────────
        pred_out = self.pred.predict()

        # ── 4. Associative memory (M55) ───────────────────────
        mem_out = self.memory.step(
            bmu_idx      = bmu_idx,
            qe_norm      = qe_norm,
            curiosity    = self._last_curiosity_delta,
            rpe_positive = self._last_rpe_positive,
        )

        # ── 5. Familiarity recall ─────────────────────────────
        recall_out  = self.memory.recall(bmu_idx)
        familiarity = recall_out['familiarity']
        self._last_familiarity = familiarity

        # ── 6. L2 sequence predictor ──────────────────────────
        l2_out = self.pred.step(
            bmu_idx         = bmu_idx,
            qe_norm         = qe_norm,
            familiarity     = familiarity,
            prediction_bias = self.thought._last_prediction_bias,
            last_action     = _NO_ACTION,
            world_moved     = True,   # always "moved" — speech is continuous
        )
        raw_error     = l2_out['prediction_error']
        raw_curiosity = l2_out['curiosity']

        # ── 7. Feedback deltas ────────────────────────────────
        surprise_signal = float(np.clip(raw_error    - self._error_ema,    0.0, 1.0))
        curiosity_delta = float(np.clip(raw_curiosity - self._curiosity_ema, 0.0, 1.0))
        self._error_ema     = ((1.0 - FEEDBACK_EMA_ALPHA) * self._error_ema
                               + FEEDBACK_EMA_ALPHA * raw_error)
        self._curiosity_ema = ((1.0 - FEEDBACK_EMA_ALPHA) * self._curiosity_ema
                               + FEEDBACK_EMA_ALPHA * raw_curiosity)
        self._last_surprise_signal  = surprise_signal
        self._last_curiosity_delta  = curiosity_delta
        self._last_prediction_error = raw_error
        self._last_curiosity        = raw_curiosity

        # ── 8. Valence (RPE + intrinsic reward) ──────────────
        v1_out = self.valence.step(
            prediction_error = raw_error,
            reward           = reward,
            familiarity      = familiarity,
        )
        self._last_rpe_positive = v1_out['pos_rpe']

        # ── 9. Neuromodulators (M66) ──────────────────────────
        # Without L4, belief entropy defaults to 0.5 (max uncertainty).
        m66_out = self.neuromod.step(
            l4_belief_entropy = 0.5,
            qe_norm           = float(qe_norm),
            surprise_signal   = float(surprise_signal),
            reward_ema        = float(v1_out['reward_ema']),
            abs_rpe           = float(abs(v1_out['rpe'])),
        )

        # ── 10. Attention ─────────────────────────────────────
        attn_out = self.attention.step(
            bmu_idx                  = bmu_idx,
            qe_norm                  = qe_norm,
            familiarity              = familiarity,
            surprise_signal          = surprise_signal,
            curiosity_delta          = curiosity_delta,
            thought_confidence_delta = self.thought._last_confidence_delta,
        )

        # ── 11. Thought ───────────────────────────────────────
        thought_out = self.thought.step(
            attended_bmu = attn_out['attended_bmu'],
            bmu_idx      = bmu_idx,
            pred         = self.pred,
            salience     = attn_out['salience'],
            memory       = self.memory,
        )

        # ── 12. Working Memory (M58) ──────────────────────────
        # ── 12. Working Memory (M58) ──────────────────────────
        # freq_idx replaced with phoneme familiarity bucket (0-7).
        # High familiarity = well-known phoneme region = low "boredom".
        wm_out = self.wm.step(
            freq_idx = int(familiarity * 7),
            action   = _NO_ACTION,
            reward   = reward,
        )

        # ── 13. Global Workspace ──────────────────────────────
        # freq_idx: phoneme familiarity bucket (0-7) replaces zone index.
        gws_out = self.gws.step(
            qe_norm            = qe_norm,
            familiarity        = familiarity,
            freq_idx           = int(familiarity * 7),
            prediction_error   = raw_error,
            thought_confidence = thought_out['thought_confidence'],
            rpe                = v1_out['rpe'],
            intrinsic_rwd      = v1_out['intrinsic_reward'],
            corridor_boredom   = wm_out['corridor_boredom'],
            steps_since_reward = wm_out['steps_since_reward'],
            salience           = attn_out['salience'],
            l4_top_prob        = 0.0,
        )

        # ── 14. Self-Model (M59) ──────────────────────────────
        sm_out = self.selfmodel.step(
            arousal            = gws_out['arousal'],
            coherence          = gws_out['coherence'],
            valence_tone       = gws_out['valence_tone'],
            curiosity_pull     = gws_out['curiosity_pull'],
            tension            = gws_out['tension'],
            surprise_debt      = gws_out['surprise_debt'],
            thought_confidence = thought_out['thought_confidence'],
            familiarity        = familiarity,
            salience           = attn_out['salience'],
            l4_top_prob        = 0.0,
            reward             = reward,
        )

        # ── 15. Question Memory (M60) ─────────────────────────
        # freq_idx: phoneme familiarity bucket — tracks which phoneme
        # regions are generating the most open questions.
        q60_out = self.questions.step(
            prediction_error = raw_error,
            freq_idx         = int(familiarity * 7),
            bmu_idx          = bmu_idx,
            predicted_bmu    = thought_out['expected_bmu'],
            l4_top_prob      = 0.0,
            sm_state_vec     = sm_out['self_state_vec'],
            sm_state_label   = sm_out['state_label'],
            reward           = reward,
        )

        # ── 16. Consistency Checker (M62) ─────────────────────
        m62_out = self.consistency.step(
            sim_history      = [],
            thought_iters    = [],
            q60_open_count   = q60_out['open_count'],
            q60_zone_pull    = q60_out['zone_pull'],
            sm_state_label   = sm_out['state_label'],
            planning_active  = False,
        )

        # ── 17. Episodic Trace (M63) ──────────────────────────
        # freq_idx: phoneme familiarity bucket.
        # current_action: unused without navigation — kept as stub.
        m63_out = self.episodic.step(
            q60_just_created  = q60_out['just_created'],
            q60_just_resolved = q60_out['just_resolved'],
            q60_zone_pull     = q60_out['zone_pull'],
            current_action    = _NO_ACTION,
            freq_idx          = int(familiarity * 7),
            sm_state_vec      = sm_out['self_state_vec'],
            sm_state_label    = sm_out['state_label'],
            prediction_error  = raw_error,
        )

        # ── 18. Language Seed (M64) ───────────────────────────
        # freq_idx: phoneme familiarity bucket.
        m64_out = self.language.step(
            sm_state_label    = sm_out['state_label'],
            sm_state_vec      = sm_out['self_state_vec'],
            freq_idx          = int(familiarity * 7),
            current_action    = _NO_ACTION,
            prediction_error  = raw_error,
            m63_just_closed   = len(m63_out['just_closed']) > 0,
            m63_episode       = m63_out['just_closed'][0] if m63_out['just_closed'] else None,
            q60_just_resolved = q60_out['just_resolved'],
        )

        # ── 19. Temporal Context (M67) ────────────────────────
        m67_out = self.temporal.step(
            global_step        = self.t,
            steps_since_reward = float(wm_out['steps_since_reward']),
            reward_ema         = float(v1_out['reward_ema']),
            got_reward         = (reward > 0.0),
        )

        # ── 20. Semantic Binding (M73) ────────────────────────
        # zone_idx: phoneme familiarity bucket (0-7).
        # High-familiarity phonemes bind to familiar "zones".
        m73_out = self.binding.step(
            phoneme_bmu  = bmu_idx,
            is_voiced    = m71_out['is_voiced'],
            state_vec    = sm_out['self_state_vec'],
            zone_idx     = int(familiarity * 7),
            reward       = reward,
            sound_bmu    = 0,
        )

        # ── 21. Vocal Output (M74) ────────────────────────────
        # zone_idx: familiarity bucket — vocal production biased toward
        # familiar phoneme regions (the brain speaks what it knows).
        m74_out = self.vocal.step(
            state_vec   = sm_out['self_state_vec'],
            state_label = sm_out['state_label'],
            zone_idx    = int(familiarity * 7),
            gws_arousal = float(gws_out['arousal']),
            ne_level    = float(m66_out['ne_level']),
            binding     = self.binding,
            phoneme_seq = self.phoneme_seq,
        )

        self.t += 1

        # ── Return unified state dict ─────────────────────────
        return {
            # M71: speech cortex (primary sensory)
            'm71_phoneme_bmu':        bmu_idx,
            'm71_qe_norm':            qe_norm,
            'm71_is_voiced':          m71_out['is_voiced'],
            'm71_phoneme_familiarity':m71_out['phoneme_familiarity'],
            # M72: phoneme sequence / word boundary
            'm72_tp':                 m72_out['tp_prev_curr'],
            'm72_boundary_strength':  m72_out['boundary_strength'],
            'm72_just_boundary':      m72_out['just_boundary'],
            'm72_n_word_candidates':  m72_out['n_word_candidates'],
            'm72_phoneme_pred_error': m72_out['phoneme_pred_error'],
            # M55: associative memory
            'familiarity':            familiarity,
            'top_associations':       recall_out['top_associations'],
            'wrote':                  mem_out['wrote'],
            # L2: sequence predictor
            'prediction_error':       raw_error,
            'correct':                l2_out['correct'],
            'predicted_bmu':          pred_out['predicted_bmu'],
            'confidence':             pred_out['confidence'],
            'curiosity':              raw_curiosity,
            # Feedback deltas
            'surprise_signal':        surprise_signal,
            'curiosity_delta':        curiosity_delta,
            'error_ema':              self._error_ema,
            'curiosity_ema':          self._curiosity_ema,
            # Attention
            'salience':               attn_out['salience'],
            'salience_ema':           attn_out['salience_ema'],
            'salience_delta':         attn_out['salience_delta'],
            'attention_gate':         attn_out['attention_gate'],
            'attended_bmu':           attn_out['attended_bmu'],
            # Thought
            'expected_bmu':           thought_out['expected_bmu'],
            'thought_confidence':     thought_out['thought_confidence'],
            'focus_entropy':          thought_out['focus_entropy'],
            'expectation_error':      thought_out['expectation_error'],
            # Valence
            'rpe':                    v1_out['rpe'],
            'pos_rpe':                v1_out['pos_rpe'],
            'neg_rpe':                v1_out['neg_rpe'],
            'reward_ema':             v1_out['reward_ema'],
            'total_reward':           v1_out['total_reward'],
            'intrinsic_reward':       v1_out['intrinsic_reward'],
            # Working Memory (M58)
            'wm_corridor_boredom':    wm_out['corridor_boredom'],
            'wm_steps_since_reward':  wm_out['steps_since_reward'],
            # Global Workspace
            'gws_arousal':            gws_out['arousal'],
            'gws_valence_tone':       gws_out['valence_tone'],
            'gws_curiosity_pull':     gws_out['curiosity_pull'],
            'gws_surprise_debt':      gws_out['surprise_debt'],
            'gws_coherence':          gws_out['coherence'],
            'gws_tension':            gws_out['tension'],
            'gws_ignited':            gws_out['ignited'],
            # Self-Model (M59)
            'sm_state_label':         sm_out['state_label'],
            'sm_state_vec':           sm_out['self_state_vec'],
            'sm_state_changed':       sm_out['state_changed'],
            'sm_steps_in_state':      sm_out['steps_in_state'],
            'sm_urgency':             sm_out['self_state_dict']['urgency'],
            'sm_clarity':             sm_out['self_state_dict']['clarity'],
            'sm_drive':               sm_out['self_state_dict']['drive'],
            'sm_novelty':             sm_out['self_state_dict']['novelty'],
            'sm_stability':           sm_out['self_state_dict']['stability'],
            'sm_confidence':          sm_out['self_state_dict']['confidence'],
            'sm_frustration':         sm_out['self_state_dict']['frustration'],
            'sm_engagement':          sm_out['self_state_dict']['engagement'],
            # Question Memory (M60)
            'q60_open_count':         q60_out['open_count'],
            'q60_zone_pull':          q60_out['zone_pull'],
            'q60_just_resolved':      q60_out['just_resolved'],
            'q60_just_created':       q60_out['just_created'],
            # M62: consistency
            'm62_consistency':        m62_out['consistency'],
            'm62_active':             m62_out['active'],
            # M63: episodic
            'm63_recognition':        m63_out['recognition'],
            'm63_episode_open':       m63_out['episode_open'],
            'm63_episodes_total':     m63_out['episodes_total'],
            'm63_just_closed':        m63_out['just_closed'],
            # M64: language seed
            'm64_situation_tag':      m64_out['situation_tag'],
            'm64_narrative':          m64_out['narrative'],
            'm64_tag_confidence':     m64_out['tag_confidence'],
            # M66: neuromodulators
            'm66_ach':                m66_out['ach_level'],
            'm66_ne':                 m66_out['ne_level'],
            'm66_sht':                m66_out['sht_level'],
            # M67: temporal context
            'm67_temporal_phase':     m67_out['m67_temporal_phase'],
            'm67_overdue_score':      m67_out['m67_overdue_score'],
            'm67_temporal_urgency':   m67_out['m67_temporal_urgency'],
            'm67_reward_trend':       m67_out['m67_reward_trend'],
            # M73: semantic binding
            'm73_associated_zone':    m73_out['associated_zone'],
            'm73_associated_reward':  m73_out['associated_reward'],
            'm73_binding_strength':   m73_out['binding_strength'],
            'm73_n_bound_phonemes':   m73_out['n_bound_phonemes'],
            # M74: vocal output
            'm74_speaking':           m74_out['speaking'],
            'm74_utterance':          m74_out['utterance'],
            'm74_utterance_len':      m74_out['utterance_len'],
            'm74_total_utterances':   m74_out['total_utterances'],
            # Step counter
            'step':                   self.t,
        }

    # ── Dreams ────────────────────────────────────────────────

    def dream(self, n_sequences: int = 8) -> dict:
        """
        Run one REM dream cycle (M70).

        Generates internally-driven sequences from the L2 prediction matrix
        (replay, counterfactual, novel). Weakly updates L2's weights.
        Call between waking steps — not during step().

        Returns dream report dict from DreamEngine.
        """
        return self.dreamer.dream(
            l3                = None,
            pred              = self.pred,
            food_trajectories = [],
            n_sequences       = n_sequences,
        )

    # ── Language experience recording ─────────────────────────

    # ── Word-level TP interface ─────────────────────────────────

    def hear_word(self, word: str):
        """
        Feed a word identity to the word-level transition predictor.
        Call once per word during training and live conversation.
        """
        if not hasattr(self, 'word_tp'):
            self.word_tp = WordTransitionPredictor()
        self.word_tp.observe(word)

    def hear_word_separator(self):
        """Mark input→response boundary in word TP."""
        if not hasattr(self, 'word_tp'):
            self.word_tp = WordTransitionPredictor()
        self.word_tp.observe_separator()

    def hear_word_end(self):
        """Mark end of response in word TP."""
        if not hasattr(self, 'word_tp'):
            self.word_tp = WordTransitionPredictor()
        self.word_tp.observe_end()

    def word_tp_generate(self, seed_words: list, max_len: int = 10) -> list:
        """
        Generate a response using the word-level TP.
        Returns a list of word strings.
        """
        if not hasattr(self, 'word_tp'):
            self.word_tp = WordTransitionPredictor()
        return self.word_tp.generate(
            seed_words=seed_words,
            max_len=max_len,
            rng=self.dreamer._rng,
        )

    def record_turn(self, bmus: list, reward: float, state_label: str = ''):
        """
        Record a conversation turn for later dreaming/replay.

        Called by live.py after each exchange:
            brain.record_turn(heard_bmus, turn_reward, sm_label)

        Parameters
        ----------
        bmus        : list of int — phoneme BMUs heard this turn
        reward      : float — total reward received this turn
                      (positive if user said 'good', negative if 'bad',
                       body-state reward otherwise)
        state_label : str — M59 state label during this turn
        """
        # Backwards-compat: old pickles won't have this attribute
        if not hasattr(self, '_word_trajectories'):
            from collections import deque as _deque
            self._word_trajectories = _deque(maxlen=200)
        if bmus:
            self._word_trajectories.append({
                'bmus':        list(bmus),
                'reward':      float(reward),
                'state_label': state_label,
            })

    def dream_language(self, n_sequences: int = 6) -> dict:
        """
        Language-domain dream cycle — replays stored word trajectories.

        Replaces the navigation-era brain.dream() for language mode.
        Runs three types of replay over _word_trajectories:

          REPLAY (40%)         — re-run a real turn's BMU sequence,
                                  strengthen TP transitions that occurred.
          COUNTERFACTUAL (40%) — swap one BMU in a turn with a random
                                  BMU from another turn, see if the
                                  blended sequence score is higher.
          NOVEL (20%)          — generate a new BMU sequence by walking
                                  the TP matrix from a random seed.

        All sequences weakly update:
          - phoneme_seq._P  (TP matrix) — strengthens transitions
          - binding._W_reward            — strengthens word-reward bindings
            for turns with positive reward

        Returns a summary dict.
        """
        if not hasattr(self, '_word_trajectories'):
            from collections import deque as _deque
            self._word_trajectories = _deque(maxlen=200)
        trajs = list(self._word_trajectories)
        rng   = self.dreamer._rng
        P     = self.phoneme_seq._P          # (N, N) TP matrix
        N     = P.shape[0]
        ETA   = 0.004   # gentle dream learning rate (<<waking rate)

        n_replay = n_cf = n_novel = 0
        total_value = 0.0
        n_p_updates = 0

        def _strengthen_sequence(bmu_seq, scale=1.0):
            """Strengthen TP transitions for a BMU sequence."""
            nonlocal n_p_updates
            for i in range(len(bmu_seq) - 1):
                a, b = bmu_seq[i], bmu_seq[i + 1]
                if 0 <= a < N and 0 <= b < N:
                    P[a, b] = float(np.clip(P[a, b] + ETA * scale, 0.0, 1.0))
                    n_p_updates += 1

        def _strengthen_reward_binding(bmu_seq, reward):
            """Strengthen reward binding for BMUs in a rewarded sequence."""
            if reward <= 0.0:
                return
            trace = np.zeros(N, dtype=np.float64)
            for b in bmu_seq:
                if 0 <= b < N:
                    trace[b] += 1.0
            s = trace.sum()
            if s > 0:
                trace /= s
                self.binding._W_reward = np.clip(
                    self.binding._W_reward + ETA * trace * reward,
                    0.0, None
                )

        for _ in range(n_sequences):
            r = rng.random()

            if r < 0.40 and trajs:
                # ── Replay dream ──────────────────────────────
                t = trajs[int(rng.integers(len(trajs)))]
                bmu_seq = t['bmus']
                reward  = t['reward']
                scale   = 1.0 + max(0.0, reward)   # reward scales reinforcement
                _strengthen_sequence(bmu_seq, scale=scale)
                _strengthen_reward_binding(bmu_seq, reward)
                total_value += reward
                n_replay += 1

            elif r < 0.80 and len(trajs) >= 2:
                # ── Counterfactual dream ──────────────────────
                # Splice together two real turns at a random point
                t1 = trajs[int(rng.integers(len(trajs)))]
                t2 = trajs[int(rng.integers(len(trajs)))]
                b1, b2 = t1['bmus'], t2['bmus']
                if len(b1) >= 2 and len(b2) >= 1:
                    cut = int(rng.integers(1, len(b1)))
                    blended = b1[:cut] + b2[:max(1, len(b2) - cut)]
                    # Blended value = average of both turns' rewards
                    blend_reward = 0.5 * (t1['reward'] + t2['reward'])
                    _strengthen_sequence(blended, scale=0.6)
                    _strengthen_reward_binding(blended, blend_reward)
                    total_value += blend_reward
                n_cf += 1

            else:
                # ── Novel dream: TP-walk from random seed ─────
                seed_bmu = int(rng.integers(N))
                novel_seq = [seed_bmu]
                for _ in range(8):
                    row = P[novel_seq[-1]].copy()
                    row = np.clip(row, 0.0, None)
                    s = row.sum()
                    if s < 1e-9:
                        break
                    row /= s
                    next_b = int(rng.choice(N, p=row))
                    novel_seq.append(next_bmu := next_b)
                _strengthen_sequence(novel_seq, scale=0.3)
                n_novel += 1

        return {
            'n_replay':       n_replay,
            'n_counterfactual': n_cf,
            'n_novel':        n_novel,
            'n_p_updates':    n_p_updates,
            'mean_value':     total_value / max(1, n_replay + n_cf),
            'trajectories_stored': len(trajs),
        }

    # ── Convenience ───────────────────────────────────────────

    def reset_feedback(self):
        """Reset feedback EMA state (use after a major context shift)."""
        self._error_ema             = float(FEEDBACK_EMA_INIT)
        self._curiosity_ema         = float(FEEDBACK_EMA_INIT)
        self._last_surprise_signal  = 0.0
        self._last_curiosity_delta  = 0.0
        self._last_rpe_positive     = 0.0
        self._last_familiarity      = 0.0
        self._last_prediction_error = 0.0
        self._last_curiosity        = 0.0

    def summary(self):
        """Print a brief status summary."""
        print(f"\n  Brain v13 — step {self.t}")
        print(f"  Primary sensory: M71 SpeechCortex (SOM step {self.speech_cortex.t})")
        print(f"  Feedback EMA:  error={self._error_ema:.4f}  curiosity={self._curiosity_ema:.4f}")
        print(f"  Language:")
        print(f"    Proto-words discovered: {len(self.phoneme_seq._word_candidates)}")
        n_bound = int(np.sum(self.binding._binding_strength > 0.05))
        print(f"    Bound phonemes:         {n_bound}")
        print(f"    Total utterances:       {self.vocal._total_utterances}")
        print(f"  Self-state: {self.selfmodel._last_label}")

    def save(self, path: str = 'brain_trained.pkl'):
        """Pickle the entire brain state to disk."""
        import pickle, pathlib
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Brain saved → {pathlib.Path(path).resolve()}")

    @staticmethod
    def load(path: str = 'brain_trained.pkl') -> 'Brain':
        """Load a pickled brain from disk."""
        import pickle
        with open(path, 'rb') as f:
            brain = pickle.load(f)
        # Backward compatibility: brains saved before M75 won't have semantic
        if not hasattr(brain, 'semantic'):
            brain.semantic = SemanticMemory()
        print(f"  Brain loaded ← {path}  (step {brain.t})")
        return brain
