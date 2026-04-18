"""
brain_fast.py — FastBrain: C++ core (FastSOM + FastTP) with Python WordTP
=========================================================================

Replaces the pure-Python Brain for 5000-neuron scale.

Architecture:
  FastSOM  (C++)  — 100×50 = 5000 neurons, OpenMP BMU search, ~5μs/step
  FastTP   (C++)  — sparse transition matrix, ~1μs accumulate
  WordTP   (Py)   — word-string → word-string transitions (no BMU aliasing)

Usage:
  brain = FastBrain()
  brain.hear(mfcc_vec)         # inject 13-dim MFCC frame
  brain.step(reward=0.0)       # one cognitive cycle → returns state dict
  brain.dream(n_sequences=20)  # offline TP consolidation
  brain.save('brain.pkl')
  brain = FastBrain.load('brain.pkl')
"""

import pickle
import numpy as np
from collections import defaultdict

# ── C++ core ──────────────────────────────────────────────────────────────────
try:
    import brain_core
    _CPP = True
except ImportError:
    _CPP = False
    print("WARNING: brain_core.so not found. Run ./build.sh first.")

# ── Dimensions ────────────────────────────────────────────────────────────────
SOM_ROWS  = 100
SOM_COLS  = 50
N_NEURONS = SOM_ROWS * SOM_COLS   # 5000
N_MFCC    = 13                    # MFCC feature dims

# ── Silence vector (used between words) ───────────────────────────────────────
SILENCE = np.zeros(N_MFCC, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# WORD-LEVEL TRANSITION PREDICTOR  (pure Python — strings, no BMU aliasing)
# ═══════════════════════════════════════════════════════════════════════════════

class WordTP:
    """
    Learns word → next-word transition probabilities from experience.

    Why kept in Python:
    - Transitions are between string tokens, not integer BMUs.
    - The vocabulary is ≤ 500 words → counts fit in a tiny dict.
    - No inner-product loops → no SIMD benefit.

    The 'separator' concept:
    When the brain hears input words and then must reply, we insert a
    <SEP> token between the input sequence and the response sequence.
    This teaches the TP to transition from "last heard word" → "first
    spoken word", which is the critical conditioned response.
    """

    SEP = '<SEP>'
    END = '<END>'

    def __init__(self):
        # counts[from_word][to_word] = float  (reward-weighted)
        self._counts: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._prev: str | None = None

    def observe(self, word: str, weight: float = 1.0):
        """Record that `word` followed the previous word."""
        if self._prev is not None:
            self._counts[self._prev][word] += weight
        self._prev = word

    def separator(self):
        """Mark transition from input sequence to response sequence."""
        self.observe(self.SEP)

    def end(self):
        """Mark end of exchange."""
        self.observe(self.END)
        self._prev = None

    def generate(self, context_words: list[str], max_len: int = 8,
                 temperature: float = 1.0) -> list[str]:
        """
        Generate a response word sequence given input context.

        Algorithm:
          1. Build a context distribution: weight each word after SEP by how
             much the context words predict it (direct TP lookup, not global sum).
          2. Sample the first response word from that context-weighted dist.
          3. Chain forward from the chosen word using the TP.

        The key fix vs. the old version: context has DOMINANT weight for the
        first word. This means "how are you" → first word is likely "i" or "good"
        (learned from training), not a random SEP follower.
        """
        if not self._counts:
            return []

        # ── Step 1: Build context-conditioned first-word distribution ──────
        # For each context word w, look at what follows SEP *intersected* with
        # what follows w directly. This finds words that are both (a) a common
        # response start and (b) semantically near the input.
        sep_followers: dict[str, float] = dict(self._counts.get(self.SEP, {}))

        ctx_boost: dict[str, float] = defaultdict(float)
        for w in context_words:
            if w in self._counts:
                for nxt, cnt in self._counts[w].items():
                    if nxt not in (self.SEP, self.END):
                        ctx_boost[nxt] += cnt

        # Combine: sep distribution × (1 + context boost)
        first_candidates: dict[str, float] = defaultdict(float)
        for w, cnt in sep_followers.items():
            if w in (self.SEP, self.END):
                continue
            boost = 1.0 + ctx_boost.get(w, 0.0) * 2.0
            first_candidates[w] += cnt * boost

        # Also allow context-boosted words even if not in SEP followers
        for w, boost in ctx_boost.items():
            if w not in first_candidates and w not in (self.SEP, self.END):
                first_candidates[w] += boost * 0.5

        if not first_candidates:
            return []

        # ── Step 2: Sample first word ──────────────────────────────────────
        words  = list(first_candidates.keys())
        counts = np.array([first_candidates[w] for w in words], dtype=np.float64)
        counts = np.log(counts + 1e-9)
        counts -= counts.max()
        counts = np.exp(counts / max(temperature, 1e-3))
        counts /= counts.sum()

        cur    = np.random.choice(words, p=counts)
        result = [cur]
        seen   = {cur}

        # ── Step 3: Chain forward from first word ──────────────────────────
        for _ in range(max_len - 1):
            candidates: dict[str, float] = defaultdict(float)

            if cur in self._counts:
                for nxt, cnt in self._counts[cur].items():
                    if nxt not in (self.SEP, self.END):
                        candidates[nxt] += cnt

            # Light context pull throughout the chain
            for w, boost in ctx_boost.items():
                if w not in (self.SEP, self.END):
                    candidates[w] += boost * 0.15

            if not candidates:
                break

            end_weight = self._counts.get(cur, {}).get(self.END, 0.0)

            words  = list(candidates.keys())
            counts = np.array([candidates[w] for w in words], dtype=np.float64)
            counts = np.log(counts + 1e-9)
            counts -= counts.max()
            counts = np.exp(counts / max(temperature, 1e-3))
            counts /= counts.sum()

            # Penalise already-seen words
            for i, w in enumerate(words):
                if w in seen:
                    counts[i] *= 0.2
            total = counts.sum()
            if total < 1e-12:
                break
            counts /= total

            chosen = np.random.choice(words, p=counts)

            # Natural stop: END token probability
            if end_weight > 0:
                total_mass = end_weight + candidates.get(chosen, 1.0)
                if np.random.random() < (end_weight / total_mass):
                    break

            result.append(chosen)
            seen.add(chosen)
            cur = chosen

        return result

    def n_words(self) -> int:
        return len(self._counts)

    def n_transitions(self) -> int:
        return sum(len(v) for v in self._counts.values())

    def get_state(self) -> dict:
        return {'counts': {k: dict(v) for k, v in self._counts.items()},
                'prev': self._prev}

    def set_state(self, state: dict):
        self._counts = defaultdict(lambda: defaultdict(float))
        for k, v in state['counts'].items():
            self._counts[k] = defaultdict(float, v)
        self._prev = state.get('prev')


# ═══════════════════════════════════════════════════════════════════════════════
# FAST BRAIN
# ═══════════════════════════════════════════════════════════════════════════════

class FastBrain:
    """
    5000-neuron brain using C++ FastSOM + FastTP for the acoustic/phoneme
    layer, plus Python WordTP for word-level language learning.

    Lifecycle:
      hear(mfcc) → step(reward) called in a tight loop during training/live.
      dream(n)   called after each epoch to consolidate TP transitions.
      save/load  persists the entire state to disk.

    State dict returned by step():
      {
        'bmu':      int    — current acoustic BMU
        'reward':   float  — reward this step
        'n_steps':  int    — total steps seen
      }
    """

    def __init__(self):
        if not _CPP:
            raise RuntimeError("brain_core.so not found. Run ./build.sh first.")

        # C++ acoustic SOM (primary sensory cortex)
        self.som = brain_core.FastSOM(rows=SOM_ROWS, cols=SOM_COLS, n_dims=N_MFCC)

        # C++ phoneme-level TP
        self.tp = brain_core.FastTP(n_neurons=N_NEURONS)

        # Python word-level TP (string transitions)
        self.word_tp = WordTP()

        # State
        self._prev_prev_bmu: int = 0
        self._prev_bmu:      int = 0
        self._current_bmu:   int = 0
        self._last_mfcc:   np.ndarray = SILENCE.copy()
        self._n_steps:     int   = 0
        self._total_reward: float = 0.0

        # Dream buffer: list of (bmu_sequence, reward)
        self._dream_buffer: list[tuple[list[int], float]] = []
        self._max_dream_buffer = 500

        # BMU → word map (built during training, used for response generation)
        self.bmu_to_word: dict[int, str] = {}
        self.word_to_bmu: dict[str, int] = {}

    # ── Sensory input ─────────────────────────────────────────────────────────

    def hear(self, mfcc_vec: np.ndarray):
        """Inject a 13-dim MFCC frame. Call before step()."""
        self._last_mfcc = np.asarray(mfcc_vec, dtype=np.float32)

    def hear_word(self, word: str, mfcc_vec: np.ndarray | None = None):
        """
        Record a word in the word-level TP.
        Optionally update the BMU→word map from the provided MFCC.
        """
        self.word_tp.observe(word)
        if mfcc_vec is not None:
            bmu = self.som.find_bmu(np.asarray(mfcc_vec, dtype=np.float32))
            if word not in self.word_to_bmu:
                self.word_to_bmu[word] = bmu
                self.bmu_to_word[bmu] = word

    def hear_word_separator(self):
        """Insert input→response boundary in word TP."""
        self.word_tp.separator()

    def hear_word_end(self):
        """Mark end of exchange in word TP."""
        self.word_tp.end()

    # ── Cognitive step ────────────────────────────────────────────────────────

    def step(self, reward: float = 0.0) -> dict:
        """
        Run one cognitive cycle:
          1. Find BMU for current MFCC frame (C++ parallel search).
          2. Update SOM weights (reward-modulated Kohonen).
          3. Record TP transition from previous BMU.
          4. If reward > 0, reinforce recent transitions.

        Returns state dict.
        """
        mfcc = self._last_mfcc
        bmu  = self.som.find_bmu(mfcc)

        # Modulated SOM update: reward boosts learning rate
        reward_mod = 1.0 + reward * 2.0
        self.som.update(mfcc, bmu, reward_mod)

        # TP: observe (prev_prev, prev) → current transition
        if self._n_steps > 1:
            self.tp.observe(self._prev_prev_bmu, self._prev_bmu, bmu)

        # Reward: reinforce the (prev_prev, prev) → current transition
        if reward > 0 and self._n_steps > 1:
            self.tp.reinforce(self._prev_prev_bmu, self._prev_bmu, bmu, reward)

        self._prev_prev_bmu = self._prev_bmu
        self._prev_bmu      = bmu
        self._current_bmu   = bmu
        self._n_steps    += 1
        self._total_reward += reward

        return {
            'bmu':     bmu,
            'reward':  reward,
            'n_steps': self._n_steps,
        }

    # ── Response generation ───────────────────────────────────────────────────

    def word_tp_generate(self, context_words: list[str],
                         max_len: int = 8,
                         temperature: float = 1.0) -> list[str]:
        """Generate a word-sequence response using the word-level TP."""
        return self.word_tp.generate(context_words, max_len=max_len,
                                     temperature=temperature)

    def generate_bmus(self, seed_bmus: list[int], n_steps: int = 12,
                      temperature: float = 1.2) -> list[int]:
        """
        Walk the phoneme-level TP from seed BMUs using bi-gram context.
        """
        if not seed_bmus:
            return []

        # We need at least 2 context seeds for the 2nd-order markov chain
        current_ctx = list(seed_bmus)
        if len(current_ctx) == 1:
            current_ctx = [current_ctx[0], current_ctx[0]]

        result = []
        for _ in range(n_steps):
            dist = self.tp.get_distribution(current_ctx[-2:], temperature=temperature)
            cur_bmu = self.tp.sample(dist)
            result.append(cur_bmu)
            
            # Slide window
            current_ctx.append(cur_bmu)

        return result

    def bmus_to_words(self, bmus: list[int]) -> list[str]:
        """Map a BMU sequence to word strings using the bmu_to_word map."""
        words = []
        seen  = set()
        for b in bmus:
            w = self.bmu_to_word.get(b)
            if w and w not in seen:
                words.append(w)
                seen.add(w)
        return words

    # ── Dream / offline consolidation ────────────────────────────────────────

    def record_turn(self, bmus: list[int], reward: float):
        """Reinforce a successful interaction sequence (used in dialogue)."""
        if len(bmus) < 3:
            return
        for i in range(2, len(bmus)):
            self.tp.reinforce(bmus[i-2], bmus[i-1], bmus[i], reward)
        if len(self._dream_buffer) >= self._max_dream_buffer:
            self._dream_buffer.pop(0)
        self._dream_buffer.append((list(bmus), reward))

    def dream(self, n_sequences: int = 20):
        """
        Offline TP consolidation: replay stored BMU trajectories with
        extra reinforcement. Mimics hippocampal replay during sleep.

        Biological basis: during slow-wave sleep, the hippocampus
        'replays' recent sequences to the cortex, strengthening
        long-term synaptic traces without new sensory input.
        """
        if not self._dream_buffer:
            return

        n = min(n_sequences, len(self._dream_buffer))
        indices = np.random.choice(len(self._dream_buffer), n, replace=False)

        for idx in indices:
            bmus, reward = self._dream_buffer[idx]
            if len(bmus) < 3:
                continue
            for i in range(2, len(bmus)):
                self.tp.observe(bmus[i-2], bmus[i-1], bmus[i], weight=0.5)
                if reward > 0:
                    self.tp.reinforce(bmus[i-2], bmus[i-1], bmus[i], reward * 0.5)

    def dream_language(self, n_sequences: int = 20):
        """Alias for dream() — used by train_dialogue.py."""
        self.dream(n_sequences)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """
        Save brain state to disk.

        Uses pickle for the Python objects (WordTP counts dict, dream buffer,
        bmu maps). Uses brain_core's own __getstate__ for C++ objects —
        these are plain numpy arrays embedded inside the pickle stream.

        Security note: only load .pkl files you created yourself.
        """
        state = {
            'som_state':    self.som.__getstate__(),
            'tp_state':     self.tp.__getstate__(),
            'word_tp':      self.word_tp.get_state(),
            'bmu_to_word':  self.bmu_to_word,
            'word_to_bmu':  self.word_to_bmu,
            'prev_prev_bmu':self._prev_prev_bmu,
            'prev_bmu':     self._prev_bmu,
            'n_steps':      self._n_steps,
            'total_reward': self._total_reward,
            'dream_buffer': self._dream_buffer,
            'version':      2,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)

    @classmethod
    def load(cls, path: str) -> 'FastBrain':
        """Load brain from a .pkl file saved by save()."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        brain = cls.__new__(cls)

        if not _CPP:
            raise RuntimeError("brain_core.so not found. Run ./build.sh first.")

        brain.som = brain_core.FastSOM(rows=SOM_ROWS, cols=SOM_COLS, n_dims=N_MFCC)
        brain.som.__setstate__(state['som_state'])

        brain.tp = brain_core.FastTP(n_neurons=N_NEURONS)
        brain.tp.__setstate__(state['tp_state'])

        brain.word_tp = WordTP()
        brain.word_tp.set_state(state['word_tp'])

        brain.bmu_to_word  = state.get('bmu_to_word', {})
        brain.word_to_bmu  = state.get('word_to_bmu', {})
        brain._prev_prev_bmu = state.get('prev_prev_bmu', 0)
        brain._prev_bmu    = state.get('prev_bmu', 0)
        brain._current_bmu = brain._prev_bmu
        brain._n_steps     = state.get('n_steps', 0)
        brain._total_reward = state.get('total_reward', 0.0)
        brain._dream_buffer = state.get('dream_buffer', [])
        brain._max_dream_buffer = 500
        brain._last_mfcc   = SILENCE.copy()

        return brain

    # ── Introspection ─────────────────────────────────────────────────────────

    def status(self) -> str:
        lines = [
            f"FastBrain — {N_NEURONS} neurons ({SOM_ROWS}×{SOM_COLS})",
            f"  Steps:      {self._n_steps:,}",
            f"  Reward:     {self._total_reward:.1f}",
            f"  Word TP:    {self.word_tp.n_words()} words, "
            f"{self.word_tp.n_transitions()} transitions",
            f"  BMU map:    {len(self.bmu_to_word)} words mapped",
            f"  Dream buf:  {len(self._dream_buffer)} trajectories",
            f"  OpenMP:     {brain_core.has_openmp}",
            f"  Threads:    {brain_core.n_threads}",
        ]
        return '\n'.join(lines)
