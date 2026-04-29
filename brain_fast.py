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
from vocab import N_MFCC           # single source of truth — currently 17

# ── Silence vector (used between words) ───────────────────────────────────────
SILENCE = np.zeros(N_MFCC, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# WORD-LEVEL TRANSITION PREDICTOR  (pure Python — strings, no BMU aliasing)
# ═══════════════════════════════════════════════════════════════════════════════

class GRUWordTP:
    """
    GRU-based word sequence generator. Replaces the bigram WordTP.

    Key improvement: hidden state h carries sequence context across all steps.
    In a bigram Markov chain, P(word_n | word_{n-1}) — only one step of history.
    In this GRU, P(word_n | h_{n-1}) where h encodes the entire sequence so far.

    Consequence: "not dangerous" — "not" changes h, so the GRU predicts a
    different distribution for the next word. Operator scope is not a grammar
    rule — it emerges from the learned recurrent dynamics.

    Training: truncated BPTT(1) — gradients computed per step without
    backpropagating through h. Simple, fast, correct for short sequences (≤8 words).
    Generation: full GRU forward pass — h accumulates context across all steps.

    Warm-start: output bias initialized from existing bigram counts so the GRU
    starts as capable as the Markov chain and only improves from there.

    Backward compat: bigram _counts kept as fallback before first fit().
    Old .pkl WordTP state is detected in FastBrain.load() and migrated.
    """

    SEP = '<SEP>'
    END = '<END>'

    def __init__(self, hidden_dim: int = 48):
        self._H = hidden_dim
        self._vocab: list[str] = [self.SEP, self.END]
        self._w2i: dict[str, int] = {self.SEP: 0, self.END: 1}

        # Sequence buffer for training
        self._sequences: list[list[str]] = []
        self._cur_seq: list[str] = [self.SEP]
        self._prev: str | None = None

        # Bigram counts — fallback before GRU trained, also used for warm-start
        self._counts: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float))

        # GRU parameters — None until first fit()
        self._Wz = self._Wr = self._Wh = None   # (H, H+V) each
        self._bz = self._br = self._bh = None   # (H,) each
        self._Wo = self._bo = None               # (V, H) and (V,)
        self._trained = False
        self._V_at_train = 0

    # ── Vocabulary ────────────────────────────────────────────────────────────

    def _add_word(self, word: str) -> int:
        if word not in self._w2i:
            self._w2i[word] = len(self._vocab)
            self._vocab.append(word)
        return self._w2i[word]

    # ── Observe (same interface as old WordTP) ────────────────────────────────

    def observe(self, word: str, weight: float = 1.0):
        self._add_word(word)
        if self._prev is not None:
            self._counts[self._prev][word] += weight
        self._prev = word
        self._cur_seq.append(word)

    def separator(self):
        self.observe(self.SEP)

    def end(self):
        self.observe(self.END)
        self._prev = None
        if len(self._cur_seq) >= 4:
            self._sequences.append(list(self._cur_seq))
        self._cur_seq = [self.SEP]

    # ── GRU math ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def _step(self, idx: int, h: np.ndarray):
        """GRU forward step. Returns (h_new, logits, cache)."""
        V = len(self._vocab)
        x = np.zeros(V, dtype=np.float32)
        x[idx] = 1.0
        hx = np.concatenate([h, x])                         # (H+V,)
        z  = self._sigmoid(self._Wz @ hx + self._bz)        # update gate (H,)
        r  = self._sigmoid(self._Wr @ hx + self._br)        # reset gate  (H,)
        hx_r    = np.concatenate([r * h, x])                # (H+V,)
        h_tilde = np.tanh(self._Wh @ hx_r + self._bh)      # candidate   (H,)
        h_new   = (1.0 - z) * h_tilde + z * h              # new hidden  (H,)
        logits  = self._Wo @ h_new + self._bo               # raw scores  (V,)
        cache   = dict(x=x, h=h, hx=hx, z=z, r=r,
                       hx_r=hx_r, h_tilde=h_tilde, h_new=h_new)
        return h_new, logits, cache

    def _step_grad(self, cache: dict, target: int,
                   logits: np.ndarray, lr: float):
        """
        Truncated BPTT(1): update GRU weights for one step.
        Gradient does not flow back through previous h — treats h as constant.
        This eliminates vanishing/exploding gradient risk for our short sequences.
        """
        h, x, hx, z, r, hx_r, h_tilde, h_new = (
            cache['h'], cache['x'], cache['hx'], cache['z'], cache['r'],
            cache['hx_r'], cache['h_tilde'], cache['h_new'])

        # ── Output layer ──────────────────────────────────────────────────────
        probs = np.exp(logits - logits.max()).astype(np.float64)
        probs /= probs.sum()
        dl = probs.astype(np.float32)
        dl[target] -= 1.0                                    # dL/dlogits (V,)

        dh = self._Wo.T @ dl                                 # dL/dh_new  (H,)
        self._Wo -= lr * np.outer(dl, h_new)
        self._bo -= lr * dl

        # ── Backprop through h_new = (1-z)*h_tilde + z*h ─────────────────────
        dh_tilde = dh * (1.0 - z)
        dz       = dh * (h - h_tilde)

        # ── Backprop through h_tilde = tanh(Wh @ hx_r + bh) ─────────────────
        dh_tilde_pre = dh_tilde * (1.0 - h_tilde ** 2)
        self._Wh -= lr * np.outer(dh_tilde_pre, hx_r)
        self._bh -= lr * dh_tilde_pre

        # ── Backprop through z gate ───────────────────────────────────────────
        dz_pre = dz * z * (1.0 - z)
        self._Wz -= lr * np.outer(dz_pre, hx)
        self._bz -= lr * dz_pre

        # ── Backprop through r gate ───────────────────────────────────────────
        dr_h   = (self._Wh.T @ dh_tilde_pre)[:self._H]     # dL/d(r*h)
        dr     = dr_h * h
        dr_pre = dr * r * (1.0 - r)
        self._Wr -= lr * np.outer(dr_pre, hx)
        self._br -= lr * dr_pre

    # ── Training ──────────────────────────────────────────────────────────────

    def _init_weights(self):
        V, H = len(self._vocab), self._H
        s = float(np.sqrt(2.0 / (H + V)))
        self._Wz = (np.random.randn(H, H + V) * s).astype(np.float32)
        self._Wr = (np.random.randn(H, H + V) * s).astype(np.float32)
        self._Wh = (np.random.randn(H, H + V) * s).astype(np.float32)
        self._bz = np.zeros(H, dtype=np.float32)
        self._br = np.zeros(H, dtype=np.float32)
        self._bh = np.zeros(H, dtype=np.float32)
        self._Wo = (np.random.randn(V, H) * float(np.sqrt(2.0 / (V + H)))).astype(np.float32)
        self._bo = np.zeros(V, dtype=np.float32)
        # Warm-start: output bias from bigram log-probs so GRU starts
        # as capable as the Markov chain and only improves from there.
        for fw, nexts in self._counts.items():
            total = sum(nexts.values()) + 1e-9
            for tw, cnt in nexts.items():
                if tw in self._w2i:
                    self._bo[self._w2i[tw]] += float(np.log(cnt / total + 1e-7)) * 0.08
        self._V_at_train = V

    def _resize_weights(self):
        """Extend matrices when vocabulary grows after last fit()."""
        V_old, V_new, H = self._V_at_train, len(self._vocab), self._H
        if V_new <= V_old:
            return
        extra = V_new - V_old
        s = float(np.sqrt(2.0 / (H + V_new)))
        for attr in ('_Wz', '_Wr', '_Wh'):
            W = getattr(self, attr)
            setattr(self, attr, np.hstack(
                [W, (np.random.randn(H, extra) * s).astype(np.float32)]))
        ext_rows = (np.random.randn(extra, H) * s).astype(np.float32)
        self._Wo = np.vstack([self._Wo, ext_rows])
        self._bo = np.concatenate([self._bo, np.zeros(extra, dtype=np.float32)])
        self._V_at_train = V_new

    def fit(self, epochs: int = 15, lr: float = 0.02, max_sequences: int = 800):
        """
        Train GRU on buffered sequences.
        max_sequences: cap training buffer to most-recent N sequences so fit
        time stays constant regardless of total run length.
        """
        if len(self._sequences) < 3:
            return
        if self._Wz is None:
            self._init_weights()
        elif len(self._vocab) > self._V_at_train:
            self._resize_weights()

        seqs = (self._sequences[-max_sequences:]
                if len(self._sequences) > max_sequences
                else self._sequences)

        H = self._H
        for ep in range(epochs):
            ep_lr = lr * (0.95 ** ep)
            perm = np.random.permutation(len(seqs))
            for si in perm:
                seq = seqs[si]
                idxs = [self._w2i.get(w, 0) for w in seq]
                h = np.zeros(H, dtype=np.float32)
                for t in range(len(idxs) - 1):
                    h, logits, cache = self._step(idxs[t], h)
                    self._step_grad(cache, idxs[t + 1], logits, ep_lr)
        self._trained = True
        # Trim stored buffer to avoid unbounded memory growth
        if len(self._sequences) > max_sequences:
            self._sequences = self._sequences[-max_sequences:]

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, context_words: list[str], max_len: int = 8,
                 temperature: float = 1.0,
                 forbidden: set | None = None) -> list[str]:
        """
        Generate word sequence.
        forbidden: words suppressed during generation — constrained decoding.
                   Brain will never output a word it knows is false in context.
        Falls back to bigram if GRU not yet trained.
        """
        fb = forbidden or set()
        if self._trained:
            if len(self._vocab) > self._V_at_train:
                self._resize_weights()
            return self._generate_gru(context_words, max_len, temperature, fb)
        return self._generate_bigram(context_words, max_len, temperature, fb)

    def _generate_gru(self, context_words: list[str], max_len: int,
                      temperature: float, forbidden: set) -> list[str]:
        V, H = len(self._vocab), self._H
        sep_i = self._w2i[self.SEP]
        end_i = self._w2i[self.END]

        # Prime hidden state: run SEP then each context word through GRU
        h = np.zeros(H, dtype=np.float32)
        h, _, _ = self._step(sep_i, h)
        for w in context_words:
            if w in self._w2i:
                h, _, _ = self._step(self._w2i[w], h)

        result: list[str] = []
        seen:   set[str]  = set()

        for _ in range(max_len):
            cur_i = self._w2i.get(result[-1], sep_i) if result else sep_i
            h, logits, _ = self._step(cur_i, h)

            logits = logits / max(temperature, 1e-3)
            probs  = np.exp(logits - logits.max()).astype(np.float64)
            probs /= probs.sum()

            # Suppress: forbidden words + already seen + special tokens
            suppress = forbidden | seen | {self.SEP, self.END}
            for w in suppress:
                if w in self._w2i:
                    probs[self._w2i[w]] = 0.0

            end_p = float(probs[end_i])
            total = probs.sum()
            if total < 1e-12:
                break
            probs /= total

            if end_p > 0.3 and len(result) >= 2:
                break

            chosen_i = int(np.random.choice(V, p=probs))
            chosen   = self._vocab[chosen_i]
            if chosen in (self.SEP, self.END):
                break

            result.append(chosen)
            seen.add(chosen)

        return result

    def _generate_bigram(self, context_words: list[str], max_len: int,
                         temperature: float, forbidden: set) -> list[str]:
        """Bigram fallback — identical to original WordTP.generate() logic."""
        if not self._counts:
            return []

        sep_followers: dict[str, float] = dict(self._counts.get(self.SEP, {}))
        ctx_boost: dict[str, float] = defaultdict(float)
        for w in context_words:
            if w in self._counts:
                for nxt, cnt in self._counts[w].items():
                    if nxt not in (self.SEP, self.END):
                        ctx_boost[nxt] += cnt

        first_cand: dict[str, float] = defaultdict(float)
        for w, cnt in sep_followers.items():
            if w in (self.SEP, self.END) or w in forbidden:
                continue
            first_cand[w] += cnt * (1.0 + ctx_boost.get(w, 0.0) * 2.0)
        for w, boost in ctx_boost.items():
            if w not in first_cand and w not in (self.SEP, self.END) and w not in forbidden:
                first_cand[w] += boost * 0.5

        if not first_cand:
            return []

        words  = list(first_cand.keys())
        cnts   = np.array([first_cand[w] for w in words], dtype=np.float64)
        cnts   = np.exp((np.log(cnts + 1e-9) - np.log(cnts + 1e-9).max()) / max(temperature, 1e-3))
        cnts  /= cnts.sum()

        cur    = np.random.choice(words, p=cnts)
        result = [cur]
        seen   = {cur}

        for _ in range(max_len - 1):
            cand: dict[str, float] = defaultdict(float)
            if cur in self._counts:
                for nxt, cnt in self._counts[cur].items():
                    if nxt not in (self.SEP, self.END) and nxt not in forbidden:
                        cand[nxt] += cnt
            for w, boost in ctx_boost.items():
                if w not in (self.SEP, self.END) and w not in forbidden:
                    cand[w] += boost * 0.15
            if not cand:
                break

            end_w = self._counts.get(cur, {}).get(self.END, 0.0)
            ws    = list(cand.keys())
            cs    = np.array([cand[w] for w in ws], dtype=np.float64)
            cs    = np.exp((np.log(cs + 1e-9) - np.log(cs + 1e-9).max()) / max(temperature, 1e-3))
            for i, w in enumerate(ws):
                if w in seen:
                    cs[i] *= 0.2
            cs /= cs.sum() if cs.sum() > 1e-12 else 1.0
            chosen = np.random.choice(ws, p=cs)

            if end_w > 0:
                if np.random.random() < end_w / (end_w + cand.get(chosen, 1.0)):
                    break

            result.append(chosen)
            seen.add(chosen)
            cur = chosen

        return result

    # ── Compatibility ─────────────────────────────────────────────────────────

    def n_words(self) -> int:
        return max(0, len(self._vocab) - 2)

    def n_transitions(self) -> int:
        return sum(len(v) for v in self._counts.values())

    def get_state(self) -> dict:
        return {
            'gru': True,
            'vocab': self._vocab, 'w2i': self._w2i,
            'counts': {k: dict(v) for k, v in self._counts.items()},
            'sequences': self._sequences,
            'trained': self._trained, 'V_at_train': self._V_at_train,
            'Wz': self._Wz, 'Wr': self._Wr, 'Wh': self._Wh,
            'bz': self._bz, 'br': self._br, 'bh': self._bh,
            'Wo': self._Wo, 'bo': self._bo,
            'hidden_dim': self._H,
        }

    def set_state(self, state: dict):
        self._H        = state.get('hidden_dim', self._H)
        self._vocab    = state.get('vocab', [self.SEP, self.END])
        self._w2i      = state.get('w2i', {self.SEP: 0, self.END: 1})
        self._sequences = state.get('sequences', [])
        self._trained  = state.get('trained', False)
        self._V_at_train = state.get('V_at_train', 0)
        self._counts   = defaultdict(lambda: defaultdict(float))
        for k, v in state.get('counts', {}).items():
            self._counts[k] = defaultdict(float, v)
        self._Wz = state.get('Wz')
        self._Wr = state.get('Wr')
        self._Wh = state.get('Wh')
        self._bz = state.get('bz')
        self._br = state.get('br')
        self._bh = state.get('bh')
        self._Wo = state.get('Wo')
        self._bo = state.get('bo')


# Use C++ FastGRU when available; fall back to Python GRUWordTP
try:
    from brain_gru import FastGRU as _CppGRU

    class WordTP(_CppGRU):
        """C++ GRU with translation layer for legacy Python state format."""

        def generate(self, context_words: list[str], max_len: int = 8,
                     temperature: float = 1.0,
                     forbidden: set | None = None) -> list[str]:
            fb = list(forbidden) if forbidden else []
            return super().generate(context_words, max_len=max_len,
                                    temperature=float(temperature),
                                    forbidden_words=fb)

        def set_state(self, s: dict):
            t = dict(s)
            # hidden_dim → H
            if 'H' not in t:
                t['H'] = t.pop('hidden_dim', 64)
            vocab = t.get('vocab', ['<SEP>', '<END>'])
            t['V'] = len(vocab)
            w2i = {w: i for i, w in enumerate(vocab)}
            # counts: string-keyed → int-keyed
            old_counts = t.get('counts', {})
            new_counts: dict = {}
            for fk, targets in old_counts.items():
                fi = w2i.get(fk, -1) if isinstance(fk, str) else int(fk)
                if fi < 0: continue
                inner: dict = {}
                for tk, cnt in targets.items():
                    ti = w2i.get(tk, -1) if isinstance(tk, str) else int(tk)
                    if ti >= 0: inner[ti] = float(cnt)
                if inner: new_counts[fi] = inner
            t['counts'] = new_counts
            # sequences: string lists → int lists
            new_seqs = []
            for seq in t.get('sequences', []):
                if seq and isinstance(seq[0], str):
                    new_seqs.append([w2i.get(w, 0) for w in seq])
                else:
                    new_seqs.append(list(seq))
            t['sequences'] = new_seqs
            super().set_state(t)

        def get_state(self) -> dict:
            return super().get_state()

except ImportError:
    WordTP = GRUWordTP


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
                         temperature: float = 1.0,
                         forbidden: set | None = None) -> list[str]:
        """Generate a word-sequence response using the word-level TP."""
        return self.word_tp.generate(context_words, max_len=max_len,
                                     temperature=float(temperature),
                                     forbidden=forbidden)

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
