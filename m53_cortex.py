"""
M53: CORTEX — BT-11 FIX (Curiosity boost)
==========================================
Inherits all M52 fixes (SIGMA_MIN=1.5, conscience) and adds one more
surgical fix to pass BT-11 (curiosity boost magnitude >2.0×).

═══════════════════════════════════════════════════════════════════════
ROOT CAUSE OF M52 BT-11 FAILURE (peak_boost=0.82×, target >2.0×)
═══════════════════════════════════════════════════════════════════════

Two interacting bugs compound to invert the curiosity signal:

BUG A — stability_w KILLS the curiosity boost (the primary culprit)
  M52 eta formula:
    eta_raw = ETA_MIN + (ETA_BASE - ETA_MIN) * (1 + NOVELTY_BOOST * qe_norm)
    eta     = eta_raw * stability_w

  Novel input DESTABILIZES the PLV (oscillators unlock/resync) → w drops.
  This is correct behaviour at the M50 decoder level, but it means:
    novel:    qe_norm=1.0 → eta_raw=0.43, w≈0.47 → eta=0.20
    familiar: qe_norm=0.32 → eta_raw=0.25, w≈1.0 → eta=0.25
  The stability gate and curiosity boost fight each other.
  Result: boost = 0.82× — INVERTED. Novel learns LESS than familiar.

  FIX A: Split eta into two additive terms.
    eta_base      = ETA_MIN + (ETA_BASE - ETA_MIN) * stability_w
      ↑ stability gate: trust the input proportionally to signal quality
    eta_curiosity = (ETA_BASE - ETA_MIN) * NOVELTY_BOOST * qe_norm
      ↑ curiosity bonus: additive, INDEPENDENT of stability
    eta = eta_base + eta_curiosity

  Biologically: even a noisy/uncertain novel signal SHOULD trigger
  heightened plasticity. Stability only affects the base learning rate,
  not whether the cortex prioritizes exploration.

BUG B — running_max includes current QE → suppresses qe_norm
  M52 step order:
    1. compute qe
    2. append qe to _qe_window              ← current (possibly huge) novel QE
    3. running_max = max(_qe_window)         ← dominated by the novel QE just appended
    4. qe_norm = qe / running_max            ← 1.22/1.22 = 1.0 looks fine...

  BUT the running_max from the previous familiar window was already ~0.017
  (due to noise peaks in the familiar range). So qe_norm_familiar ≈ 0.32,
  making the familiar baseline eta inflated.

  FIX B: Use an EMA (exponential moving average) as the QE baseline.
    qe_ema tracks the "expected" QE for known inputs.
    qe_norm = clip((qe - qe_ema) / (qe_ema + EPS), 0, 1)
    → familiar input: qe ≈ qe_ema → qe_norm ≈ 0 → eta = eta_base only
    → novel input:    qe >> qe_ema → qe_norm → 1 → full curiosity boost

  This is the correct biological formulation: surprise = excess above
  prediction, not raw magnitude. The EMA IS the prediction.

COMBINED RESULT:
  familiar: w=1.0, qe_norm=0.0 → eta = 0.15 + 0 = 0.15
  novel:    w=0.47, qe_norm=1.0 → eta = 0.076 + 0.28 = 0.356
  boost = 0.356 / 0.15 = 2.37×  ✓  (target >2.0×)

═══════════════════════════════════════════════════════════════════════
UNCHANGED FROM M52:
  - SIGMA_MIN=1.5, conscience learning (BT-08, BT-09 fixes) ✔
  - SOM architecture, prepare_input(), M50 interface ✔
  - All 17 passing M52 tests still pass ✔

CHANGE SUMMARY vs M52:
  Removed: QE_NORM_WINDOW, QE_NORM_MIN_VAL, _qe_window deque
  Added:   QE_EMA_ALPHA=0.01 (τ≈100 steps), _qe_ema float
  step():  qe_norm = clip((qe - _qe_ema)/(qe_ema + 1e-4), 0, 1)
           eta     = eta_base(w) + eta_curiosity(qe_norm)   [additive split]
           _qe_ema updated AFTER qe_norm computed
"""

import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Grid
GRID_H = 8
GRID_W = 8
N_NEURONS = GRID_H * GRID_W   # 64

# Input
N_PLV_COMPONENTS = 20
N_SCALARS        = 3
INPUT_DIM        = N_SCALARS + N_PLV_COMPONENTS  # 23

# Normalization
FREQ_MIN_HZ = 0.41
FREQ_MAX_HZ = 2.20

# Learning
ETA_BASE      = 0.15
ETA_MIN       = 0.01
NOVELTY_BOOST = 2.0

# FIX 1: Neighborhood σ floor raised from 0.5 → 1.5
# At σ=1.5: h(d=1)=0.80, h(d=2)=0.41, h(d=3)=0.14
# Every neuron within ~3 cells of winner always gets trained.
SIGMA_MAX = 3.5    # unchanged
SIGMA_MIN = 1.5    # was 0.5 — THIS IS THE PRIMARY FIX

SURPRISE_WINDOW = 100   # samples for σ modulation (~10s at sample_interval=2)

# FIX B: EMA baseline for qe_norm (replaces running-max approach)
# qe_norm = clip((qe - qe_ema) / (qe_ema + QE_EMA_EPS), 0, 1)
# qe_ema tracks the "expected" QE for familiar inputs (slow EMA).
# Surprise = excess above prediction, NOT raw magnitude.
# → familiar: qe ≈ qe_ema  → qe_norm ≈ 0 → no curiosity boost
# → novel:    qe >> qe_ema → qe_norm → 1 → full boost
# Alpha=0.01: τ ≈ 100 steps (~10s); adapts to steady-state without
# being contaminated by a single transient spike.
QE_EMA_ALPHA = 0.01    # EMA leak rate
QE_EMA_INIT  = 0.5     # conservative start (will decay to true ss quickly)
QE_EMA_EPS   = 1e-4    # floor to prevent div-by-zero at startup

# FIX 3: Conscience factor
# d_eff[i] = d[i] * (1 + CONSCIENCE_FACTOR * (p[i] - 1/N_NEURONS))
# 0.3 gives moderate pressure without destabilizing well-formed clusters
CONSCIENCE_FACTOR = 0.3
CONSCIENCE_LEAK   = 0.002   # EMA decay for win frequency p[i]
                             # τ ≈ 1/CONSCIENCE_LEAK ≈ 500 steps ≈ 50s

# Surprise threshold (unchanged)
SURPRISE_THRESH = 0.15


# ═══════════════════════════════════════════════════════════════
# INPUT PREPARATION  (unchanged from M51)
# ═══════════════════════════════════════════════════════════════

def prepare_input(decoded_freq, stability_w, novelty_flag, plv_vector):
    freq_norm = np.clip(
        (decoded_freq - FREQ_MIN_HZ) / (FREQ_MAX_HZ - FREQ_MIN_HZ),
        0.0, 1.0
    )
    plv_abs = np.abs(plv_vector)
    if len(plv_abs) >= N_PLV_COMPONENTS:
        top_idx = np.argpartition(plv_abs, -N_PLV_COMPONENTS)[-N_PLV_COMPONENTS:]
        top_plv = plv_abs[top_idx]
    else:
        top_plv = np.pad(plv_abs, (0, N_PLV_COMPONENTS - len(plv_abs)))

    plv_max = top_plv.max()
    if plv_max > 1e-9:
        top_plv = top_plv / plv_max

    return np.concatenate([
        [freq_norm, float(stability_w), float(novelty_flag)],
        top_plv
    ]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# CORTICAL COLUMN  (unchanged from M51)
# ═══════════════════════════════════════════════════════════════

class CorticalColumn:
    def __init__(self, row, col, input_dim, rng):
        self.row     = row
        self.col     = col
        self.weights = rng.uniform(0.0, 1.0, input_dim).astype(np.float32)

    def distance(self, input_vec):
        diff = input_vec - self.weights
        return float(np.dot(diff, diff))

    def grid_distance_sq(self, other):
        dr = self.row - other.row
        dc = self.col - other.col
        return dr*dr + dc*dc


# ═══════════════════════════════════════════════════════════════
# THE CORTEX — M52
# ═══════════════════════════════════════════════════════════════

class CortexM53:
    """
    Self-Organizing Cortical Map — M53 (BT-11 fixed).

    Changes vs M52:
      3. [NEW] EMA-based qe_norm: surprise = excess above prediction,
         not raw magnitude. Familiar input → qe_norm≈0. Novel → qe_norm≈1.
      4. [NEW] Additive curiosity term: eta = eta_base(w) + curiosity(qe_norm).
         stability_w gates the base learning only; curiosity is independent.
         This prevents w-drop during novel events from cancelling the boost.

    Retains M52 fixes:
      1. SIGMA_MIN=1.5 (no dead neurons)
      2. Conscience learning (uniform representation)

    Interface identical to M51/M52 — drop-in replacement.
    """

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

        # Build grid
        self.neurons = []
        self.grid    = {}
        for r in range(GRID_H):
            for c in range(GRID_W):
                idx    = r * GRID_W + c
                neuron = CorticalColumn(r, c, INPUT_DIM, self.rng)
                self.neurons.append(neuron)
                self.grid[(r, c)] = idx

        # Precompute pairwise grid distances (fixed topology)
        self._grid_dist_sq = np.zeros((N_NEURONS, N_NEURONS), np.float32)
        for i, ni in enumerate(self.neurons):
            for j, nj in enumerate(self.neurons):
                self._grid_dist_sq[i, j] = ni.grid_distance_sq(nj)

        # Weight matrix for fast BMU search
        self._W = np.array([n.weights for n in self.neurons],
                           dtype=np.float32)  # (64, 23)

        # Surprise history for σ modulation (unchanged from M51)
        self._surprise_history = deque(maxlen=SURPRISE_WINDOW)
        self._surprise_history.append(1.0)

        # FIX B: EMA baseline for adaptive qe_norm
        # Tracks expected (familiar) QE; surprise = excess above this.
        self._qe_ema = float(QE_EMA_INIT)

        # FIX 3: Conscience — win frequency per neuron
        # p[i] = EMA of whether neuron i won each step
        # Initialized to uniform: 1/N_NEURONS
        self._p = np.full(N_NEURONS, 1.0 / N_NEURONS, dtype=np.float64)

        # History (same interface as M51)
        self.qe_history    = []
        self.bmu_history   = []
        self.sigma_history = []
        self.eta_history   = []
        self.t             = 0

    # ── Core SOM step ─────────────────────────────────────────

    def step(self, decoded_freq, stability_w, novelty_flag, plv_vector):
        """
        One online learning step. Interface identical to M51.step().
        """
        # 1. Prepare input
        x = prepare_input(decoded_freq, stability_w, novelty_flag, plv_vector)

        # 2. COMPETE — find BMU with conscience penalty
        diff = self._W - x[np.newaxis, :]       # (64, 23)
        dists = np.sum(diff * diff, axis=1)      # (64,) raw squared distances

        # FIX 3: conscience penalty
        # d_eff[i] = d[i] * (1 + C * (p[i] - 1/N))
        # Over-winners → penalty > 0 → effectively farther from input
        # Under-winners → penalty < 0 → effectively closer
        target_p   = 1.0 / N_NEURONS
        conscience = 1.0 + CONSCIENCE_FACTOR * (self._p - target_p)
        dists_eff  = dists * np.maximum(conscience, 0.01)  # never invert

        bmu_idx  = int(np.argmin(dists_eff))
        bmu_dist = float(dists[bmu_idx])    # use raw distance for QE

        qe = float(np.sqrt(bmu_dist + 1e-12))

        # FIX 3: Update win frequency EMA
        # winner: p += leak*(1 - p);  losers: p -= leak*p
        winner_mask = np.zeros(N_NEURONS, dtype=np.float64)
        winner_mask[bmu_idx] = 1.0
        self._p += CONSCIENCE_LEAK * (winner_mask - self._p)
        self._p  = np.clip(self._p, 1e-6, 1.0)

        # 3. SURPRISE-DRIVEN σ (unchanged logic, but floor is now 1.5)
        mean_surprise = float(np.mean(self._surprise_history))
        sigma = SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * mean_surprise

        # 4. FIX B: EMA-based curiosity — surprise = excess above prediction
        # qe_norm = 0 when familiar (qe ≈ ema baseline)
        # qe_norm = 1 when novel (qe >> ema baseline)
        qe_norm_now = float(np.clip(
            (qe - self._qe_ema) / (self._qe_ema + QE_EMA_EPS),
            0.0, 1.0
        ))

        # FIX A: Additive curiosity term — independent of stability_w
        # eta_base:      w-gated base learning (trust input by signal quality)
        # eta_curiosity: novelty bonus, INDEPENDENT of w
        # Biologically: novel events should trigger heightened plasticity
        # even when the current signal is noisy/transitioning.
        eta_base      = ETA_MIN + (ETA_BASE - ETA_MIN) * float(stability_w)
        eta_curiosity = (ETA_BASE - ETA_MIN) * NOVELTY_BOOST * qe_norm_now
        eta           = max(eta_base + eta_curiosity, ETA_MIN)

        # Update EMA AFTER computing qe_norm (so current novel QE doesn't
        # inflate the baseline for the current sample — only future ones)
        self._qe_ema = (1.0 - QE_EMA_ALPHA) * self._qe_ema + QE_EMA_ALPHA * qe

        # 5. COOPERATE + ADAPT
        sigma_sq_2 = 2.0 * sigma * sigma
        h = np.exp(-self._grid_dist_sq[bmu_idx] / sigma_sq_2)  # (64,)

        delta    = x[np.newaxis, :] - self._W              # (64, 23)
        self._W += eta * h[:, np.newaxis] * delta
        self._W  = np.clip(self._W, 0.0, 1.0)

        # Sync neuron objects
        for i, neuron in enumerate(self.neurons):
            neuron.weights = self._W[i]

        # 6. UPDATE SURPRISE HISTORY (same as M51 — normalized for σ control)
        qe_norm_for_sigma = float(np.clip(qe / np.sqrt(INPUT_DIM), 0.0, 1.0))
        self._surprise_history.append(qe_norm_for_sigma)

        # 7. RECORD
        self.qe_history.append(qe)
        self.bmu_history.append(bmu_idx)
        self.sigma_history.append(sigma)
        self.eta_history.append(eta)
        self.t += 1

        bmu_neuron = self.neurons[bmu_idx]
        return {
            'qe':        qe,
            'qe_norm':   qe_norm_now,
            'bmu_idx':   bmu_idx,
            'bmu_pos':   (bmu_neuron.row, bmu_neuron.col),
            'sigma':     sigma,
            'eta':       eta,
            'is_novel':  qe > SURPRISE_THRESH,
            'input_vec': x,
        }

    # ── Analysis tools (identical interface to M51) ────────────

    def get_map_state(self):
        freq_map = np.zeros((GRID_H, GRID_W))
        w_map    = np.zeros((GRID_H, GRID_W))
        for neuron in self.neurons:
            r, c = neuron.row, neuron.col
            freq_norm    = neuron.weights[0]
            freq_map[r, c] = (freq_norm * (FREQ_MAX_HZ - FREQ_MIN_HZ) + FREQ_MIN_HZ)
            w_map[r, c]    = neuron.weights[1]
        return {
            'freq_map': freq_map,
            'w_map':    w_map,
            'weights':  self._W.copy(),
            'n_steps':  self.t,
            'mean_qe':  float(np.mean(self.qe_history[-100:]))
                        if self.qe_history else 0.0,
        }

    def get_surprise_stats(self):
        if not self.qe_history:
            return {}
        recent = self.qe_history[-100:]
        return {
            'mean':          float(np.mean(recent)),
            'std':           float(np.std(recent)),
            'max':           float(np.max(recent)),
            'min':           float(np.min(recent)),
            'current_sigma': float(np.mean(list(self._surprise_history))),
        }

    def neuron_activation_counts(self):
        counts = np.zeros(N_NEURONS, dtype=int)
        for idx in self.bmu_history:
            counts[idx] += 1
        return counts.reshape(GRID_H, GRID_W)

    def find_neuron_for_freq(self, target_freq):
        freq_norm = np.clip(
            (target_freq - FREQ_MIN_HZ) / (FREQ_MAX_HZ - FREQ_MIN_HZ),
            0.0, 1.0
        )
        diffs = np.abs(self._W[:, 0] - freq_norm)
        best  = int(np.argmin(diffs))
        n     = self.neurons[best]
        return (n.row, n.col), float(diffs[best])

    def get_conscience_state(self):
        """Diagnostic: win frequency distribution across neurons."""
        p_map = self._p.reshape(GRID_H, GRID_W)
        target = 1.0 / N_NEURONS
        return {
            'p_map':     p_map,
            'p_max':     float(self._p.max()),
            'p_min':     float(self._p.min()),
            'p_std':     float(self._p.std()),
            'p_target':  target,
            'gini':      float(np.sum(np.abs(
                             np.subtract.outer(self._p, self._p)
                         )) / (2 * N_NEURONS**2 * self._p.mean() + 1e-12)),
        }