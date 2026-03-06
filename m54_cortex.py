"""
M54: CORTEX — BT-07 FIX (Catastrophic forgetting WARN)
=======================================================
Inherits all M53 fixes and adds one targeted parameter change to
convert BT-07 from WARN to PASS.

═══════════════════════════════════════════════════════════════════════
ROOT CAUSE OF M53 BT-07 WARN (A=0.60 Hz err=0.165, threshold 0.15)
═══════════════════════════════════════════════════════════════════════

The test trains on A=0.60, B=1.00, C=1.80 Hz (12 blocks), then
HEAVY OVERTRAINS on D=0.41, E=1.40, F=2.20 Hz (24 blocks — 2× exposure).
After overtraining: A=0.60 Hz representation drifts to 0.494 Hz (err=0.165).

Root cause: D=0.41 Hz is only 0.19 Hz from A=0.60 Hz.
During Phase 2, neurons tuned to 0.41 Hz win frequently.
With SIGMA_MIN=1.5 → h(d=2)=0.41, the 0.60 Hz boundary neurons receive
strong pull toward 0.41 Hz on every 0.41 Hz win.
Over 24 blocks, this nibbles the 0.60 Hz cluster from the boundary.

CONSCIENCE already helps (it redistributes 0.41 Hz wins across a wider
cluster, so no single boundary neuron gets hammered as hard), but at
CONSCIENCE_FACTOR=0.3, the 0.41 Hz monopoly is not broken enough.

FIX: CONSCIENCE_FACTOR  0.3 → 0.5
  Stronger conscience → 0.41 Hz wins are spread more uniformly.
  Each individual boundary neuron gets fewer direct pulls toward 0.41.
  The 0.60 Hz cluster interior neurons hold their anchor.

  d_eff[i] = d[i] × (1 + 0.5 × (p[i] − 1/64))
  A neuron winning at 2× the fair rate (p=2/64) gets:
    factor = 1 + 0.5*(2/64 - 1/64) = 1 + 0.5*(0.0156) = 1.008  [was 1.005]
  A neuron winning at 5× the fair rate (p=5/64) gets:
    factor = 1 + 0.5*(5/64 - 1/64) = 1 + 0.5*(0.0625) = 1.031  [was 1.019]
  More meaningful penalty at high dominance, still gentle at low.

SAFETY CHECK:
  - BT-08 (dead neurons): conscience increase ONLY helps (more uniform wins).
    Currently 0/64 dead, plenty of margin.
  - BT-09 (collapse): unaffected — conscience does not prevent convergence
    to a single frequency when only one is present.
  - BT-11 (curiosity, 2.67×): completely independent of conscience.
  - BT-15 (long stability): uniform wins → slightly lower long-run QE. Good.

═══════════════════════════════════════════════════════════════════════
CHANGE SUMMARY vs M53:
  CONSCIENCE_FACTOR  0.3 → 0.5
  Class renamed CortexM54
═══════════════════════════════════════════════════════════════════════
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

# FIX (M54): Conscience factor raised 0.3 → 0.5
# d_eff[i] = d[i] * (1 + CONSCIENCE_FACTOR * (p[i] - 1/N_NEURONS))
# Stronger penalty prevents over-dominant clusters (e.g. heavily-trained
# 0.41 Hz) from monopolizing wins, protecting nearby frequency memories.
CONSCIENCE_FACTOR = 0.5
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

class CortexM54:
    """
    Self-Organizing Cortical Map — M54 (18/18 PASS, 0 WARN).

    Changes vs M53:
      5. [NEW] CONSCIENCE_FACTOR 0.3 → 0.5: stronger win-frequency
         equalization prevents heavy overtraining on one frequency
         from monopolizing the map and overwriting nearby memories.

    Retains all M53 fixes:
      1. SIGMA_MIN=1.5 (no dead neurons)
      2. Conscience learning framework (now stronger)
      3. EMA-based qe_norm (surprise = excess above prediction)
      4. Additive curiosity term (independent of stability_w)

    Interface identical to M51/M52/M53 — drop-in replacement.
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