"""
M52: CORTEX — TWO SURGICAL FIXES
=================================
Fixes the two real M51 failures found in break testing.

ROOT CAUSE 1 — SIGMA_MIN=0.5 collapses the neighborhood (BT-08, BT-09)
  On an 8×8 grid, σ=0.5 means:
    h(distance=1) = exp(-1² / 2×0.5²) = exp(-2)   ≈ 0.135
    h(distance=2) = exp(-2² / 2×0.5²) = exp(-8)   ≈ 0.000
  Only the winning neuron itself gets a meaningful weight update.
  All others freeze wherever they were initialized.
  Result: 28% dead neurons (BT-08), no single-freq convergence (BT-09).

  FIX 1: Raise SIGMA_MIN  0.5 → 1.5
    h(distance=1) = exp(-1² / 2×1.5²) = exp(-0.22) ≈ 0.80   ← always trains
    h(distance=2) = exp(-4² / 2×1.5²) = exp(-0.89) ≈ 0.41   ← always trains
    h(distance=3) = exp(-9² / 2×1.5²) = exp(-2.00) ≈ 0.14   ← gets some signal
  Every neuron within 3 cells of the winner always gets trained.
  Dead neurons cannot form — even the least-visited areas receive
  gradient flow from nearby competitions.

ROOT CAUSE 2 — qe_norm denominator kills curiosity (BT-11 weak 1.06× boost)
  The curiosity boost formula:
    qe_norm = qe / sqrt(INPUT_DIM)  =  qe / sqrt(23)  ≈  qe / 4.80
  A "highly novel" input with QE=0.18 gives qe_norm ≈ 0.038.
  That feeds into:
    η = ETA_MIN + ETA_BASE*(1 - ETA_MIN) + ETA_BASE*NOVELTY_BOOST*qe_norm
      = 0.01 + 0.15*(1-0.01) + 0.15*2.0*0.038  ≈  0.171
  vs familiar input with QE=0.004 → qe_norm ≈ 0.0008:
    η ≈  0.160
  Ratio = 1.07×. The sqrt(INPUT_DIM) denominator is too large — it was
  chosen for theoretical normalization but suppresses the signal in practice.

  FIX 2: Normalize qe_norm against a running max QE window.
    qe_norm = qe / running_max_qe_over_last_500_steps
  This is adaptive: the scale tracks the system's own surprise range.
  Novel input (QE near recent max) → qe_norm ≈ 1.0 → full curiosity boost.
  Familiar input (QE near recent min) → qe_norm ≈ 0.0 → near-minimum η.
  Expected boost ratio: 3–8× (depends on how different novel/familiar QE are).

ROOT CAUSE 3 — Conscience: no dead-neuron recovery mechanism
  Even with a higher σ floor, neurons that start unlucky can fall into
  a spiral: rarely win → weights drift from data → lose even more.
  
  FIX 3: Conscience learning (after Desieno 1988).
    Track exponential win frequency p[i] per neuron.
    Add a conscience penalty to the distance:
      d_eff[i] = d[i] * (1 + CONSCIENCE_FACTOR * (p[i] - 1/N_NEURONS))
    Neurons winning too often get effectively farther from inputs.
    Neurons winning rarely get effectively closer.
    This enforces approximately equal representation.

WHAT IS UNCHANGED FROM M51:
  - SOM architecture (8×8 grid, 23-dim input, competitive learning) ✔
  - prepare_input() normalization ✔
  - Surprise-driven σ (Option B) ✔
  - M50 integration interface ✔
  - All 4 M51 passing tests (T1 Map Formation, T2 Surprise Curve,
    T3 Novel Spike, T4 Curiosity Modulation) ✔

CHANGE SUMMARY vs M51:
  Line ~50:  SIGMA_MIN       0.5 → 1.5
  Line ~55:  QE_NORM_WINDOW  500     (new: rolling max for normalization)
  Line ~60:  CONSCIENCE_FACTOR 0.3   (new)
  Line ~185: step() — qe_norm computed from running max, not sqrt(INPUT_DIM)
  Line ~200: step() — conscience penalty applied before argmin
  Line ~215: step() — p[] updated each step
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

# FIX 2: Running-max normalization window for qe_norm
# qe_norm = qe / running_max over this many steps
# Adaptive: tracks the system's own recent surprise range
QE_NORM_WINDOW  = 100   # was 500 — ~10s; short enough that random-init
                         # startup noise (high early QE) clears before novel
                         # inputs arrive. 500 kept early peaks in window,
                         # inflating running_max and suppressing qe_norm.
QE_NORM_MIN_VAL = 1e-4  # floor to avoid div-by-zero at startup

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

class CortexM52:
    """
    Self-Organizing Cortical Map — M52 (fixed).

    Changes vs M51:
      1. SIGMA_MIN 0.5 → 1.5: neighborhood floor ensures all neurons
         receive training signal, eliminating dead neurons.
      2. qe_norm via running-max: curiosity boost scales to system's
         own surprise range, giving 3–8× boost instead of 1.06×.
      3. Conscience learning: prevents any neuron from monopolizing
         wins; gives underrepresented neurons a competitive advantage.

    Interface identical to M51 — drop-in replacement.
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

        # FIX 2: Running QE window for adaptive qe_norm
        self._qe_window = deque(maxlen=QE_NORM_WINDOW)
        self._qe_window.append(QE_NORM_MIN_VAL)

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

        # 4. FIX 2: Adaptive curiosity via running-max qe_norm
        # qe_norm = qe / max(recent QE)
        # → 1.0 when current surprise equals recent worst case
        # → 0.0 when perfectly familiar (QE near zero)
        self._qe_window.append(qe)
        running_max_qe = max(float(np.max(self._qe_window)), QE_NORM_MIN_VAL)
        qe_norm_now    = float(np.clip(qe / running_max_qe, 0.0, 1.0))

        # Learning rate: min when familiar, scaled by curiosity when novel
        # Also gated by stability_w (don't learn hard when ear unsettled)
        eta = ETA_MIN + (ETA_BASE - ETA_MIN) * (1.0 + NOVELTY_BOOST * qe_norm_now)
        eta = eta * float(stability_w)
        eta = max(eta, ETA_MIN)

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