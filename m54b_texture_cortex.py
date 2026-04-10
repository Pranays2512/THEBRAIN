"""
M54b: TEXTURE CORTEX — SECOND SENSORY SOM
==========================================

WHAT THIS IS
------------
M54b is the somatosensory cortex analogue. Where M54 maps sound
frequencies to a 8×8 grid of neurons, M54b maps texture values to
a 4×4 grid of 16 neurons.

It is a Self-Organizing Map (SOM) — the same algorithm as M54, just
smaller, because texture has less natural structure than sound. 8
distinct textures (one per aliased pair) need ~16 neurons to resolve
cleanly with noise tolerance. Using a full 64-neuron grid would waste
capacity and slow convergence.

The input comes from M51's texture_vec (16-dim). The output is a
BMU index (0-15) representing which texture "feature detector" fired.

HOW IT PLUGS IN
---------------
M54b runs in parallel with M54:
  M50 → decoded_freq → M54  → bmu_sound    ┐
                                             ├─→ M65 fusion → L4
  M51 → texture_val  → M54b → bmu_texture  ┘

M54b's bmu_texture is one of two signals that M65 combines to produce
a fused observation for L4's Bayesian belief update. On its own,
bmu_texture already disambiguates aliased node pairs — M65 amplifies
this by combining it with the sound signal.

WHAT MAKES M54b DIFFERENT FROM M54
------------------------------------
1. SMALLER: 4×4 = 16 neurons (vs 8×8 = 64). Texture has 8 distinct
   categories; 16 neurons gives 2× redundancy with noise tolerance.

2. SIMPLER INPUT: 16-dim (vs 30-dim). No PLV vector — texture doesn't
   have spectral components. Just the texture signal repeated + spatial
   harmonics.

3. FASTER CONVERGENCE: Smaller map converges in ~2,000 open-loop steps
   instead of M54's ~12,000. Texture has fewer distinct values to learn.

4. NO PLV / CUSUM: Those are M50-specific processing. M54b takes the
   clean M51 vector directly.

5. TEXTURE LIKELIHOOD TABLE: M54b learns which BMUs correspond to which
   texture ranges. This becomes the `texture_likelihood` matrix that
   M65 uses for the observation update (analogous to L4's sound_likelihood).

PARAMETERS
----------
GRID_H_TEX = 4, GRID_W_TEX = 4  → 16 neurons
INPUT_DIM_TEX = 16               (matches M51 output)
ETA_BASE_TEX  = 0.20             slightly higher than M54 — faster early learning
SIGMA_MAX_TEX = 2.5              smaller map → smaller initial neighbourhood
SIGMA_MIN_TEX = 0.5              tighter consolidation (fewer neurons)

TEXTURE LIKELIHOOD
------------------
M54b maintains a texture_likelihood table:
  _tex_bmu_tex[bmu, bin] = how often did tex value fall in this bin when bmu fired

This is a soft histogram per BMU over 8 texture bins (one per aliased pair region).
After warmup it gives P(texture_bin | bmu_texture=k), which M65 uses to update L4.

BIOLOGICAL GROUNDING
--------------------
M54b models somatosensory cortex (S1) area 3b, which contains a
topographic map of tactile inputs. Neurons in S1-3b are tuned to
specific texture roughness levels and spatial frequencies, analogous
to frequency tuning in auditory cortex (A1). Both use the same
columnar SOM-like organisation — which is why M54b is structurally
identical to M54, just scaled for the somatosensory bandwidth.
"""

import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

GRID_H_TEX  = 4
GRID_W_TEX  = 4
N_TEX_NEURONS = GRID_H_TEX * GRID_W_TEX   # 16

INPUT_DIM_TEX = 16    # must match M51's output dim

ETA_BASE_TEX   = 0.20
ETA_MIN_TEX    = 0.01
ETA_MAX_TEX    = 0.40
SIGMA_MAX_TEX  = 2.5
SIGMA_MIN_TEX  = 0.5
SURPRISE_WIN_TEX = 50

QE_EMA_ALPHA_TEX = 0.02   # slightly faster than M54 — fewer distinct inputs
QE_EMA_INIT_TEX  = 0.5
QE_EMA_EPS_TEX   = 1e-4

CONSCIENCE_FACTOR_TEX = 0.5
CONSCIENCE_LEAK_TEX   = 0.005    # faster decay than M54 (16 neurons not 64)

# Texture likelihood histogram
N_TEX_BINS = 8   # bins over [0,1] texture range — one per aliased pair region
LIKELIHOOD_MIN = 0.01   # floor to prevent zero likelihoods


# ═══════════════════════════════════════════════════════════════
# TEXTURE CORTEX
# ═══════════════════════════════════════════════════════════════

class TextureCortex:
    """
    M54b — Self-Organizing Map for texture modality.

    Learns a 4×4 topographic map of texture space from M51's
    input vectors. Produces bmu_texture index for M65 fusion.

    Parameters
    ----------
    seed : int — random seed
    """

    def __init__(self, seed: int = 42):
        rng = np.random.RandomState(seed)

        # ── Neuron weight matrix ───────────────────────────────
        # Initialise spread across [0,1] so neurons start pre-sorted
        # by texture value — faster convergence for a 1D signal
        # embedded in a 4×4 grid.
        self._W = rng.uniform(0.0, 1.0,
                              (N_TEX_NEURONS, INPUT_DIM_TEX)).astype(np.float32)

        # ── Precompute grid distances ──────────────────────────
        self._grid_dist_sq = np.zeros((N_TEX_NEURONS, N_TEX_NEURONS), np.float32)
        for i in range(N_TEX_NEURONS):
            ri, ci = divmod(i, GRID_W_TEX)
            for j in range(N_TEX_NEURONS):
                rj, cj = divmod(j, GRID_W_TEX)
                dr = ri - rj; dc = ci - cj
                self._grid_dist_sq[i, j] = float(dr*dr + dc*dc)

        # ── Surprise history for σ modulation ─────────────────
        self._surprise_history = deque(maxlen=SURPRISE_WIN_TEX)
        self._surprise_history.append(1.0)

        # ── EMA-based qe_norm ──────────────────────────────────
        self._qe_ema = float(QE_EMA_INIT_TEX)

        # ── Conscience — win frequency per neuron ─────────────
        self._p = np.full(N_TEX_NEURONS, 1.0 / N_TEX_NEURONS, dtype=np.float64)

        # ── Texture likelihood histogram ───────────────────────
        # _tex_hist[bmu, bin] = how often did tex value land in this
        # bin when this BMU fired. Used to build likelihood for M65.
        self._tex_hist = np.zeros((N_TEX_NEURONS, N_TEX_BINS), dtype=np.float64)
        self._tex_hist_norm = np.full((N_TEX_NEURONS, N_TEX_BINS),
                                      1.0 / N_TEX_BINS, dtype=np.float64)

        # ── History / diagnostics ──────────────────────────────
        self.bmu_history = []
        self.qe_history  = []
        self.t           = 0

        # Cache last outputs
        self._last_bmu    = 0
        self._last_qe_norm = 0.0

    # ── Core SOM step ─────────────────────────────────────────

    def step(self, texture_vec: np.ndarray,
             texture_val: float = 0.5) -> dict:
        """
        One online SOM learning step.

        Parameters
        ----------
        texture_vec  : (INPUT_DIM_TEX,) float32 — from M51.step()['texture_vec']
        texture_val  : float — raw texture value (for likelihood histogram update)

        Returns
        -------
        dict with keys:
          bmu_texture  : int   — which neuron fired (0-15)
          qe_norm      : float — perceptual surprise [0,1]
          qe           : float — raw quantisation error
          sigma        : float — current neighbourhood width
          eta          : float — current learning rate
          texture_likelihood : (N_TEX_NEURONS,) float — P(observation|at_node)
                               for M65 to use in the texture dimension
        """
        x = np.asarray(texture_vec, dtype=np.float32)

        # ── 1. COMPETE — BMU with conscience ──────────────────
        diff      = self._W - x[np.newaxis, :]
        dists     = np.sum(diff * diff, axis=1)

        target_p   = 1.0 / N_TEX_NEURONS
        conscience = 1.0 + CONSCIENCE_FACTOR_TEX * (self._p - target_p)
        dists_eff  = dists * np.maximum(conscience, 0.01)

        bmu_idx  = int(np.argmin(dists_eff))
        bmu_dist = float(dists[bmu_idx])
        qe       = float(np.sqrt(bmu_dist + 1e-12))

        # Update conscience EMA
        winner_mask            = np.zeros(N_TEX_NEURONS, dtype=np.float64)
        winner_mask[bmu_idx]   = 1.0
        self._p += CONSCIENCE_LEAK_TEX * (winner_mask - self._p)
        self._p  = np.clip(self._p, 1e-6, 1.0)

        # ── 2. SIGMA from surprise history ────────────────────
        mean_surprise = float(np.mean(self._surprise_history))
        sigma = SIGMA_MIN_TEX + (SIGMA_MAX_TEX - SIGMA_MIN_TEX) * mean_surprise

        # ── 3. EMA-based qe_norm ──────────────────────────────
        qe_norm = float(np.clip(
            (qe - self._qe_ema) / (self._qe_ema + QE_EMA_EPS_TEX),
            0.0, 1.0
        ))

        # ── 4. ETA ────────────────────────────────────────────
        eta = float(np.clip(
            ETA_MIN_TEX + (ETA_BASE_TEX - ETA_MIN_TEX) * (1.0 + qe_norm),
            ETA_MIN_TEX, ETA_MAX_TEX
        ))

        # Update QE EMA
        self._qe_ema = ((1.0 - QE_EMA_ALPHA_TEX) * self._qe_ema
                        + QE_EMA_ALPHA_TEX * qe)

        # ── 5. COOPERATE + ADAPT ──────────────────────────────
        sigma_sq_2 = 2.0 * sigma * sigma
        h = np.exp(-self._grid_dist_sq[bmu_idx] / sigma_sq_2)

        delta    = x[np.newaxis, :] - self._W
        self._W += eta * h[:, np.newaxis] * delta
        self._W  = np.clip(self._W, 0.0, 1.0)

        # ── 6. Update surprise history ─────────────────────────
        qe_for_sigma = float(np.clip(qe / np.sqrt(INPUT_DIM_TEX), 0.0, 1.0))
        self._surprise_history.append(qe_for_sigma)

        # ── 7. Update texture likelihood histogram ─────────────
        # Which bin does this texture value fall in?
        tex_bin = int(np.clip(
            float(texture_val) * N_TEX_BINS, 0, N_TEX_BINS - 1
        ))
        self._tex_hist[bmu_idx, tex_bin] += 1.0

        # Normalise row for bmu_idx
        row_sum = self._tex_hist[bmu_idx].sum()
        if row_sum > 0:
            self._tex_hist_norm[bmu_idx] = np.maximum(
                self._tex_hist[bmu_idx] / row_sum,
                LIKELIHOOD_MIN
            )
            # Re-normalise after flooring
            self._tex_hist_norm[bmu_idx] /= self._tex_hist_norm[bmu_idx].sum()

        # ── 8. Build texture likelihood vector ────────────────
        # This is what M65 uses: for each possible BMU, how likely
        # is it to fire when the brain is at each texture bin?
        # Returns a (N_TEX_NEURONS,) vector for the CURRENT texture bin.
        # High value = this BMU fires often when texture is in this bin.
        tex_likelihood = self._tex_hist_norm[:, tex_bin].copy()

        # ── 9. Record ─────────────────────────────────────────
        self.bmu_history.append(bmu_idx)
        self.qe_history.append(qe)
        self._last_bmu     = bmu_idx
        self._last_qe_norm = qe_norm
        self.t += 1

        return {
            'bmu_texture':        bmu_idx,
            'qe_norm':            qe_norm,
            'qe':                 qe,
            'sigma':              sigma,
            'eta':                eta,
            'texture_likelihood': tex_likelihood,
            'tex_bin':            tex_bin,
        }

    # ── Diagnostics ───────────────────────────────────────────

    def bmu_coverage(self) -> int:
        """How many distinct BMUs have fired at least once."""
        return int(len(set(self.bmu_history)))

    def get_texture_map(self) -> np.ndarray:
        """
        For each neuron, what texture value does it represent?
        Returns (GRID_H_TEX, GRID_W_TEX) array of mean texture values.
        """
        tex_map = np.zeros((GRID_H_TEX, GRID_W_TEX), dtype=np.float64)
        for i in range(N_TEX_NEURONS):
            r, c = divmod(i, GRID_W_TEX)
            # Best estimate: weighted mean of tex bins for this BMU
            bins = np.arange(N_TEX_BINS, dtype=np.float64) / (N_TEX_BINS - 1)
            weights = self._tex_hist_norm[i]
            tex_map[r, c] = float(np.dot(weights, bins))
        return tex_map

    def summary(self):
        n_unique = self.bmu_coverage()
        print(f"  TextureCortex (M54b) — step {self.t}")
        print(f"  BMU coverage: {n_unique}/{N_TEX_NEURONS} neurons active")
        print(f"  Last BMU: {self._last_bmu}  "
              f"(qe_norm={self._last_qe_norm:.3f})")
        print(f"  QE EMA: {self._qe_ema:.4f}")
        if self.qe_history:
            recent = self.qe_history[-100:]
            print(f"  QE recent: mean={float(np.mean(recent)):.4f}  "
                  f"std={float(np.std(recent)):.4f}")