"""
M56: CORTEX — LONGRUN CONSOLIDATION FIX
========================================
Inherits all M55 fixes (freq_norm repeated 8×, INPUT_DIM=30) and adds
two parameter changes that fix persistent BMU drift in brain_longrun.py.

═══════════════════════════════════════════════════════════════════════
ROOT CAUSE OF M55 LONGRUN FAILURE (observed in brain_longrun.py)
═══════════════════════════════════════════════════════════════════════

brain_longrun.py runs 50,000 steps with a grammar-structured audio stream.
Expected behaviour: zone assignments stabilise after ~15k steps, L2
prediction accuracy trends upward, PE trends downward.

Observed behaviour: zone assignments change at every 5k snapshot through
all 50k steps. Prediction accuracy oscillates 0%→75%→25%→75%→25%.
PE never trends down. Familiarity flat at 0.647 for ALL 8 frequencies.

DIAGNOSIS:
  L3 froze its zone map at step 15k (L3=stable(13) in output) and never
  re-clustered. Yet every frequency changed zones at every snapshot after
  that. Since L3's bmu_to_zone is frozen, the only explanation is that
  the CORTEX BMU for each frequency keeps drifting — L3 sees the same
  zone boundaries but different neurons arriving at each probe.

  L2 learns transitions between BMU indices. If frequency A fires BMU 7
  at step 35k but BMU 40 at step 45k, L2's learned model for A is
  invalidated and rebuilt repeatedly — hence oscillating accuracy.

  The familiarity plateau at 0.647 is downstream of the same drift:
  M55 associations are spread across many transient BMUs, so no single
  BMU accumulates enough exposure for breadth_score to saturate.

WHY THE MAP NEVER CONSOLIDATES (M55):
  At steady state: fam ≈ 0.647, FAM_ETA_SUPPRESS = 0.5
    eta_base    = 0.01 + 0.14×0.8 = 0.122
    eta_fam     = 0.5 × 0.14 × 0.647 = 0.045
    eta_net     ≈ 0.088   ← still substantial

  SIGMA_MIN = 1.5 → h(d=1) = exp(−1/4.5) = 0.80
  Drift force on boundary neuron per step = eta × h = 0.088 × 0.80 = 0.070
  Over 50k steps, each frequency visits ~6000 times.
  Accumulated drift force = 6000 × 0.070 = 420 weight-units.
  The map never stops reorganising.

  Root cause: FAM_ETA_SUPPRESS was designed to trigger phase-2
  consolidation (familiar inputs → low plasticity), but at 0.5 it is
  not strong enough to actually freeze the map at the observed fam=0.647
  plateau. Phase 2 never engaged.

═══════════════════════════════════════════════════════════════════════
FIX 1: FAM_ETA_SUPPRESS  0.5 → 1.5
═══════════════════════════════════════════════════════════════════════

At fam=0.647 (observed longrun plateau):
  eta_fam = 1.5 × 0.14 × 0.647 = 0.136
  eta_net = 0.122 − 0.136 = −0.014 → clipped to ETA_MIN = 0.010

Routine familiar visits now barely move weights (eta at floor).
Genuine novelty still works: qe_norm spike adds up to 0.28 to eta,
overriding the suppression. A truly new input (unseen frequency) would
push eta back toward ETA_MAX = 0.30 and reorganise the map.

This is what FAM_ETA_SUPPRESS was always intended to do — it just
needed to be strong enough for the actual observed fam plateau (0.647,
not the 0.78 assumed when 0.5 was originally chosen).

═══════════════════════════════════════════════════════════════════════
FIX 2: SIGMA_MIN  1.5 → 0.8
═══════════════════════════════════════════════════════════════════════

h(d=1): exp(−1/4.5) = 0.80  →  exp(−1/1.28) = 0.54
h(d=2): exp(−4/4.5) = 0.41  →  exp(−4/1.28) = 0.044

Two cells away: bleed drops from 41% to 4%. Adjacent boundary neurons
are almost completely decoupled from a neighbour's wins.

Combined with Fix 1:
  drift force per step = 0.010 × 0.54 = 0.006   (was 0.070 — 11× lower)

The map consolidates. Zone assignments stop drifting. L2 can build a
stable transition model on consistent BMU addresses.

SIGMA_MIN=0.8 is safe at this stage: the map is already organised
(8/8 unique BMUs confirmed in longrun). Lower sigma during consolidation
PROTECTS the structure — dead neurons only arise during initial
organisation when sigma was needed for exploration.

SAFETY CHECK (vs existing breaktests):
  BT-07 (catastrophic forgetting): SIGMA_MIN lower → LESS boundary bleed
    → adjacent cluster interference is REDUCED. Strictly safer.
  BT-08 (dead neurons): map already organised before consolidation.
    No new dead neuron risk. Conscience + existing structure prevents it.
  BT-09 (collapse): conscience unaffected. Collapse test unchanged.
  BT-11 (curiosity): qe_norm path unchanged. Novelty boost unchanged.
  BT-15 (long stability): this IS the long-stability fix.

═══════════════════════════════════════════════════════════════════════
CHANGE SUMMARY vs M55:
  FAM_ETA_SUPPRESS  0.5  → 1.5
  SIGMA_MIN         1.5  → 0.8
  Class renamed CortexM56
  Docstring updated to reflect actual M55 change (freq_norm ×8) and
  this consolidation fix.
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
INPUT_DIM        = 8 + 2 + N_PLV_COMPONENTS     # 30 (8x freq_norm, stability, novelty, 20 PLV)

# Normalization
FREQ_MIN_HZ = 0.41
FREQ_MAX_HZ = 2.20

# Learning
ETA_BASE      = 0.15
ETA_MIN       = 0.01
NOVELTY_BOOST = 2.0

# Neighborhood σ bounds.
# SIGMA_MAX=3.5 for initial exploration (unchanged).
# SIGMA_MIN=0.8 for consolidation phase (FIX 2 vs M55, was 1.5):
#   h(d=1): 0.80 → 0.54  (adjacent-cell bleed halved)
#   h(d=2): 0.41 → 0.044 (2 cells away: almost no bleed)
#   Drift force per step: eta×h = 0.010×0.54 = 0.006  (was 0.070, 11× lower)
#   Map consolidates after initial organisation. Zone assignments stabilise.
SIGMA_MAX = 3.5    # unchanged
SIGMA_MIN = 0.8    # was 1.5 (M55) — consolidation fix (FIX 2)

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

# ── L2 → M54 feedback ────────────────────────────────────────
# When L2's sequence predictor was wrong (high prediction_error),
# the cortex should learn faster — something unexpected happened
# that the map should update toward.
#
# eta_sequence = (ETA_BASE - ETA_MIN) * SEQUENCE_ERROR_BOOST * prediction_error
# At prediction_error=1.0: adds up to 0.14 to eta (same ceiling as novelty boost)
# At prediction_error=0.0: adds nothing (no change vs old behaviour)
# Final eta is capped at ETA_BASE * 2 = 0.30 to prevent explosion.
#
# Biologically: L2 prediction error maps to dopamine prediction error —
# unexpected events trigger heightened synaptic plasticity in cortex.
SEQUENCE_ERROR_BOOST = 1.0    # scale factor for L2→M54 feedback
ETA_MAX              = ETA_BASE * 2   # hard ceiling including all boosts

# ── M55 → M54 familiarity suppression ────────────────────────
# When the current BMU is well-known (high familiarity from M55),
# M54 should REDUCE its plasticity — there is no reason to keep
# rewriting weights for a pattern the map has already consolidated.
#
# This solves the persistent eta-floor problem: without this,
# eta_base = ETA_MIN + (ETA_BASE-ETA_MIN)*stability_w = 0.129 at
# stability_w=0.85, which keeps M54 in moderate-plasticity state
# permanently even on patterns trained for 500+ steps.
#
# Formula:
#   eta_familiarity = -FAM_ETA_SUPPRESS * (ETA_BASE - ETA_MIN) * familiarity
#   eta = clip(eta_base + eta_curiosity + eta_sequence + eta_familiarity, ETA_MIN, ETA_MAX)
#
# At familiarity=0.0 (novel BMU):    no suppression, eta unchanged
# At familiarity=0.40 (mean):        eta_base drops from 0.129 to ~0.046
# At familiarity=0.70 (well-known):  eta_base drops to ~0.010 (ETA_MIN)
#
# The suppression is bounded: eta is always clipped to [ETA_MIN, ETA_MAX].
# Novelty/surprise boosts (eta_curiosity, eta_sequence) can still override
# suppression — if a familiar BMU suddenly fires in a surprising context,
# M54 WILL learn from it (the boosts lift eta above the suppressed baseline).
#
# Biologically: long-term depression (LTD) in well-established synapses.
# Familiarity signals from perirhinal cortex suppress primary cortex
# plasticity — memories that are already consolidated need less updating.
#
# FAM_ETA_SUPPRESS is passed into cortex.step() as optional `familiarity`
# argument (default 0.0 — backward compatible). Brain stores familiarity
# from memory.recall() at step t and feeds it into cortex.step() at step t+1.
# Same next-step pattern as surprise_signal and rpe_positive.
FAM_ETA_SUPPRESS = 1.5   # suppress factor (FIX 1 vs M55, was 0.5)
                          # At fam=0.647 (observed longrun plateau):
                          #   eta_fam = 1.5 × 0.14 × 0.647 = 0.136
                          #   eta_net = 0.122 − 0.136 → clipped to ETA_MIN = 0.010
                          # Routine familiar visits barely move weights.
                          # Genuine novelty overrides: qe_norm spike adds up to 0.28,
                          # pushing eta back toward ETA_MAX regardless of familiarity.
                          # Phase-2 consolidation now actually engages at fam≥0.55.


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

    # Repeat freq_norm 8x so it carries ~28% of input signal
    # (vs 4% before). PLV normalised per-step loses freq info,
    # so freq_norm must dominate for the SOM to separate zones.
    freq_repeated = np.full(8, freq_norm, dtype=np.float32)
    return np.concatenate([
        freq_repeated,
        [float(stability_w), float(novelty_flag)],
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

class CortexM56:
    """
    Self-Organizing Cortical Map — M56 (consolidation fix).

    Changes vs M55:
      1. [FIX] FAM_ETA_SUPPRESS 0.5 → 1.5: at observed fam plateau of 0.647,
         eta now drops to ETA_MIN floor for familiar inputs. Phase-2
         consolidation engages. BMU drift stops. Zone assignments stabilise.
      2. [FIX] SIGMA_MIN 1.5 → 0.8: boundary bleed drops 11×. Combined with
         Fix 1, drift force per step falls from 0.070 → 0.006.

    Retains all M55 fixes:
      - freq_norm repeated 8× (INPUT_DIM=30): frequency signal 4.3%→26.7%
      - CONSCIENCE_FACTOR=0.5 (win-frequency equalisation)
      - EMA-based qe_norm (surprise = excess above prediction)
      - Additive curiosity term (independent of stability_w)
      - FAM_ETA_SUPPRESS (now correctly tuned)

    Interface identical to M54/M55 — drop-in replacement.
    brain.py import: from m56_cortex import CortexM56
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

    def step(self, decoded_freq, stability_w, novelty_flag, plv_vector,
             prediction_error: float = 0.0,
             familiarity:      float = 0.0):
        """
        One online learning step.

        Parameters
        ----------
        prediction_error : float [0,1]
            L2 sequence prediction error from previous step (as surprise_signal
            delta, not raw). 0 = L2 predicted correctly. 1 = fully wrong.
            Default 0.0 preserves old behaviour when feedback not wired.
        familiarity : float [0,1]
            M55 familiarity signal from previous step.
            High familiarity → suppress eta (the map has already learned this).
            Default 0.0 (no suppression) preserves old behaviour when not wired.
            Biologically: perirhinal cortex → primary cortex LTD modulation.
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
        # eta_base:         w-gated base learning (trust input by signal quality)
        # eta_curiosity:    novelty bonus, INDEPENDENT of w
        # eta_sequence:     L2 surprise boost — unexpected sequence → learn more
        # eta_familiarity:  M55 suppression — known patterns need less updating
        #
        # Biologically:
        #   eta_base      = baseline cortical plasticity
        #   eta_curiosity = acetylcholine / noradrenaline novelty boost
        #   eta_sequence  = dopamine prediction error → heightened plasticity
        #   eta_familiarity = perirhinal → cortex LTD (suppress known patterns)
        eta_base        = ETA_MIN + (ETA_BASE - ETA_MIN) * float(stability_w)
        eta_curiosity   = (ETA_BASE - ETA_MIN) * NOVELTY_BOOST * qe_norm_now
        eta_sequence    = (ETA_BASE - ETA_MIN) * SEQUENCE_ERROR_BOOST * float(prediction_error)
        eta_familiarity = FAM_ETA_SUPPRESS * (ETA_BASE - ETA_MIN) * float(familiarity)
        eta             = float(np.clip(
            eta_base + eta_curiosity + eta_sequence - eta_familiarity,
            ETA_MIN, ETA_MAX
        ))

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
            'is_novel':          qe > SURPRISE_THRESH,
            'input_vec':         x,
            'prediction_error':  float(prediction_error),
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