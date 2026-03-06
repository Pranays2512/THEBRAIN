"""
M51: THE SELF-ORGANIZING CORTEX
================================
Sits above M50 (the ear). Receives M50's output stream
and builds a live map of the input space from experience alone.
No labels. No hand-crafted categories. Pure self-organization.

ARCHITECTURE:
  Input:   23-dimensional vector from M50 each timestep
             [decoded_freq, stability_w, novelty, plv_top20]
  Layer 1: 8×8 grid of cortical columns (64 neurons)
             Each neuron has a 23-dim weight vector
             Initialized randomly, organized by experience
  Output:  surprise signal (quantization error)
             low  = familiar, map covers this well
             high = novel, map has no good representation

MECHANISM: Kohonen SOM with surprise-driven plasticity

  Standard SOM:
    σ(t) = σ_max × exp(-t / decay)   ← time-based, fixed
  
  M51 (Option B — surprise-based):
    σ(t) = σ_min + (σ_max - σ_min) × mean_recent_surprise
    
    Brain stays plastic while world is surprising.
    Hardens when world becomes familiar.
    Re-opens if genuine novelty appears.
    This IS the curiosity mechanism.

  Learning rate also modulated by M50 novelty:
    effective_η = base_η × (1 + novelty_boost × novelty)
    Novel input → learn faster (curiosity)
    Familiar input → learn slower (efficiency)

CONNECTION TO M50:
  M50 is the cochlea. M51 is the auditory cortex.
  M50 tells M51: "here's what I'm hearing and how confident I am"
  M51 tells M50: "here's how surprised I am by that"
  That feedback loop IS attention.
"""

import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Grid
GRID_H = 8
GRID_W = 8
N_NEURONS = GRID_H * GRID_W   # 64 cortical columns

# Input
N_PLV_COMPONENTS = 20         # top-20 PLV values from M50
N_SCALARS        = 3          # decoded_freq, stability_w, novelty
INPUT_DIM        = N_SCALARS + N_PLV_COMPONENTS  # 23

# Normalization ranges for scalar inputs
FREQ_MIN_HZ = 0.41
FREQ_MAX_HZ = 2.20

# Learning
ETA_BASE       = 0.15    # base learning rate
ETA_MIN        = 0.01    # minimum (never fully stops learning)
NOVELTY_BOOST  = 2.0     # curiosity multiplier on novel inputs

# Neighborhood (surprise-driven plasticity — Option B)
SIGMA_MAX      = 3.5     # broad neighborhood when surprised
                          # (covers ~half the 8×8 map)
SIGMA_MIN      = 0.5     # tight neighborhood when familiar
                          # (only immediate neighbors)
SURPRISE_WINDOW = 100    # samples for mean surprise estimate
                          # (~10s at feature_sample_interval=2, dt=0.05)

# Surprise
QE_HISTORY_LEN  = 500    # for running statistics
SURPRISE_THRESH = 0.15   # above this = "genuinely novel"


# ═══════════════════════════════════════════════════════════════
# INPUT PREPARATION
# ═══════════════════════════════════════════════════════════════

def prepare_input(decoded_freq, stability_w, novelty_flag,
                  plv_vector):
    """
    Build the 23-dim input vector from M50 outputs.
    
    Normalization:
      - decoded_freq: [FREQ_MIN, FREQ_MAX] → [0, 1]
      - stability_w:  already [0, 1]
      - novelty_flag: already {0, 1}
      - plv_vector:   top-20 by magnitude, normalized to [0,1]
    
    Why normalize?
      SOM distance metric treats all dims equally.
      If freq is in [0.4, 2.2] and PLV in [0, 1], the freq
      would dominate the distance. Normalization gives each
      dimension equal voice.
    """
    # Scalar part
    freq_norm = np.clip(
        (decoded_freq - FREQ_MIN_HZ) / (FREQ_MAX_HZ - FREQ_MIN_HZ),
        0.0, 1.0
    )
    
    # PLV part — top-20 by magnitude
    plv_abs = np.abs(plv_vector)
    if len(plv_abs) >= N_PLV_COMPONENTS:
        top_idx = np.argpartition(plv_abs, -N_PLV_COMPONENTS)[-N_PLV_COMPONENTS:]
        top_plv = plv_abs[top_idx]
    else:
        top_plv = np.pad(plv_abs, (0, N_PLV_COMPONENTS - len(plv_abs)))
    
    # Normalize PLV to [0,1]
    plv_max = top_plv.max()
    if plv_max > 1e-9:
        top_plv = top_plv / plv_max
    
    return np.concatenate([
        [freq_norm, float(stability_w), float(novelty_flag)],
        top_plv
    ]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# CORTICAL COLUMN (single neuron)
# ═══════════════════════════════════════════════════════════════

class CorticalColumn:
    """
    One neuron in the 8×8 cortical map.
    Has a position (row, col) and a weight vector.
    The weight vector is its "preferred pattern" —
    the input it responds to most strongly.
    """
    def __init__(self, row, col, input_dim, rng):
        self.row = row
        self.col = col
        # Initialize weights uniformly random
        # They will self-organize from experience
        self.weights = rng.uniform(0.0, 1.0, input_dim).astype(np.float32)
    
    def distance(self, input_vec):
        """Euclidean distance between input and this neuron's weights."""
        diff = input_vec - self.weights
        return float(np.dot(diff, diff))  # squared distance, faster
    
    def grid_distance_sq(self, other):
        """Squared grid distance to another neuron (for neighborhood)."""
        dr = self.row - other.row
        dc = self.col - other.col
        return dr*dr + dc*dc


# ═══════════════════════════════════════════════════════════════
# THE CORTEX — M51
# ═══════════════════════════════════════════════════════════════

class CortexM51:
    """
    Self-Organizing Cortical Map.
    
    Receives M50's output stream continuously.
    Builds a 2D map of the input space from experience.
    No labels. No supervision. Pure competitive learning.
    
    The map topology is meaningful:
      - Nearby neurons respond to similar inputs
      - Distance on the map ≈ distance in input space
      - This topology emerges automatically
    
    Surprise signal (quantization error):
      - How well does the best matching neuron represent this input?
      - Low  = the map covers this input well (familiar)
      - High = the map has no good match (novel/surprising)
    """
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
        # Build grid of cortical columns
        self.neurons = []
        self.grid = {}  # (row,col) → neuron index
        for r in range(GRID_H):
            for c in range(GRID_W):
                idx = r * GRID_W + c
                neuron = CorticalColumn(r, c, INPUT_DIM, self.rng)
                self.neurons.append(neuron)
                self.grid[(r,c)] = idx
        
        # Precompute all pairwise grid distances (fixed topology)
        self._grid_dist_sq = np.zeros((N_NEURONS, N_NEURONS), np.float32)
        for i, ni in enumerate(self.neurons):
            for j, nj in enumerate(self.neurons):
                self._grid_dist_sq[i,j] = ni.grid_distance_sq(nj)
        
        # Surprise history for plasticity control
        self._surprise_history = deque(maxlen=SURPRISE_WINDOW)
        self._surprise_history.append(1.0)  # start maximally plastic
        
        # Full QE history for analysis
        self.qe_history    = []
        self.bmu_history   = []   # which neuron won each timestep
        self.sigma_history = []   # how plastic was the map
        self.eta_history   = []   # effective learning rate
        self.t             = 0    # timestep counter
        
        # Build weight matrix for fast BMU search
        # Shape: (N_NEURONS, INPUT_DIM)
        self._W = np.array([n.weights for n in self.neurons],
                           dtype=np.float32)
    
    # ── Core SOM step ─────────────────────────────────────────
    
    def step(self, decoded_freq, stability_w, novelty_flag,
             plv_vector):
        """
        One online learning step.
        
        Args:
            decoded_freq:  scalar Hz, from M50
            stability_w:   scalar [0,1], M50 confidence
            novelty_flag:  scalar {0,1}, M50 CUSUM output
            plv_vector:    (500,) array, M50 PLV magnitudes
        
        Returns:
            result dict with surprise, BMU location, etc.
        """
        # 1. Prepare input
        x = prepare_input(decoded_freq, stability_w,
                          novelty_flag, plv_vector)
        
        # 2. COMPETE — find best matching unit
        # Vectorized: compute all distances at once
        diff   = self._W - x[np.newaxis, :]   # (64, 23)
        dists  = np.sum(diff * diff, axis=1)   # (64,)
        bmu_idx = int(np.argmin(dists))
        bmu_dist = float(dists[bmu_idx])       # squared distance
        
        # Quantization error = sqrt of min squared distance
        qe = float(np.sqrt(bmu_dist + 1e-12))
        
        # 3. SURPRISE-DRIVEN PLASTICITY (Option B)
        # σ tracks mean recent surprise — broad when novel,
        # narrow when familiar
        mean_surprise = float(np.mean(self._surprise_history))
        sigma = (SIGMA_MIN +
                 (SIGMA_MAX - SIGMA_MIN) * mean_surprise)
        
        # 4. CURIOSITY — driven by the cortex's OWN surprise (QE)
        # This is more brain-like: the cortex itself decides
        # when to learn hard, based on how surprised it is.
        # High QE (novel input)    → learn fast, big weight update
        # Low QE  (familiar input) → learn slowly, small update
        #
        # We also accept a nudge from M50's novelty flag, but
        # the primary signal is internal QE — not a binary external flag.
        #
        # qe_norm is in [0,1]: 0 = perfectly familiar, 1 = maximally novel
        # At qe_norm=0:   η = ETA_MIN  (barely learning, stable)
        # At qe_norm=0.5: η ≈ ETA_BASE (normal learning)
        # At qe_norm=1.0: η = ETA_BASE × (1 + NOVELTY_BOOST) (full curiosity)
        qe_norm_now = float(np.clip(qe / np.sqrt(INPUT_DIM), 0.0, 1.0))
        eta = ETA_MIN + (ETA_BASE - ETA_MIN + 
                         ETA_BASE * NOVELTY_BOOST * qe_norm_now)
        
        # Also modulate by M50 stability — don't learn hard when
        # the ear itself isn't confident (unsettled/sweeping)
        eta = eta * float(stability_w)
        eta = max(eta, ETA_MIN)
        
        # 5. COOPERATE + ADAPT
        # Update BMU and neighbors
        # h(i) = exp(-grid_dist²(bmu, i) / 2σ²)
        sigma_sq_2 = 2.0 * sigma * sigma
        h = np.exp(-self._grid_dist_sq[bmu_idx] / sigma_sq_2)
        # h shape: (N_NEURONS,)
        
        # Weight update: Δw = η × h × (x - w)
        delta = x[np.newaxis, :] - self._W          # (64, 23)
        self._W += eta * h[:, np.newaxis] * delta   # (64, 23)
        self._W = np.clip(self._W, 0.0, 1.0)
        
        # Sync neuron objects (for external access)
        for i, neuron in enumerate(self.neurons):
            neuron.weights = self._W[i]
        
        # 6. UPDATE SURPRISE HISTORY
        # Normalize QE to [0,1] range for plasticity control
        # QE is in [0, sqrt(INPUT_DIM)] = [0, ~4.8]
        qe_norm = float(np.clip(qe / np.sqrt(INPUT_DIM), 0.0, 1.0))
        self._surprise_history.append(qe_norm)
        
        # 7. RECORD
        self.qe_history.append(qe)
        self.bmu_history.append(bmu_idx)
        self.sigma_history.append(sigma)
        self.eta_history.append(eta)
        self.t += 1
        
        bmu_neuron = self.neurons[bmu_idx]
        return {
            'qe':          qe,           # surprise level
            'qe_norm':     qe_norm,      # normalized surprise
            'bmu_idx':     bmu_idx,      # winning neuron index
            'bmu_pos':     (bmu_neuron.row, bmu_neuron.col),
            'sigma':       sigma,        # current neighborhood size
            'eta':         eta,          # effective learning rate
            'is_novel':    qe > SURPRISE_THRESH,
            'input_vec':   x,
        }
    
    # ── Analysis tools ────────────────────────────────────────
    
    def get_map_state(self):
        """
        Returns the current state of the map as a dict.
        Useful for visualization and analysis.
        """
        # What frequency does each neuron prefer?
        # (reverse the normalization on dim 0)
        freq_map = np.zeros((GRID_H, GRID_W))
        w_map    = np.zeros((GRID_H, GRID_W))
        for neuron in self.neurons:
            r, c = neuron.row, neuron.col
            freq_norm = neuron.weights[0]
            freq_map[r,c] = (freq_norm * (FREQ_MAX_HZ - FREQ_MIN_HZ)
                             + FREQ_MIN_HZ)
            w_map[r,c]    = neuron.weights[1]  # preferred stability
        return {
            'freq_map':    freq_map,
            'w_map':       w_map,
            'weights':     self._W.copy(),
            'n_steps':     self.t,
            'mean_qe':     float(np.mean(self.qe_history[-100:]))
                           if self.qe_history else 0.0,
        }
    
    def get_surprise_stats(self):
        """Recent surprise statistics."""
        if not self.qe_history:
            return {}
        recent = self.qe_history[-100:]
        return {
            'mean':   float(np.mean(recent)),
            'std':    float(np.std(recent)),
            'max':    float(np.max(recent)),
            'min':    float(np.min(recent)),
            'current_sigma': float(np.mean(list(
                self._surprise_history))),
        }
    
    def neuron_activation_counts(self):
        """How many times each neuron has won (BMU)."""
        counts = np.zeros(N_NEURONS, dtype=int)
        for idx in self.bmu_history:
            counts[idx] += 1
        return counts.reshape(GRID_H, GRID_W)
    
    def find_neuron_for_freq(self, target_freq):
        """
        Which neuron best represents a given frequency?
        Used to check if the map has learned a frequency.
        """
        freq_norm = np.clip(
            (target_freq - FREQ_MIN_HZ) / (FREQ_MAX_HZ - FREQ_MIN_HZ),
            0.0, 1.0
        )
        # Find neuron whose weight[0] (freq dim) is closest
        diffs = np.abs(self._W[:, 0] - freq_norm)
        best  = int(np.argmin(diffs))
        n     = self.neurons[best]
        return (n.row, n.col), float(diffs[best])


# ═══════════════════════════════════════════════════════════════
# BRAIN — M50 + M51 COMBINED
# ═══════════════════════════════════════════════════════════════

class Brain:
    """
    The complete system: ear (M50) + cortex (M51).
    
    M50 handles perception — what frequency is present,
    how confident, whether something changed.
    
    M51 handles understanding — what patterns exist,
    what's familiar, what's novel, what's surprising.
    
    Usage:
        brain = Brain(cal)           # cal = M50 calibration
        result = brain.hear(t, data_slice)  # one timestep
        brain.cortex.get_map_state() # inspect the map
    """
    
    def __init__(self, raw_x_slow, true_y_slow,
                 raw_x_fast, true_y_fast,
                 cortex_seed=42):
        # M50 decoder (stateless — just lookups)
        from m50_neuron import (decode_resonance,
                                compute_stability_plv,
                                DivergenceCUSUM)
        from collections import deque
        
        self._decode_resonance   = decode_resonance
        self._compute_stability  = compute_stability_plv
        self._raw_x_slow = raw_x_slow
        self._true_y_slow = true_y_slow
        self._raw_x_fast  = raw_x_fast
        self._true_y_fast = true_y_fast
        
        # M50 state
        self._plv_hist    = deque(maxlen=20)  # PLV_STAB_WINDOW
        self._cusum       = DivergenceCUSUM()
        
        # M51
        self.cortex = CortexM51(seed=cortex_seed)
        
        # History
        self.perception_history = []
    
    def process(self, plv_fast, energy_fast,
                plv_slow, energy_slow, t):
        """
        One timestep: run M50 decoder + M51 cortex.
        
        Args:
            plv_fast, energy_fast: M50 fast stream arrays (500,)
            plv_slow, energy_slow: M50 slow stream arrays (500,)
            t: current time in seconds
        
        Returns:
            Combined perception + cortex result
        """
        # ── M50: decode ──────────────────────────────────────
        plv_fast_mag = np.abs(plv_fast)
        plv_slow_mag = np.abs(plv_slow)
        
        df = self._decode_resonance(
            plv_fast_mag, energy_fast,
            self._raw_x_fast, self._true_y_fast)
        ds = self._decode_resonance(
            plv_slow_mag, energy_slow,
            self._raw_x_slow, self._true_y_slow)
        
        # Stability weight
        max_plv = float(np.max(plv_slow_mag))
        self._plv_hist.append(max_plv)
        w = self._compute_stability(self._plv_hist)
        
        # Change detection
        _, novelty = self._cusum.update(df, ds, t, w=w)
        
        # Fused frequency
        f_fused = w * ds + (1.0 - w) * df
        
        # ── M51: cortex step ─────────────────────────────────
        cortex_result = self.cortex.step(
            decoded_freq  = f_fused,
            stability_w   = w,
            novelty_flag  = float(novelty),
            plv_vector    = plv_slow_mag   # use slow PLV — richer
        )
        
        result = {
            # M50 outputs
            'df':        df,
            'ds':        ds,
            'f_fused':   f_fused,
            'w':         w,
            'novelty':   novelty,
            't':         t,
            # M51 outputs
            'surprise':  cortex_result['qe'],
            'is_novel':  cortex_result['is_novel'],
            'bmu_pos':   cortex_result['bmu_pos'],
            'sigma':     cortex_result['sigma'],
            'eta':       cortex_result['eta'],
        }
        self.perception_history.append(result)
        return result