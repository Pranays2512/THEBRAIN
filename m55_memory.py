"""
M55: ASSOCIATIVE MEMORY — PURE HEBBIAN PATTERN STORAGE
=======================================================

WHAT THIS IS
------------
A biologically-grounded associative memory layer that sits on top of
M54 Cortex. M54 is never modified — M55 only reads M54's output.

Memory lives entirely in a single (64 × 64) weight matrix W.
Nothing is written to disk. Nothing is stored explicitly.
The memory IS the weights — exactly like the brain.

HOW IT WORKS
------------
Every time M54 fires a BMU, M55:
  1. Updates an eligibility trace — a decaying record of recent activity
  2. Strengthens W[i][j] for all pairs (i, j) that were recently co-active
     (Hebbian: neurons that fire together, wire together)
  3. Slowly decays all weights toward zero (forgetting unused memories)
  4. Normalizes weights per neuron (prevents explosion — homeostasis)

On recall:
  Give M55 a partial/noisy pattern (e.g. current BMU).
  It runs a few settling steps.
  The network relaxes toward the nearest stored attractor.
  Returns the settled pattern + familiarity depth (how fast/deep it settled).

ADAPTIVE ELIGIBILITY TRACE
---------------------------
The trace window is NOT fixed. It adapts to surprise:
  - Novel input (high qe_norm):  window extends  (~40 steps, ~2s)
  - Familiar input (low qe_norm): window shrinks  (~10 steps, ~0.5s)
  - Default:                      medium window   (~20 steps, ~1s)

This means surprising events bind richer, more detailed memories.
Routine events leave only faint traces.
This emerges naturally from M54's existing qe_norm signal — no new
signal needed, no change to M54.

BIOLOGICAL RULES IMPLEMENTED
-----------------------------
1. Hebbian Potentiation
   ΔW[i,j] = η × trace[i] × trace[j]
   Co-active neurons strengthen their connection.

2. Synaptic Decay
   W ← W × (1 - decay)
   All weights drift toward zero unless reinforced.
   This is forgetting. Necessary and useful.

3. Weight Normalization (Homeostatic)
   Each row normalized so max weight stays bounded.
   Prevents any single memory from dominating.
   Mirrors synaptic scaling in real cortex.

4. Adaptive Trace (Neuromodulatory)
   trace_decay = f(qe_norm)
   Surprise extends the eligibility window.
   Mirrors acetylcholine/norepinephrine modulation in brain.

FAMILIARITY SIGNAL
------------------
After recall settling, M55 returns:
  'depth':      energy drop during settling (deep = familiar)
  'speed':      steps to converge (fast = familiar)
  'familiarity': combined score [0, 1]  (1 = deeply familiar)

This signal can feed Layer 2 as a "recognition prior."

SPACE COST
----------
W matrix:        64 × 64 × float32 = 16 KB
Trace vector:    64 × float32       = 256 bytes
Everything else: negligible
Total:           ~16 KB RAM. Zero bytes disk.

INTERFACE
---------
mem = AssociativeMemory()

# In your loop, after cortex.step():
mem.step(bmu_idx, qe_norm)

# To recall from current BMU:
result = mem.recall(bmu_idx)
# result['familiarity']  → float [0,1]
# result['settled_pattern'] → array (64,) — recalled activation
# result['depth']         → energy drop
# result['speed']         → steps to settle

# Diagnostics (for ExperienceBuffer / debugging):
mem.get_state()      → dict with W snapshot, trace, stats
mem.memory_map()     → (8×8) array — memory strength per cortex neuron
mem.strongest_associations(bmu_idx) → top-k associated neurons
"""

import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# Grid (must match M54)
GRID_H    = 8
GRID_W    = 8
N_NEURONS = GRID_H * GRID_W   # 64

# Hebbian learning rate
# Small — memory accumulates over many exposures, not one shot
ETA_HEBB = 0.04

# Synaptic decay rate (per step)
# τ_decay = 1/DECAY ≈ 2000 steps ≈ ~100s of experience before fading
# Familiar memories resist decay because they're constantly re-strengthened
DECAY_RATE = 0.0005

# Weight normalization ceiling (per neuron row-max)
W_MAX = 1.0

# ── Adaptive eligibility trace ──────────────────────────────────
# Base decay: τ = 1/TRACE_DECAY_BASE ≈ 20 steps (~1s at dt=0.05)
TRACE_DECAY_BASE    = 0.05    # familiar input → fast decay → short window

# Novel decay: τ = 1/TRACE_DECAY_NOVEL ≈ 40 steps (~2s)
TRACE_DECAY_NOVEL   = 0.025   # novel input → slow decay → long window

# Minimum decay floor (longest possible window)
TRACE_DECAY_MIN     = 0.02    # ~50 steps

# How strongly qe_norm modulates the trace decay
# decay = TRACE_DECAY_BASE - NOVELTY_MODULATION * qe_norm
# At qe_norm=1.0: decay = 0.05 - 0.03 = 0.02 (longest window)
# At qe_norm=0.0: decay = 0.05          (shortest window)
NOVELTY_MODULATION  = 0.03

# ── Recall settling ──────────────────────────────────────────────
# Max iterations for pattern completion
RECALL_MAX_STEPS    = 10

# Convergence threshold — stop if pattern change below this
RECALL_CONVERGE_TOL = 1e-4

# Softmax temperature for recall update
# Lower = sharper (winner-take-all), Higher = softer (distributed)
RECALL_TEMPERATURE  = 0.5

# ── Familiarity scoring ──────────────────────────────────────────
# Energy drop needed to count as "familiar" (not novel)
FAMILIARITY_DEPTH_THRESH = 0.05

# How many steps to settle to count as "fast recall"
FAMILIARITY_SPEED_THRESH = 4

# ── Write threshold ──────────────────────────────────────────────
# Minimum trace value to participate in Hebbian update
# Prevents near-zero traces from adding noise to W
MIN_TRACE_TO_WRITE = 0.05

# ── L2 → M55 feedback ────────────────────────────────────────
# When L2's curiosity is high (unfamiliar sequence territory),
# memories should be written more strongly — novel experiences
# deserve stronger consolidation.
#
# eta_effective = ETA_HEBB * (1 + CURIOSITY_HEBB_BOOST * curiosity)
# At curiosity=1.0: ETA_HEBB * 2.0 = 0.08 (double write strength)
# At curiosity=0.0: ETA_HEBB * 1.0 = 0.04 (normal, unchanged)
#
# Biologically: acetylcholine release during novel/uncertain states
# enhances hippocampal encoding — exactly what this implements.
CURIOSITY_HEBB_BOOST = 1.0    # max multiplier: doubles ETA_HEBB at curiosity=1.0


# ═══════════════════════════════════════════════════════════════
# ASSOCIATIVE MEMORY
# ═══════════════════════════════════════════════════════════════

class AssociativeMemory:
    """
    Pure Hebbian associative memory for M54 cortex output.

    Sits above M54. Reads bmu_idx and qe_norm each step.
    Never modifies M54. Never writes to disk.
    Memory = weight matrix W (64×64). That's it.

    Compatible with M54's existing interface — drop-in addition.
    ExperienceBuffer continues to work unchanged as your debugger.
    """

    def __init__(self, seed=42, n_neurons: int = N_NEURONS):
        self._rng = np.random.RandomState(seed)
        self._n   = n_neurons  # actual neuron count (may differ from module default)

        # ── The memory itself ─────────────────────────────────
        # W[i, j] = associative strength between neuron i and neuron j
        # Initialized near zero — no memories at birth
        # Symmetric: W[i,j] == W[j,i] (mutual association)
        self._W = np.zeros((n_neurons, n_neurons), dtype=np.float32)

        # ── Eligibility trace ─────────────────────────────────
        # trace[i] = how recently neuron i was active
        # Decays exponentially each step
        # Adaptive decay rate controlled by qe_norm from M54
        self._trace = np.zeros(n_neurons, dtype=np.float32)

        # Current adaptive decay rate (updated each step)
        self._trace_decay = TRACE_DECAY_BASE

        # ── Diagnostics ──────────────────────────────────────
        self.t = 0                          # step counter
        self._write_events = 0              # how many Hebbian updates
        self._energy_history = deque(maxlen=200)
        self._familiarity_history = deque(maxlen=200)

        # Per-neuron write count (how often each neuron participated
        # in a Hebbian update — shows which memories are strongest)
        self._neuron_write_count = np.zeros(n_neurons, dtype=np.int32)

        # Per-BMU activation count — ground truth exposure counter.
        # Incremented every time that BMU fires in step().
        # This is the direct measure of familiarity: "I've seen this N times."
        # Breadth and depth are proxies; this is the fact.
        self._bmu_exposure = np.zeros(n_neurons, dtype=np.int32)

    # ── Core step ─────────────────────────────────────────────

    def step(self, bmu_idx: int, qe_norm: float,
             curiosity:    float = 0.0,
             rpe_positive: float = 0.0) -> dict:
        """
        One memory step. Call after every cortex.step().

        Parameters
        ----------
        bmu_idx : int
            Winning neuron index from M54 (0–63)
        qe_norm : float
            Normalized surprise from M54 [0, 1]
            0 = completely familiar, 1 = maximally novel
        curiosity : float [0,1]
            L2 sustained sequence novelty signal from previous step.
            0 = sequences are familiar, write at base rate.
            1 = sequences are novel, write at up to 2× base rate.
            Default 0.0 preserves old behaviour when feedback not wired.
        rpe_positive : float [0,1]
            Positive reward prediction error from V1 (Valence) module.
            "This outcome was better than expected — consolidate more."
            Boosts Hebbian write rate on top of curiosity boost.
            Formula: eta = ETA_HEBB * (1 + curiosity_boost + rpe_boost)
            Default 0.0 preserves old behaviour when V1 not wired.
            Biologically: dopamine burst → hippocampal LTP enhancement.

        Returns
        -------
        dict with:
            'wrote':         bool   — Hebbian update happened this step
            'trace_peak':    float  — highest trace value (activity level)
            'trace_decay':   float  — current window size parameter
            'w_mean':        float  — mean weight across matrix
            'w_max':         float  — strongest single association
        """
        # 1. Adapt trace decay to surprise
        #    Novel → slower decay → longer eligibility window
        #    Familiar → faster decay → shorter window
        self._trace_decay = max(
            TRACE_DECAY_MIN,
            TRACE_DECAY_BASE - NOVELTY_MODULATION * float(qe_norm)
        )

        # 2. Decay existing trace (all neurons fade toward zero)
        self._trace *= (1.0 - self._trace_decay)

        # 3. Activate current BMU (set its trace to 1.0)
        self._trace[bmu_idx] = 1.0

        # Count this exposure — ground truth familiarity signal
        self._bmu_exposure[bmu_idx] += 1

        # 4. Hebbian update — only where trace is meaningful
        #    ΔW[i,j] = η × trace[i] × trace[j]
        #    This is the outer product of trace with itself
        #    Only neurons that recently co-fired get strengthened
        active = self._trace >= MIN_TRACE_TO_WRITE
        wrote  = False

        if active.sum() > 1:
            # Outer product of active traces
            # L2 → M55 feedback: curiosity scales write strength
            # Novel sequences (high curiosity) → stronger memory consolidation
            # V1 → M55 feedback: positive RPE also boosts write strength
            # Better-than-expected outcomes → stronger
            from evaluators import RPE_M55_BOOST
            eta_effective = ETA_HEBB * (
                1.0
                + CURIOSITY_HEBB_BOOST * float(curiosity)
                + RPE_M55_BOOST * float(rpe_positive)
            )
            t_active = self._trace * active.astype(np.float32)
            delta_W  = eta_effective * np.outer(t_active, t_active)

            # No self-connections (diagonal = 0)
            np.fill_diagonal(delta_W, 0.0)

            self._W += delta_W

            # Track which neurons participated
            self._neuron_write_count[active] += 1
            self._write_events += 1
            wrote = True

        # 5. Synaptic decay — all weights drift toward zero
        #    Memories fade unless constantly refreshed
        self._W *= (1.0 - DECAY_RATE)

        # 6. Homeostatic normalization — keep weights bounded
        #    Row-normalize: no neuron dominates all others
        row_max = self._W.max(axis=1, keepdims=True)
        scale   = np.where(row_max > W_MAX, W_MAX / (row_max + 1e-9), 1.0)
        self._W *= scale

        # Enforce symmetry (W[i,j] == W[j,i])
        # Biological: mutual association, no directed edges
        self._W = (self._W + self._W.T) * 0.5

        # 7. Record energy (mean weight — proxy for total memory load)
        w_mean = float(self._W.mean())
        self._energy_history.append(w_mean)

        self.t += 1

        return {
            'wrote':       wrote,
            'trace_peak':  float(self._trace.max()),
            'trace_decay': float(self._trace_decay),
            'w_mean':      w_mean,
            'w_max':       float(self._W.max()),
            'curiosity':   float(curiosity),
        }

    # ── Recall ────────────────────────────────────────────────

    def recall(self, bmu_idx: int) -> dict:
        """
        Pattern completion from a single neuron cue.

        Starts from a partial activation (just bmu_idx = 1.0, rest = 0)
        and lets the associative weights pull the pattern toward the
        nearest stored attractor.

        This is recall as reconstruction — not retrieval.

        Parameters
        ----------
        bmu_idx : int
            Seed neuron to recall from (typically current BMU from M54)

        Returns
        -------
        dict with:
            'settled_pattern': ndarray (64,) — recalled activation
            'depth':           float — energy drop (bigger = more familiar)
            'speed':           int   — steps to converge
            'familiarity':     float [0,1] — combined familiarity score
            'converged':       bool  — did it settle cleanly?
            'top_associations': list of (neuron_idx, strength) — top 5
        """
        # Seed pattern: cue neuron active, rest zero
        pattern = np.zeros(self._n, dtype=np.float32)
        pattern[bmu_idx] = 1.0

        # Initial energy
        energy_start = self._pattern_energy(pattern)

        # ── Settling loop ─────────────────────────────────────
        # Speed measures how quickly the pattern reaches a CONFIDENT state —
        # meaning the peak activation rises above a threshold.
        #
        # Root cause of old MT-08 failure:
        #   Novel input → W near zero → softmax of near-zero = uniform →
        #   delta between steps ≈ 0 immediately → "converges" in 1-2 steps.
        #   This is NOT fast recall. It's the absence of recall.
        #
        #   Familiar input → W has real weights → softmax produces shaped
        #   distribution → pattern shifts each step → takes more steps
        #   before delta < TOL. Appeared "slower" but was actually working.
        #
        # Fix: convergence = peak activation exceeds CONFIDENCE_THRESH.
        #   Familiar → strong W → peak rises fast → low step count (fast).
        #   Novel    → weak W  → peak never rises → hits RECALL_MAX_STEPS (slow).
        CONFIDENCE_THRESH = 0.15   # peak activation must exceed this to "converge"

        steps       = 0
        peak_step   = RECALL_MAX_STEPS   # steps until confident (default = slow)

        for step in range(RECALL_MAX_STEPS):
            activation  = self._W @ pattern          # (64,) association pull

            exp_act     = np.exp(activation / (RECALL_TEMPERATURE + 1e-9))
            new_pattern = exp_act / (exp_act.sum() + 1e-9)

            # Keep cue neuron anchored
            new_pattern[bmu_idx] = max(new_pattern[bmu_idx], 0.5)

            s = new_pattern.sum()
            if s > 1e-9:
                new_pattern /= s

            steps   += 1
            pattern  = new_pattern

            # Confident when the recalled pattern has a strong non-cue peak
            # i.e. a real associated memory pulled something up
            non_cue         = new_pattern.copy()
            non_cue[bmu_idx] = 0.0
            if non_cue.max() > CONFIDENCE_THRESH:
                peak_step = steps
                break

        # Final energy
        energy_end = self._pattern_energy(pattern)
        depth      = float(max(energy_start - energy_end, 0.0))

        # ── Familiarity score ─────────────────────────────────
        #
        # WHAT WAS WRONG (root cause of MT-09 failure):
        #
        # Attempt 1 (depth): W hits ceiling W_MAX after ~40 steps.
        #   depth saturates at rep 1. Never grows. Familiarity flatlines.
        #
        # Attempt 2 (breadth): breadth = fraction of neurons activated
        #   above threshold during recall. Theory: after many exposures,
        #   transitive Hebb (A→B, B→C) spreads the pattern wider.
        #   REALITY: transitive spread only works if C has EVER co-fired
        #   with B. In MT-09, only BMU 22 and 23 ever fire. W is a 2-node
        #   graph. No third neuron ever enters. Breadth is structurally
        #   capped at 2/64 forever. Familiarity flatlines again.
        #
        # ROOT CAUSE: both metrics tried to infer familiarity from W geometry.
        #   But W geometry saturates. You cannot get a growing signal from
        #   a saturating source.
        #
        # CORRECT APPROACH: familiarity = direct exposure count.
        #   self._bmu_exposure[bmu_idx] is incremented every time this BMU
        #   fires in step(). It grows monotonically with experience.
        #   It never saturates. It is the ground truth.
        #
        # W-based signals (depth, breadth) are still useful as secondary
        # signals — they reflect the QUALITY of the stored association,
        # not just how often it was seen. But the PRIMARY signal must be
        # the exposure count, because that's what actually grows.
        #
        # Biologically: the brain uses both — familiarity is signaled by
        # perirhinal cortex (exposure-based, fast) while recollection
        # uses hippocampal pattern completion (W-based, slow). We mirror
        # this dual-process architecture here.

        exposure    = int(self._bmu_exposure[bmu_idx])

        # Exposure score: saturates at ~200 exposures (log scale)
        # log(1 + exposure) / log(1 + 200) gives ~0 at 0, ~1 at 200
        import math
        exposure_score = min(math.log1p(exposure) / math.log1p(200), 1.0)

        # Breadth score: fraction of neurons recalled above threshold
        BREADTH_THRESH = 0.02
        breadth        = float((pattern > BREADTH_THRESH).sum()) / self._n
        breadth_score  = min(breadth / 0.1, 1.0)

        # Depth score: energy drop during settling
        depth_score    = min(depth / (FAMILIARITY_DEPTH_THRESH + 1e-9), 1.0)

        # Speed score: how fast pattern reached confidence
        speed_score    = max(0.0, 1.0 - peak_step / RECALL_MAX_STEPS)

        # Weighted combination:
        #   exposure is primary  (0.5) — grows monotonically, never saturates
        #   breadth  is secondary (0.3) — reflects association richness
        #   depth    is tertiary  (0.1) — reflects attractor strength
        #   speed    is tertiary  (0.1) — reflects recall sharpness
        familiarity = float(np.clip(
            0.5 * exposure_score +
            0.3 * breadth_score  +
            0.1 * depth_score    +
            0.1 * speed_score,
            0.0, 1.0
        ))

        self._familiarity_history.append(familiarity)

        # Top associations for this BMU
        assoc_strengths = self._W[bmu_idx].copy()
        assoc_strengths[bmu_idx] = 0.0  # exclude self
        top_k_idx = np.argsort(assoc_strengths)[::-1][:5]
        top_associations = [
            (int(idx), float(assoc_strengths[idx]))
            for idx in top_k_idx
            if assoc_strengths[idx] > 1e-4
        ]

        return {
            'settled_pattern':  pattern,
            'depth':            depth,
            'speed':            peak_step,
            'familiarity':      familiarity,
            'converged':        peak_step < RECALL_MAX_STEPS,
            'top_associations': top_associations,
        }

    # ── Internal ──────────────────────────────────────────────

    def _pattern_energy(self, pattern: np.ndarray) -> float:
        """
        Hopfield-style energy: E = -½ Σ W[i,j] s[i] s[j]
        Lower energy = pattern fits a stored attractor well.
        Familiar patterns have low energy (deep attractor valley).
        Novel patterns have high energy (no valley nearby).
        """
        return float(-0.5 * pattern @ self._W @ pattern)

    # ── Diagnostics (for ExperienceBuffer / debugging) ────────

    def get_state(self) -> dict:
        """
        Full diagnostic snapshot. Feed into ExperienceBuffer tags
        or print for debugging. Never needed for core operation.
        """
        return {
            't':                   self.t,
            'write_events':        self._write_events,
            'w_mean':              float(self._W.mean()),
            'w_max':               float(self._W.max()),
            'w_nonzero_fraction':  float((self._W > 1e-4).mean()),
            'trace_peak':          float(self._trace.max()),
            'trace_decay':         float(self._trace_decay),
            'mean_familiarity':    float(np.mean(self._familiarity_history))
                                   if self._familiarity_history else 0.0,
            'W_snapshot':          self._W.copy(),
            'neuron_write_counts': self._neuron_write_count.copy(),
        }

    def memory_map(self) -> np.ndarray:
        """
        (8×8) array — total associative strength per cortex neuron.
        Bright spots = neurons with strong memories.
        Dark spots = neurons with little experience.
        Use to visualize what the system has learned.
        """
        strength = self._W.sum(axis=1)   # (64,) — total outgoing weight
        return strength.reshape(GRID_H, GRID_W)

    def strongest_associations(self, bmu_idx: int, k: int = 5) -> list:
        """
        Top-k neurons most strongly associated with bmu_idx.
        Returns list of (neuron_idx, weight, grid_pos).
        Use to answer: "what does this memory connect to?"
        """
        weights = self._W[bmu_idx].copy()
        weights[bmu_idx] = 0.0
        top_idx = np.argsort(weights)[::-1][:k]
        return [
            {
                'neuron_idx': int(i),
                'weight':     float(weights[i]),
                'grid_pos':   (int(i // GRID_W), int(i % GRID_W)),
            }
            for i in top_idx
            if weights[i] > 1e-4
        ]

    def summary(self):
        """Human-readable state summary. For debugging."""
        s = self.get_state()
        print(f"  AssociativeMemory (M55) — step {s['t']}")
        print(f"  Write events:      {s['write_events']}")
        print(f"  W mean / max:      {s['w_mean']:.5f} / {s['w_max']:.4f}")
        print(f"  W nonzero:         {s['w_nonzero_fraction']*100:.1f}% of matrix")
        print(f"  Trace peak:        {s['trace_peak']:.3f}")
        print(f"  Trace decay:       {s['trace_decay']:.4f}  "
              f"(window ≈ {1/s['trace_decay']:.0f} steps)")
        print(f"  Mean familiarity:  {s['mean_familiarity']:.3f}")

        # Most written neurons
        top_neurons = np.argsort(self._neuron_write_count)[::-1][:5]
        print(f"  Most-written neurons: "
              + "  ".join(
                  f"N{i}({self._neuron_write_count[i]}×)"
                  for i in top_neurons
                  if self._neuron_write_count[i] > 0
              ))