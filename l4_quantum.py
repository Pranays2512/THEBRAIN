"""
L4Q: QUANTUM-INSPIRED POSITION BELIEF
======================================

WHY QUANTUM MECHANICS?
----------------------
The classical L4 (l4_position.py) uses a probability vector — at each step
it multiplies non-matching nodes by zero (hard collapse) and renormalises.
This destroys path history: after hearing fi=0 (ambiguous A/I), the belief
is exactly 50/50 regardless of what came before.

Schrödinger's equation does something fundamentally different:
  - State is a COMPLEX AMPLITUDE vector ψ, not a probability vector
  - |ψ[node]|² is the probability of being at that node
  - The amplitude carries PHASE — directional information about how
    probability is flowing and which paths led here
  - Paths INTERFERE: ψ_A + ψ_B can be larger or smaller than either alone
    depending on phase alignment (constructive vs destructive interference)

HOW THIS SOLVES ALIASING
------------------------
A and I both produce fi=0. Their single-step transitions are symmetric.
Classical L4 is stuck at 50/50.

With phase:
  - Each action accumulates a phase shift exp(iθ) on the amplitude
  - The C→B→A path accumulates phase θ(fi=2,East) + θ(fi=1,West)
  - The K→J→I path accumulates phase θ(fi=4,West) + θ(fi=1,West)
  - These are DIFFERENT phase histories
  - When the amplitudes from both paths arrive at fi=0, they interfere:
    if phases differ significantly → partial destructive interference
    → one node's amplitude is suppressed
  - After 3-4 steps the phase signature of the current path reinforces
    the correct node constructively

THREE KEY DIFFERENCES FROM CLASSICAL L4
-----------------------------------------
1. AMPLITUDE not probability: ψ ∈ ℂⁿ, probabilities = |ψ|²
2. SOFT OBSERVATION: non-matching nodes get amplitude *= SOFT_DECAY (0.15)
   instead of hard zero. Path history persists through ambiguous steps.
3. PHASE EVOLUTION: each (fi, action) pair has a fixed phase angle.
   Consistent path history → coherent phase → constructive interference.
   Crossed paths → phase mismatch → partial cancellation.

PARAMETERS
----------
L4Q_SOFT_DECAY       = 0.15   # amplitude multiplier for non-matching obs
                               # (vs 0.0 in classical — that's the key change)
L4Q_PHASE_SCALE      = 0.6    # how strongly action history rotates phase
                               # 0 = no phase (reduces to soft-decay only)
                               # π = full half-rotation per step
L4Q_BELIEF_DECAY     = 0.015  # per-step amplitude decay toward uniform
L4Q_TM_WARMUP        = 25     # min transitions before TM trusted
L4Q_CTM_WARMUP       = 12     # min context-pair transitions before CTM trusted
L4Q_CONFIDENCE_THRESH= 0.60   # min top_prob for confident output
L4Q_ENTROPY_SMOOTH   = 0.05   # EMA smoothing for entropy output

INTERFACE
---------
Drop-in replacement for PositionBelief. Same step() signature, same output keys.
Brain.py just needs: from l4_quantum import QuantumPositionBelief
and:                  self.l4 = QuantumPositionBelief(node_fi=node_fi)
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

L4Q_SOFT_DECAY        = 0.10   # amplitude kept for non-matching nodes
                               # Lowered from 0.15: reduces leakage through ambiguous obs,
                               # so aliased nodes (A/I, B/J, D/L, E/K) don't reach false
                               # confidence that poisons Q_n updates in M56.
L4Q_PHASE_SCALE       = 0.6    # radians of phase per (fi, action) step
L4Q_BELIEF_DECAY      = 0.015
L4Q_TM_WARMUP         = 25
L4Q_CTM_WARMUP        = 12
L4Q_CONFIDENCE_THRESH = 0.65   # raised from 0.60: requires stronger phase-derived signal
                               # before claiming confident node belief. Prevents premature
                               # Q_n blending on still-ambiguous aliased nodes.
L4Q_ENTROPY_SMOOTH    = 0.05


# ═══════════════════════════════════════════════════════════════
# QUANTUM POSITION BELIEF
# ═══════════════════════════════════════════════════════════════

class QuantumPositionBelief:
    """
    Complex-amplitude position tracker.

    Uses Schrödinger-inspired state evolution:
      - ψ ∈ ℂⁿ  (complex amplitude per node)
      - probability[node] = |ψ[node]|²
      - Observation: soft amplitude decay (non-matching nodes stay at
        SOFT_DECAY × amplitude, not zeroed)
      - Phase: each (fi, action) step rotates ψ by exp(i*θ), where θ
        depends on the fi and action. Consistent paths build coherent
        phase; mixed paths develop phase mismatch and partial cancellation.

    Parameters
    ----------
    node_fi   : dict[str, int]  node → freq_index
    n_freqs   : int             number of distinct frequencies (default 8)
    n_actions : int             number of actions (default 4)
    """

    def __init__(self, node_fi: dict, n_freqs: int = 8, n_actions: int = 4):
        self.node_fi   = dict(node_fi)
        self.nodes     = list(node_fi.keys())
        self.n_nodes   = len(self.nodes)
        self.n_freqs   = n_freqs
        self.n_actions = n_actions
        self._node_to_idx = {n: i for i, n in enumerate(self.nodes)}

        # ── Complex amplitude vector ───────────────────────────
        # Start uniform and real. Phase accumulates over time.
        self._psi = (np.ones(self.n_nodes, dtype=np.complex128)
                     / np.sqrt(self.n_nodes))

        # ── Sound likelihood ───────────────────────────────────
        # P(fi | node): 1.0 if node sounds like fi, else 0.
        # Used in observation step — applied to amplitudes, not probs.
        self._sound_match = np.zeros((self.n_nodes, n_freqs), dtype=np.float64)
        for i, node in enumerate(self.nodes):
            self._sound_match[i, self.node_fi[node]] = 1.0

        # ── Phase table: fixed per (fi, action) pair ──────────
        # Each (fi, action) combination has a unique phase angle.
        # Derived from a deterministic hash so it's reproducible and
        # spread across [0, 2π] without requiring learning.
        # Nodes that share fi but are reached via different action histories
        # will accumulate different phase signatures over time.
        rng = np.random.default_rng(seed=314159)   # fixed seed — not learned
        raw_phases = rng.uniform(0, 2 * np.pi, size=(n_freqs, n_actions))
        self._phase_table = raw_phases * L4Q_PHASE_SCALE  # shape (n_freqs, n_actions)

        # Per-node phase accumulator — tracks how much phase each node
        # has accumulated based on the path leading to it.
        # Updated every step based on which fi×action pair was just taken.
        self._node_phase = np.zeros(self.n_nodes, dtype=np.float64)

        # ── Transition models (same as classical L4) ──────────
        # TM:  zone-level 1-step
        self._TM      = np.zeros((n_freqs, n_actions, n_freqs), dtype=np.float64)
        self._TM_norm = np.ones((n_freqs, n_actions, n_freqs),
                                 dtype=np.float64) / n_freqs
        self._TM_n    = np.zeros((n_freqs, n_actions), dtype=np.int32)

        # CTM: 2-step context
        N_CTX = n_freqs * n_freqs
        self._N_CTX    = N_CTX
        self._CTM      = np.zeros((N_CTX, n_actions, n_freqs), dtype=np.float64)
        self._CTM_norm = np.ones((N_CTX, n_actions, n_freqs),
                                  dtype=np.float64) / n_freqs
        self._CTM_n    = np.zeros((N_CTX, n_actions), dtype=np.int32)

        # NTM: node-level belief-weighted
        self._NTM      = np.zeros((self.n_nodes, n_actions, n_freqs),
                                   dtype=np.float64)
        self._NTM_norm = np.ones((self.n_nodes, n_actions, n_freqs),
                                  dtype=np.float64) / n_freqs
        self._NTM_n    = np.zeros((self.n_nodes, n_actions), dtype=np.int32)

        # ── State ─────────────────────────────────────────────
        self._prev_fi      = -1
        self._prev_prev_fi = -1
        self._prev_action  = -1
        self.t             = 0

        # ── Output cache ──────────────────────────────────────
        self._top_node       = self.nodes[0]
        self._top_prob       = 1.0 / self.n_nodes
        self._belief_entropy = 1.0
        self._entropy_ema    = 1.0
        self._confident      = False

    # ── Helpers ───────────────────────────────────────────────

    def _prob(self) -> np.ndarray:
        """Convert complex amplitudes to real probabilities."""
        p = np.abs(self._psi) ** 2
        s = p.sum()
        if s < 1e-12:
            return np.ones(self.n_nodes) / self.n_nodes
        return p / s

    def _normalise_psi(self):
        """Normalise amplitude vector so |ψ|² sums to 1."""
        n = np.linalg.norm(self._psi)
        if n > 1e-12:
            self._psi /= n

    # ── Main update ───────────────────────────────────────────

    def step(self, curr_fi: int, action: int = -1,
             world_moved: bool = True) -> dict:
        """
        Update quantum belief given current observed frequency index.

        Parameters
        ----------
        curr_fi     : int  — bucketed freq index (0-7) heard this step
        action      : int  — action taken this step (-1 = unknown)
        world_moved : bool — False on wall hits; models don't update
        """
        if curr_fi < 0 or curr_fi >= self.n_freqs:
            # Unknown observation: decay toward uniform
            self._psi = ((1.0 - L4Q_BELIEF_DECAY) * self._psi
                         + L4Q_BELIEF_DECAY / self.n_nodes)
            self._normalise_psi()
            self._prev_action = action
            self.t += 1
            return self._make_output()

        # ── 1. Learn transition models from last move ─────────
        if world_moved and self._prev_fi >= 0 and self._prev_action >= 0:
            fi_p = self._prev_fi
            a    = self._prev_action
            prob = self._prob()

            # TM
            self._TM[fi_p, a, curr_fi] += 1.0
            row = self._TM[fi_p, a]; s = row.sum()
            if s > 0: self._TM_norm[fi_p, a] = row / s
            self._TM_n[fi_p, a] += 1

            # CTM
            if self._prev_prev_fi >= 0:
                ctx = self._prev_prev_fi * self.n_freqs + fi_p
                self._CTM[ctx, a, curr_fi] += 1.0
                row = self._CTM[ctx, a]; s = row.sum()
                if s > 0: self._CTM_norm[ctx, a] = row / s
                self._CTM_n[ctx, a] += 1

            # NTM — belief-weighted
            for ni in range(self.n_nodes):
                if self.node_fi[self.nodes[ni]] != fi_p: continue
                w = float(prob[ni])
                if w < 1e-6: continue
                self._NTM[ni, a, curr_fi] += w
                row = self._NTM[ni, a]; s = row.sum()
                if s > 0: self._NTM_norm[ni, a] = row / s
                self._NTM_n[ni, a] += 1

        # ── 2. Transition prior: propagate amplitude forward ──
        if world_moved and self._prev_action >= 0 and self._prev_fi >= 0:
            a      = self._prev_action
            fi_prv = self._prev_fi

            new_psi = np.zeros(self.n_nodes, dtype=np.complex128)
            new_phase = np.zeros(self.n_nodes, dtype=np.float64)
            prob = self._prob()

            for ni in range(self.n_nodes):
                amp_i = self._psi[ni]
                if abs(amp_i) < 1e-8: continue
                fi_i = self.node_fi[self.nodes[ni]]

                # Pick deepest available transition model
                tm_row = None
                if self._prev_prev_fi >= 0 and fi_i == fi_prv:
                    ctx = self._prev_prev_fi * self.n_freqs + fi_i
                    if self._CTM_n[ctx, a] >= L4Q_CTM_WARMUP:
                        tm_row = self._CTM_norm[ctx, a]
                if tm_row is None and self._NTM_n[ni, a] >= L4Q_TM_WARMUP:
                    tm_row = self._NTM_norm[ni, a]
                if tm_row is None:
                    tm_row = self._TM_norm[fi_i, a]

                # Phase this node contributes to its successors
                phase_contribution = self._node_phase[ni] + self._phase_table[fi_i, a]

                # Distribute amplitude to successor nodes with fi_j.
                # Split uniformly among all nodes sharing fi_j — the phase
                # mechanism handles disambiguation over time.
                for fi_j in range(self.n_freqs):
                    p = float(tm_row[fi_j])
                    if p < 1e-8: continue
                    amp_contribution = amp_i * np.sqrt(p)
                    candidates = [nj for nj in range(self.n_nodes)
                                  if self.node_fi[self.nodes[nj]] == fi_j]
                    share = 1.0 / max(len(candidates), 1)
                    for nj in candidates:
                        new_psi[nj] += amp_contribution * share
                        new_phase[nj] += phase_contribution * float(prob[ni])

            n = np.linalg.norm(new_psi)
            if n > 1e-9:
                self._psi = new_psi / n
                # Update phase accumulators for the new distribution
                # Normalise phase contributions by probability mass
                prob_new = np.abs(self._psi) ** 2
                for nj in range(self.n_nodes):
                    if prob_new[nj] > 1e-6:
                        self._node_phase[nj] = new_phase[nj]

        # ── 3. Phase rotation ─────────────────────────────────
        # Apply the phase accumulated for this (fi, action) step.
        # Only on real world moves — wall hits don't advance position.
        if world_moved and action >= 0 and curr_fi >= 0:
            phase_shift = self._phase_table[curr_fi, action]
            for ni in range(self.n_nodes):
                fi_ni = self.node_fi[self.nodes[ni]]
                if fi_ni == curr_fi:
                    self._psi[ni] *= np.exp(1j * phase_shift)
                    self._node_phase[ni] += phase_shift
                else:
                    self._psi[ni] *= np.exp(1j * phase_shift * 0.3)

        # ── 4. Soft observation update ────────────────────────
        # THE KEY QUANTUM DIFFERENCE:
        # Classical: belief[non_match] *= 0.0   (hard collapse)
        # Quantum:   psi[non_match]   *= SOFT_DECAY  (soft amplitude decay)
        #
        # This preserves path history through ambiguous observations.
        # If L4 was 80% confident we're at A (from C path), and hears fi=0
        # (ambiguous A/I), classical resets to 50/50. Quantum stays ~70/30
        # because A's amplitude history is larger and only partially decayed.
        #
        # UNIFORM FLOOR: add a tiny uniform amplitude before observation so
        # that nodes with near-zero amplitude can still receive probability
        # mass from matching observations (avoids hard zero from reset_to_node).
        uniform_floor = 0.01 / np.sqrt(self.n_nodes)
        self._psi += uniform_floor
        self._normalise_psi()

        # Only update observation on real moves
        if world_moved:
            obs_weights = np.where(
                self._sound_match[:, curr_fi] > 0.5,
                1.0,
                L4Q_SOFT_DECAY
            )
            self._psi *= obs_weights
            self._normalise_psi()

        # ── 5. Decay toward uniform ───────────────────────────
        # Slow drift prevents overconfidence from locking in wrong belief.
        uniform_amp = 1.0 / np.sqrt(self.n_nodes)
        self._psi = ((1.0 - L4Q_BELIEF_DECAY) * self._psi
                     + L4Q_BELIEF_DECAY * uniform_amp)
        self._normalise_psi()

        # ── 6. Advance state ──────────────────────────────────
        self._prev_prev_fi = self._prev_fi
        self._prev_fi      = curr_fi
        self._prev_action  = action
        self.t            += 1

        return self._make_output()

    # ── Output ────────────────────────────────────────────────

    def _make_output(self) -> dict:
        prob           = self._prob()
        top_idx        = int(np.argmax(prob))
        top_prob       = float(prob[top_idx])
        self._top_node = self.nodes[top_idx]
        self._top_prob = top_prob

        b     = np.clip(prob, 1e-12, 1.0)
        raw_e = float(np.clip(
            -np.sum(b * np.log(b)) / np.log(self.n_nodes + 1e-9),
            0.0, 1.0))
        self._entropy_ema    = ((1.0 - L4Q_ENTROPY_SMOOTH) * self._entropy_ema
                                + L4Q_ENTROPY_SMOOTH * raw_e)
        self._belief_entropy = self._entropy_ema
        self._confident      = top_prob >= L4Q_CONFIDENCE_THRESH

        return {
            'top_node':       self._top_node,
            'top_prob':       top_prob,
            'belief_entropy': self._belief_entropy,
            'belief':         {n: float(prob[i])
                               for i, n in enumerate(self.nodes)},
            'confident':      self._confident,
            'belief_vector':  prob.copy(),
            # Quantum-specific diagnostics
            'phase_coherence': float(np.abs(np.mean(
                np.exp(1j * self._node_phase)))),
        }

    # ── Accessors ─────────────────────────────────────────────

    def get_top_node(self) -> str:   return self._top_node
    def get_top_prob(self) -> float: return self._top_prob
    def is_confident(self) -> bool:  return self._confident
    def get_belief_vector(self) -> np.ndarray: return self._prob().copy()

    def reset_to_node(self, node: str) -> None:
        if node in self._node_to_idx:
            self._psi[:] = 0.0
            self._psi[self._node_to_idx[node]] = 1.0
            self._node_phase[:] = 0.0

    def reset_uniform(self) -> None:
        self._psi[:] = 1.0 / np.sqrt(self.n_nodes)
        self._node_phase[:] = 0.0
        self._prev_fi      = -1
        self._prev_prev_fi = -1
        self._prev_action  = -1

    # ── Diagnostics ───────────────────────────────────────────

    def summary(self, ground_truth_node: str = None) -> str:
        prob  = self._prob()
        lines = [f"  L4Q QuantumPositionBelief — step {self.t}"]
        lines.append(f"  Top: {self._top_node} ({self._top_prob:.3f})  "
                     f"entropy={self._belief_entropy:.3f}  "
                     f"confident={self._confident}")
        coherence = float(np.abs(np.mean(np.exp(1j * self._node_phase))))
        lines.append(f"  Phase coherence: {coherence:.3f}")
        if ground_truth_node:
            true_idx  = self._node_to_idx.get(ground_truth_node, -1)
            true_prob = float(prob[true_idx]) if true_idx >= 0 else 0.0
            correct   = (self._top_node == ground_truth_node)
            lines.append(f"  True: {ground_truth_node}  "
                         f"P(true)={true_prob:.3f}  "
                         f"{'✓' if correct else '✗'}")
        sig = sorted(
            [(self.nodes[i], float(prob[i]))
             for i in range(self.n_nodes) if prob[i] > 0.05],
            key=lambda x: -x[1])
        lines.append("  Belief: " +
                     "  ".join(f"{n}={p:.3f}" for n, p in sig[:6]))
        return "\n".join(lines)

    def tm_coverage(self) -> dict:
        covered_tm  = int((self._TM_n  >= L4Q_TM_WARMUP).sum())
        covered_ctm = int((self._CTM_n >= L4Q_CTM_WARMUP).sum())
        return {
            'tm_covered':  covered_tm,  'tm_total':  self.n_freqs * self.n_actions,
            'ctm_covered': covered_ctm, 'ctm_total': self._N_CTX  * self.n_actions,
        }