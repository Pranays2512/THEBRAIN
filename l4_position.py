"""
L4: POSITION BELIEF — BAYESIAN LOCATION TRACKING
=================================================

WHAT THIS MODULE DOES
---------------------
L4 maintains a probability distribution over all possible locations
(nodes in the world). Every step it asks: "given what I just heard and
what action I just took, where am I most likely to be?"

This is the missing piece that separates REACTIVE navigation (hear a
sound → act) from DELIBERATE navigation (know where I am → plan a route).

WHY IT'S NEEDED
---------------
The brain currently has no sense of position. M56's Q-table is indexed
on (prev_bmu, curr_bmu) — which fires the same entry for node A and
node I because they produce identical sounds (0.5Hz). L2's TC table
helps by conditioning on the previous frequency, but it only looks one
step back at zone granularity.

L4 does something fundamentally different: it tracks a running belief
distribution forward through time, using the FULL history of sounds
and actions. The belief at step t is:

  belief[node] ∝ P(heard_sound | at_node)
                × Σ_prev [ P(reached_node | was_at_prev, took_action)
                           × belief_prev[prev_node] ]

This is the standard Hidden Markov Model (HMM) / particle filter update.
After 3-4 steps on a unique path, the distribution collapses to a single
node — the brain knows exactly where it is.

HOW DISAMBIGUATION WORKS — THE ALIASING PROBLEM
-------------------------------------------------
A and I both sound like 0.5Hz. Their single-step forward transitions are
also symmetric: both go East to 0.7Hz. So one step of context is not enough.

Two steps are enough:
  C(0.9Hz)→West→B(0.7Hz)→West→A(0.5Hz): context chain fi=2→fi=1→fi=0
  K(1.3Hz)→West→J(0.7Hz)→West→I(0.5Hz): context chain fi=4→fi=1→fi=0

The (prev_fi=2, curr_fi=1) pair is NEVER observed on the I path.
The (prev_fi=4, curr_fi=1) pair is NEVER observed on the A path.
The CTM (context transition model) tracks exactly this 2-step pattern.

THREE TRANSITION MODELS (deepest model with sufficient data wins)
------------------------------------------------------------------
1. CTM[(prev_fi*8+curr_fi), action, next_fi]  — 2-step context (best)
2. NTM[node_idx, action, next_fi]             — node-level, belief-weighted
3. TM[fi, action, next_fi]                    — zone-level fallback

PARAMETERS
----------
L4_BELIEF_DECAY      = 0.02   # per-step decay toward uniform
L4_TM_WARMUP         = 25     # min transitions before TM trusted
L4_CTM_WARMUP        = 12     # lower — context pairs visited less often
L4_CONFIDENCE_THRESH = 0.60   # min top_prob for confident output
L4_ENTROPY_SMOOTH    = 0.05   # EMA smoothing for entropy

# ── World 5 scaling overrides ──────────────────────────────────
# 16-node worlds need slower decay (0.02 leaks belief to uniform too
# fast — after 50 steps belief is nearly flat) and lower CTM warmup
# (64 context pairs vs 32 in World 3, each visited less often).
# These are set by Brain.__init__ when node_fi has > 12 nodes.
L4_BELIEF_DECAY_LARGE_WORLD = 0.005
L4_CTM_WARMUP_LARGE_WORLD   = 6
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

L4_BELIEF_DECAY      = 0.02
L4_TM_WARMUP         = 25
L4_CTM_WARMUP        = 12
L4_CONFIDENCE_THRESH = 0.80
L4_ENTROPY_SMOOTH    = 0.05

# World-5 scaling overrides (applied by Brain for n_nodes > 12)
L4_BELIEF_DECAY_LARGE_WORLD = 0.005
L4_CTM_WARMUP_LARGE_WORLD   = 6


# ═══════════════════════════════════════════════════════════════
# POSITION BELIEF MODULE
# ═══════════════════════════════════════════════════════════════

class PositionBelief:
    """
    Bayesian position tracker for a discrete-node soundscape.

    Parameters
    ----------
    node_fi   : dict[str, int]  node name → freq_index
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

        # ── Belief ────────────────────────────────────────────
        self._belief = np.ones(self.n_nodes, dtype=np.float64) / self.n_nodes
        self._prev_belief_for_nntm = np.ones(self.n_nodes, dtype=np.float64) / self.n_nodes

        # P(fi | node): exact — 1.0 if node sounds like fi, else 0
        self._sound_likelihood = np.zeros((self.n_nodes, n_freqs), dtype=np.float64)
        for i, node in enumerate(self.nodes):
            self._sound_likelihood[i, self.node_fi[node]] = 1.0

        # ── TM: zone-level 1-step transition model ────────────
        self._TM      = np.zeros((n_freqs, n_actions, n_freqs), dtype=np.float64)
        self._TM_norm = np.ones((n_freqs, n_actions, n_freqs),
                                 dtype=np.float64) / n_freqs
        self._TM_n    = np.zeros((n_freqs, n_actions), dtype=np.int32)

        # ── CTM: 2-step context transition model ──────────────
        # Indexed by (prev_fi * n_freqs + curr_fi) — 64 context pairs
        # This is the key to aliased-node disambiguation.
        N_CTX = n_freqs * n_freqs
        self._N_CTX    = N_CTX
        self._CTM      = np.zeros((N_CTX, n_actions, n_freqs), dtype=np.float64)
        self._CTM_norm = np.ones((N_CTX, n_actions, n_freqs),
                                  dtype=np.float64) / n_freqs
        self._CTM_n    = np.zeros((N_CTX, n_actions), dtype=np.int32)

        # ── NNTM: Node-to-Node topological transition model ──
        self._NNTM      = np.zeros((self.n_nodes, n_actions, self.n_nodes), dtype=np.float64)
        self._NNTM_norm = np.ones((self.n_nodes, n_actions, self.n_nodes), dtype=np.float64) / self.n_nodes
        self._NNTM_n    = np.zeros((self.n_nodes, n_actions), dtype=np.float64)

        self._locked_transitions = {}  # (node_idx, action) -> node_idx: confirmed deterministic edges

        # EXPERIMENTAL PROOF: Provide actual topology to NNTM instead of learning it via noisy forward aliased signals.
        # This proves that if L4 knew the topological edges (which it mathematically cannot learn in a purely forward online pass with fully aliased observations), tracking would be perfect.
        # if self.n_nodes == 16:
        #     try:
        #         import world5
        #         action_names = ['North', 'East', 'South', 'West']
        #         for src, edges in world5.ADJACENCY.items():
        #             ni = self._node_to_idx.get(src, -1)
        #             if ni >= 0:
        #                 for act_str, dest in edges.items():
        #                     a = action_names.index(act_str)
        #                     nj = self._node_to_idx.get(dest, -1)
        #                     if nj >= 0:
        #                         self._NNTM_n[ni, a] = 10.0  # warmed up
        #                         self._NNTM_norm[ni, a, :] = 0.0
        #                         self._NNTM_norm[ni, a, nj] = 1.0
        #     except ImportError:
        #         pass

        # ── WM: Wall model ────────────────────────────────────
        self._WM_hits  = np.zeros((self.n_nodes, n_actions), dtype=np.float64)
        self._WM_tries = np.zeros((self.n_nodes, n_actions), dtype=np.float64)

        # ── State ─────────────────────────────────────────────
        self._prev_fi      = -1
        self._prev_prev_fi = -1   # TWO steps back — for CTM context
        self._prev_action  = -1
        self.t             = 0

        # Hippocampal replay buffer for backward NNTM learning
        from collections import deque
        self._history = deque(maxlen=16)

        # ── Output cache ──────────────────────────────────────
        self._top_node       = self.nodes[0]
        self._top_prob       = 1.0 / self.n_nodes
        self._belief_entropy = 1.0
        self._entropy_ema    = 1.0
        self._confident      = False

    # ── Main update ───────────────────────────────────────────

    def step(self, curr_fi: int, action: int = -1,
             world_moved: bool = True) -> dict:
        """
        Update belief given the current observed frequency index.

        Parameters
        ----------
        curr_fi     : int  — bucketed freq index (0-7) heard this step
        action      : int  — action taken this step (-1 = unknown / open loop)
        world_moved : bool — False on wall hits; belief stays, models don't update
        """
        if curr_fi < 0 or curr_fi >= self.n_freqs:
            self._belief = ((1.0 - L4_BELIEF_DECAY) * self._belief
                            + L4_BELIEF_DECAY / self.n_nodes)
            self._belief /= self._belief.sum()
            self._prev_action = action
            self.t += 1
            return self._make_output()

        # ── 1. Update Wall Model ──────────────────────────────
        if action >= 0:
            for ni in range(self.n_nodes):
                w = float(self._belief[ni])
                if w < 1e-4: continue
                self._WM_tries[ni, action] += w
                if not world_moved:
                    self._WM_hits[ni, action] += w

        # ── 1b. Learn from the just-completed transition ───────
        # prev_fi --[action]--> curr_fi (we just arrived at curr_fi)
        if world_moved and self._prev_fi >= 0 and action >= 0:
            fi_p = self._prev_fi
            a    = action

            # TM update
            self._TM[fi_p, a, curr_fi] += 1.0
            row = self._TM[fi_p, a]; s = row.sum()
            if s > 0: self._TM_norm[fi_p, a] = row / s
            self._TM_n[fi_p, a] += 1

            # CTM update — needs prev_prev_fi
            if self._prev_prev_fi >= 0:
                ctx = self._prev_prev_fi * self.n_freqs + fi_p
                self._CTM[ctx, a, curr_fi] += 1.0
                row = self._CTM[ctx, a]; s = row.sum()
                if s > 0: self._CTM_norm[ctx, a] = row / s
                self._CTM_n[ctx, a] += 1

        # ── 2a. Wall Prior update ─────────────────────────────
        if action >= 0:
            wall_likelihood = np.ones(self.n_nodes, dtype=np.float64)
            for ni in range(self.n_nodes):
                t_w = self._WM_tries[ni, action]
                if t_w >= 1.0:
                    p_wall = self._WM_hits[ni, action] / t_w
                    p_obs  = p_wall if not world_moved else (1.0 - p_wall)
                    wall_likelihood[ni] = max(0.01, p_obs)
            self._belief *= wall_likelihood
            if self._belief.sum() > 1e-12:
                self._belief /= self._belief.sum()

        # ── 2b. Transition prior: propagate belief forward ─────
        if world_moved and action >= 0 and self._prev_fi >= 0:
            a      = action
            fi_prv = self._prev_fi

            new_prior = np.zeros(self.n_nodes, dtype=np.float64)

            for ni in range(self.n_nodes):
                b_i = float(self._belief[ni])
                if b_i < 1e-8: continue
                fi_i = self.node_fi[self.nodes[ni]]

                # Top priority: Topological Node-to-Node Model (NNTM)
                # Requires sufficient specific observations of this origin node + action
                # Check for locked deterministic edges first
                key = (ni, a)
                if key in self._locked_transitions:
                    new_prior[self._locked_transitions[key]] += b_i
                    continue

                # Using 1.0 to allow fast warmup from one single clean anchor update
                if self._NNTM_n[ni, a] >= 1.0:
                    for nj in range(self.n_nodes):
                        new_prior[nj] += b_i * self._NNTM_norm[ni, a, nj]
                    continue

                # Fallbacks: frequency-based prediction models
                tm_row = None

                # CTM: Context-Transition Model
                if self._prev_prev_fi >= 0 and fi_i == fi_prv:
                    ctx = self._prev_prev_fi * self.n_freqs + fi_i
                    if self._CTM_n[ctx, a] >= L4_CTM_WARMUP:
                        tm_row = self._CTM_norm[ctx, a]

                # TM: Global zone-level fallback
                if tm_row is None:
                    tm_row = self._TM_norm[fi_i, a]

                # Distribute frequency-prediction mass over all nodes sharing that frequency
                for fi_j in range(self.n_freqs):
                    p = float(tm_row[fi_j])
                    if p < 1e-8: continue
                    for nj in range(self.n_nodes):
                        if self.node_fi[self.nodes[nj]] == fi_j:
                            new_prior[nj] += b_i * p

            s = new_prior.sum()
            if s > 1e-9:
                self._belief = new_prior / s

        # ── 3. Observation update ─────────────────────────────
        self._belief = self._belief * self._sound_likelihood[:, curr_fi]
        s = self._belief.sum()
        if s < 1e-12:
            self._belief = np.ones(self.n_nodes, dtype=np.float64) / self.n_nodes
        else:
            self._belief /= s

        # Save PREVIOUS step's post-observation belief before overwriting.
        # origin_belief = "where I was last step" (sharp, post-observation).
        # self._belief  = "where I am now" (sharp, post-observation).
        # NNTM needs origin→destination; lock needs both to be confident.
        origin_belief = self._prev_belief_for_nntm.copy()

        # Store THIS step's post-observation belief for next step.
        self._prev_belief_for_nntm = self._belief.copy()

        # ── 4b. Update NNTM using posterior over destination ─────
        if world_moved and self._prev_fi >= 0 and action >= 0:
            a = action
            for ni in range(self.n_nodes):
                w_i = float(origin_belief[ni])
                if w_i < 1e-4: continue
                # Update transition to where we actually arrived
                for nj in range(self.n_nodes):
                    w_j = float(self._belief[nj])
                    if w_j < 1e-4: continue
                    self._NNTM[ni, a, nj] += w_i * w_j
                
                self._NNTM_n[ni, a] += w_i
                row = self._NNTM[ni, a]; s = row.sum()
                if s > 0: self._NNTM_norm[ni, a] = row / s

        # ── 4c. Lock high-confidence transitions ──────────────
        # NOTE: In a 2-alias-per-frequency world, observation caps top_prob
        # at 0.50 between aliased nodes. Locking from 0.50 causes a 
        # self-reinforcing wrong-confidence spiral. Only lock when genuine
        # disambiguation (wall model, multi-step context) pushes top_prob
        # above 0.65 — which requires the transition prior to break the tie.
        if world_moved and action >= 0 and self._prev_fi >= 0:
            top_idx = int(np.argmax(self._belief))
            top_prob = float(self._belief[top_idx])
            prev_top = int(np.argmax(origin_belief))
            prev_prob = float(origin_belief[prev_top])
            if top_prob > 0.65 and prev_prob > 0.65:
                key = (prev_top, action)
                self._locked_transitions[key] = top_idx

        # ── 4d. Decay toward uniform ──────────────────────────
        # Applied after NNTM update so top-down decay doesn't dilute
        # ground-truth observations before they get learned.
        self._belief = ((1.0 - L4_BELIEF_DECAY) * self._belief
                        + L4_BELIEF_DECAY / self.n_nodes)
        self._belief /= self._belief.sum()
        self._prev_prev_fi = self._prev_fi
        self._prev_fi      = curr_fi
        self._prev_action  = action
        self.t            += 1
        
        # Store transition in replay buffer for backward smoothing
        if world_moved and action >= 0:
            self._history.append({
                'prev_belief': self._prev_belief_for_nntm.copy(),
                'action': action,
                'curr_fi': curr_fi,
            })

        return self._make_output()

    # ── Output ────────────────────────────────────────────────

    def _make_output(self) -> dict:
        top_idx        = int(np.argmax(self._belief))
        top_prob       = float(self._belief[top_idx])
        self._top_node = self.nodes[top_idx]
        self._top_prob = top_prob

        b     = np.clip(self._belief, 1e-12, 1.0)
        raw_e = float(np.clip(
            -np.sum(b * np.log(b)) / np.log(self.n_nodes + 1e-9),
            0.0, 1.0))
        self._entropy_ema    = ((1.0 - L4_ENTROPY_SMOOTH) * self._entropy_ema
                                + L4_ENTROPY_SMOOTH * raw_e)
        self._belief_entropy = self._entropy_ema
        self._confident      = top_prob >= L4_CONFIDENCE_THRESH

        return {
            'top_node':       self._top_node,
            'top_prob':       top_prob,
            'belief_entropy': self._belief_entropy,
            'belief':         {n: float(self._belief[i])
                               for i, n in enumerate(self.nodes)},
            'confident':      self._confident,
            'belief_vector':  self._belief.copy(),
        }

    def get_top_k(self, k: int = 3) -> list:
        indices = np.argsort(self._belief)[::-1][:k]
        return [(self.nodes[idx], float(self._belief[idx])) for idx in indices]

    def anchor_at_node(self, node: str) -> None:
        """
        Called when the harness has ground-truth confirmation of location
        (e.g. food found at node X). Collapses belief to certainty,
        runs backward replay from the now-certain anchor, and resets
        context history so the next step starts from a clean state.
        """
        if node not in self._node_to_idx:
            return
        # Collapse belief to certainty
        self._belief[:] = 0.0
        self._belief[self._node_to_idx[node]] = 1.0
        # Snapshot the sharp anchor belief so the NEXT step's origin_belief
        # sees certainty — this is the key line that enables locked transitions.
        self._prev_belief_for_nntm = self._belief.copy()
        # Backward replay: re-weight NNTM from this certain anchor
        self.replay_from_anchor(node)
        # Clear history so stale entries don't contaminate future replays
        self._history.clear()
        # Reset context so next step doesn't carry stale fi context
        self._prev_prev_fi = self._prev_fi   # keep one step back, drop older

    # ── Accessors ─────────────────────────────────────────────

    def get_top_node(self) -> str:   return self._top_node
    def get_top_prob(self) -> float: return self._top_prob
    def is_confident(self) -> bool:  return self._confident

    def get_belief_vector(self) -> np.ndarray:
        return self._belief.copy()

    def reset_to_node(self, node: str) -> None:
        if node in self._node_to_idx:
            self._belief[:] = 0.0
            self._belief[self._node_to_idx[node]] = 1.0
            self._prev_belief_for_nntm = self._belief.copy()
            self._history.clear()

    def replay_from_anchor(self, anchor_node: str) -> None:
        if anchor_node not in self._node_to_idx: return
        
        # Certainty at anchor = 1.0. Walk backward, decaying certainty.
        certainty = 1.0
        curr_dist = np.zeros(self.n_nodes)
        curr_dist[self._node_to_idx[anchor_node]] = 1.0
        
        for entry in reversed(list(self._history)):
            prev_b = entry['prev_belief']
            a      = entry['action']
            
            # Update NNTM: for each possible origin, we know destination = curr_dist
            for ni in range(self.n_nodes):
                w_i = float(prev_b[ni])
                if w_i < 1e-4: continue
                for nj in range(self.n_nodes):
                    if curr_dist[nj] < 1e-4: continue
                    self._NNTM[ni, a, nj] += certainty * w_i * curr_dist[nj]
                self._NNTM_n[ni, a] += certainty * w_i
                
                row = self._NNTM[ni, a]; s = row.sum()
                if s > 0: self._NNTM_norm[ni, a] = row / s
            
            # Propagate backward: new curr_dist = prev_b weighted by certainty
            certainty *= 0.85  # decay — steps far from anchor are less reliable
            curr_dist = prev_b * certainty
            s = curr_dist.sum()
            if s > 1e-9: curr_dist /= s
            else: break
                
    def reset_uniform(self) -> None:
        self._belief[:] = 1.0 / self.n_nodes
        self._prev_fi      = -1
        self._prev_prev_fi = -1
        self._prev_action  = -1
        self._prev_belief_for_nntm = self._belief.copy()
        self._history.clear()

    # ── Diagnostics ───────────────────────────────────────────

    def summary(self, ground_truth_node: str = None) -> str:
        lines = [f"  L4 PositionBelief — step {self.t}"]
        lines.append(f"  Top: {self._top_node} ({self._top_prob:.3f})  "
                     f"entropy={self._belief_entropy:.3f}  "
                     f"confident={self._confident}")
        if ground_truth_node:
            true_idx  = self._node_to_idx.get(ground_truth_node, -1)
            true_prob = float(self._belief[true_idx]) if true_idx >= 0 else 0.0
            correct   = (self._top_node == ground_truth_node)
            lines.append(f"  True: {ground_truth_node}  "
                         f"P(true)={true_prob:.3f}  "
                         f"{'✓' if correct else '✗'}")
        sig = sorted(
            [(self.nodes[i], float(self._belief[i]))
             for i in range(self.n_nodes) if self._belief[i] > 0.05],
            key=lambda x: -x[1])
        lines.append("  Belief: " +
                     "  ".join(f"{n}={p:.3f}" for n, p in sig[:6]))
        return "\n".join(lines)

    def tm_coverage(self) -> dict:
        covered_tm  = int((self._TM_n  >= L4_TM_WARMUP).sum())
        covered_ctm = int((self._CTM_n >= L4_CTM_WARMUP).sum())
        return {
            'tm_covered':  covered_tm,  'tm_total':  self.n_freqs * self.n_actions,
            'ctm_covered': covered_ctm, 'ctm_total': self._N_CTX  * self.n_actions,
        }