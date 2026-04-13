"""
M68: INFERENCE ENGINE — LOGICAL REASONING VIA ZONE-CHAIN INFERENCE
====================================================================

Gives the brain the ability to chain learned facts into multi-hop logical
conclusions — analogous to transitive inference in hippocampal-PFC systems.

BIOLOGICAL BASIS
----------------
Animals solve transitive inference problems: if A>B and B>C, they can
infer A>C without ever having directly compared A and C. Biologically,
this requires hippocampal relational memory combined with PFC-mediated
inference. The hippocampus stores (A→B) and (B→C) as episodic facts;
PFC chains them into a conclusion during deliberation.

Here the "relational facts" are L3's action-conditioned zone transitions:
  T_norm[zone_A, action_East, zone_B] ≈ 0.9  → fact: "East from A leads to B"

Logical chaining:
  Known: A →(East)→ B     (high probability transition)
  Known: B →(East)→ C     (high probability transition)
  Known: C has food        (zone_value[C] is high)
  Infer: East from A is the logical action (reaches food in 2 hops)

This is NOT new learning — it's inference over L3's existing model.
The brain uses existing knowledge compositionally. This is the hallmark
of logical, rule-based reasoning vs. pattern matching.

ALGORITHM
---------
BFS (Breadth-First Search) over the zone graph defined by T_norm.
Each edge: (zone_i, action) → zone_j has weight P(zone_j | zone_i, action).
Path score = product_of_probabilities × discount^depth × zone_value[terminal].

Pruning:
  - Low-probability edges (< EDGE_PROB_THRESH) are pruned for efficiency.
  - Zones where the model is unverified (low TPE accuracy) are skipped.
  - Cycles are avoided (no zone appears twice in a path).
  - Paths too long (> MAX_CHAIN_DEPTH) are cut off.

Output: the first action in the highest-scoring path = the "logical" action.

WHAT SEPARATES THIS FROM M57
-----------------------------
M57 (Planner): greedy depth-limited lookahead using Q-values and zone values.
  Picks the best NEXT action. Does not explicitly chain rules.

M68 (Inference): explicit multi-hop chain reasoning over T_norm.
  Finds the best PATH (not just next step). Each hop is a distinct "fact".
  Returns chain_depth so downstream modules know how many steps ahead
  the reasoning projects. Can reason about paths M57 never simulated.

CALL ORDER
----------
Called after L3 step (step 5b) and L4 (step 5c) in Brain.step().
Requires l3._zones_stable before meaningful inference is possible.

Outputs → brain.step() return dict:
  m68_inferred_action      : int [0,3]   — logically best first action
  m68_inference_confidence : float [0,1] — probability of the inferred path
  m68_chain_depth          : int         — how many hops the chain covers
  m68_inferred_path_zones  : list[int]   — zone sequence of inferred path
  m68_reasoning_active     : bool        — True = confident inference found
"""

import numpy as np
from collections import deque

# ── Inference parameters ──────────────────────────────────────
# Maximum chain length. Beyond 5 hops, compounding uncertainty
# (each edge p ≤ 1) makes path probability negligibly small.
# At 5 hops with avg edge p=0.7: path p ≈ 0.7^5 ≈ 0.17 (still meaningful).
MAX_CHAIN_DEPTH = 5

# Minimum edge probability to follow. Edges with p < this are too
# uncertain to reason through. At 0.05 we include most plausible edges
# while pruning noise (T_norm is uniform 1/8 for untrained zones → 0.125,
# so we'd prune half the edges of an untrained zone — correct behavior).
EDGE_PROB_THRESH = 0.05

# Minimum path probability for inference to count as "active".
# At 0.15: a 2-hop chain with both edges at p=0.39 = 0.15, which is
# meaningful but not certain. Below this, inference is too uncertain
# to override other signals.
CONFIDENCE_THRESH = 0.15

# Minimum observed transitions from a zone before we trust its model.
# Below this, T_norm is near-uniform (uninformative).
MIN_TRANSITIONS = 10

# Minimum TPE accuracy for a zone to be trusted in chain reasoning.
# Zones that have many transitions but keep predicting wrong are
# unreliable — the brain should not reason through them.
MIN_TPE_ACCURACY = 0.40

# Bellman discount applied at each hop. Matches M57's GAMMA=0.85.
# Near-food paths get full value; distant paths get discounted.
CHAIN_GAMMA = 0.85


class InferenceEngine:
    """
    M68: Logical reasoning via multi-hop zone-chain inference.

    Queries L3's T_norm transition model to find the optimal path
    to high-value zones using BFS, chaining learned zone→action→zone
    facts into multi-step logical conclusions.
    """

    def __init__(self, n_zones: int = 8, n_actions: int = 4):
        self.n_zones   = n_zones
        self.n_actions = n_actions
        self.t         = 0
        self._last_action = 0

    def step(self, l3, curr_zone: int) -> dict:
        """
        Run multi-hop BFS inference from curr_zone.

        Parameters
        ----------
        l3        : ConceptLayer object — provides T_norm, _zone_value,
                    _tpe_accuracy, _T (raw counts for trust check)
        curr_zone : int — current zone index (freq_idx), -1 if unknown
        """
        self.t += 1

        null_out = {
            'm68_inferred_action':       self._last_action,
            'm68_inference_confidence':  0.0,
            'm68_chain_depth':           0,
            'm68_inferred_path_zones':   [],
            'm68_reasoning_active':      False,
        }

        # Can't reason if zone unknown or L3 not yet stable
        if curr_zone < 0 or not l3._zones_stable:
            return null_out

        # Need enough transitions from current zone to trust the model
        zone_n_trans = float(l3._T[curr_zone].sum())
        if zone_n_trans < MIN_TRANSITIONS:
            return null_out

        # ── BFS over zone graph ───────────────────────────────
        # Queue entry: (zone, path_prob, path_list, first_action, depth)
        # path_list: zones visited (to detect cycles)
        # first_action: the first action taken from curr_zone (what we return)
        queue = deque()

        # Seed: try all first actions from current zone
        for a0 in range(self.n_actions):
            row = l3._T_norm[curr_zone, a0]  # shape: (n_zones,)
            for z_next in range(self.n_zones):
                p = float(row[z_next])
                if p < EDGE_PROB_THRESH:
                    continue
                # Check starting zone's accuracy
                tpe = float(l3._tpe_accuracy[curr_zone])
                if tpe < MIN_TPE_ACCURACY and zone_n_trans > 40:
                    # Skip only if zone has enough data to judge accuracy
                    continue
                queue.append((z_next, p, [curr_zone, z_next], a0, 1))

        best_score   = -1.0
        best_action  = self._last_action
        best_path    = []
        best_depth   = 0
        best_conf    = 0.0
        visited_keys = set()

        while queue:
            zone, path_prob, path, first_action, depth = queue.popleft()

            # Deduplicate: (zone, first_action) — we want the best arrival
            # at each zone per first action, not every route
            key = (zone, first_action)
            if key in visited_keys:
                continue
            visited_keys.add(key)

            if depth > MAX_CHAIN_DEPTH:
                continue

            # Score this endpoint: discounted zone value × path probability
            z_val = float(l3._zone_value[zone]) if hasattr(l3, '_zone_value') else 0.0
            score = path_prob * (CHAIN_GAMMA ** depth) * (z_val + 0.01)

            if score > best_score:
                best_score  = score
                best_action = first_action
                best_path   = path[:]
                best_depth  = depth
                best_conf   = path_prob

            # Expand if not at depth limit
            if depth < MAX_CHAIN_DEPTH:
                z_n_trans = float(l3._T[zone].sum())
                if z_n_trans < MIN_TRANSITIONS:
                    continue
                for a in range(self.n_actions):
                    row = l3._T_norm[zone, a]
                    for z_next in range(self.n_zones):
                        if z_next in path:  # no cycles
                            continue
                        p = float(row[z_next])
                        if p < EDGE_PROB_THRESH:
                            continue
                        new_prob = path_prob * p
                        if new_prob < 0.001:
                            continue  # prune near-zero probability paths
                        queue.append(
                            (z_next, new_prob, path + [z_next], first_action, depth + 1)
                        )

        # Determine if inference produced a confident enough result
        reasoning_active = (best_conf >= CONFIDENCE_THRESH and best_score > 0.0)

        if reasoning_active:
            self._last_action = best_action

        return {
            'm68_inferred_action':       int(best_action),
            'm68_inference_confidence':  float(np.clip(best_conf, 0.0, 1.0)),
            'm68_chain_depth':           int(best_depth),
            'm68_inferred_path_zones':   list(best_path),
            'm68_reasoning_active':      bool(reasoning_active),
        }
