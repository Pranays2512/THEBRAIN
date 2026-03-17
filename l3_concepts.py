"""
L3: CONCEPT LAYER — DIRECT OWNERSHIP ZONES + HEBBIAN TRANSITIONS
=================================================================

DESIGN RATIONALE
----------------
All previous clustering approaches failed because the SOM places
8 frequency response regions as neighbours on a 64-neuron grid.
Their spatial contexts inevitably overlap — no unsupervised method
(k-means on activity fingerprints, weight-space clustering, etc.)
can cleanly separate them without knowing ground-truth labels.

THE CORRECT APPROACH: DIRECT OWNERSHIP
---------------------------------------
brain_longrun.py already tracks freq_bmu_counters: for each frequency
index fi, how many times has each BMU fired while that frequency was
playing. This is ground truth.

Zone assignment: zone[bmu] = argmax over frequencies of visit counts.
  bmu_visits[fi, bmu] = freq_bmu_counters[fi][bmu]
  zone[bmu] = argmax_fi(bmu_visits[:, bmu])

Zone index = frequency index (0-7), so the Z matrix stays coherent
across reassignments with zero label-remapping complexity.

ZONE TRANSITION MATRIX
-----------------------
Z updated ONLY on zone *changes* (inter-freq transitions).
Updating every step floods Z[i,i] with self-transitions because
most steps are intra-block dwell — making every zone predict itself.

  on zone change (prev != curr):
      Z[prev, curr] += Z_ETA
  every step:
      Z *= (1 - Z_DECAY)

PARAMETERS
----------
N_ZONES = 8                  (one per frequency; zone index = freq index)
ZONE_UPDATE_INTERVAL = 2000  (reassign from counters every N steps)
ZONE_UPDATE_WARMUP   = 5000  (wait for all 8 freqs to accumulate data)
Z_ETA   = 0.10               (zone transition learning rate)
Z_DECAY = 0.0002             (tau ~5000 steps; rare transitions survive)
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

N_NEURONS = 64
N_ZONES   = 8     # one per frequency; zone index = frequency index
GRID_W    = 8

ZONE_UPDATE_INTERVAL = 2000
ZONE_UPDATE_WARMUP   = 5000

Z_ETA   = 0.10
Z_DECAY = 0.0002   # tau ~5000 steps

MIN_VISITS_FOR_ZONE = 10   # BMU needs this many visits to get a zone

# ── Zone curiosity bonus ──────────────────────────────────────
# Unvisited / undervisited zones get an intrinsic reward bonus so
# M57 is steered toward unexplored territory even when one food
# source is already found.
#
# ZONE_CURIOSITY_WEIGHT: how much bonus an unvisited zone contributes
#   to zone_value via value iteration. Scaled by (1 - visit_fraction)
#   so zones visited 0 times get full bonus, fully visited zones get 0.
#   0.15 keeps it below food reward (1.0) but meaningful enough to
#   pull the brain toward the H★ path after finding E★.
#
# ZONE_VISIT_TAU: EMA decay for zone visit counts → visit_fraction.
#   Slow (0.001) so the curiosity bonus persists across many steps
#   and doesn't evaporate after a single visit. A zone must be visited
#   repeatedly over ~1000 steps before its curiosity bonus fully fades.
ZONE_CURIOSITY_WEIGHT = 0.15
ZONE_VISIT_TAU        = 0.001   # EMA decay for zone visit fraction


# ═══════════════════════════════════════════════════════════════
# CONCEPT LAYER
# ═══════════════════════════════════════════════════════════════

class ConceptLayer:
    """
    L3: Direct ownership zone assignment + inter-freq transition learning.
    Zone index = frequency index (0-7). No k-means, no label remapping.
    """

    def __init__(self, n_zones: int = N_ZONES):
        self.n_zones = n_zones

        # zone[bmu] = freq index that visited it most (-1 = unassigned)
        self._bmu_to_zone    = -np.ones(N_NEURONS, dtype=np.int32)
        # confidence = top_freq_visits / total_visits for each BMU
        self._bmu_confidence = np.zeros(N_NEURONS, dtype=np.float32)

        # Zone transition matrix — updated only on zone changes
        self._Z     = np.zeros((n_zones, n_zones), dtype=np.float32)
        self._Z_ctx = -1   # zone at previous step

        self.t              = 0
        self._zones_stable  = False
        self._n_assignments = 0

        # Last output cache
        self._last_zone_idx       = 0
        self._last_zone_conf      = 0.0
        self._last_zone_probs     = np.ones(n_zones) / n_zones
        self._last_top_zone_pred  = 0
        self._last_zone_pred_conf = 0.0

        # Zone reward EMA — updated by Brain when external reward arrives.
        self._zone_reward_ema     = np.zeros(n_zones, dtype=np.float32)
        self._zone_reward_alpha   = 0.05   # slow EMA — reward signal is sparse

        # Zone value — Bellman backup of reward through transition matrix Z.
        # _zone_value[i] = expected future reward reachable from zone i.
        # Recomputed whenever zone_reward_ema changes (after each food event).
        # M57 uses zone_value (not raw zone_reward_ema) for scoring, so that
        # zones on the PATH to food score highly, not just the food zone itself.
        self._zone_value          = np.zeros(n_zones, dtype=np.float32)
        self._value_gamma         = 0.85   # Bellman discount (matches M57 GAMMA)
        self._value_iters         = 20     # VI iterations — converges fast on 8 zones

        # Action-conditioned zone transition model.
        # T[zone_i, action, zone_j] = count of times taking action `a` from
        # zone `i` led to zone `j` (on real world moves, world_moved=True).
        # T_norm[i, a, :] = normalised row = P(next_zone | zone_i, action_a).
        # M57 uses T_norm @ zone_value to score each action: pick the action
        # that leads to the highest expected zone value in expectation.
        # This is the action-conditioned cognitive map the brain builds by
        # observing which directions lead to which zones during navigation.
        # Size: n_zones × N_ACTIONS × n_zones = 8×4×8 = 256 entries.
        self._N_ACTIONS   = 4
        self._T           = np.zeros((n_zones, 4, n_zones), dtype=np.float32)
        self._T_norm      = np.ones((n_zones, 4, n_zones), dtype=np.float32) / n_zones
        self._T_prev_zone = -1   # zone at the step BEFORE current (for update)
        self._T_prev_action = -1

        # Zone visit EMA — tracks how much each zone has been visited.
        # Rises toward 1.0 as a zone is visited repeatedly.
        # Used to compute curiosity bonus: unvisited zones get a pull
        # from M57's planning so the brain explores the full map.
        # Initialised to 0 (no visits yet — all zones maximally curious).
        self._zone_visit_ema = np.zeros(n_zones, dtype=np.float32)

    # ── Zone reward tracking ──────────────────────────────────

    def update_zone_reward(self, zone_idx: int, reward: float) -> None:
        """
        Update EMA of reward received in zone_idx, then recompute zone values.
        Called by Brain when external reward arrives.
        """
        if 0 <= zone_idx < self.n_zones:
            alpha = self._zone_reward_alpha
            self._zone_reward_ema[zone_idx] = (
                (1.0 - alpha) * self._zone_reward_ema[zone_idx]
                + alpha * float(np.clip(reward, 0.0, 1.0))
            )
            self._recompute_zone_values()

    def update_zone_visit(self, zone_idx: int) -> None:
        """
        Record a visit to zone_idx. Updates visit EMA and recomputes zone
        values (curiosity bonus changes as zones become more familiar).
        Call from Brain.step() every step, passing the current zone.
        """
        if 0 <= zone_idx < self.n_zones:
            # Decay all zones toward 0, then boost the visited zone toward 1.
            # This gives a running estimate of visit frequency per zone.
            self._zone_visit_ema *= (1.0 - ZONE_VISIT_TAU)
            self._zone_visit_ema[zone_idx] = min(
                1.0,
                self._zone_visit_ema[zone_idx] + ZONE_VISIT_TAU
            )
            # Recompute zone values so curiosity bonus reflects new visit counts.
            # Only recompute every 100 calls to avoid per-step overhead.
            if self.t % 100 == 0:
                self._recompute_zone_values()

    def update_action_transition(self, prev_zone: int, action: int,
                                  curr_zone: int) -> None:
        """
        Update the action-conditioned zone transition model.
        Call from Brain on every real world move (world_moved=True).

        prev_zone : zone before taking action
        action    : action taken (0-3)
        curr_zone : zone after taking action (ground-truth freq_idx)
        """
        if (0 <= prev_zone < self.n_zones and
                0 <= action < self._N_ACTIONS and
                0 <= curr_zone < self.n_zones):
            self._T[prev_zone, action, curr_zone] += 1.0
            # Renormalise this row
            row = self._T[prev_zone, action]
            s = row.sum()
            if s > 0:
                self._T_norm[prev_zone, action] = row / s

    def action_value(self, zone: int) -> np.ndarray:
        """
        For each action, compute expected zone_value of the predicted next zone.
        Returns ndarray of shape (N_ACTIONS,).
        Used by M57 to select the action pointing toward highest-value zone.
        """
        if not (0 <= zone < self.n_zones):
            return np.zeros(self._N_ACTIONS, dtype=np.float32)
        # P(next_zone | zone, action) @ zone_value
        # Shape: (N_ACTIONS, n_zones) @ (n_zones,) → (N_ACTIONS,)
        return self._T_norm[zone] @ self._zone_value   # (N_ACTIONS,)

    def _recompute_zone_values(self) -> None:
        """
        Value iteration on zone transition matrix, including curiosity bonus.

        R_effective[i] = zone_reward_ema[i]
                       + ZONE_CURIOSITY_WEIGHT * (1 - zone_visit_ema[i])

        The curiosity term gives unvisited zones intrinsic value.
        V[i] = R_effective[i] + gamma * sum_j( T[i,j] * V[j] )

        This propagates the curiosity pull backward through the zone graph:
        zones on the PATH to unvisited zones also gain value, steering M57
        toward unexplored territory even from several steps away.

        Example: H★ zone unvisited (visit_ema≈0) → R_eff[H]≈0.15
                 G zone visit_ema≈0 → R_eff[G]≈0.15
                 After VI: V[F]≈0.15*0.85, V[D]≈0.15*0.85^2
                 → M57 sees D→F→G→H★ path has value even before food found there.
        """
        Z = self._Z.copy()
        row_sums = Z.sum(axis=1, keepdims=True) + 1e-9
        T = Z / row_sums   # (n_zones, n_zones)

        # Effective reward = food reward + curiosity bonus for unvisited zones
        curiosity_bonus = ZONE_CURIOSITY_WEIGHT * (1.0 - self._zone_visit_ema)
        R_eff = np.clip(self._zone_reward_ema + curiosity_bonus, 0.0, 1.0)

        V = R_eff.copy()
        gamma = self._value_gamma
        for _ in range(self._value_iters):
            V_new = R_eff + gamma * (T @ V)
            np.clip(V_new, 0.0, 1.0 / (1.0 - gamma + 1e-9), out=V_new)
            V = V_new
        self._zone_value = V.astype(np.float32)

    # ── Zone assignment ───────────────────────────────────────

    def assign_zones_from_counters(self, freq_bmu_counters) -> None:
        """
        Assign zone[bmu] = argmax_freq(visit_count).
        Call periodically from brain_longrun after ZONE_UPDATE_WARMUP.

        freq_bmu_counters: list of Counter, one per frequency.
        """
        n_freqs = len(freq_bmu_counters)
        visits = np.zeros((n_freqs, N_NEURONS), dtype=np.float32)
        for fi, counter in enumerate(freq_bmu_counters):
            for bmu_idx, count in counter.items():
                if 0 <= bmu_idx < N_NEURONS:
                    visits[fi, bmu_idx] = float(count)

        total_per_bmu = visits.sum(axis=0)

        for bmu in range(N_NEURONS):
            tot = total_per_bmu[bmu]
            if tot >= MIN_VISITS_FOR_ZONE:
                top_fi = int(np.argmax(visits[:, bmu]))
                self._bmu_to_zone[bmu]    = top_fi
                self._bmu_confidence[bmu] = float(visits[top_fi, bmu] / tot)
            # else stays -1

        self._zones_stable   = True
        self._n_assignments += 1

    # ── Main step ─────────────────────────────────────────────

    def step(self, bmu_idx: int,
             l2_scores:      np.ndarray,
             familiarity:    float = 0.0,
             cortex_weights: np.ndarray = None,  # ignored, kept for API compat
             cortex:         object = None,
             freq_idx:       int = -1) -> dict:
        """
        Update L3 state for the current step.

        Parameters
        ----------
        bmu_idx    : current BMU from M54
        l2_scores  : (64,) L2 prediction score distribution
        familiarity: M55 familiarity signal (informational)
        freq_idx   : ground-truth frequency index (0-7) for this step.
                     When provided, used directly for Z-matrix updating
                     instead of bmu_to_zone lookup — eliminates zone-drift
                     contamination when BMU ownership reassigns.
        """
        # ── 1. Get zone for display / L2 blending ────────────
        if self._bmu_to_zone[bmu_idx] >= 0:
            zone_idx  = int(self._bmu_to_zone[bmu_idx])
            zone_conf = float(self._bmu_confidence[bmu_idx])
        else:
            zone_idx  = int((bmu_idx // GRID_W) * self.n_zones // 8)
            zone_conf = 0.0

        # ── 2. Zone transition counting using ground-truth freq_idx ──
        # Use freq_idx (ground truth from stream) when available.
        # Falls back to bmu_to_zone if freq_idx not passed.
        z_for_transition = freq_idx if freq_idx >= 0 else zone_idx

        if self._Z_ctx >= 0 and z_for_transition != self._Z_ctx:
            self._Z[self._Z_ctx, z_for_transition] += 1.0

        self._Z_ctx = z_for_transition








        # ── 3. Zone-level prediction: pure Z-matrix (no L2 blend) ──
        # L2 blend was adding noise: L2 scores are per-BMU but Z is per-zone.
        # The raw count matrix is ground truth; blend only introduces artifacts
        # when zone_idx is ambiguous (unassigned BMUs) or during warmup.
        z_row = self._Z[z_for_transition].copy() if z_for_transition >= 0 \
                else np.zeros(self.n_zones, dtype=np.float32)

        z_s = z_row.sum()
        if z_s > 1e-9:
            zone_probs = z_row / z_s
        else:
            # No transitions seen yet — blend with L2 as fallback
            l2_zone = np.zeros(self.n_zones, dtype=np.float32)
            if self._zones_stable and l2_scores is not None:
                for bmu in range(N_NEURONS):
                    z = self._bmu_to_zone[bmu]
                    if z >= 0:
                        l2_zone[z] += float(l2_scores[bmu])
                l2_s = l2_zone.sum()
                if l2_s > 1e-9:
                    zone_probs = l2_zone / l2_s
                else:
                    zone_probs = np.ones(self.n_zones, dtype=np.float32) / self.n_zones
            else:
                zone_probs = np.ones(self.n_zones, dtype=np.float32) / self.n_zones

        top_zone_pred  = int(np.argmax(zone_probs))
        zone_pred_conf = float(zone_probs[top_zone_pred])

        self._last_zone_idx       = zone_idx
        self._last_zone_conf      = zone_conf
        self._last_zone_probs     = zone_probs
        self._last_top_zone_pred  = top_zone_pred
        self._last_zone_pred_conf = zone_pred_conf

        self.t += 1

        return {
            'zone_idx':        zone_idx,
            'zone_confidence': zone_conf,
            'zone_probs':      zone_probs,
            'top_zone_pred':   top_zone_pred,
            'zone_pred_conf':  zone_pred_conf,
            'zones_stable':    self._zones_stable,
            'n_clusterings':   self._n_assignments,
        }

    # ── Accessors ─────────────────────────────────────────────

    def get_bmu_zone(self, bmu_idx: int) -> int:
        return int(self._bmu_to_zone[bmu_idx])

    def get_zone_probs(self, zone_idx: int) -> np.ndarray:
        row = self._Z[zone_idx].copy()
        s   = row.sum()
        if s > 1e-9: row /= s
        return row

    # ── Diagnostics ───────────────────────────────────────────

    def zone_summary(self, freq_per_bmu=None) -> dict:
        if not self._zones_stable:
            return {'status': 'not yet assigned'}
        summary = {}
        for z in range(self.n_zones):
            mask = self._bmu_to_zone == z
            n    = int(mask.sum())
            if n == 0:
                continue
            bmus = np.where(mask)[0].tolist()
            rows = [b // GRID_W for b in bmus]
            cols = [b % GRID_W  for b in bmus]
            z_row = self._Z[z].copy()
            if z_row.sum() > 1e-9: z_row /= z_row.sum()
            top_next = int(np.argmax(z_row))
            top_conf = float(z_row[top_next])
            summary[z] = {
                'n_bmus': n, 'bmus': bmus,
                'mean_row': float(np.mean(rows)),
                'mean_col': float(np.mean(cols)),
                'top_next': top_next, 'top_conf': top_conf,
                'z_probs': z_row.tolist(),
                'mean_conf': float(self._bmu_confidence[mask].mean()),
            }
        return summary

    def transition_matrix_str(self) -> str:
        Z = self._Z.copy()
        row_sums = Z.sum(axis=1, keepdims=True) + 1e-9
        Z_norm   = Z / row_sums
        lines = ["  Zone transition matrix (row=from, col=to):"]
        header = "      " + " ".join(f"  Z{j}" for j in range(self.n_zones))
        lines.append(header)
        for i in range(self.n_zones):
            row = " ".join(f"{Z_norm[i,j]:4.2f}" for j in range(self.n_zones))
            lines.append(f"  Z{i} |  {row}")
        return "\n".join(lines)

    def summary(self):
        print(f"  ConceptLayer (L3) — step {self.t}")
        print(f"  Zones stable: {self._zones_stable}  (assignments={self._n_assignments})")
        print(f"  Z matrix: sum={self._Z.sum():.3f}  max={self._Z.max():.4f}")
        if self._zones_stable:
            zc = np.bincount(self._bmu_to_zone[self._bmu_to_zone >= 0],
                             minlength=self.n_zones)
            mc = self._bmu_confidence[self._bmu_to_zone >= 0].mean()
            print(f"  BMUs per zone: {zc.tolist()}  mean_conf={mc:.3f}")
        print(f"  Last: zone={self._last_zone_idx}  conf={self._last_zone_conf:.3f}  "
              f"pred→Z{self._last_top_zone_pred}  pred_conf={self._last_zone_pred_conf:.3f}")