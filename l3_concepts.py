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

# ── Transition Prediction Error (TPE) ────────────────────────
# TPE measures how accurately L3's action-conditioned model (_T_norm)
# predicts the next zone given (current_zone, action).
#
# Every time the brain makes a real move (world_moved=True), L3:
#   1. Looks up its prediction: predicted_zone = argmax T_norm[zone, action]
#   2. Observes the actual outcome: actual_zone (from bucketed_fi)
#   3. Computes error: 0.0 if correct, 1.0 if wrong
#   4. Updates a per-zone model accuracy EMA: _tpe_accuracy[zone]
#
# M57 reads _tpe_accuracy[current_zone] before planning:
#   - High accuracy (close to 1.0): model is reliable → M57 may plan
#   - Low accuracy (close to 0.0): model is unreliable → M57 defers to M56
#
# This prevents M57 from planning using stale or noisy transition data.
# The model must earn trust through verified predictions, not just accumulate
# counts. A zone that has many transitions but keeps getting them wrong
# (e.g. noisy bucketed_fi) will have low accuracy → M57 stays silent.
#
# TPE_ACCURACY_ALPHA: EMA decay for model accuracy per zone.
#   Slow (0.05) — accuracy should reflect sustained model quality,
#   not just the last few steps. At 0.05, tau ≈ 20 transitions per zone.
#
# TPE_MIN_TRANSITIONS: minimum real transitions observed from a zone
#   before TPE accuracy is meaningful. Below this, accuracy defaults to 0
#   (untrusted) so M57 doesn't plan from newly-entered zones.
#
# TPE_ACCURACY_THRESH: minimum accuracy M57 requires before planning.
#   At 0.50: model must be right more than half the time. This is a soft
#   threshold — M57's planning weight is multiplied by accuracy, so a
#   0.51-accurate zone gets very little M57 influence.
TPE_ACCURACY_ALPHA  = 0.05
TPE_MIN_TRANSITIONS = 10
TPE_ACCURACY_THRESH = 0.50


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

        # ── Context-aware transition table (TC) ─────────────
        # TC[(prev_fi * N_ZONES + curr_fi), action, next_fi]
        # When the brain knows both the previous frequency and current
        # frequency, it can distinguish aliased nodes:
        #   A (fi=0, reached from E fi=4) → context_zone = 4*8+0 = 32
        #   I (fi=0, reached from H fi=7) → context_zone = 7*8+0 = 56
        # TC has N_ZONES^2 = 64 context-zones × 4 actions × N_ZONES.
        # M57 queries TC first; falls back to T when TC has < TC_MIN_TRANS.
        # TC_MIN_TRANS is higher than T_MIN_TRANSITIONS because context-zones
        # are visited less often (need prev+curr match, not just curr).
        N_CTX = self.n_zones * self.n_zones   # 64 context zones
        self._N_CTX       = N_CTX
        self._TC          = np.zeros((N_CTX, 4, self.n_zones), dtype=np.float32)
        self._TC_norm     = np.ones((N_CTX, 4, self.n_zones), dtype=np.float32) / self.n_zones
        self._TC_accuracy = np.zeros(N_CTX, dtype=np.float32)
        self._TC_n_trans  = np.zeros(N_CTX, dtype=np.int32)
        TC_MIN_TRANS      = 20   # lower bar — context zones are rarer
        self._TC_MIN_TRANS = TC_MIN_TRANS

        # ── Transition Prediction Error (TPE) state ───────────
        # Per-zone model accuracy EMA — how often T_norm correctly
        # predicted the next zone for each starting zone.
        # Initialised to 0 (untrusted) — must earn accuracy through
        # verified predictions before M57 is allowed to plan from it.
        self._tpe_accuracy      = np.zeros(n_zones, dtype=np.float32)

        # Per-zone transition count — how many real moves have been
        # observed from each zone. TPE accuracy is only meaningful
        # once this exceeds TPE_MIN_TRANSITIONS.
        self._tpe_n_transitions = np.zeros(n_zones, dtype=np.int32)

        # Last TPE outcome per zone (for diagnostics)
        self._tpe_last_correct  = np.zeros(n_zones, dtype=np.float32)
        self._tpe_last_predicted_zone = -np.ones(n_zones, dtype=np.int32)

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
                                  curr_zone: int,
                                  prev_prev_zone: int = -1) -> None:
        """
        Update the action-conditioned zone transition model.
        Call from Brain on every real world move (world_moved=True).

        prev_zone      : zone before taking action (freq_idx)
        action         : action taken (0-3)
        curr_zone      : zone after taking action (ground-truth freq_idx)
        prev_prev_zone : zone TWO steps back (freq_idx), used for TC context.
                         When supplied, also updates the context-aware TC table
                         indexed by (prev_prev_zone * n_zones + prev_zone).
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

        # Context-aware TC update — needs two history steps
        if (0 <= prev_prev_zone < self.n_zones and
                0 <= prev_zone < self.n_zones and
                0 <= action < self._N_ACTIONS and
                0 <= curr_zone < self.n_zones):
            ctx = int(prev_prev_zone * self.n_zones + prev_zone)
            self._TC[ctx, action, curr_zone] += 1.0
            row = self._TC[ctx, action]
            s = row.sum()
            if s > 0:
                self._TC_norm[ctx, action] = row / s

    def record_transition_outcome(self, prev_zone: int, action: int,
                                    actual_zone: int,
                                    prev_prev_zone: int = -1) -> float:
        """
        Record whether L3's transition prediction was correct.

        Call AFTER update_action_transition, every real world move.

        Compares T_norm[prev_zone, action]'s top prediction to actual_zone.
        Updates _tpe_accuracy[prev_zone] EMA with 1.0 (correct) or 0.0 (wrong).
        When prev_prev_zone supplied, also updates TC accuracy.

        Returns the prediction error (0.0=correct, 1.0=wrong) for diagnostics.
        """
        if not (0 <= prev_zone < self.n_zones and
                0 <= action < self._N_ACTIONS and
                0 <= actual_zone < self.n_zones):
            return 1.0

        # What did the model predict for this (zone, action)?
        predicted_zone = int(np.argmax(self._T_norm[prev_zone, action]))
        self._tpe_last_predicted_zone[prev_zone] = predicted_zone

        # Was it correct?
        correct = float(predicted_zone == actual_zone)
        self._tpe_last_correct[prev_zone] = correct

        # Update transition count
        self._tpe_n_transitions[prev_zone] += 1

        # Update accuracy EMA — only after minimum transitions
        if self._tpe_n_transitions[prev_zone] >= TPE_MIN_TRANSITIONS:
            self._tpe_accuracy[prev_zone] = (
                (1.0 - TPE_ACCURACY_ALPHA) * self._tpe_accuracy[prev_zone]
                + TPE_ACCURACY_ALPHA * correct
            )

        # TC accuracy update
        if 0 <= prev_prev_zone < self.n_zones:
            ctx = int(prev_prev_zone * self.n_zones + prev_zone)
            tc_pred = int(np.argmax(self._TC_norm[ctx, action]))
            tc_correct = float(tc_pred == actual_zone)
            self._TC_n_trans[ctx] += 1
            if self._TC_n_trans[ctx] >= self._TC_MIN_TRANS:
                self._TC_accuracy[ctx] = (
                    (1.0 - TPE_ACCURACY_ALPHA) * self._TC_accuracy[ctx]
                    + TPE_ACCURACY_ALPHA * tc_correct
                )

        return 1.0 - correct   # return error (0=correct, 1=wrong)

    def get_tpe_accuracy(self, zone: int) -> float:
        """
        Return model accuracy for zone, or 0.0 if insufficient data.
        Used by M57 to gate planning confidence.
        """
        if not (0 <= zone < self.n_zones):
            return 0.0
        if self._tpe_n_transitions[zone] < TPE_MIN_TRANSITIONS:
            return 0.0   # untrusted — not enough data
        return float(self._tpe_accuracy[zone])

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

    def action_value_ctx(self, prev_zone: int, curr_zone: int) -> np.ndarray:
        """
        Context-aware action value: uses TC table indexed by (prev_zone, curr_zone).
        Falls back to action_value(curr_zone) if TC has insufficient data.
        Returns ndarray of shape (N_ACTIONS,).
        """
        if not (0 <= prev_zone < self.n_zones and 0 <= curr_zone < self.n_zones):
            return self.action_value(curr_zone)
        ctx = int(prev_zone * self.n_zones + curr_zone)
        if self._TC_n_trans[ctx] < self._TC_MIN_TRANS:
            return self.action_value(curr_zone)   # fallback
        return self._TC_norm[ctx] @ self._zone_value   # (N_ACTIONS,)

    def get_tpe_accuracy_ctx(self, prev_zone: int, curr_zone: int) -> float:
        """
        Return TC model accuracy for context (prev_zone, curr_zone).
        Returns 0.0 if insufficient data — M57 will fall back to T.
        """
        if not (0 <= prev_zone < self.n_zones and 0 <= curr_zone < self.n_zones):
            return 0.0
        ctx = int(prev_zone * self.n_zones + curr_zone)
        if self._TC_n_trans[ctx] < self._TC_MIN_TRANS:
            return 0.0
        return float(self._TC_accuracy[ctx])

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