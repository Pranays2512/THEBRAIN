"""
M70: DREAM ENGINE — OFFLINE COUNTERFACTUAL SIMULATION
======================================================

Gives the brain the ability to dream: internally-generated sequences of
zone activations that run offline (no sensory input), explore counterfactual
alternatives to real experiences, and weakly strengthen L2 and Q-learning
through imagined transitions.

BIOLOGICAL BASIS
----------------
REM sleep is characterized by:

  Hippocampal replay (forward AND backward) — sequences from recent
    experience are replayed, but with creative recombination. The
    hippocampus doesn't just replay A→B→C; it generates A→C→B or
    novel paths through known territory.

  Counterfactual simulation — "what if I had gone left instead of right?"
    Rodent studies show place cells replay alternative paths during sleep,
    not just the routes actually taken. This is how animals learn to
    avoid dead-ends they haven't encountered yet.

  Neuromodulator profile shift — during REM:
    ACh HIGH (basal forebrain fires strongly → high plasticity)
    NE  LOW  (locus coeruleus silent → no physical arousal response)
    5-HT LOW (dorsal raphe silent → reduced emotional gating)
  This creates a distinctive learning state: high plasticity but
  decoupled from real-world consequences.

  Emotional tagging — amygdala remains active during REM. Dreams with
    high-value or high-threat content are emotionally tagged and
    prioritized for consolidation.

WHAT THIS IS NOT
----------------
- `idle_step()`: replays actual food trajectories (real experience)
- `sleep_consolidate()`: strengthens M63 episodes → pi_bmu (real episodes)
- M69 (Imagination): waking creative simulation (sensory-anchored zone)

M70 is different: called explicitly between waking steps, it:
  1. Generates purely internal sequences starting from L2's predictor
     (no real bmu_idx input — the brain dreams its own continuation)
  2. Creates counterfactual variants of real food trajectories
     (alternate paths — "what if I'd gone East?")
  3. Runs all sequences with REM neuromodulator profile (high ACh, low NE/5-HT)
  4. Weakly updates L2 from internally-generated transitions
  5. Returns a "dream report" — emotional tone, dominant zones, counterfactual quality

DREAM TYPES
-----------
Three types of dreams, chosen probabilistically:

  REPLAY DREAM (40%) — replay a stored food trajectory with small random
    perturbations. Like reliving a memory but slightly differently each time.
    → Strengthens known good paths. Dream content: reward-positive.

  COUNTERFACTUAL DREAM (40%) — take a stored food trajectory, swap one
    action at a random step, and simulate the counterfactual path using L3's
    T_norm. Did the alternate path find food? Learn from the answer:
    if yes → slightly reinforce the counterfactual action in Q
    if no  → nothing (the brain already did better)
    → Discovers alternative routes. Dream content: mixed.

  NOVEL DREAM (20%) — pure imagination: generate a new path from scratch
    using M69-style stochastic sampling with ACh-boosted exploration.
    No anchor in real experience. Pure recombination.
    → Explores regions not yet well-mapped. Dream content: neutral/curious.

CALL INTERFACE
--------------
Called explicitly from the harness between waking steps:

  dream_report = brain.dream(n_sequences=5)

or once per sleep cycle:

  brain.sleep_consolidate()        # slow-wave sleep
  dream_report = brain.dream()     # REM follows SWS

Returns a dict with dream statistics. Does NOT modify brain.t.
"""

import numpy as np
from collections import namedtuple

# ── Dream parameters ──────────────────────────────────────────

# Number of dream sequences per dream() call (default).
N_DREAM_SEQUENCES     = 8

# Max steps in a single dream sequence.
DREAM_SEQUENCE_LENGTH = 10

# Weak learning rate during dreaming — much smaller than online ALPHA_ACTOR.
# Biology: synaptic changes during sleep are gentler than waking plasticity.
# At 0.004: takes ~25 dream sequences to equal one waking online update.
DREAM_L2_ALPHA   = 0.004
DREAM_Q_ALPHA    = 0.003

# Probabilities for each dream type.
P_REPLAY_DREAM        = 0.40
P_COUNTERFACTUAL_DREAM= 0.40
P_NOVEL_DREAM         = 0.20   # remainder

# REM neuromodulator overrides during dreaming.
# High ACh → high plasticity (more learning from dreams).
# Low NE  → no temperature boost (calm simulation, not panicked).
# Low 5-HT→ patient dreaming (high gamma, forward-looking).
DREAM_ACH_LEVEL  = 0.80
DREAM_NE_LEVEL   = 0.05
DREAM_SHT_LEVEL  = 0.30

# Minimum zone value for a dream sequence to be considered "positive"
# (emotionally rewarding dream).
DREAM_POSITIVE_VALUE_THRESH = 0.20

# Minimum trajectory length to use for counterfactual dreaming.
# Short trajectories don't have enough context for meaningful swaps.
MIN_TRAJ_LEN_FOR_CF = 4

# Discount during dream simulation — same as M57/M68/M69.
DREAM_GAMMA = 0.85


DreamSequence = namedtuple(
    'DreamSequence',
    ['dream_type', 'zone_sequence', 'bmu_sequence',
     'cumulative_value', 'emotional_tone', 'counterfactual_action']
)


class DreamEngine:
    """
    M70: Offline dream simulation.

    Generates three types of dreams (replay, counterfactual, novel),
    weakly updates L2 prediction and Q-values from internally-generated
    sequences, and returns a dream report.

    Designed to be called from brain.dream() — NOT from brain.step().
    """

    def __init__(self, n_zones: int = 8, n_actions: int = 4, seed: int = 42):
        self.n_zones   = n_zones
        self.n_actions = n_actions
        self._rng      = np.random.default_rng(seed)
        self.t_dreams  = 0   # total dream() calls

        # Dream memory: ring buffer of last 20 dream reports
        # (so the brain has a history of its own dreams)
        self._dream_history    = []
        self._max_dream_history = 20

    # ── Internal generators ───────────────────────────────────

    def _sample_zone(self, l3, from_zone: int, action: int) -> int:
        """Stochastically sample next zone from T_norm."""
        if not hasattr(l3, '_T_norm'):
            return int(self._rng.integers(0, self.n_zones))
        probs = l3._T_norm[from_zone, action].copy().astype(np.float64)
        probs = np.clip(probs, 0.0, None)
        s = probs.sum()
        if s < 1e-9:
            return int(self._rng.integers(0, self.n_zones))
        probs /= s
        return int(self._rng.choice(self.n_zones, p=probs))

    def _zone_value(self, l3, zone: int) -> float:
        """Get zone value from L3, with fallback."""
        if hasattr(l3, '_zone_value') and 0 <= zone < len(l3._zone_value):
            return float(l3._zone_value[zone])
        return 0.0

    def _emotional_tone(self, cumulative_value: float) -> str:
        """Classify dream emotional tone from cumulative value."""
        if cumulative_value > DREAM_POSITIVE_VALUE_THRESH:
            return 'rewarding'
        elif cumulative_value < -0.05:
            return 'aversive'
        else:
            return 'neutral'

    # ── Dream type generators ─────────────────────────────────

    def _replay_dream(self, food_trajectories, l3) -> DreamSequence:
        """
        Replay an existing food trajectory with small perturbations.

        Replays the zone sequence from a stored food trajectory
        (from brain._food_trajectories), adding occasional random
        deviations. This strengthens the known good path while
        exploring slight variations.
        """
        if not food_trajectories:
            return self._novel_dream(l3, seed_zone=-1)

        ft = food_trajectories[int(self._rng.integers(0, len(food_trajectories)))]
        trajectory = list(ft['trajectory'])
        if not trajectory:
            return self._novel_dream(l3, seed_zone=-1)

        zone_seq = []
        bmu_seq  = []
        cum_val  = 0.0

        for i, entry in enumerate(trajectory[:DREAM_SEQUENCE_LENGTH]):
            prev_bmu = entry[0]
            curr_bmu = entry[1]
            fi       = entry[3] if len(entry) > 3 else -1

            bmu_seq.append(curr_bmu)
            if 0 <= fi < self.n_zones:
                zone_seq.append(fi)
                # Small perturbation: occasionally jump to a neighbor zone
                if self._rng.random() < 0.15 and i > 0:
                    alt_action = int(self._rng.integers(0, self.n_actions))
                    curr_zone  = zone_seq[-1]
                    if 0 <= curr_zone < self.n_zones:
                        zone_seq[-1] = self._sample_zone(l3, curr_zone, alt_action)
                cum_val += (DREAM_GAMMA ** i) * self._zone_value(l3, zone_seq[-1])

        # Add the food reward at the end
        cum_val += ft['reward'] * (DREAM_GAMMA ** len(zone_seq))

        return DreamSequence(
            dream_type            = 'replay',
            zone_sequence         = zone_seq,
            bmu_sequence          = bmu_seq,
            cumulative_value      = cum_val,
            emotional_tone        = self._emotional_tone(cum_val),
            counterfactual_action = -1,
        )

    def _counterfactual_dream(self, food_trajectories, l3) -> DreamSequence:
        """
        Generate a counterfactual: replay a real trajectory but swap one action.

        Takes a stored food trajectory, randomly selects a step, and replaces
        the action taken with a random alternative. Then simulates where the
        alternative path leads using L3's T_norm.

        This is "what if I had done X instead?" — the core cognitive operation
        hypothesized to occur during REM sleep.

        If the counterfactual path reaches higher-value zones, a tiny Q update
        is queued (the caller applies it). If worse, nothing — the real path
        was already optimal.
        """
        if not food_trajectories:
            return self._novel_dream(l3, seed_zone=-1)

        ft = food_trajectories[int(self._rng.integers(0, len(food_trajectories)))]
        trajectory = list(ft['trajectory'])
        if len(trajectory) < MIN_TRAJ_LEN_FOR_CF:
            return self._novel_dream(l3, seed_zone=-1)

        # Choose a random swap point (not the very last step — that's the food)
        swap_idx = int(self._rng.integers(0, max(1, len(trajectory) - 1)))
        swap_entry  = trajectory[swap_idx]
        real_action = int(swap_entry[2]) if len(swap_entry) > 2 else 0
        real_fi     = int(swap_entry[3]) if len(swap_entry) > 3 else -1

        # Pick a different action (not the one actually taken)
        alt_actions = [a for a in range(self.n_actions) if a != real_action]
        if not alt_actions:
            alt_actions = list(range(self.n_actions))
        cf_action = int(self._rng.choice(alt_actions))

        # Simulate from the swap point using the counterfactual action
        zone_seq = []
        cum_val  = 0.0
        curr_zone = real_fi if 0 <= real_fi < self.n_zones else 0

        for d in range(DREAM_SEQUENCE_LENGTH):
            action = cf_action if d == 0 else int(self._rng.integers(0, self.n_actions))
            if not hasattr(l3, '_zones_stable') or not l3._zones_stable:
                break
            next_zone = self._sample_zone(l3, curr_zone, action)
            zone_seq.append(next_zone)
            cum_val += (DREAM_GAMMA ** d) * self._zone_value(l3, next_zone)
            curr_zone = next_zone

        return DreamSequence(
            dream_type            = 'counterfactual',
            zone_sequence         = zone_seq,
            bmu_sequence          = [],   # no real BMUs — internally generated
            cumulative_value      = cum_val,
            emotional_tone        = self._emotional_tone(cum_val),
            counterfactual_action = int(cf_action),
        )

    def _novel_dream(self, l3, seed_zone: int = -1) -> DreamSequence:
        """
        Generate a novel dream: random walk through zones, no real-experience anchor.

        Pure recombination — the brain wanders through its internal zone map
        in a way it may never have navigated while awake.

        With ACh elevated (REM state), exploration is wider and learning from
        novel transitions is stronger. This can create new zone→zone associations
        the brain hasn't built from waking experience.
        """
        if seed_zone < 0 or seed_zone >= self.n_zones:
            seed_zone = int(self._rng.integers(0, self.n_zones))

        zone_seq = [seed_zone]
        cum_val  = 0.0

        for d in range(DREAM_SEQUENCE_LENGTH):
            curr = zone_seq[-1]
            action = int(self._rng.integers(0, self.n_actions))
            if hasattr(l3, '_zones_stable') and l3._zones_stable:
                next_zone = self._sample_zone(l3, curr, action)
            else:
                next_zone = int(self._rng.integers(0, self.n_zones))
            zone_seq.append(next_zone)
            cum_val += (DREAM_GAMMA ** d) * self._zone_value(l3, next_zone)

        return DreamSequence(
            dream_type            = 'novel',
            zone_sequence         = zone_seq,
            bmu_sequence          = [],
            cumulative_value      = cum_val,
            emotional_tone        = self._emotional_tone(cum_val),
            counterfactual_action = -1,
        )

    # ── L2 dream update ───────────────────────────────────────

    def _update_l2_from_dream(self, pred, zone_sequence: list, l3):
        """
        Weakly update L2's zone transition predictions from dream zone sequence.

        For each adjacent pair in the dream zone sequence, if L3 has BMUs
        associated with those zones, apply a tiny positive update to L2's
        prediction matrix for the representative BMU transition.

        This is how dreaming generalizes: the brain strengthens zone-level
        associations it explored in the dream, even if those exact BMU
        transitions weren't experienced awake.

        Only updates if L3 is stable (zones have real assignments).
        Uses DREAM_L2_ALPHA << ALPHA (gentle, doesn't overwrite waking learning).
        """
        if not hasattr(l3, '_zones_stable') or not l3._zones_stable:
            return 0
        if not hasattr(l3, '_bmu_to_zone'):
            return 0
        if not hasattr(pred, '_P') or pred._P is None:
            return 0

        n_updates = 0
        bmu_to_zone = l3._bmu_to_zone  # shape: (N_NEURONS,)

        for i in range(len(zone_sequence) - 1):
            from_zone = zone_sequence[i]
            to_zone   = zone_sequence[i + 1]
            if from_zone == to_zone:
                continue

            # Find representative BMUs for each zone
            from_bmus = [b for b in range(len(bmu_to_zone))
                         if int(bmu_to_zone[b]) == from_zone]
            to_bmus   = [b for b in range(len(bmu_to_zone))
                         if int(bmu_to_zone[b]) == to_zone]

            if not from_bmus or not to_bmus:
                continue

            # Use the most representative BMU (first in list — deterministic)
            from_bmu = from_bmus[0]
            to_bmu   = to_bmus[0]

            # Tiny update to L2's prediction matrix P[from_bmu, to_bmu]
            n = pred._P.shape[0]
            if from_bmu < n and to_bmu < n:
                pred._P[from_bmu, to_bmu] = float(np.clip(
                    pred._P[from_bmu, to_bmu] + DREAM_L2_ALPHA,
                    0.0, 1.0
                ))
                n_updates += 1

        return n_updates

    # ── Main dream() entry point ──────────────────────────────

    def dream(self,
              l3,
              pred,
              food_trajectories,
              n_sequences: int = N_DREAM_SEQUENCES,
              ) -> dict:
        """
        Run one dream cycle: generate diverse sequences, apply weak updates.

        Parameters
        ----------
        l3                : ConceptLayer — zone model for sampling
        pred              : SequencePredictor — L2, for weak updates
        food_trajectories : list-like — brain._food_trajectories
        n_sequences       : int — number of dream sequences to generate

        Returns
        -------
        dict — dream report with:
          dream_sequences    : list of DreamSequence namedtuples
          dominant_emotion   : most common emotional tone across sequences
          mean_dream_value   : average cumulative value across dreams
          n_cf_dreams        : number of counterfactual dreams generated
          n_replay_dreams    : number of replay dreams
          n_novel_dreams     : number of novel/creative dreams
          n_l2_updates       : weak L2 updates applied
          best_cf_action     : most common counterfactual action across dreams
          dream_valence      : overall dream valence [-1, +1]
          t_dream            : dream call count
        """
        self.t_dreams += 1
        traj_list = list(food_trajectories) if food_trajectories else []

        sequences       = []
        total_l2_updates = 0
        type_counts     = {'replay': 0, 'counterfactual': 0, 'novel': 0}
        cf_actions      = []

        for _ in range(n_sequences):
            r = self._rng.random()
            if r < P_REPLAY_DREAM and traj_list:
                seq = self._replay_dream(traj_list, l3)
            elif r < P_REPLAY_DREAM + P_COUNTERFACTUAL_DREAM and traj_list:
                seq = self._counterfactual_dream(traj_list, l3)
            else:
                seed_z = int(self._rng.integers(0, self.n_zones))
                seq    = self._novel_dream(l3, seed_zone=seed_z)

            sequences.append(seq)
            type_counts[seq.dream_type] += 1
            if seq.counterfactual_action >= 0:
                cf_actions.append(seq.counterfactual_action)

            # Weak L2 update from this dream's zone sequence
            n_up = self._update_l2_from_dream(pred, seq.zone_sequence, l3)
            total_l2_updates += n_up

        # ── Aggregate dream report ────────────────────────────
        emotions = [s.emotional_tone for s in sequences]
        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'

        values = [s.cumulative_value for s in sequences]
        mean_value = float(np.mean(values)) if values else 0.0

        # dream_valence: normalize mean_value to [-1, +1]
        # food reward ≈ 1.0 per event, zone_value ≈ 0-0.5 typically
        # so clip at [-1, +1] after scaling by 0.5
        dream_valence = float(np.clip(mean_value * 0.5, -1.0, 1.0))

        best_cf_action = -1
        if cf_actions:
            # Mode of counterfactual actions
            action_freq = {}
            for a in cf_actions:
                action_freq[a] = action_freq.get(a, 0) + 1
            best_cf_action = max(action_freq, key=action_freq.get)

        report = {
            'dream_sequences':   sequences,
            'dominant_emotion':  dominant_emotion,
            'mean_dream_value':  mean_value,
            'n_cf_dreams':       type_counts['counterfactual'],
            'n_replay_dreams':   type_counts['replay'],
            'n_novel_dreams':    type_counts['novel'],
            'n_l2_updates':      total_l2_updates,
            'best_cf_action':    best_cf_action,
            'dream_valence':     dream_valence,
            't_dream':           self.t_dreams,
            # REM neuromodulator state during dreaming
            'rem_ach':           DREAM_ACH_LEVEL,
            'rem_ne':            DREAM_NE_LEVEL,
            'rem_sht':           DREAM_SHT_LEVEL,
        }

        # Store in dream history ring buffer
        self._dream_history.append({
            'valence':          dream_valence,
            'dominant_emotion': dominant_emotion,
            'n_cf':             type_counts['counterfactual'],
            't':                self.t_dreams,
        })
        if len(self._dream_history) > self._max_dream_history:
            self._dream_history.pop(0)

        return report
