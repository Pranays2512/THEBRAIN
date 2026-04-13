"""
M69: IMAGINATION ENGINE — CREATIVE MENTAL SIMULATION
=====================================================

Gives the brain the ability to imagine novel futures and exhibit
creativity by generating diverse, non-routine action sequences.

BIOLOGICAL BASIS
----------------
Imagination in neuroscience = "offline simulation" or "mental time travel":
the hippocampus replays and recombines past experiences to simulate futures
that haven't happened yet.

  Default Mode Network (DMN) — activates during mind-wandering and
    imagination. Generates diverse, internally-driven simulations that
    are not constrained by current sensory input.

  Hippocampal replay (forward) — generates sequences of possible
    future states by pattern-completing from partial cues. Each "imagined
    step" activates the next expected location in the cognitive map.

  Prefrontal evaluation — the OFC (orbitofrontal cortex) evaluates imagined
    futures for reward value. The PFC selects among competing imaginations
    based on goals and current motivational state.

  Creativity (lateral PFC + DMN) — when habitual behavior fails (boredom,
    being stuck), the brain shifts into a more exploratory mode: random
    variations on known patterns, novel recombinations of familiar elements.
    This is "creative divergence" from routine behavior.

WHAT SEPARATES THIS FROM M57
-----------------------------
M57 (Planner): greedy lookahead from current state, depth=3, SINGLE thread.
  Picks actions that maximize expected Q-value. Pure exploitation of known
  good paths. Limited to the single most-likely next BMU at each step.

M69 (Imagination): K=12 diverse random-action simulations, depth=5.
  Explicitly rewards NOVELTY over familiarity via a novelty_bonus term.
  Uses stochastic sampling from T_norm (not argmax) to generate varied paths.
  In creativity mode (high boredom), prefers paths through unfamiliar zones.
  Can discover directions M57 would never consider because they look
  suboptimal one step ahead but open unexplored territory.

CREATIVITY MODE
---------------
Triggered when corridor_boredom > BOREDOM_CREATIVITY_THRESH (default 0.60).
In creativity mode:
  - novelty_bonus weight increases from 0.30 → 0.75
  - The brain strongly prefers paths through zones it rarely visits recently
  - imagination_action may differ from M57/M68's suggestion
  - creativity_mode output signals how creative the brain is being

CALL ORDER
----------
Called after M57 (step 10) in Brain.step().
Inputs: l3 (zone model), curr_zone, zone_recency (from WM), corridor_boredom.

Outputs → brain.step() return dict:
  m69_imagination_action     : int       — suggested action from imagination
  m69_imagination_score      : float     — best imagined path score (value+novelty)
  m69_imagination_novelty    : float [0,1] — how novel is the best imagined path
  m69_creativity_mode        : float [0,1] — 0=exploit, 1=full creative mode
  m69_imagination_path_zones : list[int] — simulated zone sequence of best path
"""

import numpy as np

# ── Simulation parameters ─────────────────────────────────────
# Number of imagined paths to generate per step.
# 12 paths gives good diversity coverage of all 4 actions × 3 depths
# without being computationally expensive (each path is O(depth × zones)).
N_IMAGINED_PATHS = 12

# Depth of each imagined path (steps ahead to simulate).
# 5 steps: enough to "see around corners" in most worlds.
# Beyond 5, T_norm uncertainty compounds too much to be meaningful.
IMAGINE_DEPTH = 5

# ── Novelty weighting ─────────────────────────────────────────
# Base weight for novelty vs value when NOT in creativity mode.
# At 0.30: score = 0.70 × path_value + 0.30 × novelty
# The brain explores somewhat but primarily values reward-rich paths.
NOVELTY_WEIGHT_BASE = 0.30

# Weight when in full creativity mode (high boredom).
# At 0.75: score = 0.25 × path_value + 0.75 × novelty
# The brain strongly prefers novel paths even if they have lower value.
NOVELTY_WEIGHT_BORED = 0.75

# Boredom threshold to enter creativity mode.
# corridor_boredom is a Gini coefficient [0,1] from WM.
# At 0.60: brain has been visiting the same 1-2 zones repeatedly → stuck.
BOREDOM_CREATIVITY_THRESH = 0.60

# Minimum transitions for a zone to be usable in imagination.
# Below this, T_norm is near-uniform → samples are random noise.
MIN_TRANSITIONS_FOR_IMAGINATION = 8

# Bellman discount per imagined hop — same as M57 and M68.
IMAGINE_GAMMA = 0.85


class ImaginationEngine:
    """
    M69: Creative mental simulation via diverse path generation.

    Generates K imagined futures using L3's zone transition model,
    scores each by value × novelty (weighted by creativity mode),
    and returns the best imagined action.

    In creativity mode (high boredom), novelty is weighted more heavily
    than value, causing the brain to seek novel/unexplored territory
    rather than its usual food-hunting path.
    """

    def __init__(self, n_zones: int = 8, n_actions: int = 4, seed: int = 42):
        self.n_zones   = n_zones
        self.n_actions = n_actions
        self._rng      = np.random.default_rng(seed)
        self.t         = 0
        self._last_action = 0

    def _sample_next_zone(self, l3, zone: int, action: int) -> int:
        """
        Stochastically sample next zone from T_norm distribution.

        Uses the learned transition probabilities — not argmax.
        This generates diverse paths (each call may return different zones)
        which is essential for creative exploration: even low-probability
        transitions get sampled occasionally, exploring the "long tail"
        of possible futures.
        """
        probs = l3._T_norm[zone, action].copy().astype(np.float64)
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total < 1e-9:
            return int(self._rng.integers(0, self.n_zones))
        probs /= total
        return int(self._rng.choice(self.n_zones, p=probs))

    def _path_novelty(self, path_zones: list, zone_recency: np.ndarray) -> float:
        """
        Compute novelty of a path relative to recent zone history.

        Novelty = 1 − (average zone_recency along the path).

        zone_recency from WM measures how recently/often each zone was
        visited in the last N steps. Low recency = unfamiliar = novel.

        A path through zones the brain has barely visited recently = novel.
        A path through zones the brain is stuck in = not novel.
        """
        if not path_zones:
            return 0.5
        if zone_recency is None or len(zone_recency) == 0:
            return 0.5
        recencies = [
            float(zone_recency[z])
            for z in path_zones
            if 0 <= z < len(zone_recency)
        ]
        if not recencies:
            return 0.5
        avg_recency = float(np.mean(recencies))
        # Low recency → high novelty
        return float(np.clip(1.0 - avg_recency, 0.0, 1.0))

    def step(self,
             l3,
             curr_zone:        int,
             zone_recency:     np.ndarray,
             corridor_boredom: float,
             ) -> dict:
        """
        Generate diverse imagined paths and select the most creative/valuable.

        Parameters
        ----------
        l3               : ConceptLayer — provides T_norm, _zone_value, _T
        curr_zone        : int — current zone index (freq_idx), -1 if unknown
        zone_recency     : (n_zones,) float — WM zone recency EMA
        corridor_boredom : float [0,1] — WM boredom signal (Gini of recency)
        """
        self.t += 1

        null_out = {
            'm69_imagination_action':     self._last_action,
            'm69_imagination_score':      0.0,
            'm69_imagination_novelty':    0.0,
            'm69_creativity_mode':        0.0,
            'm69_imagination_path_zones': [],
        }

        if curr_zone < 0 or not l3._zones_stable:
            return null_out

        if float(l3._T[curr_zone].sum()) < MIN_TRANSITIONS_FOR_IMAGINATION:
            return null_out

        # ── Creativity mode: how much to weight novelty ───────
        in_creativity_mode = corridor_boredom >= BOREDOM_CREATIVITY_THRESH
        novelty_w = (NOVELTY_WEIGHT_BORED if in_creativity_mode
                     else NOVELTY_WEIGHT_BASE)
        value_w   = 1.0 - novelty_w

        # creativity_level: 0 at threshold, 1 at full boredom (1.0)
        if in_creativity_mode:
            creativity_level = float(np.clip(
                (corridor_boredom - BOREDOM_CREATIVITY_THRESH)
                / max(1.0 - BOREDOM_CREATIVITY_THRESH, 0.01),
                0.0, 1.0
            ))
        else:
            creativity_level = 0.0

        # ── Generate K imagined paths ─────────────────────────
        best_score   = -1.0
        best_action  = self._last_action
        best_path    = []
        best_novelty = 0.0

        for _ in range(N_IMAGINED_PATHS):
            # Sample a random first action to maximize diversity across paths.
            # Each of the 12 paths starts with a uniformly random action,
            # ensuring we sample at least 3 paths per action on average.
            first_action = int(self._rng.integers(0, self.n_actions))

            path_zones = [curr_zone]
            zone       = curr_zone
            path_value = 0.0

            for d in range(IMAGINE_DEPTH):
                # First hop uses first_action; subsequent hops are random.
                # This models: "what if I take action A first, then explore?"
                action = first_action if d == 0 else int(
                    self._rng.integers(0, self.n_actions)
                )

                # Only sample if zone has enough data
                if float(l3._T[zone].sum()) < 3:
                    break

                next_zone = self._sample_next_zone(l3, zone, action)
                path_zones.append(next_zone)

                # Accumulate discounted zone value
                z_val = float(l3._zone_value[next_zone]) if hasattr(l3, '_zone_value') else 0.0
                path_value += (IMAGINE_GAMMA ** (d + 1)) * z_val

                zone = next_zone

            if len(path_zones) < 2:
                continue

            # Compute novelty of this path (skip start zone — we're already here)
            novelty = self._path_novelty(path_zones[1:], zone_recency)

            # Combined score: value (exploitation) + novelty (exploration)
            score = value_w * path_value + novelty_w * novelty

            if score > best_score:
                best_score   = score
                best_action  = first_action
                best_path    = path_zones[:]
                best_novelty = novelty

        self._last_action = best_action

        return {
            'm69_imagination_action':     int(best_action),
            'm69_imagination_score':      float(max(best_score, 0.0)),
            'm69_imagination_novelty':    float(best_novelty),
            'm69_creativity_mode':        float(creativity_level),
            'm69_imagination_path_zones': list(best_path),
        }
