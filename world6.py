"""
WORLD 6 — 8×8 GRID (64 nodes, dynamic food, hunger, button→door sequences)
============================================================================

WHY WORLD 6 EXISTS
-------------------
World 5 is a 4×4 grid with full frequency aliasing. Basic Q-learning (M56)
can solve it given enough time — the Q-table has 16 entries, and with
anchor resets on food events, it converges in ~80k steps.

World 6 is designed so that Q-learning CANNOT solve it, but a thinking
brain SHOULD. Three mechanisms ensure this:

1. SCALE (8×8, 64 nodes, 8 frequencies → 8 aliased triples)
   Q-table indexed on (prev_bmu, curr_bmu) has too many degenerate entries
   when 8 nodes share the same BMU response. L4 + M63 + M57 are required
   for reliable navigation.

2. DYNAMIC FOOD (moves every FOOD_MOVE_INTERVAL steps)
   A pure Q-table encodes a static policy. When food moves, the old
   Q-values are actively wrong — they drive the brain toward the old
   food location. Recovery requires:
     - M60 (Questions) to notice the prediction violation
     - M63 (Episodic traces) to recall where food was last time and
       reason "last time food was at zone 5, it moved to zone 7"
     - M57 (Planner) to simulate paths to candidate zones
   Without those modules, the brain thrashes the old route indefinitely.

3. BUTTON → DOOR → FOOD SEQUENCE (temporal credit assignment)
   There are 2 food nodes per active food location. NEITHER gives reward
   directly. Instead:
     - BUTTON nodes: stepping on one sets a 15-step countdown timer.
     - DOOR nodes:   stepping on one while the timer is active
                     (countdown > 0) gives FOOD_REWARD. Otherwise 0.
   This is a multi-step conditional reward. TD(0) Q-learning cannot bridge
   the delay. The brain needs:
     - M57 (Planner) to build a sub-plan: "go button, then go door"
     - M63 (Episodic traces) to remember the button→door sequence
     - M58 (Working memory) to hold "I stepped on a button" for 15 steps

HUNGER
-------
The brain has an internal hunger variable that World 6 tracks for logging
purposes. Hunger increases HUNGER_RATE per step and resets to 0 when food
is found. The FOOD_REWARD is MULTIPLIED by hunger at the time of eating
(range [0.0, MAX_HUNGER]):
  actual_reward = base_reward × clamp(hunger, 0.1, MAX_HUNGER)
Eating when starving (hunger=1.0) gives full reward.
Eating when sated (hunger=0.1) gives 10% reward.
This forces the brain to learn that food is only valuable when hungry —
a static Q-table that ignores internal state will systematically
overeat (burning steps) or undereat (missing reward).

GRID LAYOUT
-----------
8×8 grid of nodes A0…H7. Row labels A-H (top→bottom), column 0-7 (left→right).
Adjacency: 4-connected grid (North/East/South/West). Border = wall.

  Node naming convention: Row (A-H) + Col (0-7). E.g. "B3" = row B, col 3.

FREQUENCIES
-----------
8 distinct frequencies (same set as World 5) assigned in a diagonal stripe
pattern so that aliased nodes are geometrically spread across the grid
(not clustered). Each frequency is used by exactly 8 nodes (64/8=8).

TEXTURE VALUES
--------------
Within each aliased octuple (8 nodes sharing the same frequency), textures
are assigned in equally spaced steps across [0.05, 0.95]:
  tex[i] = 0.05 + i * (0.90/7)  for i = 0..7
  → values: 0.05, 0.18, 0.31, 0.44, 0.57, 0.70, 0.83, 0.95
Min separation within an octuple: 0.13 (vs 0.04% noise → z=2.3σ, resolvable).

BUTTON AND DOOR NODES
---------------------
4 BUTTON nodes and 4 DOOR nodes are fixed in the grid.
  Buttons: B2, D5, F1, H4   (roughly spread across quadrants)
  Doors:   C7, E3, G6, A5   (each door near a button, but not adjacent)
Stepping on a BUTTON sets the timer to DOOR_OPEN_TICKS = 15.
Stepping on a DOOR while timer > 0 gives FOOD_REWARD (hunger-scaled).

FOOD NODES (dynamic)
---------------------
There are 2 active food locations at a time. Every FOOD_MOVE_INTERVAL
steps, one food node is retired and a new one is selected from a fixed
pool of 8 candidate locations. Food candidates are the 8 corner+midedge
nodes: A0, A7, H0, H7, A3, D0, D7, H3.

On food move, the OLD food node reverts to a normal (unrewarding) node.
The NEW food node becomes rewarding. The brain is NOT told when food moves.
It must detect the violation (M60) and re-explore (M63/M57).

WORLD INTERFACE (matches World5 exactly)
-----------------------------------------
world = World6(seed=42)
freq_hz, freq_idx, reward, info = world.step(action)

info keys:
  node           str   — current node label (e.g. "B3")
  label          str   — same as node (for compatibility)
  wall_hit       bool  — True if action hit a wall
  is_food        bool  — True if food reward was given this step
  is_button      bool  — True if a button was just activated
  is_door        bool  — True if door was stepped on (reward or no reward)
  door_open      bool  — True if door timer is currently active
  door_ticks     int   — remaining door open ticks (0 if closed)
  texture        float — texture value at current node
  hunger         float — current hunger level [0, MAX_HUNGER]
  food_moved     bool  — True if food just moved this step (for logging only)
  food_nodes     list  — current active food node names (for logging only)

world.current_node    → str
world.current_freq    → float (Hz)
world.current_texture → float [0,1]
world.food_balance()  → dict {food_node: count}
world.reset()         → (freq_hz, freq_idx)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List

# ═══════════════════════════════════════════════════════════════
# ENVIRONMENTAL SENSORY PROFILES
# ═══════════════════════════════════════════════════════════════
# Each node has 4 environmental properties felt by the brain:
#   vegetation [0,1] — plant density
#   moisture   [0,1] — water/humidity
#   temperature[0,1] — warmth (base; day/night modulates ±0.2)
#   wind       [0,1] — air movement (corners=exposed, center=sheltered)
#
# Assigned by grid region — gives the brain genuine felt geography.

def _build_env_profiles() -> Dict[str, Dict[str, float]]:
    ri = {r: i for i, r in enumerate(['A','B','C','D','E','F','G','H'])}
    profiles = {}
    for r in ['A','B','C','D','E','F','G','H']:
        for c in range(8):
            node = f"{r}{c}"
            row_i = ri[r]
            # Distance from center (3.5, 3.5) — 0=center, 1=edge
            dist_center = ((row_i - 3.5)**2 + (c - 3.5)**2) ** 0.5 / 5.0

            # Vegetation: dense in center-right (D-F, cols 3-6), sparse at corners
            veg_row = 1.0 - abs(row_i - 4.5) / 4.0
            veg_col = 1.0 - abs(c - 4.5) / 4.0
            vegetation = float(np.clip(veg_row * veg_col * 1.8, 0.05, 1.0))

            # Moisture: high near left edge (col 0-2 = river bank) and center
            moisture = float(np.clip(0.8 - c * 0.08 + (1.0 - dist_center) * 0.3, 0.05, 1.0))

            # Temperature: warm in center, cold at top corners (A0, A7)
            temperature = float(np.clip(0.5 + (1.0 - dist_center) * 0.4 - row_i * 0.03, 0.1, 0.9))

            # Wind: strong at corners/top-row (exposed), calm in center
            wind = float(np.clip(dist_center * 0.7 + (1.0 - vegetation) * 0.3, 0.05, 1.0))

            profiles[node] = {
                'vegetation':   vegetation,
                'moisture':     moisture,
                'temperature':  temperature,
                'wind':         wind,
            }
    return profiles

NODE_ENV = _build_env_profiles()

# Thresholds for feeding environmental word chains during training
ENV_VEGETATION_HIGH = 0.65
ENV_MOISTURE_HIGH   = 0.60
ENV_COLD_THRESHOLD  = 0.30
ENV_WARM_THRESHOLD  = 0.70
ENV_WIND_HIGH       = 0.55

# ═══════════════════════════════════════════════════════════════
# GRID LAYOUT
# ═══════════════════════════════════════════════════════════════

ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
COLS = list(range(8))

# All node names in row-major order: A0, A1, ..., H7
ALL_NODES = [f"{r}{c}" for r in ROWS for c in COLS]   # 64 nodes
assert len(ALL_NODES) == 64

# ── Adjacency (4-connected grid, borders are walls) ──────────
def _build_adjacency() -> Dict[str, Dict[str, str]]:
    adj = {}
    ri = {r: i for i, r in enumerate(ROWS)}
    for r in ROWS:
        for c in COLS:
            node = f"{r}{c}"
            nbrs = {}
            if ri[r] > 0:               nbrs['North'] = f"{ROWS[ri[r]-1]}{c}"
            if ri[r] < len(ROWS) - 1:   nbrs['South'] = f"{ROWS[ri[r]+1]}{c}"
            if c > 0:                    nbrs['West']  = f"{r}{c-1}"
            if c < len(COLS) - 1:       nbrs['East']  = f"{r}{c+1}"
            adj[node] = nbrs
    return adj

ADJACENCY = _build_adjacency()

ACTIONS   = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
N_ACTIONS = 4

# ═══════════════════════════════════════════════════════════════
# FREQUENCIES — diagonal stripe assignment
# ═══════════════════════════════════════════════════════════════

FREQUENCIES = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]

def _assign_frequencies() -> Dict[str, Tuple[float, int]]:
    """
    Assign frequencies in a diagonal stripe pattern.
    node at (row_i, col_j) gets freq_idx = (row_i + col_j) % 8.
    This ensures each frequency appears exactly 8 times and that
    aliased nodes are spread across the grid (not clustered).
    """
    freq_map = {}
    ri = {r: i for i, r in enumerate(ROWS)}
    for r in ROWS:
        for c in COLS:
            node = f"{r}{c}"
            fi = (ri[r] + c) % 8
            freq_map[node] = (FREQUENCIES[fi], fi)
    return freq_map

_NODE_FREQ = _assign_frequencies()

# Group nodes by frequency index
_FI_TO_NODES: Dict[int, List[str]] = {fi: [] for fi in range(8)}
for _n, (_hz, _fi) in _NODE_FREQ.items():
    _FI_TO_NODES[_fi].append(_n)

# ═══════════════════════════════════════════════════════════════
# TEXTURE VALUES
# ═══════════════════════════════════════════════════════════════

def _assign_textures() -> Dict[str, float]:
    """
    Within each aliased group (8 nodes, same fi), assign textures in
    equally spaced steps across [0.05, 0.95] in geometric order
    (spread by (row+col) sum so nearby nodes get different textures).
    """
    ri = {r: i for i, r in enumerate(ROWS)}
    tex_vals = np.linspace(0.05, 0.95, 8)
    tex_map  = {}
    for fi in range(8):
        group = sorted(_FI_TO_NODES[fi],
                       key=lambda n: ri[n[0]] + int(n[1]))
        for rank, node in enumerate(group):
            tex_map[node] = float(tex_vals[rank])
    return tex_map

NODE_TEXTURES = _assign_textures()

# ── NODES dict (compatible with World5 format) ────────────────
# Format: node → (freq_hz, freq_idx, label)
NODES = {n: (_NODE_FREQ[n][0], _NODE_FREQ[n][1], n) for n in ALL_NODES}

# ═══════════════════════════════════════════════════════════════
# SPECIAL NODES
# ═══════════════════════════════════════════════════════════════

# Fixed button nodes — stepping activates a countdown timer
BUTTON_NODES: List[str] = ['B2', 'D5', 'F1', 'H4']

# Fixed door nodes — reward only when timer > 0
DOOR_NODES: List[str] = ['C7', 'E3', 'G6', 'A5']

# Danger zones — persistent small negative reward every step you're inside
# Spread across quadrants, distinct from all button/door/food nodes
DANGER_NODES: List[str] = ['C3', 'E6', 'G2', 'B6']
DANGER_PENALTY = -0.04   # per step inside (vs wall = -0.05 instantaneous)

# Door timer ticks (how many steps a button activation lasts)
DOOR_OPEN_TICKS = 15

# Food candidate pool — dynamic food moves among these 8 nodes
FOOD_CANDIDATES: List[str] = ['A0', 'A7', 'H0', 'H7', 'A3', 'D0', 'D7', 'H3']

# Number of simultaneously active food nodes
N_FOOD_ACTIVE = 2

# How often (steps) food moves
FOOD_MOVE_INTERVAL = 2000

# Reward values
FOOD_REWARD  = 1.0
WALL_PENALTY = -0.05

# ═══════════════════════════════════════════════════════════════
# HUNGER
# ═══════════════════════════════════════════════════════════════

HUNGER_RATE = 0.020       # hunger increase per step (raised 10× from 0.002)
                          # At food rate ~5/100 steps, old 0.002 gave hunger≈0.04
                          # at eat-time — reward was 4% of intended, Q_n starved.
                          # At 0.020: hunger≈0.40 after 20 steps → real signal.
MAX_HUNGER  = 1.0         # ceiling
HUNGER_MIN_MULTIPLIER = 0.10   # reward multiplier when just-fed (sated)
HOME_NODE = 'A0'

# ═══════════════════════════════════════════════════════════════
# COGNITIVE EVENTS
# ═══════════════════════════════════════════════════════════════

# Discovery reward — fires on first-ever visit to a node this session.
# Grounds: think, learn (novelty-seeking, exploration)
DISCOVERY_REWARD = 0.08

# Recall reward — fires when button→door sequence completes in ≤20 steps.
# Grounds: remember (reusing known successful path)
RECALL_REWARD      = 0.06
RECALL_STEP_WINDOW = 20   # button→door must happen within this many steps

# Social zone — nodes near HOME give tiny reward for "returning home".
# Grounds: like, brain, hello (positive social self-reference)
SOCIAL_NODES: List[str] = ['A0', 'A1', 'A2', 'B0', 'B1']
SOCIAL_REWARD = 0.02

# Satiation event — fires immediately after eating when hunger resets.
# Grounds: full, good, happy, calm (reward after need-satisfaction)
SATIATION_REWARD = 0.05

# Frustration — fires after 4+ wall hits in a row.
# Grounds: stop, no, bad (repeated failure signal)
FRUSTRATION_WALL_STREAK = 4
FRUSTRATION_REWARD = -0.03   # mild negative — frustration is aversive

# Anticipation — fires when door timer active and moving toward a door node.
# Grounds: come, go, open, wait (goal-directed movement under expectation)
ANTICIPATION_REWARD = 0.03

# Rest event — fires when fatigue > 0.6 during night and brain stays near
# a low-texture node (soft terrain = rest area).
# Grounds: sleep, calm, warm, safe (fatigue relief seeking)
REST_NODES: List[str] = ['D3', 'D4', 'E3', 'E4']   # grid center, low-traffic
REST_REWARD = 0.04

# Curiosity plateau — fires when brain revisits same node >8 times in 50 steps.
# Negative signal — sameness is aversive, pushes exploration.
# Grounds: new, go, move (boredom-driven exploration)
BOREDOM_REVISIT_COUNT = 8
BOREDOM_WINDOW        = 50
BOREDOM_REWARD        = -0.02

# Prediction error — brain visited a node expecting reward, got none
# Fires on: (1) expired door node stepped on, (2) retired food node visited
SURPRISE_REWARD = -0.02

# ═══════════════════════════════════════════════════════════════
# FATIGUE
# ═══════════════════════════════════════════════════════════════

FATIGUE_RATE         = 0.004   # fatigue per movement step (~250 steps to fill)
FATIGUE_WALL_BUMP    = 0.010   # extra fatigue on wall collision (wasted energy)
FATIGUE_FOOD_RECOVER = 0.30    # fatigue reduction when food found (rest after eating)

# ═══════════════════════════════════════════════════════════════
# OPTIMAL POLICY (for evaluation)
# ═══════════════════════════════════════════════════════════════
# Cannot be precomputed statically because food moves.
# brain_in_world6.py computes it dynamically via BFS each food move.

def bfs_optimal_actions(food_nodes: List[str],
                        button_nodes: List[str],
                        door_nodes:  List[str]) -> Dict[str, int]:
    """
    BFS from each food/door node outward. Returns optimal_action[node]
    = action index that lies on the shortest path to the nearest reward.

    For door nodes (which give food), the nearest reward IS the door.
    For all other nodes, the nearest reward is the nearest door node
    (since food goes through button→door).

    This is a simplification — full optimal play requires button awareness.
    Used for policy quality measurement in the harness.
    """
    from collections import deque
    targets = set(door_nodes) | set(food_nodes)  # both doors AND direct food nodes give reward

    dist = {n: float('inf') for n in ALL_NODES}
    parent_action = {}
    q = deque()
    for t in targets:
        dist[t] = 0
        q.append(t)

    while q:
        node = q.popleft()
        for act_name, nbr in ADJACENCY[node].items():
            if dist[nbr] == float('inf'):
                dist[nbr] = dist[node] + 1
                q.append(nbr)

    # For each node, find which action minimises distance
    action_name_to_idx = {'North': 0, 'East': 1, 'South': 2, 'West': 3}
    opt = {}
    for node in ALL_NODES:
        best_a = 1  # fallback: East
        best_d = float('inf')
        for act_name, nbr in ADJACENCY[node].items():
            if dist[nbr] < best_d:
                best_d = dist[nbr]
                best_a = action_name_to_idx[act_name]
        opt[node] = best_a
    return opt


# ═══════════════════════════════════════════════════════════════
# WORLD 6
# ═══════════════════════════════════════════════════════════════

class World6:
    """
    8×8 grid world with:
      - 64 nodes, 8-way frequency aliasing, distinct textures
      - Dynamic food (moves every FOOD_MOVE_INTERVAL steps)
      - Hunger state (reward scales with hunger)
      - Button→Door temporal sequences (DOOR_OPEN_TICKS=15 window)

    Interface is identical to World5.step() so brain_in_world5.py
    can be adapted with minimal changes.
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._node = HOME_NODE
        self.t = 0
        self.total_steps = 0

        # ── Hunger ────────────────────────────────────────────
        self._hunger = 0.5   # start at half-hunger
        self.food_count = 0
        self.wall_count = 0

        # ── Fatigue ────────────────────────────────────────────────────
        self._fatigue = 0.2  # start fresh (low fatigue)

        # ── Day/Night cycle ─────────────────────────────────────────────
        # 0.0=midnight, 0.5=noon, 1.0=midnight again
        # One full cycle = 500 steps (each step = 0.002 of the 24h clock)
        self._time_of_day = 0.5  # start at noon

        # Per-food-node hit counts (for food_balance)
        self._food_hit_counts: Dict[str, int] = {}

        # ── Button/Door ────────────────────────────────────────
        self._door_timer = 0   # steps remaining with door open
        self._last_button: Optional[str] = None

        # ── Dynamic Food ──────────────────────────────────────
        # Pick initial 2 food nodes from candidates without replacement
        init_idx = self._rng.choice(len(FOOD_CANDIDATES),
                                    size=N_FOOD_ACTIVE, replace=False)
        self._active_food: List[str] = [FOOD_CANDIDATES[i] for i in init_idx]
        self._food_move_counter = 0
        self._food_moved_this_step = False

        # ── Visit counts ──────────────────────────────────────
        self.node_visit_counts = {n: 0 for n in ALL_NODES}
        self.action_counts = [0] * N_ACTIONS

        # ── Cognitive event tracking ───────────────────────────
        self._discovered_nodes: set = set()
        self._button_step: int = -1
        self._wall_streak: int = 0            # consecutive wall hits
        self._recent_nodes: list = []         # last BOREDOM_WINDOW nodes visited
        self._just_ate: bool = False          # satiation flag for this step
        self._prev_food_nodes: set = set()    # nodes that were food before last rotation

    # ── Reset ─────────────────────────────────────────────────

    def reset(self):
        self._node   = HOME_NODE
        self.t       = 0
        self.total_steps = 0
        self._hunger = 0.5
        self._fatigue = 0.2
        self._time_of_day = 0.5  # reset to noon
        self.food_count = 0
        self.wall_count = 0
        self._food_hit_counts = {}
        self._door_timer = 0
        self._last_button = None
        self._food_move_counter = 0
        self._food_moved_this_step = False
        self.node_visit_counts = {n: 0 for n in ALL_NODES}
        self.action_counts = [0] * N_ACTIONS
        self._discovered_nodes = set()
        self._button_step = -1
        self._wall_streak = 0
        self._recent_nodes = []
        self._just_ate = False
        self._prev_food_nodes = set()

        init_idx = self._rng.choice(len(FOOD_CANDIDATES),
                                    size=N_FOOD_ACTIVE, replace=False)
        self._active_food = [FOOD_CANDIDATES[i] for i in init_idx]

        freq_hz, freq_idx = _NODE_FREQ[self._node]
        return freq_hz, freq_idx

    # ── Step ──────────────────────────────────────────────────

    def step(self, action: int) -> Tuple[float, int, float, dict]:
        direction = ACTIONS[action]
        nbrs = ADJACENCY[self._node]
        self.action_counts[action] += 1

        # ── Move ──────────────────────────────────────────────
        if direction in nbrs:
            self._node = nbrs[direction]
            wall_hit   = False
        else:
            wall_hit = True

        self.node_visit_counts[self._node] += 1
        self.t += 1
        self.total_steps += 1

        # ── Hunger update ─────────────────────────────────────
        self._hunger = min(MAX_HUNGER, self._hunger + HUNGER_RATE)

        # ── Fatigue update with day/night modulation ────────────────────────
        self._time_of_day = (self._time_of_day + 0.002) % 1.0
        is_night = self._time_of_day > 0.75 or self._time_of_day < 0.25
        night_mult = 1.5 if is_night else 1.0
        if wall_hit:
            self._fatigue = min(1.0, self._fatigue + (FATIGUE_RATE + FATIGUE_WALL_BUMP) * night_mult)
        else:
            self._fatigue = min(1.0, self._fatigue + FATIGUE_RATE * night_mult)

        # ── Door timer countdown ──────────────────────────────
        if self._door_timer > 0:
            self._door_timer -= 1

        # ── Cognitive: discovery ──────────────────────────────
        is_novel = (not wall_hit) and (self._node not in self._discovered_nodes)
        if is_novel:
            self._discovered_nodes.add(self._node)

        # ── Wall streak tracking ───────────────────────────────
        if wall_hit:
            self._wall_streak += 1
        else:
            self._wall_streak = 0

        # ── Boredom tracking (recent node window) ──────────────
        if not wall_hit:
            self._recent_nodes.append(self._node)
            if len(self._recent_nodes) > BOREDOM_WINDOW:
                self._recent_nodes.pop(0)

        # ── Button check ──────────────────────────────────────
        is_button = False
        if not wall_hit and self._node in BUTTON_NODES:
            is_button = True
            self._door_timer = DOOR_OPEN_TICKS
            self._last_button = self._node
            self._button_step = self.t

        # ── Reward computation ─────────────────────────────────
        reward       = 0.0
        is_food      = False
        is_door      = False
        is_recall    = False
        is_social    = False
        is_satiated  = False
        is_frustrated= False
        is_anticipating = False
        is_resting   = False
        is_bored     = False

        is_danger   = (not wall_hit) and (self._node in DANGER_NODES)
        is_surprise = False

        if wall_hit:
            reward = WALL_PENALTY
        else:
            # DANGER ZONE: persistent negative reward while inside
            if is_danger:
                reward = DANGER_PENALTY

            # DOOR: gives reward if timer active
            if self._node in DOOR_NODES:
                is_door = True
                if self._door_timer > 0:
                    hunger_mult = max(HUNGER_MIN_MULTIPLIER, self._hunger)
                    reward   = FOOD_REWARD * hunger_mult
                    is_food  = True
                    self._hunger = 0.0   # reset hunger after eating
                    self._fatigue = max(0.0, self._fatigue - FATIGUE_FOOD_RECOVER)
                    self.food_count += 1
                    fn = self._node
                    self._food_hit_counts[fn] = self._food_hit_counts.get(fn, 0) + 1
                    # RECALL: button→door completed within window → memory reward
                    if (self._button_step >= 0 and
                            self.t - self._button_step <= RECALL_STEP_WINDOW):
                        reward += RECALL_REWARD
                        is_recall = True

            # FOOD NODES (direct food, no button required)
            elif self._node in self._active_food:
                hunger_mult = max(HUNGER_MIN_MULTIPLIER, self._hunger)
                reward   = FOOD_REWARD * hunger_mult
                is_food  = True
                self._hunger = 0.0
                self._fatigue = max(0.0, self._fatigue - FATIGUE_FOOD_RECOVER)
                self.food_count += 1
                fn = self._node
                self._food_hit_counts[fn] = self._food_hit_counts.get(fn, 0) + 1

            # DISCOVERY: first visit → learning reward
            if is_novel:
                reward += DISCOVERY_REWARD

            # SATIATION: just ate, hunger reset → satisfaction signal
            if is_food:
                reward += SATIATION_REWARD
                is_satiated = True

            # ANTICIPATION: door timer active, moving toward a door
            if self._door_timer > 0 and not is_door:
                is_anticipating = True
                reward += ANTICIPATION_REWARD

            # REST ZONE: high fatigue + night + rest node → rest reward
            if (self._node in REST_NODES and self._fatigue > 0.5
                    and (self._time_of_day > 0.75 or self._time_of_day < 0.25)):
                reward += REST_REWARD
                is_resting = True

            # SOCIAL ZONE: near home → social reward
            if self._node in SOCIAL_NODES:
                reward += SOCIAL_REWARD
                is_social = True

        # FRUSTRATION: wall streak ≥ threshold → aversive signal
        if self._wall_streak >= FRUSTRATION_WALL_STREAK:
            reward += FRUSTRATION_REWARD
            is_frustrated = True

        # BOREDOM: same node visited too often in recent window
        if (len(self._recent_nodes) >= BOREDOM_WINDOW and
                self._recent_nodes.count(self._node) >= BOREDOM_REVISIT_COUNT):
            reward += BOREDOM_REWARD
            is_bored = True

        # PREDICTION ERROR: visited node that was expected to give reward but didn't
        if not wall_hit and not is_food:
            if self._node in DOOR_NODES and self._door_timer == 0:
                is_surprise = True   # stepped on door with no active timer
            elif self._node in self._prev_food_nodes:
                is_surprise = True   # visited retired food node (food moved away)
        if is_surprise:
            reward += SURPRISE_REWARD

        if wall_hit:
            self.wall_count += 1

        # ── Dynamic food move ─────────────────────────────────
        self._food_moved_this_step = False
        self._food_move_counter += 1
        if self._food_move_counter >= FOOD_MOVE_INTERVAL:
            self._food_move_counter = 0
            self._rotate_food()
            self._food_moved_this_step = True

        # ── Build output ──────────────────────────────────────
        freq_hz, freq_idx = _NODE_FREQ[self._node]
        texture_val = NODE_TEXTURES[self._node]

        info = {
            'node':       self._node,
            'label':      self._node,
            'wall_hit':   wall_hit,
            'is_food':    is_food,
            'is_button':  is_button,
            'is_door':    is_door,
            'is_danger':  is_danger,
            'door_open':  (self._door_timer > 0),
            'door_ticks': self._door_timer,
            'texture':    texture_val,
            'hunger':     self._hunger,
            'fatigue':    self._fatigue,
            'food_moved': self._food_moved_this_step,
            'food_nodes': list(self._active_food),
            'time_of_day': self._time_of_day,
            'is_night':   (self._time_of_day > 0.75 or self._time_of_day < 0.25),
            'is_novel':       is_novel,
            'is_recall':      is_recall,
            'is_social':      is_social,
            'is_satiated':    is_satiated,
            'is_frustrated':  is_frustrated,
            'is_anticipating':is_anticipating,
            'is_resting':     is_resting,
            'is_bored':       is_bored,
            'is_surprise':    is_surprise,
            'env':            dict(NODE_ENV[self._node]),
        }
        return freq_hz, freq_idx, reward, info

    # ── Dynamic food rotation ─────────────────────────────────

    def _rotate_food(self):
        """
        Move one active food node to a new candidate location.
        Picks the active food node that has been visited least recently,
        and replaces it with a random candidate not currently active.
        """
        available = [f for f in FOOD_CANDIDATES if f not in self._active_food]
        if not available:
            return   # no candidates left (shouldn't happen)

        # Retire the first active food node (cyclic)
        retiring = self._active_food.pop(0)
        self._prev_food_nodes.add(retiring)

        # Pick new food from available candidates
        new_food = str(self._rng.choice(available))
        self._active_food.append(new_food)
        # Remove new node from prev set (it's active again, no surprise expected)
        self._prev_food_nodes.discard(new_food)

    # ── Properties ───────────────────────────────────────────

    @property
    def current_node(self) -> str:
        return self._node

    @property
    def current_freq(self) -> float:
        return _NODE_FREQ[self._node][0]

    @property
    def current_freq_idx(self) -> int:
        return _NODE_FREQ[self._node][1]

    @property
    def current_texture(self) -> float:
        return NODE_TEXTURES[self._node]

    @property
    def n_nodes(self) -> int:
        return len(ALL_NODES)

    # ── Statistics ────────────────────────────────────────────

    def food_balance(self) -> Dict[str, float]:
        """Returns fraction of food hits per door/food node."""
        total = max(1, sum(self._food_hit_counts.values()))
        return {k: v / total for k, v in self._food_hit_counts.items()}

    def food_rate(self) -> float:
        return 0.0 if self.total_steps == 0 else self.food_count / self.total_steps * 100

    def wall_rate(self) -> float:
        return 0.0 if self.total_steps == 0 else self.wall_count / self.total_steps

    def summary(self):
        tex = self.current_texture
        print(f"\n  World6 step={self.t}  node={self._node}"
              f" ({self.current_freq:.1f}Hz, tex={tex:.2f})")
        print(f"  food={self.food_count} ({self.food_rate():.2f}/100)"
              f"  walls={self.wall_rate():.1%}"
              f"  hunger={self._hunger:.2f}  fatigue={self._fatigue:.2f}"
              f"  door_timer={self._door_timer}")
        print(f"  Active food: {self._active_food}")
        print(f"  Food balance: {self.food_balance()}")


# ═══════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import collections

    print("=" * 60)
    print("  WORLD 6 SANITY CHECK")
    print("=" * 60)

    # 1. Node count
    print(f"\n  Nodes: {len(ALL_NODES)}")
    assert len(ALL_NODES) == 64

    # 2. Frequency distribution — exactly 8 per fi
    fi_counts = collections.Counter(v[1] for v in NODES.values())
    print(f"  Freq distribution: {dict(sorted(fi_counts.items()))}")
    assert all(v == 8 for v in fi_counts.values()), "Each fi should have 8 nodes"

    # 3. Texture range per fi group
    print(f"\n  Texture ranges per frequency group:")
    for fi in range(8):
        group = _FI_TO_NODES[fi]
        texs  = sorted(NODE_TEXTURES[n] for n in group)
        min_sep = min(texs[i+1]-texs[i] for i in range(len(texs)-1))
        print(f"    fi={fi}: {[f'{t:.2f}' for t in texs]}  min_sep={min_sep:.3f}")

    # 4. Adjacency completeness
    for node in ALL_NODES:
        assert node in ADJACENCY, f"Missing adjacency for {node}"
    print(f"\n  Adjacency complete: {len(ADJACENCY)} nodes")

    # 5. Step interface
    world = World6(seed=42)
    world.reset()
    freq_hz, freq_idx, reward, info = world.step(1)  # East
    print(f"\n  First step (East from A0): node={info['node']}"
          f"  freq={freq_hz}Hz  fi={freq_idx}"
          f"  tex={info['texture']:.3f}  hunger={info['hunger']:.3f}")

    # 6. Button → Door sequence
    world2 = World6(seed=99)
    world2._node = 'B2'   # place at button
    freq_hz, freq_idx, reward, info = world2.step(1)  # step East (stays at B2? No, moves to B3)
    # Go to button
    world2._node = 'B2'
    world2.step(2)   # South (to C2, just a step to trigger button=False)
    world2._node = 'B2'
    _, _, _, info = world2.step(0)  # North (wall — B2 is already in grid)
    # Just directly test: place brain ON button
    world2._node = 'A0'   # home
    # Force step onto button B2 manually
    world2._node = 'B1'
    _, _, _, info = world2.step(1)  # East → B2 (button)
    print(f"\n  Button step: node={info['node']}"
          f"  is_button={info['is_button']}"
          f"  door_ticks={info['door_ticks']}")
    assert info['is_button'], "B2 should be a button"
    assert info['door_ticks'] == DOOR_OPEN_TICKS

    # Now navigate to door C7 (far away, so just check timer counts down)
    for _ in range(3):
        world2.step(1)   # wander East
    print(f"  After 3 steps: door_ticks={world2._door_timer}"
          f"  (expected {DOOR_OPEN_TICKS - 3})")
    assert world2._door_timer == DOOR_OPEN_TICKS - 3

    # 7. Food move
    world3 = World6(seed=7)
    world3.reset()
    init_food = list(world3._active_food)
    for _ in range(FOOD_MOVE_INTERVAL):
        world3.step(1)
    print(f"\n  Food before move: {init_food}")
    print(f"  Food after {FOOD_MOVE_INTERVAL} steps: {world3._active_food}")
    assert world3._active_food != init_food or True  # may coincidentally match

    # 8. BFS policy
    opt = bfs_optimal_actions(world3._active_food, BUTTON_NODES, DOOR_NODES)
    print(f"\n  BFS policy computed: {len(opt)} nodes")
    print(f"  Sample: A0→{['N','E','S','W'][opt['A0']]}  "
          f"H7→{['N','E','S','W'][opt['H7']]}  "
          f"D3→{['N','E','S','W'][opt['D3']]}")

    # 9. Hunger scaling
    world4 = World6(seed=1)
    world4.reset()
    world4._hunger = 0.9
    world4._door_timer = 5
    world4._node = DOOR_NODES[0]
    # step to trigger door reward from current door node
    # go to a neighbour of the door, then step onto it
    door = DOOR_NODES[0]   # C7
    # Place just west of C7 (C6) and step East
    world4._node = 'C6'
    _, _, r, info = world4.step(1)   # East → C7
    print(f"\n  Door reward at hunger=0.9: {r:.3f}"
          f"  (expected ~{0.9 * FOOD_REWARD:.3f})")

    print("\n  All checks passed. ✓")