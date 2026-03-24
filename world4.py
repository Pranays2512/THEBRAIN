"""
WORLD 4 — Y-FORK TOPOLOGY (12 nodes, harder disambiguation)
============================================================

World 4 tests three things World 3 could not:

1. NO EASY PATH: Both food sources are 4-5 steps from home.
   No short fallback. The brain must work for every reward.

2. FORK JUNCTION WITH ALIASING: Node C is both the critical
   fork decision point (North to F★ or East to H★) AND shares
   its frequency with K (a spur escape node below home).
   Different optimal actions at same sound — hardest aliasing yet.

3. SYMMETRIC ARMS: Both food sources equidistant and equal reward.
   No dominant path to lock onto. M58 boredom must drive alternation.

MAP (vertical Y-fork):

       [F★]            <- food, 5 steps North from home
        |
       [E]
        |
       [D]
        |
       [C] ─── [G] ─── [H★]  <- food, 4 steps East from home
        |
       [B] ─── [I]     <- I is dead end (only exit: West)
        |
       [A] HOME
        |
       [J] ─── [K] ─── [L]   <- spur below home (L dead end)

HOME: A   FOOD: F★ (North, 5 steps), H★ (East, 4 steps)

NODE FREQUENCIES (12 nodes, 8 frequencies — 4 aliased pairs):
  A: 0.5Hz (fi=0)  shared with L  — home vs dead end
  B: 0.7Hz (fi=1)  shared with J  — on-path vs spur
  C: 0.9Hz (fi=2)  shared with K  — fork vs spur escape  [hardest]
  D: 1.1Hz (fi=3)  unique
  E: 1.3Hz (fi=4)  unique
  F: 1.5Hz (fi=5)  shared with H  — both food nodes
  G: 1.7Hz (fi=6)  unique
  H: 1.5Hz (fi=5)  shared with F
  I: 2.0Hz (fi=7)  unique dead end
  J: 0.7Hz (fi=1)  shared with B
  K: 0.9Hz (fi=2)  shared with C
  L: 0.5Hz (fi=0)  shared with A

DISAMBIGUATION (all aliases have different optimal actions):
  A(fi=0)->North  vs  L(fi=0)->West   DIFFERENT
  B(fi=1)->North  vs  J(fi=1)->West   DIFFERENT
  C(fi=2)->N/E    vs  K(fi=2)->West   VERY DIFFERENT (fork vs escape)
  F(fi=5)->South  vs  H(fi=5)->West   mild (both return from food)

OPTIMAL POLICY:
  A: North   B: North   C: North (or East — both valid)
  D: North   E: North   F: South (return)
  G: East    H: West (return)
  I: West (escape)   J: West   K: West   L: West (escape)

RANDOM BASELINES:  Food ~6.2/100   Wall ~54.2%
"""

import numpy as np

FREQUENCIES = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]

NODES = {
    'A': (0.5, 0, 'A'),
    'B': (0.7, 1, 'B'),
    'C': (0.9, 2, 'C'),
    'D': (1.1, 3, 'D'),
    'E': (1.3, 4, 'E'),
    'F': (1.5, 5, 'F'),
    'G': (1.7, 6, 'G'),
    'H': (1.5, 5, 'H'),
    'I': (2.0, 7, 'I'),
    'J': (0.7, 1, 'J'),
    'K': (0.9, 2, 'K'),
    'L': (0.5, 0, 'L'),
}

FOOD_NODES = {'F', 'H'}
HOME_NODE  = 'A'

ADJACENCY = {
    'A': {'North': 'B', 'South': 'J'},
    'B': {'North': 'C', 'South': 'A', 'East': 'I'},
    'C': {'North': 'D', 'South': 'B', 'East': 'G'},
    'D': {'North': 'E', 'South': 'C'},
    'E': {'North': 'F', 'South': 'D'},
    'F': {'South': 'E'},
    'G': {'East': 'H', 'West': 'C'},
    'H': {'West': 'G'},
    'I': {'West': 'B'},
    'J': {'North': 'A', 'East': 'K'},
    'K': {'West': 'J', 'East': 'L'},
    'L': {'West': 'K'},
}

ACTIONS   = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
N_ACTIONS = 4
FOOD_REWARD  = 1.0
WALL_PENALTY = -0.05

OPTIMAL_ACTION = {
    'A': 0, 'B': 0, 'C': 0,  # North (C also accepts East)
    'D': 0, 'E': 0, 'F': 2,  # North, North, South
    'G': 1, 'H': 3,           # East, West
    'I': 3, 'J': 3, 'K': 3, 'L': 3,  # West
}
FORK_NODE  = 'C'
FORK_VALID = {0, 1}  # North and East both correct at C


class World4:
    def __init__(self, seed=42):
        self._rng              = np.random.default_rng(seed)
        self._node             = HOME_NODE
        self.t                 = 0
        self.food_count        = 0
        self.wall_count        = 0
        self.total_steps       = 0
        self.node_visit_counts = {n: 0 for n in NODES}
        self.action_counts     = [0] * N_ACTIONS
        self.food_north        = 0
        self.food_east         = 0

    def reset(self):
        self._node             = HOME_NODE
        self.t                 = 0
        self.food_count        = 0
        self.wall_count        = 0
        self.total_steps       = 0
        self.node_visit_counts = {n: 0 for n in NODES}
        self.action_counts     = [0] * N_ACTIONS
        self.food_north        = 0
        self.food_east         = 0
        return self._obs()

    def step(self, action):
        direction  = ACTIONS[action]
        neighbours = ADJACENCY[self._node]
        self.action_counts[action] += 1

        if direction in neighbours:
            self._node = neighbours[direction]
            wall_hit   = False
            reward     = FOOD_REWARD if self._node in FOOD_NODES else 0.0
        else:
            wall_hit = True
            reward   = WALL_PENALTY
            self.wall_count += 1

        if self._node in FOOD_NODES and not wall_hit:
            self.food_count += 1
            if   self._node == 'F': self.food_north += 1
            elif self._node == 'H': self.food_east  += 1

        self.node_visit_counts[self._node] += 1
        self.t           += 1
        self.total_steps += 1

        freq_hz, freq_idx, label = NODES[self._node]
        info = {
            'node': self._node, 'label': label,
            'wall_hit': wall_hit,
            'is_food': (self._node in FOOD_NODES and not wall_hit),
        }
        return freq_hz, freq_idx, reward, info

    def _obs(self):
        freq_hz, freq_idx, label = NODES[self._node]
        return freq_hz, freq_idx

    @property
    def current_node(self):     return self._node
    @property
    def current_freq(self):     return NODES[self._node][0]
    @property
    def current_freq_idx(self): return NODES[self._node][1]

    def food_rate(self, window=None):
        s = window if window else self.total_steps
        return 0.0 if s == 0 else self.food_count / s * 100

    def arm_balance(self):
        total = max(1, self.food_north + self.food_east)
        return self.food_north / total, self.food_east / total

    def wall_rate(self):
        return 0.0 if self.total_steps == 0 else self.wall_count / self.total_steps

    def summary(self):
        n_f, e_f = self.arm_balance()
        print(f"\n  World4 step={self.t}  node={self._node}({self.current_freq:.1f}Hz)")
        print(f"  food={self.food_count} ({self.food_rate():.2f}/100)"
              f"  N={n_f:.0%} E={e_f:.0%}  walls={self.wall_rate():.1%}")