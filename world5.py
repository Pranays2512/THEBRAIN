"""
WORLD 5 — 4×4 GRID (16 nodes, all frequencies aliased)
=======================================================

World 5 is the scaling test: does the architecture work when
every frequency is shared by exactly two nodes?

In Worlds 3 and 4, some nodes had unique frequencies — Q_f could
give unambiguous action signals for those. In World 5 there are
zero unique-frequency nodes. Q_f[fi] is shared by exactly two
nodes for every fi. The brain must use sequence context (L2/L4)
to know which of the two nodes it is at before Q_n can help.

MAP (4×4 grid, home top-left, food top-right and bottom-right):

    [A]─[B]─[C]─[D]
     |   |   |   |
    [E]─[F]─[G]─[H★]   <- food (4 steps from home)
     |   |   |   |
    [I]─[J]─[K]─[L]
     |   |   |   |
    [M]─[N]─[O]─[P★]   <- food (6 steps from home)

HOME: A (top-left)
FOOD: H★ (row 1, col 3 — 4 steps), P★ (row 3, col 3 — 6 steps)

NODE FREQUENCIES (16 nodes, 8 frequencies — every fi shared by 2 nodes):
  Row 0:  A=0.5Hz(fi=0)  B=0.7Hz(fi=1)  C=0.9Hz(fi=2)  D=1.1Hz(fi=3)
  Row 1:  E=1.3Hz(fi=4)  F=1.5Hz(fi=5)  G=1.7Hz(fi=6)  H=2.0Hz(fi=7)★
  Row 2:  I=0.5Hz(fi=0)  J=0.7Hz(fi=1)  K=0.9Hz(fi=2)  L=1.1Hz(fi=3)
  Row 3:  M=1.3Hz(fi=4)  N=1.5Hz(fi=5)  O=1.7Hz(fi=6)  P=2.0Hz(fi=7)★

Aliased pairs (same fi, different rows):
  fi=0: A/I   fi=1: B/J   fi=2: C/K   fi=3: D/L
  fi=4: E/M   fi=5: F/N   fi=6: G/O   fi=7: H/P  (both food)

WHAT MAKES THIS HARDER THAN W3/W4:
  1. 16 nodes (vs 12) — larger belief space for L4
  2. Zero unique-frequency nodes — Q_f cannot give unambiguous signal alone
  3. Multiple paths to each food source — brain must learn which route is faster
  4. 4×4 grid topology — novel spatial structure
  5. Two food sources at different distances (4 vs 6 steps, like W3)
     M58 boredom must drive coverage of the longer P path

WHAT MAKES IT TRACTABLE:
  Most aliased pairs share the same optimal action (East for most nodes).
  Q_f[fi, East] accumulates signal for ALL nodes in a column, and
  East is correct for most of them. The brain can navigate by
  Q_f alone without full disambiguation. L4/Q_n refine the policy
  for nodes where the correct action differs from the majority.

OPTIMAL POLICY (shortest path to nearest food):
  Top half (rows 0-1): go East toward H★
    A:E  B:E  C:E  D:S  E:E  F:E  G:E  H:N(return)
  Bottom half (rows 2-3): go East toward P★
    I:E  J:E  K:E  L:S  M:E  N:E  O:E  P:N(return)

RANDOM BASELINES:
  16 nodes, 4 actions, 2 food nodes
  Food rate  ≈ 6.5/100   Wall rate ≈ 21.8%
  (lower wall rate than W3/W4 — grid has more valid moves per node)
"""

import numpy as np

FREQUENCIES = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]

NODES = {
    'A': (0.5, 0, 'A'), 'B': (0.7, 1, 'B'), 'C': (0.9, 2, 'C'), 'D': (1.1, 3, 'D'),
    'E': (1.3, 4, 'E'), 'F': (1.5, 5, 'F'), 'G': (1.7, 6, 'G'), 'H': (2.0, 7, 'H'),
    'I': (0.5, 0, 'I'), 'J': (0.7, 1, 'J'), 'K': (0.9, 2, 'K'), 'L': (1.1, 3, 'L'),
    'M': (1.3, 4, 'M'), 'N': (1.5, 5, 'N'), 'O': (1.7, 6, 'O'), 'P': (2.0, 7, 'P'),
}

FOOD_NODES = {'H', 'P'}
HOME_NODE  = 'A'

ADJACENCY = {
    'A': {'East': 'B', 'South': 'E'},
    'B': {'West': 'A', 'East': 'C', 'South': 'F'},
    'C': {'West': 'B', 'East': 'D', 'South': 'G'},
    'D': {'West': 'C', 'South': 'H'},
    'E': {'North': 'A', 'East': 'F', 'South': 'I'},
    'F': {'North': 'B', 'West': 'E', 'East': 'G', 'South': 'J'},
    'G': {'North': 'C', 'West': 'F', 'East': 'H', 'South': 'K'},
    'H': {'North': 'D', 'West': 'G', 'South': 'L'},   # food
    'I': {'North': 'E', 'East': 'J', 'South': 'M'},
    'J': {'North': 'F', 'West': 'I', 'East': 'K', 'South': 'N'},
    'K': {'North': 'G', 'West': 'J', 'East': 'L', 'South': 'O'},
    'L': {'North': 'H', 'West': 'K', 'South': 'P'},
    'M': {'North': 'I', 'East': 'N'},
    'N': {'North': 'J', 'West': 'M', 'East': 'O'},
    'O': {'North': 'K', 'West': 'N', 'East': 'P'},
    'P': {'North': 'L', 'West': 'O'},                 # food
}

ACTIONS   = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
N_ACTIONS = 4
FOOD_REWARD  = 1.0
WALL_PENALTY = -0.05

# Optimal action per node (for scoring — oracle only, never fed to brain)
# Top half heads East toward H; bottom half heads East toward P.
# D and L go South (one step to their respective food).
OPTIMAL_ACTION = {
    'A': 1, 'B': 1, 'C': 1, 'D': 2,   # East, East, East, South
    'E': 1, 'F': 1, 'G': 1, 'H': 0,   # East, East, East, North (return)
    'I': 1, 'J': 1, 'K': 1, 'L': 2,   # East, East, East, South
    'M': 1, 'N': 1, 'O': 1, 'P': 0,   # East, East, East, North (return)
}


class World5:
    """
    16-node 4×4 grid soundscape — maximal frequency aliasing.
    Drop-in for World3 and World4 in any harness.
    """

    def __init__(self, seed: int = 42):
        self._rng              = np.random.default_rng(seed)
        self._node             = HOME_NODE
        self.t                 = 0
        self.food_count        = 0
        self.wall_count        = 0
        self.total_steps       = 0
        self.node_visit_counts = {n: 0 for n in NODES}
        self.action_counts     = [0] * N_ACTIONS
        self.food_h            = 0   # H★ visits
        self.food_p            = 0   # P★ visits

    def reset(self):
        self._node             = HOME_NODE
        self.t                 = 0
        self.food_count        = 0
        self.wall_count        = 0
        self.total_steps       = 0
        self.node_visit_counts = {n: 0 for n in NODES}
        self.action_counts     = [0] * N_ACTIONS
        self.food_h            = 0
        self.food_p            = 0
        return self._obs()

    def step(self, action: int):
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
            if   self._node == 'H': self.food_h += 1
            elif self._node == 'P': self.food_p += 1

        self.node_visit_counts[self._node] += 1
        self.t           += 1
        self.total_steps += 1

        freq_hz, freq_idx, label = NODES[self._node]
        info = {
            'node':     self._node,
            'label':    label,
            'wall_hit': wall_hit,
            'is_food':  (self._node in FOOD_NODES and not wall_hit),
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

    def food_balance(self):
        """(h_frac, p_frac) of total food. H is near, P is far."""
        total = max(1, self.food_h + self.food_p)
        return self.food_h / total, self.food_p / total

    def wall_rate(self):
        return 0.0 if self.total_steps == 0 else self.wall_count / self.total_steps

    def summary(self):
        h_f, p_f = self.food_balance()
        print(f"\n  World5 step={self.t}  node={self._node}({self.current_freq:.1f}Hz)")
        print(f"  food={self.food_count} ({self.food_rate():.2f}/100)"
              f"  [H={h_f:.0%} P={p_f:.0%}]  walls={self.wall_rate():.1%}")