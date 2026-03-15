"""
WORLD — Soundscape Navigation Environment
==========================================

A fixed 8-node graph where each node has a characteristic frequency.
The brain navigates by taking directional actions (N/E/S/W).
Food nodes deliver external reward. Wall hits deliver a small penalty.

MAP (3×3 grid, one corner missing):

  [A] — [B] — [C]
   |           |
  [D]         [E]★
   |
  [F] — [G] — [H]★

★ = food nodes (reward=1.0 on arrival)

Node frequencies are drawn from the 8-frequency set used in training:
  0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0 Hz

Home node: A (freq 0.5 Hz, index 0)
Food nodes: E (freq 1.3 Hz, index 4) and H (freq 2.0 Hz, index 7)

ACTIONS: 0=North, 1=East, 2=South, 3=West

REWARD:
  +1.0  on arriving at a food node
  -0.05 on hitting a wall (no movement)
   0.0  otherwise
"""

import numpy as np

# ── Node definitions ──────────────────────────────────────────

FREQUENCIES = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]

# Node: (frequency_hz, freq_index, label)
NODES = {
    'A': (0.5, 0, 'A'),
    'B': (0.7, 1, 'B'),
    'C': (0.9, 2, 'C'),
    'D': (1.1, 3, 'D'),
    'E': (1.3, 4, 'E'),
    'F': (1.5, 5, 'F'),
    'G': (1.7, 6, 'G'),
    'H': (2.0, 7, 'H'),
}

FOOD_NODES = {'E', 'H'}
HOME_NODE  = 'A'

# Adjacency: node → {direction: neighbour}
# Missing directions = walls
ADJACENCY = {
    'A': {'East': 'B', 'South': 'D'},
    'B': {'East': 'C', 'West': 'A'},
    'C': {'South': 'E', 'West': 'B'},
    'D': {'North': 'A', 'South': 'F'},
    'E': {'North': 'C'},
    'F': {'North': 'D', 'East': 'G'},
    'G': {'East': 'H', 'West': 'F'},
    'H': {'West': 'G'},
}

ACTIONS    = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
N_ACTIONS  = 4
FOOD_REWARD  = 1.0
WALL_PENALTY = -0.05


class World:
    """
    Soundscape navigation environment.

    Usage
    -----
    world = World(seed=42)
    world.reset()

    for step in range(N):
        action = brain_decides()
        freq, freq_idx, reward, info = world.step(action)
        # feed freq into M50 / brain.step(decoded_freq=freq, freq_idx=freq_idx, reward=reward)
    """

    def __init__(self, seed: int = 42):
        self._rng     = np.random.default_rng(seed)
        self._node    = HOME_NODE
        self.t        = 0

        # Counters
        self.food_count  = 0
        self.wall_count  = 0
        self.total_steps = 0

        # History
        self.node_visit_counts = {n: 0 for n in NODES}
        self.action_counts     = [0] * N_ACTIONS

    def reset(self):
        self._node    = HOME_NODE
        self.t        = 0
        self.food_count  = 0
        self.wall_count  = 0
        self.total_steps = 0
        self.node_visit_counts = {n: 0 for n in NODES}
        self.action_counts     = [0] * N_ACTIONS
        return self._obs()

    def step(self, action: int):
        """
        Take action (0–3) and return (freq_hz, freq_idx, reward, info).

        Parameters
        ----------
        action : int  0=North 1=East 2=South 3=West

        Returns
        -------
        freq_hz  : float  — frequency the brain hears at new node
        freq_idx : int    — frequency index (0-7) for L3
        reward   : float  — +1.0 food, -0.05 wall, 0.0 otherwise
        info     : dict   — diagnostic info
        """
        direction = ACTIONS[action]
        neighbours = ADJACENCY[self._node]

        self.action_counts[action] += 1

        if direction in neighbours:
            # Valid move
            self._node = neighbours[direction]
            wall_hit   = False
            reward     = FOOD_REWARD if self._node in FOOD_NODES else 0.0
        else:
            # Wall
            wall_hit   = True
            reward     = WALL_PENALTY
            self.wall_count += 1

        if self._node in FOOD_NODES and not wall_hit:
            self.food_count += 1

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
    def current_node(self):
        return self._node

    @property
    def current_freq(self):
        return NODES[self._node][0]

    @property
    def current_freq_idx(self):
        return NODES[self._node][1]

    def food_rate(self, window=None):
        """Food events per 100 steps."""
        steps = window if window else self.total_steps
        if steps == 0:
            return 0.0
        return self.food_count / steps * 100

    def wall_rate(self):
        if self.total_steps == 0:
            return 0.0
        return self.wall_count / self.total_steps

    def summary(self):
        print(f"\n  World — step {self.t}")
        print(f"  Current node: {self._node}  ({self.current_freq:.1f} Hz)")
        print(f"  Food events:  {self.food_count}  ({self.food_rate():.2f}/100 steps)")
        print(f"  Wall rate:    {self.wall_rate():.1%}")
        print(f"  Node visits:  { {n: self.node_visit_counts[n] for n in NODES} }")
        print(f"  Actions:      N={self.action_counts[0]} E={self.action_counts[1]} "
              f"S={self.action_counts[2]} W={self.action_counts[3]}")