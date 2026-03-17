"""
WORLD 3 — LARGER SOUNDSCAPE (12 nodes, frequency reuse)
========================================================

A 12-node graph that tests three things the 8-node world could not:

1. FREQUENCY REUSE: 12 nodes but only 8 frequencies.
   Nodes I and J share frequencies with A and B (0.5 and 0.7 Hz).
   The brain cannot identify its location from sound alone — it must
   use sequence context (what did I hear before?) to disambiguate.
   This is the first genuinely hard perceptual challenge for M54/L2.

2. ASYMMETRIC FOOD: two food sources at very different distances.
   E★ is 3 steps from home (short path: A→B→C→E).
   K★ is 6 steps from home (long path: A→D→F→G→H→J→K).
   Tests whether the brain finds both or only exploits the easy one.

3. DEAD END: node L hangs off K with no further exits.
   The brain must learn that L is not worth visiting (no food, no path).
   Tests whether Q-learning correctly discounts dead-end nodes.

MAP (4×3 grid):

  [A] — [B] — [C]
   |           |
  [D]         [E]★
   |
  [F] — [G] — [H]
               |
              [I] — [J] — [K]★
                          |
                         [L]  (dead end)

Home: A
Food: E★ (3 steps), K★ (6 steps)

NODE FREQUENCIES (12 nodes, 8 frequencies — reuse necessary):
  A: 0.5 Hz  (index 0)   ← shared with I
  B: 0.7 Hz  (index 1)   ← shared with J
  C: 0.9 Hz  (index 2)
  D: 1.1 Hz  (index 3)
  E: 1.3 Hz  (index 4)   ← food
  F: 1.5 Hz  (index 5)
  G: 1.7 Hz  (index 6)
  H: 2.0 Hz  (index 7)
  I: 0.5 Hz  (index 0)   ← same as A — disambiguation required
  J: 0.7 Hz  (index 1)   ← same as B — disambiguation required
  K: 1.3 Hz  (index 4)   ← same as E — disambiguation required
  L: 1.1 Hz  (index 3)   ← same as D — dead end

DISAMBIGUATION CHALLENGE:
  A and I both sound like 0.5 Hz. At A, East→B and South→D are valid.
  At I, East→J is the only good exit (West→H is a return).
  The brain must use sequence context — "I heard 2.0Hz (H) just before
  this 0.5Hz, so I'm probably at I, not A" — to learn the right action.

OPTIMAL POLICY:
  A: East → B → C → E★     (3 steps to food)
  B: East → C → E★
  C: South → E★
  D: South → F → G → H → I → J → K★  (longer path)
  E: North → C (return from food)
  F: East → G
  G: East → H
  H: South → I
  I: East → J
  J: East → K★
  K: West → J (return from food)
  L: North → K (return from dead end — only exit)

RANDOM BASELINES:
  12 nodes, 4 actions, 2 food nodes.
  Random food rate ≈ 8.3/100 (lower than 8-node world due to more nodes).
  Random wall rate ≈ 58.3% (similar to 8-node world).
"""

import numpy as np

# ── Frequencies ───────────────────────────────────────────────
# 8 distinct frequencies — shared across 12 nodes
FREQUENCIES = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]

# Node: (frequency_hz, freq_index, label)
# freq_index is the index into FREQUENCIES — shared nodes have same index
NODES = {
    'A': (0.5, 0, 'A'),
    'B': (0.7, 1, 'B'),
    'C': (0.9, 2, 'C'),
    'D': (1.1, 3, 'D'),
    'E': (1.3, 4, 'E'),   # food
    'F': (1.5, 5, 'F'),
    'G': (1.7, 6, 'G'),
    'H': (2.0, 7, 'H'),
    'I': (0.5, 0, 'I'),   # same freq as A — ambiguous
    'J': (0.7, 1, 'J'),   # same freq as B — ambiguous
    'K': (1.3, 4, 'K'),   # same freq as E — ambiguous
    'L': (1.1, 3, 'L'),   # same freq as D — dead end
}

FOOD_NODES = {'E', 'K'}
HOME_NODE  = 'A'

# Adjacency: node → {direction: neighbour}
ADJACENCY = {
    'A': {'East': 'B', 'South': 'D'},
    'B': {'East': 'C', 'West': 'A'},
    'C': {'South': 'E', 'West': 'B'},
    'D': {'North': 'A', 'South': 'F'},
    'E': {'North': 'C'},               # food — only exit is back to C
    'F': {'North': 'D', 'East': 'G'},
    'G': {'East': 'H', 'West': 'F'},
    'H': {'South': 'I', 'West': 'G'},
    'I': {'North': 'H', 'East': 'J'},
    'J': {'East': 'K', 'West': 'I'},
    'K': {'West': 'J', 'South': 'L'}, # food — exits: return West or dead end South
    'L': {'North': 'K'},               # dead end — only exit is back to K
}

ACTIONS      = {0: 'North', 1: 'East', 2: 'South', 3: 'West'}
N_ACTIONS    = 4
FOOD_REWARD  = 1.0
WALL_PENALTY = -0.05

# Optimal action per node (for policy scoring — oracle only, never fed to brain)
OPTIMAL_ACTION = {
    'A': 1,   # East
    'B': 1,   # East
    'C': 2,   # South → E★
    'D': 2,   # South
    'E': 0,   # North (return from food)
    'F': 1,   # East
    'G': 1,   # East
    'H': 2,   # South
    'I': 1,   # East
    'J': 1,   # East → K★
    'K': 3,   # West (return from food)
    'L': 0,   # North (escape dead end)
}


class World3:
    """
    12-node soundscape navigation environment with frequency reuse.

    Same interface as World — drop-in replacement for brain_in_world2.
    """

    def __init__(self, seed: int = 42):
        self._rng    = np.random.default_rng(seed)
        self._node   = HOME_NODE
        self.t       = 0
        self.food_count  = 0
        self.wall_count  = 0
        self.total_steps = 0
        self.node_visit_counts = {n: 0 for n in NODES}
        self.action_counts     = [0] * N_ACTIONS

    def reset(self):
        self._node   = HOME_NODE
        self.t       = 0
        self.food_count  = 0
        self.wall_count  = 0
        self.total_steps = 0
        self.node_visit_counts = {n: 0 for n in NODES}
        self.action_counts     = [0] * N_ACTIONS
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
    def current_node(self):   return self._node
    @property
    def current_freq(self):   return NODES[self._node][0]
    @property
    def current_freq_idx(self): return NODES[self._node][1]

    def food_rate(self, window=None):
        steps = window if window else self.total_steps
        return 0.0 if steps == 0 else self.food_count / steps * 100

    def wall_rate(self):
        return 0.0 if self.total_steps == 0 else self.wall_count / self.total_steps

    def summary(self):
        print(f"\n  World3 — step {self.t}")
        print(f"  Current node: {self._node}  ({self.current_freq:.1f} Hz)")
        print(f"  Food: {self.food_count}  ({self.food_rate():.2f}/100)")
        print(f"  Walls: {self.wall_rate():.1%}")
        print(f"  Visits: { {n: self.node_visit_counts[n] for n in NODES} }")