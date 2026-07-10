#!/usr/bin/env python3
"""
brain_planner.py — the two engines in one loop.

Binding memory KNOWS things (facts taught online). Tree search REASONS to a
goal. This joins them: the brain stores a world model as facts, and the search
plans over that model by *querying the brain* — then explains the plan.

The world is taught as facts:
    smelt requires ore     smelt produces iron
    chop  requires axe     chop  produces wood
    forge requires iron    forge requires wood    forge produces sword

Then: "you have ore and axe, get a sword." The planner asks the brain what each
action needs and makes (via query_all), searches for an order that works, and
reads back the plan. This is genuine multi-precondition planning — forge needs
iron AND wood, which come from two different branches — so it is strictly more
than transitive closure.

Honest about the join: facts live in the real binding memory and are retrieved
through the brain's query_all; the search is tree_reason's general engine. The
brain is the knowledge; the search is the reasoning over it.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import brain2
from core.reasoning.tree_reason import SearchProblem, solve

N_DIMS = 64


class FactWorld:
    """Holds the brain + a token<->vector registry, teaches and reads facts."""

    def __init__(self):
        self.b = brain2.Brain(som_rows=32, som_cols=32, n_dims=N_DIMS, hidden_dim=128, seed=1)
        self.vecs = {}
        self.actions = set()

    def vec(self, token):
        if token not in self.vecs:
            h = abs(hash(token)) % (2**32)
            self.vecs[token] = np.random.default_rng(h).standard_normal(N_DIMS).astype(np.float32)
        return self.vecs[token]

    def teach(self, subj, rel, obj):
        self.b.bind_triple(self.vec(subj), self.vec(rel), self.vec(obj))
        if rel in ("requires", "produces"):
            self.actions.add(subj)

    def _decode(self, v):
        v = np.asarray(v, dtype=np.float32)
        if v.size == 0 or np.linalg.norm(v) < 1e-8:
            return None
        toks = sorted(self.vecs)
        M = np.stack([self.vecs[t] for t in toks])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        s = M @ (v / np.linalg.norm(v))
        return toks[int(np.argmax(s))]

    def facts_of(self, action):
        """Retrieve all (relation, object) facts for an action FROM THE BRAIN."""
        flat = self.b.binding.query_all(self.vec(action), 0.5)   # [rel0, obj0, rel1, obj1, ...]
        out = []
        for i in range(0, len(flat) - 1, 2):
            rel = self._decode(flat[i])
            obj = self._decode(flat[i + 1])
            if rel and obj:
                out.append((rel, obj))
        return out

    def requires(self, action):
        return frozenset(o for r, o in self.facts_of(action) if r == "requires")

    def produces(self, action):
        return frozenset(o for r, o in self.facts_of(action) if r == "produces")


class CraftPlan(SearchProblem):
    """Plan over the brain's learned world: reach `goal` from items `have`."""

    def __init__(self, world, have, goal):
        self.w = world
        self.have = frozenset(have)
        self.goal = goal

    def initial(self):
        return self.have

    def is_goal(self, state):
        return self.goal in state

    def key(self, state):
        return state

    def heuristic(self, state):
        return 0 if self.goal in state else 1

    def moves(self, state):
        for a in sorted(self.w.actions):
            req, prod = self.w.requires(a), self.w.produces(a)
            if req <= state and not (prod <= state):     # applicable AND makes progress
                label = f"{a}: use {sorted(req)} -> get {sorted(prod)}"
                yield (label, state | prod, 1)


def plan_and_show(world, have, goal):
    prob = CraftPlan(world, have, goal)
    path, cost, nodes = solve(prob)
    print(f"  have {sorted(have)}, want {goal}")
    if path is None:
        print("  -> no plan found from what it knows.\n")
        return
    for i, (label, state) in enumerate(path, 1):
        print(f"    {i}. {label}")
    print(f"  -> achieved {goal} in {len(path)} steps "
          f"(searched {nodes} states over its OWN learned facts)\n")


def main():
    print("=== brain_planner — it knows facts, it reasons to a goal, it explains ===\n")
    w = FactWorld()
    world = [
        ("smelt", "requires", "ore"), ("smelt", "produces", "iron"),
        ("chop", "requires", "axe"),  ("chop", "produces", "wood"),
        ("forge", "requires", "iron"), ("forge", "requires", "wood"),
        ("forge", "produces", "sword"),
    ]
    print("Teaching the world (stored online in binding memory):")
    for s, r, o in world:
        w.teach(s, r, o)
        print(f"    {s} {r} {o}")
    print()

    print("Goal: make a sword from ore and an axe.")
    print("(forge needs iron AND wood — two branches — so this is real planning,")
    print(" not a single relation chain)\n")
    plan_and_show(w, ["ore", "axe"], "sword")

    print("Online learning changes the reasoning: teach a shortcut, replan.")
    print("    buy requires coin    buy produces sword")
    w.teach("buy", "requires", "coin")
    w.teach("buy", "produces", "sword")
    plan_and_show(w, ["coin"], "sword")

    print("And it only plans from what it actually knows:")
    plan_and_show(w, ["ore"], "sword")   # no axe -> no wood -> cannot forge


if __name__ == "__main__":
    main()
