#!/usr/bin/env python3
"""
planning_engine.py — hardened "Knowledge + search joined" (milestone #4).

The first layer that composes two already-hardened layers: actions and their
preconditions/effects are stored as facts in the KnowledgeEngine, and the
hardened search engine plans a valid ordering to reach a goal — then explains
it. Genuine multi-precondition planning (an action can require several inputs
from different branches), not single-relation chaining.

    pe = PlanningEngine()
    pe.define_action("smelt", requires=["ore"],          produces=["iron"])
    pe.define_action("chop",  requires=["axe"],          produces=["wood"])
    pe.define_action("forge", requires=["iron", "wood"], produces=["sword"])
    plan = pe.plan(have=["ore", "axe"], goal="sword")
    plan.found     -> True
    plan.explain() -> chop -> smelt -> forge, with data-flow

Actions live in the KnowledgeEngine (persisted, validated); the plan comes from
the optimal, deterministic search engine. Both hardened, with their own tests.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engines.knowledge.knowledge_engine import KnowledgeEngine, KnowledgeError
from engines.reasoning.tree_reason import SearchProblem, solve


class PlanningError(ValueError):
    """Invalid planning input."""


class Plan:
    def __init__(self, found, steps, goal):
        self.found = found
        self.steps = steps          # list of dicts: {action, uses, makes}
        self.goal = goal

    def explain(self):
        if not self.found:
            return f"  no plan reaches {self.goal} from what is known"
        lines = []
        for i, s in enumerate(self.steps, 1):
            uses = sorted(s["uses"]) or ["-"]
            lines.append(f"  {i}. {s['action']}: use {uses} -> get {sorted(s['makes'])}")
        return "\n".join(lines)

    def __repr__(self):
        return f"Plan(found={self.found}, steps={[s['action'] for s in self.steps]})"


class _CraftPlan(SearchProblem):
    def __init__(self, engine, have, goal):
        self.e = engine
        self.have = have
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
        for a in sorted(self.e.actions()):
            req, prod = self.e.requires(a), self.e.produces(a)
            if req <= state and not (prod <= state):     # applicable AND progresses
                yield (a, state | prod, 1)


class PlanningEngine:
    def __init__(self, kb=None):
        self.kb = kb or KnowledgeEngine()

    # ── defining actions (stored as facts) ───────────────────────────────────
    def define_action(self, name, requires=(), produces=()):
        name = KnowledgeEngine._norm(name)
        produces = [KnowledgeEngine._norm(p) for p in produces]
        requires = [KnowledgeEngine._norm(r) for r in requires]
        if not produces:
            raise PlanningError(f"action {name!r} must produce something")
        for r in requires:
            self.kb.learn(name, "requires", r)
        for p in produces:
            self.kb.learn(name, "produces", p)

    def actions(self):
        return {s for (s, r, _) in self.kb.facts if r in ("requires", "produces")}

    def requires(self, action):
        return frozenset(o for (s, r, o) in self.kb.facts
                         if s == action and r == "requires")

    def produces(self, action):
        return frozenset(o for (s, r, o) in self.kb.facts
                         if s == action and r == "produces")

    # ── planning ─────────────────────────────────────────────────────────────
    def plan(self, have, goal, max_nodes=200_000):
        if isinstance(have, str):
            raise PlanningError("have must be a collection of items, not a string")
        have = frozenset(KnowledgeEngine._norm(h) for h in have)
        goal = KnowledgeEngine._norm(goal)

        path, _, _ = solve(_CraftPlan(self, have, goal), max_nodes)
        if path is None:
            return Plan(False, [], goal)
        steps = []
        for action, _ in path:
            steps.append({"action": action,
                          "uses": set(self.requires(action)),
                          "makes": set(self.produces(action))})
        return Plan(True, steps, goal)

    # ── persistence (actions are facts) ──────────────────────────────────────
    def save(self, path):
        self.kb.save(path)

    @classmethod
    def load(cls, path):
        return cls(kb=KnowledgeEngine.load(path))


def _demo():
    pe = PlanningEngine()
    pe.define_action("smelt", requires=["ore"], produces=["iron"])
    pe.define_action("chop", requires=["axe"], produces=["wood"])
    pe.define_action("forge", requires=["iron", "wood"], produces=["sword"])
    print("PlanningEngine demo: make a sword from ore + axe")
    plan = pe.plan(have=["ore", "axe"], goal="sword")
    print(plan.explain())


if __name__ == "__main__":
    _demo()
