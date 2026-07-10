#!/usr/bin/env python3
"""
test_planning_engine.py — hardening tests for "Knowledge + search joined" (#4).

Multi-precondition planning over actions stored as facts: correctness, no-plan
handling, online change, distractors, validation, persistence, optimality.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engines.reasoning.planning_engine import PlanningEngine, PlanningError

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def craft():
    pe = PlanningEngine()
    pe.define_action("smelt", requires=["ore"], produces=["iron"])
    pe.define_action("chop", requires=["axe"], produces=["wood"])
    pe.define_action("forge", requires=["iron", "wood"], produces=["sword"])
    return pe


def run():
    print("\nPlanningEngine — hardening tests")
    pe = craft()

    # 1. multi-precondition plan (forge needs iron AND wood from two branches)
    plan = pe.plan(have=["ore", "axe"], goal="sword")
    actions = [s["action"] for s in plan.steps]
    check("plan found", plan.found)
    check("plan ends in forge", actions[-1] == "forge")
    check("plan includes both branches", {"smelt", "chop"} <= set(actions))

    # 2. optimality / minimality (no wasted steps)
    check("plan is minimal (3 steps)", len(plan.steps) == 3)

    # 3. goal already in hand -> empty plan
    plan0 = pe.plan(have=["sword"], goal="sword")
    check("already-have -> empty plan", plan0.found and len(plan0.steps) == 0)

    # 4. unreachable goal -> no plan
    plan_no = pe.plan(have=["ore"], goal="sword")   # no axe -> no wood -> no forge
    check("unreachable -> no plan", not plan_no.found)
    check("explain on no-plan is graceful", "no plan" in plan_no.explain())

    # 5. online change: add a shortcut, replan
    pe.define_action("buy", requires=["coin"], produces=["sword"])
    plan_buy = pe.plan(have=["coin"], goal="sword")
    check("online action enables new plan", plan_buy.found and
          [s["action"] for s in plan_buy.steps] == ["buy"])

    # 6. distractors: many irrelevant actions present
    pe2 = craft()
    for i in range(200):
        pe2.define_action(f"junk{i}", requires=[f"a{i}"], produces=[f"b{i}"])
    plan_d = pe2.plan(have=["ore", "axe"], goal="sword")
    check("correct under 200 distractor actions", plan_d.found and
          [s["action"] for s in plan_d.steps][-1] == "forge")

    # 7. validation
    try:
        pe.plan(have="ore", goal="sword"); check("reject string `have`", False)
    except PlanningError:
        check("reject string `have`", True)
    try:
        pe.define_action("noop", requires=["x"], produces=[]); check("reject no-produces action", False)
    except PlanningError:
        check("reject no-produces action", True)

    # 8. persistence round-trip
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "world.json")
        pe.save(path)
        pe_l = PlanningEngine.load(path)
        p1 = pe.plan(have=["ore", "axe"], goal="sword")
        p2 = pe_l.plan(have=["ore", "axe"], goal="sword")
        check("save/load preserves planning",
              p2.found and [s["action"] for s in p2.steps] == [s["action"] for s in p1.steps])

    print(f"\nPlanning layer: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
