#!/usr/bin/env python3
"""
test_policy_engine.py — regression for the C++ PolicyEngine (the ported crisp
reasoner). Run with the interpreter that can import brain2 (the build's venv):

    venv2/bin/python3 test_policy_engine.py

Checks the C++ engine reproduces the Python means_ends/policy_proposer results:
solve + means-ends decomposition, conjecture->verify->admit, the groundability
proposer, and the bridge to the Python ReasoningEngine as a fact source.
"""

import brain2


def base_mem():
    m = brain2.PolicyMemory()
    m.add("force", ["mass", "accel"], ["*", "mass", "accel"])
    m.add("energy", ["mass", "speed"], ["*", 0.5, ["*", "mass", ["^", "speed", 2]]])
    m.add("power", ["force", "speed"], ["*", "force", "speed"])
    return m


def test_solve_and_learn():
    facts = {("rocket", "mass"): 1000.0, ("rocket", "accel"): 20.0,
             ("rocket", "speed"): 300.0}
    eng = brain2.PolicyEngine(base_mem(), lambda e, r: facts.get((e, r)), True)
    assert eng.solve("rocket", "force") == 20000.0
    assert eng.solve("rocket", "energy") == 45000000.0
    assert eng.solve("rocket", "power") == 6000000.0
    assert eng.solve("rocket", "wisdom") is None          # honest unknown
    assert eng.learn("power", "rocket") is True           # conjecture verified + admitted
    print("solve + conjecture/verify: OK")


def test_proposer():
    m = brain2.PolicyMemory()
    m.add("force", ["mass", "accel"], ["*", "mass", "accel"])
    m.add("energy", ["mass", "speed"], ["*", 0.5, ["*", "mass", ["^", "speed", 2]]])
    m.add("power", ["energy", "time"], ["/", "energy", "time"])   # dead (no time)
    m.add("power", ["force", "speed"], ["*", "force", "speed"])   # live
    facts = {("rocket", "mass"): 1000.0, ("rocket", "accel"): 20.0,
             ("rocket", "speed"): 300.0}
    fn = lambda e, r: facts.get((e, r))
    blind = brain2.PolicyEngine(m, fn, False); blind.solve("rocket", "power")
    prop = brain2.PolicyEngine(m, fn, True);   prop.solve("rocket", "power")
    assert prop.work < blind.work, (prop.work, blind.work)
    print(f"proposer prunes search: blind {blind.work} -> proposer {prop.work}: OK")


def test_bridge_to_python_kb():
    from reasoning_engine import ReasoningEngine
    kb = ReasoningEngine()
    for s, r, o in [("rocket", "mass", "1000"), ("rocket", "accel", "20"),
                    ("rocket", "speed", "300")]:
        kb.learn(s, r, o)

    def fact(e, r):
        ans, _ = kb.ask(e, r)
        return float(ans) if ans is not None else None

    eng = brain2.PolicyEngine(base_mem(), fact, True)
    assert eng.solve("rocket", "power") == 6000000.0
    print("bridge C++ engine <- Python ReasoningEngine facts: OK")


if __name__ == "__main__":
    test_solve_and_learn()
    test_proposer()
    test_bridge_to_python_kb()
    print("\nall PolicyEngine tests passed (crisp reasoner now native in brain2.so)")
