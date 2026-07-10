#!/usr/bin/env python3
"""
test_physics_engine.py — apply laws, solve for any variable by isolation.

Pins that the engine rearranges a law for each variable correctly, shows the
formula, and refuses variables that aren't in the law.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.math.physics_engine import PhysicsEngine, PhysicsError

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nPhysicsEngine — apply laws, solve for any variable")
    pe = PhysicsEngine()
    pe.add_law("newton2", "F", ("*", "m", "a"))                  # F = m*a
    pe.add_law("speed", "v", ("/", "d", "t"))                    # v = d/t
    pe.add_law("kinetic", "KE", ("*", 0.5, ("*", "m", ("^", "v", 2))))

    # 1. solve each variable of F = m*a
    check("forward: F from m,a", pe.solve("newton2", "F", m=3, a=4)[0] == 12)
    check("rearrange: a from F,m", pe.solve("newton2", "a", F=12, m=3)[0] == 4)
    check("rearrange: m from F,a", pe.solve("newton2", "m", F=12, a=4)[0] == 3)

    # 2. the rearranged formula is shown
    _, steps = pe.solve("newton2", "a", F=12, m=3)
    check("formula shown: a = F/m", steps[0] == "a = F/m")

    # 3. division law, solve the denominator
    check("v=d/t -> solve t", pe.solve("speed", "t", d=100, v=20)[0] == 5)

    # 4. power law: KE = 1/2 m v^2 -> solve v (inverse of square)
    check("KE -> solve v (sqrt)", pe.solve("kinetic", "v", KE=100, m=2)[0] == 10)

    # 5. honest errors
    try:
        pe.solve("newton2", "q", F=1, m=1)
        check("reject variable not in law", False)
    except PhysicsError:
        check("reject variable not in law", True)
    try:
        pe.solve("nope", "F", m=1, a=1)
        check("reject unknown law", False)
    except PhysicsError:
        check("reject unknown law", True)

    print(f"\nPhysics engine: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
