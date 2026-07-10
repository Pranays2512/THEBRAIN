#!/usr/bin/env python3
"""
dimensional_verify.py — a new VERIFIER: dimensional analysis.

The architecture's reach = the reach of its verifiers. This adds one: units. A
formula must be DIMENSIONALLY consistent (force is kg·m/s², not kg+m/s²), checked
by unit algebra over base dimensions [Mass, Length, Time]. It's a SECOND,
independent gate on induced policies: a formula that fits the data numerically but
is dimensional nonsense (a coincidental overfit) gets rejected — without running a
single extra data point.

  units("mass*accel") = (1,1,-2) = force  -> consistent
  units("mass+accel") = mismatch          -> INVALID (can't add kg and m/s²)
  units("mass*speed") = (1,1,-1) = momentum != force -> rejected for 'force'

Pairs with policy_induction: induce by fit, then KEEP only if dimensionally sound.

    python3 dimensional_verify.py
"""

# base dimensions: (Mass, Length, Time)
UNITS = {
    "mass": (1, 0, 0), "accel": (0, 1, -2), "speed": (0, 1, -1),
    "dist": (0, 1, 0), "height": (0, 1, 0), "time": (0, 0, 1),
    "area": (0, 2, 0), "volume": (0, 3, 0), "gravity": (0, 1, -2),
}
TARGET = {
    "force": (1, 1, -2), "momentum": (1, 1, -1), "energy": (1, 2, -2),
    "pressure": (1, -1, -2), "density": (1, -3, 0), "work": (1, 2, -2),
}


class DimError(Exception):
    pass


def _add(a, b): return tuple(x + y for x, y in zip(a, b))
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _scale(a, n): return tuple(int(x * n) for x in a)


def units(e):
    """Dimension vector of an expression, or raise DimError on +/- of unlike units."""
    if isinstance(e, (int, float)):
        return (0, 0, 0)                              # dimensionless constant
    if isinstance(e, str):
        if e not in UNITS:
            raise DimError(f"unknown quantity '{e}'")
        return UNITS[e]
    op, a, b = e[0], units(e[1]), (units(e[2]) if len(e) > 2 else None)
    if op == "*":
        return _add(a, b)
    if op == "/":
        return _sub(a, b)
    if op in ("+", "-"):
        if a != b:
            raise DimError(f"cannot {op} unlike units {a} and {b}")
        return a
    if op == "^":
        return _scale(a, e[2])
    raise DimError(f"unknown op {op}")


def dimensionally_sound(expr, target):
    """True iff expr's units match the target quantity's units (and are valid)."""
    try:
        return units(expr) == TARGET[target]
    except DimError:
        return False


def _demo():
    print("=== dimensional_verify — units as a second gate on formulas ===\n")
    cases = [
        ("force",  ("*", "mass", "accel"),  "the right law"),
        ("force",  ("+", "mass", "accel"),  "dimensional nonsense (kg + m/s²)"),
        ("force",  ("*", "mass", "speed"),  "fits some data by luck, but = momentum"),
        ("energy", ("*", 0.5, ("*", "mass", ("^", "speed", 2))), "½mv²"),
        ("pressure", ("/", ("*", "mass", "accel"), "area"), "force/area"),
    ]
    for target, expr, note in cases:
        ok = dimensionally_sound(expr, target)
        try:
            u = units(expr)
        except DimError as e:
            u = f"INVALID ({e})"
        print(f"  {target:9s} <- {note}")
        print(f"      units {u}  ->  {'SOUND ✓' if ok else 'REJECTED ✗'}\n")
    print("  a formula must fit the data AND be dimensionally sound — two independent")
    print("  gates. The unit gate catches coincidental overfits for free.")


if __name__ == "__main__":
    _demo()
