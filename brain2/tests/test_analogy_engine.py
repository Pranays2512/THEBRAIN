#!/usr/bin/env python3
"""
test_analogy_engine.py — structure mapping by relational role (bounded idea #2).

Pins that the correspondence is found from shared-relation structure, that a
missing analog is transferred as a prediction, and that structure-poor or
ambiguous domains honestly yield no (or partial) mapping.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.events.analogy_engine import AnalogyEngine

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nAnalogyEngine — structure mapping")
    ae = AnalogyEngine()

    water = [("pump", "increases", "flow"), ("pipe", "resists", "flow"),
             ("flow", "depends_on", "pressure"), ("pump", "raises", "pressure")]
    elec = [("battery", "increases", "current"), ("resistor", "resists", "current"),
            ("current", "depends_on", "voltage")]
    mapping, transfers = ae.map_domains(water, elec)

    # 1. correspondence aligns relational roles (not surface names)
    check("pump ~ battery", mapping.get("pump") == "battery")
    check("flow ~ current", mapping.get("flow") == "current")
    check("pipe ~ resistor", mapping.get("pipe") == "resistor")
    check("pressure ~ voltage", mapping.get("pressure") == "voltage")

    # 2. the missing analog is transferred as a prediction
    preds = {(s, r, o) for s, r, o, _ in transfers}
    check("predicts 'battery raises voltage'", ("battery", "raises", "voltage") in preds)
    check("does not re-predict facts already present",
          ("battery", "increases", "current") not in preds)

    # 3. no shared structure -> no mapping
    m2, t2 = ae.map_domains([("a", "likes", "b")], [("c", "eats", "d")])
    check("no shared relations -> empty mapping", m2 == {} and t2 == [])

    # 4. ambiguous correspondence is left unmapped (honest)
    m3, _ = ae.map_domains([("a", "r", "b")],
                           [("c", "r", "d"), ("e", "r", "d")])
    check("ambiguous source object not mapped", "a" not in m3)
    check("unambiguous object still mapped", m3.get("b") == "d")

    print(f"\nAnalogy engine: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
