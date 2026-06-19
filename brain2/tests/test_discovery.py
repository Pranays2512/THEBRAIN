#!/usr/bin/env python3
"""
test_discovery.py — the discovery cycle: two proposers, one verifier, one memory.

Pins that induction and analogy both feed the same verify-before-promote gate,
that unconfirmed transfers are NOT promoted, and that the engine reasons over
everything it discovered.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from discovery import DiscoverySystem

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


WATER = [("pump", "increases", "flow"), ("pipe", "resists", "flow"),
         ("flow", "depends_on", "pressure"), ("pump", "raises", "pressure")]
ELEC = [("battery", "increases", "current"), ("resistor", "resists", "current"),
        ("current", "depends_on", "voltage")]


def episodes():
    train = [["switch", "battery", "current"]] * 3 + [["short", "spark"]] * 3
    test = [["switch", "battery", "current"]] * 2 + [["short", "spark"]] * 2
    return train, test


def run():
    print("\nDiscoverySystem — induction + analogy, one verifier")
    train, test = episodes()

    # confirmed transfer
    ds = DiscoverySystem()
    mined, ok, no = ds.discover(train, test, WATER, ELEC,
                                experiments=[["battery", "voltage"]] * 3)
    mset = {(r.a, r.b) for r in mined}
    check("induction mined switch -> battery", ("switch", "battery") in mset)
    check("induction mined battery -> current", ("battery", "current") in mset)
    check("reasons transitively over mined rules",
          ds.engine.reaches("switch", "leads_to", "current")[0])

    okset = {(s, r, o) for s, r, o, _ in ok}
    check("analogy transfer confirmed by experiment",
          ("battery", "raises", "voltage") in okset)
    check("confirmed transfer is usable in reasoning",
          ds.engine.ask("battery", "raises")[0] == "voltage")

    # SAME transfer, but NO experiment -> must not be promoted
    ds2 = DiscoverySystem()
    _, ok2, no2 = ds2.discover(train, test, WATER, ELEC, experiments=[])
    noset = {(s, r, o) for s, r, o, _ in no2}
    check("unconfirmed transfer kept as hypothesis", ("battery", "raises", "voltage") in noset)
    check("unconfirmed transfer NOT promoted", ds2.engine.ask("battery", "raises")[0] is None)
    check("unconfirmed transfer not in the confirmed list",
          ("battery", "raises", "voltage") not in {(s, r, o) for s, r, o, _ in ok2})

    print(f"\nDiscovery: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
