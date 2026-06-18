#!/usr/bin/env python3
"""
test_synthesis_engine.py — hardening tests for Verifiable synthesis (#6).

Pins the defining property: every returned program is CORRECT on the spec by
construction, and it generalizes. Plus composition, honest failure, shortest
program, determinism, and validation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synthesis_engine import SynthesisEngine, SynthesisError

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nSynthesisEngine — hardening tests")
    se = SynthesisEngine()

    # 1. basic synthesis
    r = se.synthesize([("John Smith", "JS"), ("Mary Jane", "MJ")])
    check("synthesizes initials", r.found and r.program == ("initials",))

    # 2. verification: reproduces ALL given examples (by construction)
    ex = [("John Smith", "JOHN"), ("bob dylan", "BOB"), ("Ada Lovelace", "ADA")]
    r = se.synthesize(ex)
    check("verified on all training examples", r.found and all(r.apply(i) == o for i, o in ex))

    # 3. multi-op composition
    check("composes 2-op program", len(r.program) == 2)

    # 4. generalization to held-out inputs
    held = [("grace hopper", "GRACE"), ("kit kat", "KIT")]
    check("generalizes to new inputs", all(r.apply(i) == o for i, o in held))

    # 5. honest failure outside the DSL
    r_no = se.synthesize([("John Smith", "Smith, John")])
    check("honest failure outside DSL", not r_no.found)
    try:
        r_no.apply("x"); check("apply on not-found raises", False)
    except SynthesisError:
        check("apply on not-found raises", True)

    # 6. shortest program (search optimality): lower is 1 op, not 2
    r_low = se.synthesize([("ABC", "abc"), ("XY", "xy")])
    check("returns shortest program", r_low.found and len(r_low.program) == 1)

    # 7. determinism
    a = se.synthesize([("John Smith", "JS")])
    b = se.synthesize([("John Smith", "JS")])
    check("deterministic synthesis", a.program == b.program)

    # 8. validation
    for bad in ([], [("only_one",)], [("a", 5)], "notalist_pair"):
        try:
            se.synthesize(bad if bad != "notalist_pair" else [("a", "b"), bad])
            check(f"reject malformed {bad!r}", False)
        except SynthesisError:
            check(f"reject malformed {bad!r}", True)
    try:
        SynthesisEngine(max_len=0); check("reject bad max_len", False)
    except SynthesisError:
        check("reject bad max_len", True)

    print(f"\nSynthesis engine: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
