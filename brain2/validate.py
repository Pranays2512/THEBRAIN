#!/usr/bin/env python3
"""
validate.py — the promotion gate.

Basic infra for the prototype -> validate -> promote workflow. Run this before
moving any capability up the hierarchy: it exercises every validated mechanism
and checks the headline results still hold. Nothing gets promoted if this
regresses.

    python3 validate.py            # core correctness gate (a few minutes)
    python3 validate.py --full     # also run the slow performance sweeps

Each check runs a script and asserts a robust marker of its known-good result.
"""

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (name, script, [regex asserts on output]); empty asserts => just must run clean
CORE = [
    ("knowledge engine (HARDENED)", "tests/test_knowledge_engine.py",
     [r"Knowledge layer: READY"]),
    ("reasoning engine (HARDENED)", "tests/test_reasoning_engine.py",
     [r"Reasoning layer: READY"]),
    ("search engine (HARDENED)", "tests/test_search_engine.py",
     [r"Search engine: READY"]),
    ("planning engine (HARDENED)", "tests/test_planning_engine.py",
     [r"Planning layer: READY"]),
    ("learned guidance (HARDENED)", "tests/test_learned_guidance.py",
     [r"Learned guidance: READY"]),
    ("synthesis engine (HARDENED)", "tests/test_synthesis_engine.py",
     [r"Synthesis engine: READY"]),
    ("consolidation (HARDENED)", "tests/test_consolidation.py",
     [r"Consolidation: READY"]),
    ("knowledge+reasoning", "reasoning_suite.py",
     [r"transitive_derived_acc': 1\.0", r"relation_composition_acc': 1\.0"]),
    ("dream consolidation", "component_validation.py",
     [r"winner': 'faithful'"]),
    ("general search (algebra+bridge)", "tree_reason.py",
     [r"ANSWER: x = 6", r"MINIMUM TOTAL TIME"]),
    ("general search (domains)", "tree_domains.py",
     [r"N-QUEENS", r"reached 4L", r"derived"]),
    ("knowledge+search joined", "brain_planner.py",
     [r"forge.*->.*sword|forge: use"]),
    ("verifiable synthesis", "program_synth.py",
     [r"SYNTHESIZED program:\s+initials"]),
    ("dual cognition", "dual_process.py",
     [r"faster once practiced|dual-process"]),
    ("unit tests", "tests/run_all.py",
     [r"1[2-9] passed"]),
]

FULL = [
    ("learned heuristic", "tree_learn.py", [r"LEARNED heuristic"]),
    ("guided synthesis", "program_synth_guided.py", [r"guided search explores"]),
    ("policy synthesis", "program_synth_policy.py", [r"goal-conditioned POLICY"]),
    ("tree policy", "program_synth_tree.py", [r"DECISION-TREE policy"]),
]


def run_check(name, script, asserts):
    t = time.time()
    try:
        r = subprocess.run([PY, script], cwd=HERE, capture_output=True,
                           text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - t, "exceeded 1200s"
    out = r.stdout + r.stderr
    for a in asserts:
        if not re.search(a, out):
            return "FAIL", time.time() - t, f"missing /{a}/"
    if not asserts and r.returncode != 0:
        return "FAIL", time.time() - t, f"exit {r.returncode}"
    return "PASS", time.time() - t, ""


def main():
    checks = CORE + (FULL if "--full" in sys.argv else [])
    print("=== brain2 validation gate ===")
    print("(run before promoting any prototype up the hierarchy)\n")
    results = []
    for name, script, asserts in checks:
        status, dt, note = run_check(name, script, asserts)
        results.append(status)
        print(f"  [{status:7s}] {name:26s} {dt:5.0f}s   {note}")
    npass = results.count("PASS")
    print(f"\n{npass}/{len(results)} validations passed", end="")
    print("  — GATE OPEN, safe to promote" if npass == len(results)
          else "  — GATE CLOSED, fix before promoting")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
