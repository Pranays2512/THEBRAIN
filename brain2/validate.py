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
    ("dual cognition (HARDENED)", "tests/test_dual_process.py",
     [r"Dual cognition: READY"]),
    ("semantic memory (NEW)", "tests/test_semantic_memory.py",
     [r"Semantic memory: READY"]),
    ("appraisal/emotion (NEW)", "tests/test_appraisal_engine.py",
     [r"Appraisal engine: READY"]),
    ("conversation loop (CAPSTONE)", "tests/test_conversation_engine.py",
     [r"Conversation loop: READY"]),
    ("learn-by-reading (NEW)", "tests/test_fact_extractor.py",
     [r"Fact extractor: READY"]),
    ("domain demo (read->reason->why)", "domain_demo.py",
     [r"great_grandparent ada", r"don.t know anyone named Zara"]),
    ("world knowledge (ConceptNet)", "world_demo.py",
     [r"dog -> pet -> animal", r"never heard of a zorblax"]),
    ("world chat (conversation)", "world_chat.py",
     [r"Yes . dog -> pet -> animal", r"don.t know anything about zorblax"]),
    ("causal how/why (chain)", "causal_demo.py",
     [r"leads to photosynthesis, which leads to sugar", r"It also helps",
      r"don.t know how rock works"]),
    ("word problem (arithmetic)", "word_math.py",
     [r"7 apples. \(10 - 3 = 7\)", r"9 marbles"]),
    ("query planner (decompose+compose)", "tests/test_query_planner.py",
     [r"Query planner: READY"]),
    ("calculus (differentiation)", "tests/test_calculus_engine.py",
     [r"Calculus engine: READY"]),
    ("physics (solve any variable)", "tests/test_physics_engine.py",
     [r"Physics engine: READY"]),
    ("algebra (solve for x)", "tests/test_algebra_engine.py",
     [r"Algebra engine: READY"]),
    ("integration (verified by inverse)", "tests/test_integral_engine.py",
     [r"Integral engine: READY"]),
    ("code generation (3 languages)", "tests/test_code_gen.py",
     [r"Code generator: READY"]),
    ("math parser (notation)", "tests/test_math_parser.py",
     [r"Math parser: READY"]),
    ("math chat (ask the engines)", "tests/test_math_chat.py",
     [r"Math chat: READY"]),
    ("brain chat (unified router)", "tests/test_brain_chat.py",
     [r"Brain chat: READY"]),
    ("inductive learning (mine+verify)", "tests/test_inductive_engine.py",
     [r"Inductive engine: READY"]),
    ("curiosity loop (idle learning)", "tests/test_curiosity_loop.py",
     [r"Curiosity loop: READY"]),
    ("analogy (structure mapping)", "tests/test_analogy_engine.py",
     [r"Analogy engine: READY"]),
    ("discovery (propose+verify+reason)", "tests/test_discovery.py",
     [r"Discovery: READY"]),
    ("neuro bridge (eyes/brain/mouth)", "tests/test_neuro_bridge.py",
     [r"Neuro bridge: READY"]),
    ("eval harness (trust metrics)", "tests/test_eval_harness.py",
     [r"Eval harness: READY"]),
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
     [r"\d\d passed"]),
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
