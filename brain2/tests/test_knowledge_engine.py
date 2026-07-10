#!/usr/bin/env python3
"""
test_knowledge_engine.py — hardening tests for the Knowledge layer.

Covers what the demo never did: unknowns, idempotency, persistence round-trip,
input validation, cycles, multi-relation isolation, distractor robustness.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engines.knowledge.knowledge_engine import KnowledgeEngine, KnowledgeError

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def org():
    kb = KnowledgeEngine()
    for s, r, o in [("alice", "manages", "bob"), ("bob", "manages", "carol"),
                    ("carol", "manages", "dave"), ("dave", "manages", "erin")]:
        kb.learn(s, r, o)
    return kb


def run():
    print("\nKnowledgeEngine — hardening tests")

    # 1. direct fact
    kb = org()
    obj, conf = kb.ask("alice", "manages")
    check("direct fact retrieved", obj == "bob" and conf > 0.9)

    # 2. transitive derivation (never stored)
    obj, _ = kb.ask("alice", "manages", hops=3)
    check("3-hop derived (alice -> dave)", obj == "dave")
    check("derive chain correct", kb.derive("alice", "manages") ==
          ["alice", "bob", "carol", "dave", "erin"])
    check("knows() transitive yes", kb.knows("alice", "manages", "erin"))
    check("knows() false for absent", not kb.knows("alice", "manages", "zoe"))

    # 3. unknown subject -> graceful, no crash
    obj, conf = kb.ask("nobody", "manages")
    check("unknown subject -> (None, 0)", obj is None and conf == 0.0)
    check("explain on dead end -> None", kb.explain("erin", "manages") is None)

    # 4. idempotency
    n_before = len(kb.all_facts())
    added = kb.learn("alice", "manages", "bob")
    check("re-learning a fact is a no-op", added is False and len(kb.all_facts()) == n_before)

    # 5. input validation
    for bad in ("", "   ", None, 5):
        try:
            kb.learn(bad, "rel", "obj")
            check(f"reject invalid token {bad!r}", False)
        except KnowledgeError:
            check(f"reject invalid token {bad!r}", True)

    # 6. multi-relation isolation (different relations don't cross)
    kb2 = KnowledgeEngine()
    kb2.learn("rome", "capital_of", "italy")
    kb2.learn("rome", "older_than", "milan")
    cap, _ = kb2.ask("rome", "capital_of")
    old, _ = kb2.ask("rome", "older_than")
    check("relations stay isolated", cap == "italy" and old == "milan")

    # 7. cycle does not loop forever
    kb3 = KnowledgeEngine()
    kb3.learn("a", "r", "b")
    kb3.learn("b", "r", "a")
    chain = kb3.derive("a", "r")          # must terminate
    check("cycle terminates", chain[0] == "a" and len(chain) <= 3)

    # 8. distractor robustness (many unrelated facts present)
    kb4 = org()
    for i in range(300):
        kb4.learn(f"x{i}", "noise", f"y{i}")
    obj, _ = kb4.ask("alice", "manages", hops=3)
    check("correct under 300 distractors", obj == "dave")

    # 9. persistence round-trip
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "kb.json")
        kb.save(path)
        kb_l = KnowledgeEngine.load(path)
        check("save/load preserves facts", kb_l.all_facts() == kb.all_facts())
        o1, _ = kb.ask("alice", "manages", hops=3)
        o2, _ = kb_l.ask("alice", "manages", hops=3)
        check("save/load preserves answers", o1 == o2 == "dave")

    print(f"\nKnowledge layer: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
