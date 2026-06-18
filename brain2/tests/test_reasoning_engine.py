#!/usr/bin/env python3
"""
test_reasoning_engine.py — hardening tests for the Reasoning layer.

Covers composition rules, nested rules, transitive relations, cycle safety,
rule idempotency, persistence, validation — the inference layer above Knowledge.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from knowledge_engine import KnowledgeError
from reasoning_engine import ReasoningEngine

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def family():
    re = ReasoningEngine()
    for s, r, o in [("tom", "parent", "sam"), ("sam", "parent", "kid"),
                    ("kid", "parent", "baby")]:
        re.learn(s, r, o)
    re.add_rule("parent", "parent", "grandparent")
    re.add_rule("parent", "grandparent", "great_grandparent")
    return re


def run():
    print("\nReasoningEngine — hardening tests")
    re = family()

    # 1. direct relation still works
    ans, _ = re.ask("tom", "parent")
    check("direct relation", ans == "sam")

    # 2. composition rule (NEW vs Knowledge layer)
    ans, why = re.ask("tom", "grandparent")
    check("composition: parent∘parent -> grandparent", ans == "kid")
    check("explanation cites the rule", why and "rule" in why and "sam" in why)

    # 3. nested rule (great_grandparent uses grandparent)
    ans, _ = re.ask("tom", "great_grandparent")
    check("nested rule composition", ans == "baby")

    # 4. no derivation -> (None, None)
    ans, why = re.ask("baby", "grandparent")
    check("no derivation -> None", ans is None and why is None)

    # 5. transitive relation
    re2 = ReasoningEngine()
    for a, b in [("a", "b"), ("b", "c"), ("c", "d")]:
        re2.learn(a, "before", b)
    re2.set_transitive("before")
    ans, _ = re2.ask("a", "before")
    check("transitive relation chains to end", ans == "d")

    # 6. cycle safety (rule + cyclic facts must terminate)
    re3 = ReasoningEngine()
    re3.learn("x", "r", "y")
    re3.learn("y", "r", "x")
    re3.add_rule("r", "r", "loop")
    ans, _ = re3.ask("x", "loop")          # must return, not hang
    check("cycle terminates", ans in ("x", "y", None))

    # 7. rule idempotency
    n = len(re.rules)
    re.add_rule("parent", "parent", "grandparent")
    check("duplicate rule ignored", len(re.rules) == n)

    # 8. rule input validation
    try:
        re.add_rule("", "parent", "x")
        check("reject empty rule term", False)
    except KnowledgeError:
        check("reject empty rule term", True)

    # 9. knows()
    check("knows() via composition", re.knows("tom", "grandparent", "kid"))
    check("knows() false when underivable", not re.knows("tom", "grandparent", "tom"))

    # 10. persistence round-trip (facts + rules)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "re.json")
        re.save(path)
        re_l = ReasoningEngine.load(path)
        a1, _ = re.ask("tom", "great_grandparent")
        a2, _ = re_l.ask("tom", "great_grandparent")
        check("save/load preserves rules+facts", a1 == a2 == "baby")

    print(f"\nReasoning layer: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
