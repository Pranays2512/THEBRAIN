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

    # 9b. MULTI-PARENT transitive closure (the promotion).
    # A concept with MANY parents — binding memory's single-best recall can't
    # do this; closure over the fact graph can.
    tax = ReasoningEngine()
    for s, o in [("dog", "pet"), ("dog", "mammal"), ("dog", "canine"),
                 ("pet", "animal"), ("mammal", "animal"), ("animal", "organism")]:
        tax.learn(s, "isa", o)
    tax.set_transitive("isa")

    anc = tax.derive_all("dog", "isa")
    check("closure finds ALL ancestors (multi-parent)",
          set(anc) == {"pet", "mammal", "canine", "animal", "organism"})

    reach, path = tax.reaches("dog", "isa", "animal")
    check("reaches target through an intermediate", reach)
    check("path is a real chain dog..animal",
          path and path[0] == "dog" and path[-1] == "animal" and "animal" in path)
    check("shortest path (dog->pet->animal, not longer)", len(path) == 3)

    reach2, _ = tax.reaches("dog", "isa", "vehicle")
    check("underivable membership -> False", not reach2)

    # closure must terminate on a cycle
    cyc = ReasoningEngine()
    cyc.learn("a", "isa", "b"); cyc.learn("b", "isa", "a")
    cyc.set_transitive("isa")
    anc_c = cyc.derive_all("a", "isa")     # must return, not hang
    check("closure terminates on cycle (self excluded)", set(anc_c) == {"b"})

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
