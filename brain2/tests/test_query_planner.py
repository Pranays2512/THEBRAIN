#!/usr/bin/env python3
"""
test_query_planner.py — intent + slot-fill + compound decomposition + compose.

Pins the planner: a multi-part question becomes a STACK of structured sub-queries,
each answered over the reasoning engine, then composed into one reply.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faculties.query_planner import QueryPlanner

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def world():
    qp = QueryPlanner()
    for s in ("orange", "lemon", "strawberry"):
        qp.learn(s, "provides", "vitamin_c")
        qp.learn(s, "isa", "fruit")
    for s, o in [("vitamin_c", "immune_system"), ("immune_system", "fighting_infection"),
                 ("vitamin_c", "energy")]:
        qp.learn(s, "helps", o)
    qp.set_transitive("isa")
    qp.set_transitive("helps")
    return qp


def run():
    print("\nQueryPlanner — decompose -> reason -> compose")
    qp = world()

    # 1. slot-filling: a single question parses to one structured query
    q = qp.parse("in how many ways can we get vitamin C?")[0]
    check("intent recognized as question", q.intent == "question")
    check("quantifier = count", q.quantifier == "count")
    check("subject = vitamin_c (multi-word folded)", q.subject == "vitamin_c")
    check("relation inferred/mapped = provides", q.relation == "provides")

    # 2. count query executes over the engine
    a1 = qp.answer("in how many ways can we get vitamin C?")
    check("count answer says 3 ways", "3 ways" in a1)
    check("count lists the sources", "orange" in a1 and "lemon" in a1)

    # 3. which + category filter
    a2 = qp.answer("which fruit provides vitamin C?")
    check("which lists the fruits", "fruit" in a2.lower() and "strawberry" in a2)

    # 4. COMPOUND question -> stack of two -> composed
    qs = qp.parse("in how many ways can we get vitamin C? which fruit provides vitamin C?")
    check("compound decomposes into two sub-queries", len(qs) == 2)
    a3 = qp.answer("in how many ways can we get vitamin C? which fruit provides vitamin C?")
    check("composed answer covers both", "3 ways" in a3 and "Specifically" in a3)

    # 5. branching list
    a4 = qp.answer("in which ways does vitamin C help?")
    check("list enumerates the branches", "2 ways" in a4 and "energy" in a4)

    # 6. honest: unknown subject
    check("unknown subject -> honest",
          "can't answer" in qp.answer("how many ways can we get unobtainium?").lower())

    print(f"\nQuery planner: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
