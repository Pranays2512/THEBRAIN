#!/usr/bin/env python3
"""
test_brainql.py — Tests for BrainQL: the structured LLM↔Brain query language.

Tests cover:
  1. BrainQL parser (parse_bql / parse_bql_block)
  2. The generalization fix: HCL → acid → turns_litmus → red
  3. Auto-inheritance rule registration in ReasoningEngine.learn()
  4. BrainQLExecutor: all 8 ops
  5. End-to-end pipeline: StubClient → BrainQLEyes → BrainQL → Brain → BrainQLMouth
  6. BrainInterface unified entry point
  7. Regression: existing ReasoningEngine tests still pass

Run from brain2/:
    python3 -m tests.test_brainql
    python3 tests/test_brainql.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


# ── helpers ───────────────────────────────────────────────────────────────────

def check(condition, msg=""):
    if not condition:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  ✓ {msg}")


# ── 1. Parser tests ───────────────────────────────────────────────────────────

def test_parser():
    print("\n[1] BrainQL Parser")
    from engines.reasoning.brainql import parse_bql, parse_bql_block, BrainQLParseError

    # LOOKUP
    q = parse_bql("LOOKUP acid turns_litmus")
    check(q.op == "LOOKUP", "LOOKUP op")
    check(q.subj == "acid", "LOOKUP subj")
    check(q.rel == "turns_litmus", "LOOKUP rel")

    # CHAIN with hops
    q = parse_bql("CHAIN dog isa hops=5")
    check(q.op == "CHAIN", "CHAIN op")
    check(q.subj == "dog", "CHAIN subj")
    check(q.hops == 5, "CHAIN hops=5")

    # INHERIT (lowercase normalisation handled by executor, not parser)
    q = parse_bql("INHERIT HCL turns_litmus")
    check(q.op == "INHERIT", "INHERIT op")
    check(q.subj == "HCL", "INHERIT subj preserved")

    # TEACH
    q = parse_bql("TEACH HCL isa acid")
    check(q.op == "TEACH", "TEACH op")
    check(q.subj == "HCL" and q.rel == "isa" and q.obj == "acid", "TEACH fields")

    # TEACH_RULE
    q = parse_bql("TEACH_RULE isa turns_litmus -> turns_litmus")
    check(q.op == "TEACH_RULE", "TEACH_RULE op")
    check(q.prem1 == "isa" and q.prem2 == "turns_litmus" and q.concl == "turns_litmus", "TEACH_RULE fields")

    # COMPUTE
    q = parse_bql("COMPUTE rocket force")
    check(q.op == "COMPUTE" and q.subj == "rocket" and q.rel == "force", "COMPUTE")

    # EXPLAIN
    q = parse_bql("EXPLAIN hcl turns_litmus")
    check(q.op == "EXPLAIN", "EXPLAIN op")

    # Comment stripping
    q = parse_bql("LOOKUP acid turns_litmus  # check this")
    check(q.op == "LOOKUP", "comment stripped")

    # Block parse
    block = """
    # comment line
    TEACH acid turns_litmus red
    TEACH HCL isa acid
    INHERIT HCL turns_litmus
    """
    qs = parse_bql_block(block)
    check(len(qs) == 3, "block parse: 3 instructions")
    check(qs[0].op == "TEACH" and qs[2].op == "INHERIT", "block order preserved")

    # Error cases
    try:
        parse_bql("UNKNOWN foo bar")
        check(False, "should raise on unknown op")
    except BrainQLParseError:
        check(True, "unknown op → BrainQLParseError")

    try:
        parse_bql("TEACH_RULE isa turns_litmus turns_litmus")  # no ->
        check(False, "should raise on TEACH_RULE without ->")
    except BrainQLParseError:
        check(True, "TEACH_RULE without -> → BrainQLParseError")


# ── 2. Generalization fix: auto-inheritance in ReasoningEngine ────────────────

def test_auto_inheritance():
    print("\n[2] Auto-inheritance (the generalization fix)")
    from engines.reasoning.reasoning_engine import ReasoningEngine

    re = ReasoningEngine()
    re.set_transitive("isa")

    # Teach the acid category
    re.learn("acid", "turns_litmus", "red")
    re.learn("acid", "ph", "low")
    re.learn("HCL", "isa", "acid")
    re.learn("H2SO4", "isa", "acid")

    # auto-inheritance rule must be present
    check(("isa", "turns_litmus", "turns_litmus") in re.rules,
          "auto-rule isa∘turns_litmus→turns_litmus registered")
    check(("isa", "ph", "ph") in re.rules,
          "auto-rule isa∘ph→ph registered")

    # HCL should inherit turns_litmus
    obj, expl = re.ask("HCL", "turns_litmus")
    check(obj == "red", f"HCL turns_litmus → red (got {obj!r})")
    check(expl is not None and "hcl" in expl.lower(), "explanation mentions HCL")

    # H2SO4 should also inherit it
    obj2, _ = re.ask("H2SO4", "turns_litmus")
    check(obj2 == "red", f"H2SO4 turns_litmus → red via acid (got {obj2!r})")

    # ph as well
    ph, _ = re.ask("HCL", "ph")
    check(ph == "low", f"HCL ph → low (got {ph!r})")

    # Multi-hop: HCL isa strong_acid isa acid
    re.learn("strong_acid", "isa", "acid")
    re.learn("HCL", "isa", "strong_acid")
    obj3, expl3 = re.ask("HCL", "turns_litmus")
    check(obj3 == "red", "multi-hop: HCL isa strong_acid isa acid → turns_litmus red")

    # mark_non_inheritable: structural relations should not auto-inherit
    re2 = ReasoningEngine()
    re2.mark_non_inheritable("example_of", "synonym")
    re2.learn("X", "example_of", "Y")
    check(("isa", "example_of", "example_of") not in re2.rules,
          "marked non_inheritable: example_of does not get auto-rule")

    # isa itself must never be its own premise (default exclusion)
    re3 = ReasoningEngine()
    re3.learn("cat", "isa", "animal")
    check(("isa", "isa", "isa") not in re3.rules,
          "isa not in auto-rules (default non-inheritable)")


# ── 3. BrainQLExecutor: all 8 ops ─────────────────────────────────────────────

def test_executor():
    print("\n[3] BrainQLExecutor — all 8 ops")
    from engines.reasoning.reasoning_engine import ReasoningEngine
    from engines.reasoning.brainql import BrainQLExecutor, parse_bql

    re = ReasoningEngine()
    re.set_transitive("isa")
    exec_ = BrainQLExecutor(re)

    # TEACH
    r = exec_.run(parse_bql("TEACH acid turns_litmus red"))
    check(r.known and r.op == "TEACH", "TEACH: confirmed")

    r2 = exec_.run(parse_bql("TEACH HCL isa acid"))
    check(r2.known, "TEACH HCL isa acid: confirmed")

    # LOOKUP (direct)
    r = exec_.run(parse_bql("LOOKUP acid turns_litmus"))
    check(r.known and r.value == "red", f"LOOKUP acid turns_litmus → red (got {r.value!r})")

    # LOOKUP (unknown)
    r = exec_.run(parse_bql("LOOKUP HCL colour"))
    check(not r.known, "LOOKUP unknown → not known")

    # INHERIT — the key op
    r = exec_.run(parse_bql("INHERIT HCL turns_litmus"))
    check(r.known and r.value == "red", f"INHERIT HCL turns_litmus → red (got {r.value!r})")
    check(len(r.chain) > 0, "INHERIT: has chain")

    # CHAIN (transitive closure of isa)
    r = exec_.run(parse_bql("CHAIN HCL isa"))
    check(r.known and "acid" in r.value, f"CHAIN HCL isa includes 'acid' (got {r.value})")

    # DERIVE (full composition rules)
    r = exec_.run(parse_bql("DERIVE HCL turns_litmus"))
    check(r.known and r.value == "red", f"DERIVE HCL turns_litmus → red (got {r.value!r})")

    # TEACH_RULE
    r = exec_.run(parse_bql("TEACH_RULE parent parent -> grandparent"))
    check(r.known and "grandparent" in r.value, "TEACH_RULE: registered")
    exec_.run(parse_bql("TEACH tom parent sam"))
    exec_.run(parse_bql("TEACH sam parent kid"))
    r = exec_.run(parse_bql("DERIVE tom grandparent"))
    check(r.known and r.value == "kid", f"DERIVE tom grandparent → kid (got {r.value!r})")

    # EXPLAIN
    r = exec_.run(parse_bql("EXPLAIN HCL turns_litmus"))
    check(r.known and r.value == "red", "EXPLAIN: returns value")
    check(len(r.chain) > 0, "EXPLAIN: has chain")

    # COMPUTE (no MeansEndsSolver wired → honest abstention)
    r = exec_.run(parse_bql("COMPUTE rocket force"))
    check(not r.known, "COMPUTE without solver → not known (honest abstention)")
    check("COMPUTE" in r.op, "COMPUTE op preserved in result")

    # Block execution
    block = [parse_bql("TEACH cat isa animal"), parse_bql("INHERIT cat can")]
    results = exec_.run_block(block)
    check(len(results) == 2, "block: 2 results")
    # cat doesn't have 'can' yet, so second result should be not known
    check(not results[1].known, "INHERIT cat can → not known (no facts about animal.can)")


# ── 4. End-to-end pipeline with StubClient ───────────────────────────────────

def test_end_to_end():
    print("\n[4] End-to-end pipeline: StubClient → BrainQLEyes → Brain → BrainQLMouth")
    from engines.reasoning.reasoning_engine import ReasoningEngine
    from engines.reasoning.brainql import BrainQLExecutor, BrainQLResult
    from adapters.llm_adapter import StubClient, BrainQLEyes, BrainQLMouth

    # Stub: LLM translates NL → BrainQL
    stub = StubClient({
        "turn litmus":     "INHERIT hcl turns_litmus",
        "what is hcl":    "CHAIN hcl isa",
        "teach acid":     "TEACH acid turns_litmus red",
    })

    eyes = BrainQLEyes(stub)
    mouth = BrainQLMouth(stub)  # stub returns '' for mouth → fallback renderer

    re = ReasoningEngine()
    re.set_transitive("isa")
    exec_ = BrainQLExecutor(re)

    # Pre-teach
    exec_.run_block([
        __import__("engines.reasoning.brainql", fromlist=["parse_bql"]).parse_bql("TEACH acid turns_litmus red"),
        __import__("engines.reasoning.brainql", fromlist=["parse_bql"]).parse_bql("TEACH hcl isa acid"),
    ])

    # NL → Eyes → BrainQL → Brain → Mouth
    parsed = eyes.parse("Does HCL turn litmus red?")
    check(isinstance(parsed, list), "BrainQLEyes returned BrainQL list")
    check(parsed[0].op == "INHERIT", "Eyes: INHERIT op from stub")

    results = exec_.run_block(parsed)
    check(results[0].known and results[0].value == "red",
          f"Pipeline: INHERIT → red (got {results[0].value!r})")

    # Mouth: deterministic fallback (stub returns '' for mouth prompts)
    text = mouth.render_result(results[0])
    check("red" in text.lower() or "hcl" in text.lower(),
          f"Mouth output contains answer: {text!r}")

    # Math fallback: exact path still works
    from engines.reasoning.neuro_bridge import RuleEyes
    math_eyes = BrainQLEyes(stub, rule=RuleEyes())
    q = math_eyes.parse("differentiate sin(x^2)")
    check(not isinstance(q, list), "Math query: returns Query not BrainQL")
    check(hasattr(q, "kind") and q.kind == "differentiate", "Math query: kind=differentiate")


# ── 5. BrainInterface unified entry point ─────────────────────────────────────

def test_brain_interface():
    print("\n[5] BrainInterface unified entry point")
    from adapters.llm_adapter import StubClient
    from adapters.brain_interface import BrainInterface

    stub = StubClient({
        "turn litmus": "INHERIT hcl turns_litmus",
        "is a dog":    "CHAIN dog isa",
    })

    bi = BrainInterface(client=stub)
    bi.teach("acid", "turns_litmus", "red")
    bi.teach("HCL", "isa", "acid")
    bi.set_transitive("isa")

    # Direct BrainQL passthrough (first word is INHERIT)
    result = bi.respond("INHERIT HCL turns_litmus")
    check("red" in result.lower(), f"Direct BrainQL → contains 'red': {result!r}")

    # respond_bql returns structured results
    results = bi.respond_bql("INHERIT HCL turns_litmus")
    check(len(results) == 1, "respond_bql: 1 result")
    check(results[0].known and results[0].value == "red",
          f"respond_bql: value=red (got {results[0].value!r})")

    # Block: multi-instruction
    results2 = bi.respond_bql("LOOKUP acid turns_litmus\nINHERIT HCL turns_litmus")
    check(len(results2) == 2, "respond_bql block: 2 results")
    check(all(r.known and r.value == "red" for r in results2),
          "respond_bql block: both → red")

    # NL through stub → BrainQL → Brain → Mouth
    answer = bi.respond("Does HCL turn litmus red?")
    check(isinstance(answer, str) and len(answer) > 0, f"NL respond: got {answer!r}")


# ── 6. Regression: existing ReasoningEngine API unchanged ─────────────────────

def test_regression():
    print("\n[6] Regression: existing ReasoningEngine API")
    from engines.reasoning.reasoning_engine import ReasoningEngine

    re = ReasoningEngine()
    for s, r, o in [("tom", "parent", "sam"), ("sam", "parent", "kid"),
                    ("kid", "parent", "baby")]:
        re.learn(s, r, o)
    re.add_rule("parent", "parent", "grandparent")
    re.add_rule("parent", "grandparent", "great_grandparent")

    ans, why = re.ask("tom", "grandparent")
    check(ans == "kid", f"grandparent chain: tom → kid (got {ans!r})")
    check("sam" in why, "explanation mentions sam")

    ans2, _ = re.ask("tom", "great_grandparent")
    check(ans2 == "baby", f"great_grandparent: tom → baby (got {ans2!r})")

    # closure / BFS
    paths = re.closure("tom", "parent")
    check("sam" in paths and "kid" in paths, "closure includes sam, kid")

    # abduce_category still works
    re2 = ReasoningEngine()
    re2.learn("acid", "turns_litmus", "red")
    re2.learn("hcl", "turns_litmus", "red")
    hyps = re2.abduce_category("hcl")
    check(len(hyps) > 0 and hyps[0][0] == "acid", "abduce_category: hcl → acid hypothesis")

    # explain_chain (new method)
    re3 = ReasoningEngine()
    re3.set_transitive("isa")
    re3.learn("acid", "turns_litmus", "red")
    re3.learn("HCL", "isa", "acid")
    val, steps = re3.explain_chain("HCL", "turns_litmus")
    check(val == "red", f"explain_chain value=red (got {val!r})")
    check(len(steps) > 0, "explain_chain: has steps")


# ── 7. BrainQL demo (the HCL example end-to-end, clean) ──────────────────────

def test_hcl_demo():
    print("\n[7] HCL litmus example — the full story")
    from engines.reasoning.brainql import BrainQLExecutor, parse_bql, parse_bql_block

    from engines.reasoning.reasoning_engine import ReasoningEngine
    re = ReasoningEngine()
    re.set_transitive("isa")
    exec_ = BrainQLExecutor(re)

    # Step 1: teach domain knowledge
    teaching_session = """
    TEACH acid    turns_litmus  red
    TEACH acid    ph            low
    TEACH base    turns_litmus  blue
    TEACH HCL     isa           acid
    TEACH H2SO4   isa           acid
    TEACH NaOH    isa           base
    """
    exec_.run_block(parse_bql_block(teaching_session))

    # Step 2: generalisation queries
    queries = [
        ("INHERIT HCL turns_litmus",   "red"),
        ("INHERIT H2SO4 turns_litmus", "red"),
        ("INHERIT NaOH turns_litmus",  "blue"),
        ("INHERIT HCL ph",             "low"),
    ]
    for bql_str, expected in queries:
        r = exec_.run(parse_bql(bql_str))
        check(r.known and r.value == expected,
              f"{bql_str} → {expected} (got known={r.known} value={r.value!r})")

    # Step 3: chain / ancestry
    r = exec_.run(parse_bql("CHAIN HCL isa"))
    check(r.known and "acid" in r.value, "CHAIN HCL isa: includes acid")

    # Step 4: explain proof
    r = exec_.run(parse_bql("EXPLAIN HCL turns_litmus"))
    check(r.known and len(r.chain) > 0, "EXPLAIN: has proof chain")
    print(f"  Proof chain: {r.chain}")

    print("\n  All HCL tests passed — brain generalizes correctly!")


# ── runner ────────────────────────────────────────────────────────────────────

def run():
    tests = [
        test_parser,
        test_auto_inheritance,
        test_executor,
        test_end_to_end,
        test_brain_interface,
        test_regression,
        test_hcl_demo,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"\n  ✗ {t.__name__} FAILED: {e}")
            traceback.print_exc()

    total = len(tests)
    print(f"\n{'='*50}")
    print(f"BrainQL tests: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)
    else:
        print("All tests passed ✓")


if __name__ == "__main__":
    run()
