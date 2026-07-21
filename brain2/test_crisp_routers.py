#!/usr/bin/env python3
"""
test_crisp_routers.py — unit tests for CrispInternalRouter and CrispExternalRouter.

Run:
    python3 test_crisp_routers.py          (from brain2/)

Tests:
─── CrispInternalRouter (T1–T15) ────────────────────────────────────────────
  T1   greeting → IDLE
  T2   high-confidence verified direct answer → RETRIEVE + trigger_teach
  T3   moderate confidence factual → VERIFY
  T4   very high novelty → PROPOSE + trigger_propose + trigger_curiosity
  T5   very high curiosity_error → PROPOSE
  T6   solution_type="code" → SYNTHESIZE
  T7   solution_type="none" → SYNTHESIZE
  T8   solution_type="compute", no answer (low conf) → SYNTHESIZE
  T9   moderate curiosity, factual answer → CURIOUS + trigger_curiosity
  T10  low-bar factual fallback (confidence > 0, not meeting RETRIEVE) → RETRIEVE
  T11  domain_hint matches solution_type
  T12  RETRIEVE does NOT fire when not verified (even high confidence)
  T13  VERIFY does NOT trigger_teach when confidence < 0.75
  T14  VERIFY DOES trigger_teach when verified + confidence >= 0.75
  T15  custom thresholds: retrieve_min_confidence = 0.99 → blocks RETRIEVE

─── CrispExternalRouter (T16–T30) ────────────────────────────────────────────
  T16  push_fact — verified=True → accepted (offline buffer)
  T17  push_fact — verified=False → rejected with "unverified"
  T18  push_fact — low confidence → rejected with "low_confidence"
  T19  push_policy — verified=True, expr set → accepted (offline buffer)
  T20  push_policy — verified=False → rejected
  T21  push_policy — expr=None → rejected with "null_expr"
  T22  offline buffer drains when brain set via set_brain()
  T23  stats() counts match accept/reject counts
  T24  quarantine filled when C++ mock rejects (simulated)
  T25  unpack_signal — all fields round-trip correctly
  T26  unpack_signal — missing keys default gracefully
  T27  push_facts (batch) → all accepted, results in order
  T28  push_policies (batch) → some accepted, some rejected
  T29  audit_trail length == total pushes
  T30  quarantined() returns copy (not reference)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from routing.crisp_internal_router import (
    CrispInternalRouter, CrispMode, CrispThresholds, CrispRoutingDecision
)
from routing.crisp_external_router import (
    CrispExternalRouter, CrispFact, CrispPolicy, PushResult, CrispInboundSignal
)

# ── minimal harness ─────────────────────────────────────────────────────────
_TR = _TP = _TF = 0

def PASS(msg):
    global _TR, _TP
    _TR += 1; _TP += 1
    print(f"  [PASS] {msg}")

def FAIL(msg, detail=""):
    global _TR, _TF
    _TR += 1; _TF += 1
    print(f"  [FAIL] {msg}" + (f"  ({detail})" if detail else ""))

def CHECK(cond, msg, detail=""):
    if cond: PASS(msg)
    else:     FAIL(msg, detail)

def CHECK_EQ(a, b, msg):
    CHECK(a == b, msg, f"{a!r} != {b!r}")

def CHECK_GT(a, b, msg):
    CHECK(a > b, msg, f"{a} not > {b}")

def CHECK_LT(a, b, msg):
    CHECK(a < b, msg, f"{a} not < {b}")

def CHECK_NEAR(a, b, eps, msg):
    CHECK(abs(a - b) <= eps, msg, f"|{a}-{b}| > {eps}")


# ════════════════════════════════════════════════════════════════════════════
#  CrispInternalRouter tests
# ════════════════════════════════════════════════════════════════════════════

def test_internal_router():
    print("\n=== CrispInternalRouter tests ===\n")
    r = CrispInternalRouter()

    # ── T1: greeting → IDLE ─────────────────────────────────────────────────
    print("T1  greeting → IDLE")
    d = r.decide(appraisal_type="greeting", solution_type="none")
    CHECK_EQ(d.mode, CrispMode.IDLE, "greeting → IDLE")
    CHECK(not d.trigger_teach,    "no trigger_teach in IDLE")
    CHECK(not d.trigger_propose,  "no trigger_propose in IDLE")

    # ── T2: RETRIEVE (high-confidence verified direct) ───────────────────────
    print("\nT2  high-confidence verified direct answer → RETRIEVE + trigger_teach")
    d = r.decide(confidence=0.92, verification_depth=0,
                 is_verified=True, solution_type="factual")
    CHECK_EQ(d.mode, CrispMode.RETRIEVE, "verified+high_conf → RETRIEVE")
    CHECK(d.trigger_teach, "RETRIEVE triggers teach")
    CHECK_NEAR(d.confidence_out, 0.92, 0.001, "confidence_out preserved")

    # ── T3: VERIFY (moderate confidence factual) ─────────────────────────────
    print("\nT3  moderate-confidence factual → VERIFY")
    d = r.decide(confidence=0.60, verification_depth=3,
                 is_verified=True, solution_type="factual")
    CHECK_EQ(d.mode, CrispMode.VERIFY, "moderate conf factual → VERIFY")
    CHECK_LT(d.confidence_out, 0.60 + 0.001, "confidence_out <= input (penalised)")

    # ── T4: PROPOSE (very high novelty) ──────────────────────────────────────
    print("\nT4  very high novelty → PROPOSE + trigger_propose + trigger_curiosity")
    d = r.decide(novelty=0.80, curiosity_error=0.30,
                 solution_type="none", is_verified=False)
    CHECK_EQ(d.mode, CrispMode.PROPOSE, "high novelty → PROPOSE")
    CHECK(d.trigger_propose,   "PROPOSE sets trigger_propose")
    CHECK(d.trigger_curiosity, "PROPOSE sets trigger_curiosity")

    # ── T5: PROPOSE (high curiosity_error) ────────────────────────────────────
    print("\nT5  high curiosity_error → PROPOSE")
    d = r.decide(curiosity_error=0.75, novelty=0.20,
                 solution_type="factual", confidence=0.30)
    CHECK_EQ(d.mode, CrispMode.PROPOSE, "high curiosity → PROPOSE")

    # ── T6: SYNTHESIZE (code) ─────────────────────────────────────────────────
    print("\nT6  solution_type='code' → SYNTHESIZE")
    d = r.decide(solution_type="code", confidence=0.0, is_verified=False)
    CHECK_EQ(d.mode, CrispMode.SYNTHESIZE, "code → SYNTHESIZE")

    # ── T7: SYNTHESIZE (none — no answer) ────────────────────────────────────
    print("\nT7  solution_type='none' → SYNTHESIZE")
    d = r.decide(solution_type="none", confidence=0.0, is_verified=False)
    CHECK_EQ(d.mode, CrispMode.SYNTHESIZE, "none → SYNTHESIZE")

    # ── T8: SYNTHESIZE (compute, no confidence) ──────────────────────────────
    print("\nT8  solution_type='compute', confidence=0.1 → SYNTHESIZE")
    d = r.decide(solution_type="compute", confidence=0.10, is_verified=False)
    CHECK_EQ(d.mode, CrispMode.SYNTHESIZE, "compute + low conf → SYNTHESIZE")

    # ── T9: CURIOUS (moderate curiosity, factual answer) ─────────────────────
    print("\nT9  moderate curiosity, factual answer → CURIOUS + trigger_curiosity")
    d = r.decide(curiosity_error=0.45, confidence=0.70,
                 solution_type="factual", is_verified=True,
                 verification_depth=10)  # depth=10 > verify_max_depth=8 → skips VERIFY
    CHECK_EQ(d.mode, CrispMode.CURIOUS, "moderate curiosity → CURIOUS")
    CHECK(d.trigger_curiosity, "CURIOUS sets trigger_curiosity")

    # ── T10: RETRIEVE fallback (confidence > 0, non-strict) ──────────────────
    print("\nT10 low-bar factual fallback → RETRIEVE(fallback)")
    d = r.decide(confidence=0.30, solution_type="factual",
                 is_verified=False, verification_depth=0)
    # Should eventually land in RETRIEVE fallback (not IDLE or SYNTHESIZE)
    CHECK_EQ(d.mode, CrispMode.RETRIEVE, "non-zero conf factual → RETRIEVE fallback")
    CHECK(not d.trigger_teach, "unverified fallback: no trigger_teach")

    # ── T11: domain_hint matches solution_type ────────────────────────────────
    print("\nT11 domain_hint matches solution_type")
    for stype, expected in [("compute","physics"), ("factual","factual"),
                             ("code","code"), ("none","unknown")]:
        d = r.decide(solution_type=stype, confidence=0.5, is_verified=False)
        CHECK_EQ(d.domain_hint, expected, f"stype={stype} → domain={expected}")

    # ── T12: RETRIEVE gate — not verified → no RETRIEVE ──────────────────────
    print("\nT12 RETRIEVE requires verified=True even at high confidence")
    d = r.decide(confidence=0.99, verification_depth=0,
                 is_verified=False, solution_type="factual")
    CHECK(d.mode != CrispMode.RETRIEVE, "unverified + high conf → not RETRIEVE")

    # ── T13: VERIFY.trigger_teach = False when confidence < 0.75 ─────────────
    print("\nT13 VERIFY does NOT trigger_teach when confidence < 0.75")
    d = r.decide(confidence=0.60, verification_depth=2,
                 is_verified=True, solution_type="factual")
    CHECK_EQ(d.mode, CrispMode.VERIFY, "moderate conf → VERIFY")
    CHECK(not d.trigger_teach, "VERIFY below 0.75: no trigger_teach")

    # ── T14: VERIFY.trigger_teach = True when verified + conf >= 0.75 ─────────
    print("\nT14 VERIFY DOES trigger_teach when verified + confidence >= 0.75")
    d = r.decide(confidence=0.80, verification_depth=3,
                 is_verified=True, solution_type="factual")
    CHECK_EQ(d.mode, CrispMode.VERIFY, "conf=0.80 depth=3 → VERIFY")
    CHECK(d.trigger_teach, "VERIFY above 0.75 + verified → trigger_teach")

    # ── T15: custom thresholds ────────────────────────────────────────────────
    print("\nT15 custom threshold: retrieve_min_confidence=0.99 blocks normal RETRIEVE")
    strict = CrispThresholds(retrieve_min_confidence=0.99)
    r2 = CrispInternalRouter(thresholds=strict)
    d = r2.decide(confidence=0.92, verification_depth=0,
                  is_verified=True, solution_type="factual")
    CHECK(d.mode != CrispMode.RETRIEVE,
          "conf=0.92 < 0.99 custom threshold → not RETRIEVE")


# ════════════════════════════════════════════════════════════════════════════
#  CrispExternalRouter tests
# ════════════════════════════════════════════════════════════════════════════

def test_external_router():
    print("\n\n=== CrispExternalRouter tests ===\n")

    # ── T16: push_fact verified=True → buffered (offline mode) ───────────────
    print("T16 push_fact — verified=True, no brain → buffered offline")
    er = CrispExternalRouter(brain=None)
    r = er.push_fact(CrispFact("rocket", "mass", 1000.0, verified=True))
    CHECK(r.accepted,              "offline push: accepted=True")
    CHECK_EQ(r.reason, "buffered_offline", "offline push: reason=buffered_offline")

    # ── T17: push_fact verified=False → rejected ──────────────────────────────
    print("\nT17 push_fact — verified=False → rejected unverified")
    r = er.push_fact(CrispFact("rocket", "mass", 999.0, verified=False))
    CHECK(not r.accepted,        "unverified → rejected")
    CHECK_EQ(r.reason, "unverified", "reason=unverified")

    # ── T18: push_fact low confidence → rejected ──────────────────────────────
    print("\nT18 push_fact — confidence=0.3 → rejected low_confidence")
    r = er.push_fact(CrispFact("x", "y", 1.0, verified=True, confidence=0.3))
    CHECK(not r.accepted,             "low_confidence → rejected")
    CHECK_EQ(r.reason, "low_confidence", "reason=low_confidence")

    # ── T19: push_policy verified=True, expr set → buffered ───────────────────
    print("\nT19 push_policy — verified=True, expr set → buffered offline")
    pol = CrispPolicy("force", ("mass","accel"), ("*","mass","accel"), verified=True)
    r = er.push_policy(pol)
    CHECK(r.accepted,              "policy offline: accepted")
    CHECK_EQ(r.reason, "buffered_offline", "policy offline: reason=buffered_offline")

    # ── T20: push_policy verified=False → rejected ────────────────────────────
    print("\nT20 push_policy — verified=False → rejected")
    pol2 = CrispPolicy("momentum", ("mass","speed"), ("*","mass","speed"), verified=False)
    r = er.push_policy(pol2)
    CHECK(not r.accepted, "unverified policy → rejected")

    # ── T21: push_policy expr=None → rejected ─────────────────────────────────
    print("\nT21 push_policy — expr=None → rejected null_expr")
    pol3 = CrispPolicy("x", ("a","b"), None, verified=True)
    r = er.push_policy(pol3)
    CHECK(not r.accepted,        "null_expr → rejected")
    CHECK_EQ(r.reason, "null_expr", "reason=null_expr")

    # ── T22: offline buffer drains when set_brain() called ───────────────────
    print("\nT22 offline buffer drains when brain set via set_brain()")
    er2 = CrispExternalRouter(brain=None)
    er2.push_fact(CrispFact("dog", "legs", 4.0, verified=True, confidence=0.9))
    er2.push_fact(CrispFact("cat", "legs", 4.0, verified=True, confidence=0.9))
    CHECK_EQ(len(er2._buffer), 2, "2 facts buffered before set_brain")

    # Simulate brain with accept_fact that always accepts
    class _FakeBrain:
        def accept_fact(self, f):
            return True
        def accept_policy(self, p):
            return True
    er2.set_brain(_FakeBrain())
    CHECK_EQ(len(er2._buffer), 0, "buffer drained after set_brain")
    CHECK_EQ(er2.facts_accepted, 2, "both facts accepted after drain")

    # ── T23: stats() counts match ─────────────────────────────────────────────
    print("\nT23 stats() totals are consistent")
    er3 = CrispExternalRouter(brain=None)
    er3.push_fact(CrispFact("a","b",1.0, verified=True))
    er3.push_fact(CrispFact("c","d",2.0, verified=False))
    er3.push_fact(CrispFact("e","f",3.0, verified=True, confidence=0.2))
    s = er3.stats()
    CHECK_EQ(s["facts_accepted"], 1, "1 fact accepted")
    CHECK_EQ(s["facts_rejected"], 2, "2 facts rejected")
    CHECK_EQ(s["buffered"], 1, "1 buffered offline")

    # ── T24: quarantine filled when fuzzy rejects ──────────────────────────────
    print("\nT24 quarantine filled when C++ mock rejects")
    class _RejectBrain:
        def accept_fact(self, entity, relation, value, verified=False, source=""): return False
        def accept_policy(self, target, inputs, expr, verified=False, source=""): return False
    er4 = CrispExternalRouter(brain=_RejectBrain())
    er4.push_fact(CrispFact("x","y",1.0, verified=True, confidence=0.9))
    CHECK_EQ(len(er4._quarantine), 1, "1 fact quarantined after C++ rejection")
    CHECK_EQ(er4.facts_rejected, 1,   "rejected counter incremented")

    # ── T25: unpack_signal round-trips all fields ─────────────────────────────
    print("\nT25 unpack_signal — all fields round-trip")
    raw = {"novelty":0.7, "valence":-0.2, "arousal":0.4, "bmu":42,
           "gate_open":True, "confidence":0.85, "domain_hint":"MATH",
           "mode_name":"REASON", "episodic_stored":True}
    sig = CrispExternalRouter.unpack_signal(raw)
    CHECK_NEAR(sig.novelty,     0.70, 1e-6, "novelty round-trips")
    CHECK_NEAR(sig.valence,    -0.20, 1e-6, "valence round-trips")
    CHECK_NEAR(sig.arousal,     0.40, 1e-6, "arousal round-trips")
    CHECK_EQ(sig.bmu,           42,         "bmu round-trips")
    CHECK(sig.gate_open,                    "gate_open round-trips")
    CHECK_NEAR(sig.confidence,  0.85, 1e-6, "confidence round-trips")
    CHECK_EQ(sig.domain_hint,  "MATH",      "domain_hint round-trips")
    CHECK_EQ(sig.mode_name,    "REASON",    "mode_name round-trips")
    CHECK(sig.episodic_stored,              "episodic_stored round-trips")

    # ── T26: unpack_signal — missing keys default gracefully ─────────────────
    print("\nT26 unpack_signal — empty dict defaults gracefully")
    sig2 = CrispExternalRouter.unpack_signal({})
    CHECK_NEAR(sig2.novelty,    0.0, 1e-6, "default novelty=0.0")
    CHECK_EQ(sig2.bmu,           -1,       "default bmu=-1")
    CHECK(not sig2.gate_open,              "default gate_open=False")
    CHECK_EQ(sig2.mode_name, "PERCEIVE",   "default mode_name=PERCEIVE")

    # ── T27: push_facts (batch) ────────────────────────────────────────────────
    print("\nT27 push_facts (batch) — all accepted")
    er5 = CrispExternalRouter(brain=None)
    facts = [CrispFact(f"e{i}", "r", float(i), verified=True) for i in range(5)]
    results = er5.push_facts(facts)
    CHECK_EQ(len(results), 5, "5 results returned")
    CHECK(all(r.accepted for r in results), "all 5 accepted offline")

    # ── T28: push_policies (batch) — mixed results ────────────────────────────
    print("\nT28 push_policies (batch) — mixed results")
    er6 = CrispExternalRouter(brain=None)
    pols = [
        CrispPolicy("f1", ("a",), ("x","a"), verified=True),
        CrispPolicy("f2", ("b",), None,      verified=True),   # null expr
        CrispPolicy("f3", ("c",), ("y","c"), verified=False),  # not verified
        CrispPolicy("f4", ("d",), ("z","d"), verified=True),
    ]
    results = er6.push_policies(pols)
    accepted = [r.accepted for r in results]
    CHECK_EQ(accepted, [True, False, False, True], "batch: T,F,F,T pattern")

    # ── T29: audit_trail length == total pushes ───────────────────────────────
    print("\nT29 audit_trail length == total pushes")
    er7 = CrispExternalRouter(brain=None)
    for i in range(6):
        er7.push_fact(CrispFact(f"e{i}", "r", float(i),
                                verified=(i % 2 == 0)))
    CHECK_EQ(len(er7.audit_trail()), 6, "audit_trail length = 6 (all pushes logged)")

    # ── T30: quarantined() returns a copy ────────────────────────────────────
    print("\nT30 quarantined() returns a copy, not a reference")
    er8 = CrispExternalRouter(brain=_RejectBrain())
    er8.push_fact(CrispFact("a","b",1.0, verified=True, confidence=0.9))
    q = er8.quarantined()
    q.clear()  # modifying the returned list should not empty the real quarantine
    CHECK_EQ(len(er8.quarantined()), 1, "original quarantine unaffected by external mutation")


# ════════════════════════════════════════════════════════════════════════════
#  Integration test: InternalRouter → ExternalRouter → (mock) Brain
# ════════════════════════════════════════════════════════════════════════════

def test_integration():
    print("\n\n=== Integration: CrispInternalRouter → CrispExternalRouter ===\n")

    class _MockBrain:
        """Thin mock: accepts all, counts calls. Uses flat pybind signature."""
        def __init__(self): self.fact_calls = []; self.policy_calls = []
        def accept_fact(self, entity, relation, value, verified=False, source="python"):
            self.fact_calls.append((entity, relation, value))
            return True
        def accept_policy(self, target, inputs, expr, verified=False, source="python"):
            self.policy_calls.append(target)
            return True

    brain = _MockBrain()
    ir = CrispInternalRouter()
    er = CrispExternalRouter(brain=brain)

    # Simulate: crisp layer answers "rocket.force = 12000" with high confidence
    dec = ir.decide(confidence=0.93, verification_depth=0,
                    is_verified=True, solution_type="compute")
    CHECK_EQ(dec.mode, CrispMode.RETRIEVE, "IR: compute+high_conf → RETRIEVE")
    CHECK(dec.trigger_teach, "IR: trigger_teach set")
    print(f"  IR decision: {dec.label}")

    # Since trigger_teach is set, push the fact
    if dec.trigger_teach:
        result = er.push_fact(CrispFact("rocket", "force", 12000.0,
                                        verified=True, source="means_ends_solver"))
    CHECK(result.accepted, "ER: fact pushed to mock brain")
    CHECK_EQ(brain.fact_calls, [("rocket", "force", 12000.0)],
             "mock brain received correct fact")

    # Simulate: crisp layer gets a synthesis result → push policy too
    pol_dec = ir.decide(confidence=0.88, verification_depth=0,
                        is_verified=True, solution_type="compute")
    if pol_dec.trigger_teach:
        pr = er.push_policy(CrispPolicy("force", ("mass","accel"),
                                        ("*","mass","accel"), verified=True))
        CHECK(pr.accepted, "ER: policy pushed to mock brain")
        CHECK_EQ(brain.policy_calls, ["force"], "mock brain received policy")

    print(f"  Stats: {er.stats()}")


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Crisp Router Unit Tests ===")
    test_internal_router()
    test_external_router()
    test_integration()

    print(f"\n=== Results: {_TP}/{_TR} passed", end="")
    if _TF == 0:
        print(" — ALL PASS ✓ ===")
    else:
        print(f" — {_TF} FAILED ✗ ===")
    sys.exit(0 if _TF == 0 else 1)
