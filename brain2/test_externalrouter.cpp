/*
 * test_externalrouter.cpp — Unit tests for ExternalRouter
 *
 * Standalone C++ test: no pybind, no Python, no Brain.
 * Builds with:
 *   g++ -std=c++17 -I. test_externalrouter.cpp -o test_externalrouter && ./test_externalrouter
 *
 * Tests:
 *   T1   accept_fact — verified=true  → accepted, writer called
 *   T2   accept_fact — verified=false → rejected, writer NOT called
 *   T3   accept_fact — no writer set  → rejected with "null_writer"
 *   T4   accept_policy — verified=true  → accepted, writer called
 *   T5   accept_policy — verified=false → rejected
 *   T6   accept_policy — null expr → rejected with "null_expr"
 *   T7   pack() — high error → novelty field matches
 *   T8   pack() — gate_open true when salience > 0 and mode != IDLE
 *   T9   pack() — gate_open false when mode == IDLE
 *   T10  pack() — confidence decreases with high error
 *   T11  pack() — confidence increases with wm_load
 *   T12  domain_hint — default modulo-4 bucketing
 *   T13  domain_hint — custom map overrides default
 *   T14  domain_hint — LANGUAGE gw_winner overrides default
 *   T15  domain_hint — self_concept < 0 → "UNKNOWN"
 *   T16  Multiple accepts: counters track correctly (using multiple fact writers)
 */

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "core/externalrouter.hpp"

// ── Minimal test harness ───────────────────────────────────────────────────
static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define EXPECT_TRUE(cond, msg) do { \
    tests_run++; \
    if (cond) { \
        printf("  [PASS] %s\n", msg); \
        tests_passed++; \
    } else { \
        printf("  [FAIL] %s\n", msg); \
        tests_failed++; \
    } \
} while(0)

#define EXPECT_FALSE(cond, msg)    EXPECT_TRUE(!(cond), msg)
#define EXPECT_EQ_STR(a, b, msg)   EXPECT_TRUE((a) == (b), msg)

#define EXPECT_NEAR(a, b, eps, msg) do { \
    tests_run++; \
    float diff = (a) - (b); \
    if (diff < 0.f) diff = -diff; \
    if (diff <= (eps)) { \
        printf("  [PASS] %s  (%.4f ≈ %.4f)\n", msg, (float)(a), (float)(b)); \
        tests_passed++; \
    } else { \
        printf("  [FAIL] %s  (%.4f, expected %.4f, diff %.4f)\n", \
               msg, (float)(a), (float)(b), diff); \
        tests_failed++; \
    } \
} while(0)

#define EXPECT_GT(a, b, msg) do { \
    tests_run++; \
    if ((a) > (b)) { \
        printf("  [PASS] %s  (%.4f > %.4f)\n", msg, (float)(a), (float)(b)); \
        tests_passed++; \
    } else { \
        printf("  [FAIL] %s  (%.4f not > %.4f)\n", msg, (float)(a), (float)(b)); \
        tests_failed++; \
    } \
} while(0)

#define EXPECT_LT(a, b, msg) do { \
    tests_run++; \
    if ((a) < (b)) { \
        printf("  [PASS] %s  (%.4f < %.4f)\n", msg, (float)(a), (float)(b)); \
        tests_passed++; \
    } else { \
        printf("  [FAIL] %s  (%.4f not < %.4f)\n", msg, (float)(a), (float)(b)); \
        tests_failed++; \
    } \
} while(0)

using namespace brain2;

// ── Helper: build a simple ExprPtr (number literal) ───────────────────────
static ExprPtr make_num_expr(double v) { return num(v); }

// ── Helper: make a verified ExternalRouter with counters ──────────────────
static ExternalRouter make_wired_router(int& fact_calls, int& policy_calls) {
    ExternalRouter er;
    er.set_fact_writer([&](const std::string&, const std::string&, double) {
        fact_calls++;
    });
    er.set_policy_writer([&](const std::string&, const std::vector<std::string>&,
                              const ExprPtr&) {
        policy_calls++;
    });
    return er;
}

int main() {
    printf("=== ExternalRouter unit tests ===\n\n");

    // ── T1: accept_fact verified=true ─────────────────────────────────────
    printf("T1  accept_fact — verified=true → accepted, writer called\n");
    {
        int calls = 0, pc = 0;
        auto er = make_wired_router(calls, pc);
        InboundFact f{"rocket", "mass", 1000.0, true, "test"};
        auto dec = er.accept_fact(f);
        EXPECT_TRUE(dec.accepted,  "accepted == true");
        EXPECT_EQ_STR(dec.reason, std::string("ok"), "reason == 'ok'");
        EXPECT_TRUE(calls == 1,    "fact_writer called once");
    }

    // ── T2: accept_fact verified=false ────────────────────────────────────
    printf("\nT2  accept_fact — verified=false → rejected, writer NOT called\n");
    {
        int calls = 0, pc = 0;
        auto er = make_wired_router(calls, pc);
        InboundFact f{"rocket", "mass", 1000.0, false, "test"};
        auto dec = er.accept_fact(f);
        EXPECT_FALSE(dec.accepted, "accepted == false");
        EXPECT_EQ_STR(dec.reason, std::string("unverified"), "reason == 'unverified'");
        EXPECT_TRUE(calls == 0,    "fact_writer NOT called");
    }

    // ── T3: accept_fact no writer ─────────────────────────────────────────
    printf("\nT3  accept_fact — no writer set → rejected with 'null_writer'\n");
    {
        ExternalRouter er;  // no callbacks wired
        InboundFact f{"rocket", "mass", 1000.0, true, "test"};
        auto dec = er.accept_fact(f);
        EXPECT_FALSE(dec.accepted, "accepted == false");
        EXPECT_EQ_STR(dec.reason, std::string("null_writer"), "reason == 'null_writer'");
    }

    // ── T4: accept_policy verified=true ───────────────────────────────────
    printf("\nT4  accept_policy — verified=true → accepted, writer called\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        InboundPolicy p;
        p.target   = "force";
        p.inputs   = {"mass", "accel"};
        p.expr     = opx('*', var("mass"), var("accel"));
        p.verified = true;
        p.source   = "policy_induction";
        auto dec = er.accept_policy(p);
        EXPECT_TRUE(dec.accepted,  "accepted == true");
        EXPECT_EQ_STR(dec.reason, std::string("ok"), "reason == 'ok'");
        EXPECT_TRUE(pc == 1,       "policy_writer called once");
    }

    // ── T5: accept_policy verified=false ──────────────────────────────────
    printf("\nT5  accept_policy — verified=false → rejected\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        InboundPolicy p;
        p.target   = "force";
        p.inputs   = {"mass", "accel"};
        p.expr     = opx('*', var("mass"), var("accel"));
        p.verified = false;
        auto dec = er.accept_policy(p);
        EXPECT_FALSE(dec.accepted, "accepted == false");
        EXPECT_EQ_STR(dec.reason, std::string("unverified"), "reason == 'unverified'");
        EXPECT_TRUE(pc == 0,       "policy_writer NOT called");
    }

    // ── T6: accept_policy null expr ───────────────────────────────────────
    printf("\nT6  accept_policy — null expr → rejected with 'null_expr'\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        InboundPolicy p;
        p.target   = "force";
        p.inputs   = {"mass"};
        p.expr     = nullptr;   // <-- null
        p.verified = true;
        auto dec = er.accept_policy(p);
        EXPECT_FALSE(dec.accepted, "accepted == false");
        EXPECT_EQ_STR(dec.reason, std::string("null_expr"), "reason == 'null_expr'");
    }

    // ── T7: pack() — novelty field matches error ───────────────────────────
    printf("\nT7  pack() — novelty field == error passed in\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        auto s = er.pack(
            /*error*/0.75f, /*valence*/0.1f, /*arousal*/0.3f,
            /*salience*/0.5f, /*wm_load*/0.4f,
            /*bmu*/7, /*self_concept*/2, /*gw_winner*/(int)GWModule::SOM,
            /*episodic_stored*/false, RouteMode::PERCEIVE);
        EXPECT_NEAR(s.novelty, 0.75f, 0.001f, "novelty == error (0.75)");
        EXPECT_NEAR(s.valence, 0.1f,  0.001f, "valence field correct");
        EXPECT_NEAR(s.arousal, 0.3f,  0.001f, "arousal field correct");
        EXPECT_TRUE(s.bmu == 7, "bmu field correct");
    }

    // ── T8: gate_open when salience > 0 and mode != IDLE ──────────────────
    printf("\nT8  pack() — gate_open=true when salience > 0.05 and mode != IDLE\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        auto s = er.pack(0.3f, 0.1f, 0.3f, /*salience*/0.6f, 0.4f,
                         5, 1, (int)GWModule::SOM, false, RouteMode::PERCEIVE);
        EXPECT_TRUE(s.gate_open, "gate_open == true (salience=0.6, mode=PERCEIVE)");
    }

    // ── T9: gate_open false when mode == IDLE ─────────────────────────────
    printf("\nT9  pack() — gate_open=false when mode == IDLE\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        auto s = er.pack(0.01f, 0.01f, 0.01f, /*salience*/0.60f, 0.1f,
                         0, 0, (int)GWModule::SOM, false, RouteMode::IDLE);
        EXPECT_FALSE(s.gate_open, "gate_open == false when mode == IDLE");
    }

    // ── T10: confidence decreases with high error ──────────────────────────
    printf("\nT10 pack() — confidence lower for high-error (novel) input\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        auto hi_err = er.pack(/*error*/0.90f, 0.1f, 0.3f, 0.4f, 0.4f,
                               5, 2, (int)GWModule::SOM, false, RouteMode::PERCEIVE);
        auto lo_err = er.pack(/*error*/0.05f, 0.1f, 0.3f, 0.4f, 0.4f,
                               5, 2, (int)GWModule::SOM, false, RouteMode::PERCEIVE);
        EXPECT_LT(hi_err.confidence, lo_err.confidence,
                  "high error → lower confidence than low error");
    }

    // ── T11: confidence increases with wm_load ────────────────────────────
    printf("\nT11 pack() — confidence increases with wm_load (context-rich)\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        auto lo_wm = er.pack(0.15f, 0.1f, 0.3f, 0.4f, /*wm_load*/0.10f,
                              5, 2, (int)GWModule::SOM, false, RouteMode::REASON);
        auto hi_wm = er.pack(0.15f, 0.1f, 0.3f, 0.4f, /*wm_load*/0.80f,
                              5, 2, (int)GWModule::SOM, false, RouteMode::REASON);
        EXPECT_GT(hi_wm.confidence, lo_wm.confidence,
                  "higher wm_load → higher confidence");
    }

    // ── T12: domain_hint — default modulo-4 bucketing ─────────────────────
    printf("\nT12 domain_hint — default modulo-4 (LANGUAGE/MATH/PHYSICS/CODE)\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        // concept 0 → LANGUAGE, 1 → MATH, 2 → PHYSICS, 3 → CODE, 4 → LANGUAGE, ...
        auto s0 = er.pack(0.2f, 0.f, 0.f, 0.4f, 0.3f, 0, 0, (int)GWModule::SOM, false, RouteMode::PERCEIVE);
        auto s1 = er.pack(0.2f, 0.f, 0.f, 0.4f, 0.3f, 1, 1, (int)GWModule::SOM, false, RouteMode::PERCEIVE);
        auto s2 = er.pack(0.2f, 0.f, 0.f, 0.4f, 0.3f, 2, 2, (int)GWModule::SOM, false, RouteMode::PERCEIVE);
        auto s3 = er.pack(0.2f, 0.f, 0.f, 0.4f, 0.3f, 3, 3, (int)GWModule::SOM, false, RouteMode::PERCEIVE);
        EXPECT_EQ_STR(s0.domain_hint, std::string("LANGUAGE"), "concept 0 → LANGUAGE");
        EXPECT_EQ_STR(s1.domain_hint, std::string("MATH"),     "concept 1 → MATH");
        EXPECT_EQ_STR(s2.domain_hint, std::string("PHYSICS"),  "concept 2 → PHYSICS");
        EXPECT_EQ_STR(s3.domain_hint, std::string("CODE"),     "concept 3 → CODE");
    }

    // ── T13: custom domain map overrides default ───────────────────────────
    printf("\nT13 domain_hint — custom map overrides modulo default\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        // Override: concept 1 (normally MATH) → "BIOLOGY"
        er.set_domain_map({{1, "BIOLOGY"}, {5, "CHEMISTRY"}});
        auto s1 = er.pack(0.2f, 0.f, 0.f, 0.4f, 0.3f, 0, 1, (int)GWModule::SOM, false, RouteMode::PERCEIVE);
        auto s5 = er.pack(0.2f, 0.f, 0.f, 0.4f, 0.3f, 0, 5, (int)GWModule::SOM, false, RouteMode::PERCEIVE);
        auto s2 = er.pack(0.2f, 0.f, 0.f, 0.4f, 0.3f, 0, 2, (int)GWModule::SOM, false, RouteMode::PERCEIVE);  // not in custom map
        EXPECT_EQ_STR(s1.domain_hint, std::string("BIOLOGY"),   "concept 1 custom → BIOLOGY");
        EXPECT_EQ_STR(s5.domain_hint, std::string("CHEMISTRY"), "concept 5 custom → CHEMISTRY");
        EXPECT_EQ_STR(s2.domain_hint, std::string("PHYSICS"),   "concept 2 (not in custom) → PHYSICS (modulo)");
    }

    // ── T14: LANGUAGE gw_winner overrides modulo ───────────────────────────
    printf("\nT14 domain_hint — GWModule::LANGUAGE winner → 'LANGUAGE' override\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        // concept 3 would normally be CODE, but LANGUAGE wins the GW
        auto s = er.pack(0.2f, 0.f, 0.f, 0.4f, 0.3f, 0, 3,
                          (int)GWModule::LANGUAGE, false, RouteMode::PERCEIVE);
        EXPECT_EQ_STR(s.domain_hint, std::string("LANGUAGE"),
                      "GW::LANGUAGE winner → domain_hint='LANGUAGE'");
    }

    // ── T15: self_concept < 0 → UNKNOWN ───────────────────────────────────
    printf("\nT15 domain_hint — self_concept < 0 → 'UNKNOWN'\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        auto s = er.pack(0.2f, 0.f, 0.f, 0.4f, 0.3f, 0, /*self_concept*/-1,
                          (int)GWModule::SOM, false, RouteMode::PERCEIVE);
        EXPECT_EQ_STR(s.domain_hint, std::string("UNKNOWN"),
                      "self_concept=-1 → 'UNKNOWN'");
    }

    // ── T16: multiple facts accepted — counters track ─────────────────────
    printf("\nT16 Multiple accepts: verified/unverified mixed — count total calls\n");
    {
        int fc = 0, pc = 0;
        auto er = make_wired_router(fc, pc);
        // 3 verified facts
        for (int i = 0; i < 3; i++) {
            InboundFact f{"e" + std::to_string(i), "r", double(i), true, "test"};
            er.accept_fact(f);
        }
        // 2 unverified facts (should NOT call writer)
        for (int i = 0; i < 2; i++) {
            InboundFact f{"e" + std::to_string(i), "r", double(i), false, "test"};
            er.accept_fact(f);
        }
        // 1 verified policy
        InboundPolicy p;
        p.target = "force"; p.inputs = {"m","a"};
        p.expr = opx('*', var("m"), var("a")); p.verified = true;
        er.accept_policy(p);

        EXPECT_TRUE(fc == 3, "exactly 3 fact_writer calls (3 verified)");
        EXPECT_TRUE(pc == 1, "exactly 1 policy_writer call (1 verified)");
    }

    // ── Summary ───────────────────────────────────────────────────────────
    printf("\n=== Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed == 0) {
        printf(" — ALL PASS ✓ ===\n");
    } else {
        printf(" — %d FAILED ✗ ===\n", tests_failed);
    }
    return tests_failed == 0 ? 0 : 1;
}
