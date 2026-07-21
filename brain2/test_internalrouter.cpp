/*
 * test_internalrouter.cpp — Unit tests for InternalRouter
 *
 * Standalone C++ test: no pybind, no Python, no Brain.
 * Builds with:
 *   g++ -std=c++17 -I. test_internalrouter.cpp -o test_internalrouter && ./test_internalrouter
 *
 * Tests:
 *   T1  PERCEIVE(normal)      — typical waking input
 *   T2  PERCEIVE(novel)       — high prediction error → boosted gain
 *   T3  ATTEND(episodic)      — stored episode + high arousal
 *   T4  ATTEND(neg-emotion)   — EMOTION wins GW + negative valence
 *   T5  REASON(wm-load)       — working memory heavily loaded
 *   T6  REASON(confident)     — PREDICT wins GW + low error + some WM
 *   T7  IMAGINE               — approach mode + salience + low error
 *   T8  CONSOLIDATE           — rest-like state
 *   T9  IDLE                  — everything near zero
 *   T10 sync_symbols flag     — REASON sets sync_symbols=true
 *   T11 trigger_episodic flag — ATTEND sets trigger_episodic=true
 *   T12 trigger_replay flag   — CONSOLIDATE sets trigger_replay=true
 *   T13 imagination_gain      — scales with salience in IMAGINE
 *   T14 perception_gain boost — high error → gain > 1.0 in PERCEIVE
 *   T15 Custom thresholds     — overriding attend_arousal works
 */

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

// Include only the router header — no Brain, no pybind
#include "core/internalrouter.hpp"

// ── GWModule int values used in tests ──────────────────────────────────────
static constexpr int GW_SOM     = (int)brain2::GWModule::SOM;
static constexpr int GW_PREDICT = (int)brain2::GWModule::PREDICT;
static constexpr int GW_EMOTION = (int)brain2::GWModule::EMOTION;

// ── Minimal test harness ───────────────────────────────────────────────────
static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define EXPECT_EQ(a, b, msg) do { \
    tests_run++; \
    if ((a) == (b)) { \
        printf("  [PASS] %s\n", msg); \
        tests_passed++; \
    } else { \
        printf("  [FAIL] %s  (got '%s', expected '%s')\n", \
               msg, brain2::route_mode_name(a), brain2::route_mode_name(b)); \
        tests_failed++; \
    } \
} while(0)

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

#define EXPECT_FALSE(cond, msg) EXPECT_TRUE(!(cond), msg)

#define EXPECT_GT(a, b, msg) do { \
    tests_run++; \
    if ((a) > (b)) { \
        printf("  [PASS] %s  (%.3f > %.3f)\n", msg, (float)(a), (float)(b)); \
        tests_passed++; \
    } else { \
        printf("  [FAIL] %s  (%.3f not > %.3f)\n", msg, (float)(a), (float)(b)); \
        tests_failed++; \
    } \
} while(0)

// ── Convenience alias ──────────────────────────────────────────────────────
using RM = brain2::RouteMode;
using IR = brain2::InternalRouter;

int main() {
    printf("=== InternalRouter unit tests ===\n\n");

    IR router;  // default thresholds

    // ── T1: PERCEIVE(normal) ───────────────────────────────────────────────
    printf("T1  PERCEIVE(normal) — moderate inputs\n");
    {
        auto d = router.decide(
            /*error*/    0.30f,
            /*arousal*/  0.30f,
            /*valence*/  0.10f,
            /*salience*/ 0.35f,
            /*wm_load*/  0.40f,
            /*approach*/ false,
            /*ep_stored*/false,
            /*gw_winner*/GW_SOM);
        EXPECT_EQ(d.mode, RM::PERCEIVE, "mode == PERCEIVE");
        EXPECT_TRUE(d.perception_gain >= 1.0f, "perception_gain >= 1.0");
        EXPECT_FALSE(d.sync_symbols, "sync_symbols == false");
    }

    // ── T2: PERCEIVE(novel-high-gain) ─────────────────────────────────────
    printf("\nT2  PERCEIVE(novel-high-gain) — error > 0.45\n");
    {
        auto d = router.decide(0.80f, 0.35f, 0.05f, 0.40f, 0.20f,
                               false, false, GW_SOM);
        EXPECT_EQ(d.mode, RM::PERCEIVE, "mode == PERCEIVE");
        EXPECT_GT(d.perception_gain, 1.0f, "perception_gain > 1.0 on novel input");
    }

    // ── T3: ATTEND(episodic+arousal) ──────────────────────────────────────
    printf("\nT3  ATTEND(episodic+arousal) — stored episode AND arousal > 0.55\n");
    {
        auto d = router.decide(0.60f, 0.70f, 0.05f, 0.50f, 0.30f,
                               false, /*ep_stored=*/true, GW_SOM);
        EXPECT_EQ(d.mode, RM::ATTEND, "mode == ATTEND");
        EXPECT_TRUE(d.trigger_episodic, "trigger_episodic == true");
        EXPECT_TRUE(d.sync_symbols,     "sync_symbols == true");
        EXPECT_GT(d.perception_gain, 1.0f, "perception_gain > 1.0 (ATTEND boosts)");
    }

    // ── T4: ATTEND(neg-emotion) ───────────────────────────────────────────
    printf("\nT4  ATTEND(neg-emotion) — EMOTION wins + valence < -0.15\n");
    {
        auto d = router.decide(0.25f, 0.40f, /*valence*/-0.30f,
                               0.40f, 0.40f, false, false, GW_EMOTION);
        EXPECT_EQ(d.mode, RM::ATTEND, "mode == ATTEND");
        EXPECT_TRUE(d.trigger_episodic, "trigger_episodic == true");
    }

    // ── T5: REASON(wm-load) ───────────────────────────────────────────────
    printf("\nT5  REASON(wm-load) — wm_load > 0.70\n");
    {
        auto d = router.decide(0.20f, 0.30f, 0.10f, 0.35f,
                               /*wm_load*/0.85f, false, false, GW_SOM);
        EXPECT_EQ(d.mode, RM::REASON, "mode == REASON");
        EXPECT_TRUE(d.sync_symbols,  "sync_symbols == true");
        EXPECT_GT(d.reasoning_gain, 0.f, "reasoning_gain > 0");
    }

    // ── T6: REASON(confident-predict) ─────────────────────────────────────
    printf("\nT6  REASON(confident-predict) — PREDICT wins + low error + WM has content\n");
    {
        auto d = router.decide(/*error*/0.10f, 0.25f, 0.05f, 0.35f,
                               /*wm_load*/0.50f, false, false, GW_PREDICT);
        EXPECT_EQ(d.mode, RM::REASON, "mode == REASON");
        EXPECT_TRUE(d.sync_symbols, "sync_symbols == true");
    }

    // ── T7: IMAGINE ───────────────────────────────────────────────────────
    printf("\nT7  IMAGINE — approach=true + salience > 0.45 + error < 0.45\n");
    {
        auto d = router.decide(0.20f, 0.35f, 0.15f,
                               /*salience*/0.60f, 0.30f,
                               /*approach*/true, false, GW_SOM);
        EXPECT_EQ(d.mode, RM::IMAGINE, "mode == IMAGINE");
        EXPECT_GT(d.imagination_gain, 0.f, "imagination_gain > 0");
    }

    // ── T8: CONSOLIDATE ──────────────────────────────────────────────────────
    printf("\nT8  CONSOLIDATE — rest-like state (all signals low, salience just above IDLE floor)\n");
    {
        auto d = router.decide(/*error*/0.05f, /*arousal*/0.10f, /*valence*/0.02f,
                               /*salience*/0.12f, /*wm_load*/0.05f,
                               false, false, GW_SOM);
        EXPECT_EQ(d.mode, RM::CONSOLIDATE, "mode == CONSOLIDATE");
        EXPECT_TRUE(d.trigger_replay,     "trigger_replay == true");
        EXPECT_GT(d.consolidation_gain, 0.f, "consolidation_gain > 0");
    }

    // ── T9: IDLE ──────────────────────────────────────────────────────────
    printf("\nT9  IDLE — everything near zero\n");
    {
        auto d = router.decide(0.01f, 0.01f, 0.00f, 0.02f, 0.01f,
                               false, false, GW_SOM);
        EXPECT_EQ(d.mode, RM::IDLE, "mode == IDLE");
        EXPECT_TRUE(d.perception_gain < 1.0f, "perception_gain < 1.0 (reduced)");
    }

    // ── T10: sync_symbols only in REASON ──────────────────────────────────
    printf("\nT10 sync_symbols is true in REASON, false in PERCEIVE\n");
    {
        auto reason = router.decide(0.10f, 0.25f, 0.05f, 0.35f, 0.85f, false, false, GW_SOM);
        auto percv  = router.decide(0.30f, 0.30f, 0.10f, 0.35f, 0.40f, false, false, GW_SOM);
        EXPECT_TRUE(reason.sync_symbols,  "REASON: sync_symbols == true");
        EXPECT_FALSE(percv.sync_symbols,  "PERCEIVE: sync_symbols == false");
    }

    // ── T11: trigger_episodic in ATTEND ───────────────────────────────────
    printf("\nT11 trigger_episodic is true in ATTEND, false in PERCEIVE\n");
    {
        auto attend = router.decide(0.60f, 0.70f, 0.05f, 0.50f, 0.30f, false, true, GW_SOM);
        auto percv  = router.decide(0.30f, 0.30f, 0.10f, 0.35f, 0.40f, false, false, GW_SOM);
        EXPECT_TRUE(attend.trigger_episodic,  "ATTEND: trigger_episodic == true");
        EXPECT_FALSE(percv.trigger_episodic,  "PERCEIVE: trigger_episodic == false");
    }

    // ── T12: trigger_replay in CONSOLIDATE ────────────────────────────────────
    printf("\nT12 trigger_replay is true in CONSOLIDATE, false in REASON\n");
    {
        auto consol = router.decide(0.05f, 0.10f, 0.02f, 0.12f, 0.05f, false, false, GW_SOM);
        auto reason = router.decide(0.10f, 0.25f, 0.05f, 0.35f, 0.85f, false, false, GW_SOM);
        EXPECT_TRUE(consol.trigger_replay,  "CONSOLIDATE: trigger_replay == true");
        EXPECT_FALSE(reason.trigger_replay, "REASON: trigger_replay == false");
    }

    // ── T13: imagination_gain scales with salience ─────────────────────────
    printf("\nT13 imagination_gain scales with salience in IMAGINE\n");
    {
        auto low_sal  = router.decide(0.20f, 0.35f, 0.15f, 0.50f, 0.30f, true, false, GW_SOM);
        auto high_sal = router.decide(0.20f, 0.35f, 0.15f, 0.90f, 0.30f, true, false, GW_SOM);
        EXPECT_TRUE(low_sal.mode  == RM::IMAGINE, "low_sal: mode == IMAGINE");
        EXPECT_TRUE(high_sal.mode == RM::IMAGINE, "high_sal: mode == IMAGINE");
        EXPECT_GT(high_sal.imagination_gain, low_sal.imagination_gain,
                  "higher salience → higher imagination_gain");
    }

    // ── T14: perception_gain boost on high error ───────────────────────────
    printf("\nT14 perception_gain > 1.0 when error > novelty_high_gain threshold\n");
    {
        auto low_err  = router.decide(0.20f, 0.30f, 0.10f, 0.35f, 0.40f, false, false, GW_SOM);
        auto high_err = router.decide(0.90f, 0.30f, 0.10f, 0.35f, 0.20f, false, false, GW_SOM);
        EXPECT_TRUE(low_err.mode  == RM::PERCEIVE, "low_err: PERCEIVE");
        EXPECT_TRUE(high_err.mode == RM::PERCEIVE, "high_err: PERCEIVE");
        EXPECT_TRUE(low_err.perception_gain  == 1.0f, "low_err: gain == 1.0 (normal)");
        EXPECT_GT(high_err.perception_gain, 1.0f, "high_err: gain > 1.0 (boosted)");
    }

    // ── T15: Custom thresholds ─────────────────────────────────────────────
    printf("\nT15 Custom threshold: attend_arousal = 0.9 (very high)\n");
    {
        // With default thresholds, arousal=0.70 + episodic → ATTEND.
        // With custom threshold of 0.9, same input should NOT be ATTEND.
        brain2::RouterThresholds T;
        T.attend_arousal = 0.90f;   // very high bar
        IR strict_router(T);
        auto d = strict_router.decide(0.60f, 0.70f, 0.05f, 0.50f, 0.30f, false, true, GW_SOM);
        EXPECT_TRUE(d.mode != RM::ATTEND,
                    "custom threshold: arousal=0.70 < 0.90 → not ATTEND");
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
