#pragma once
/*
 * internalrouter.hpp — Internal Signal Router for the Fuzzy Brain
 *
 * After every perceive() call the brain has a PerceiveResult + InternalState.
 * InternalRouter reads those signals and emits a single RoutingDecision that
 * tells all downstream modules what cognitive mode they are in for this step.
 *
 * This replaces the scattered `if (pc_bg.should_propagate()) ...` chains
 * throughout brain.hpp with one inspectable, testable decision point.
 *
 * ── RouteMode priority (evaluated top-down; first match wins) ───────────────
 *
 *   ATTEND      high-arousal novel event (episodic stored) OR threat (neg valence).
 *               WM + Episodic both gate hard; symbolic context pulled in.
 *
 *   REASON      WM heavily loaded (working a problem) OR Predictor very confident
 *               in familiar territory → time for deduction not more sensation.
 *               BG/LogicEngine gets priority; scratchpad ↔ symbolic sync fires.
 *
 *   IMAGINE     Approach mode with salience: explore a plausible continuation
 *               offline via Imagination rather than consuming more real input.
 *
 *   CONSOLIDATE WM quiet + low error + low arousal: rest-like state.
 *               Trigger dream-replay so episodic → predictive patterns harden.
 *
 *   PERCEIVE    Default waking perception.  perception_gain > 1 on high novelty.
 *
 *   IDLE        Every signal below threshold; minimal processing.
 *
 * ── Dependency policy ────────────────────────────────────────────────────────
 *   Includes only standalone core headers (no brain.hpp → no circular deps).
 *   decide() takes plain scalar inputs so it can be unit-tested without Brain.
 *   Brain.hpp wires decide_from() as an inline helper after PerceiveResult
 *   and InternalState are both visible.
 */

#include <cmath>
#include <string>

#include "global_workspace.hpp"   // GWModule enum

namespace brain2 {

// ─────────────────────────────────────────────────────────────────────────────
// RouteMode
// ─────────────────────────────────────────────────────────────────────────────
enum class RouteMode : int {
    PERCEIVE    = 0,  // SOM+PC pipeline, normal waking gain
    ATTEND      = 1,  // high-salience novel event: WM + Episodic both gate
    REASON      = 2,  // BG/LogicEngine active; WM provides scratchpad context
    IMAGINE     = 3,  // Imagination offline; approach + salience driven
    CONSOLIDATE = 4,  // rest-phase: episodic replay drives Predictor
    IDLE        = 5,  // all signals below threshold
};

inline const char* route_mode_name(RouteMode m) noexcept {
    switch (m) {
        case RouteMode::PERCEIVE:    return "PERCEIVE";
        case RouteMode::ATTEND:      return "ATTEND";
        case RouteMode::REASON:      return "REASON";
        case RouteMode::IMAGINE:     return "IMAGINE";
        case RouteMode::CONSOLIDATE: return "CONSOLIDATE";
        case RouteMode::IDLE:        return "IDLE";
    }
    return "UNKNOWN";
}

// ─────────────────────────────────────────────────────────────────────────────
// RoutingDecision — what the router emits each step
// ─────────────────────────────────────────────────────────────────────────────
struct RoutingDecision {
    RouteMode   mode               = RouteMode::PERCEIVE;

    // Module-level gain knobs: 0=off, 1=normal, >1=boosted
    float       perception_gain    = 1.f;  // SOM/PC modulation
    float       imagination_gain   = 0.f;  // Imagination step intensity
    float       consolidation_gain = 0.f;  // Replay intensity
    float       reasoning_gain     = 0.f;  // BG/LogicEngine priority

    // Action flags
    bool        sync_symbols       = false; // trigger scratchpad ↔ symbolic sync
    bool        trigger_episodic   = false; // force episodic commit this step
    bool        trigger_replay     = false; // trigger dream replay this step

    std::string label;                      // human-readable (logging / tests)
};

// ─────────────────────────────────────────────────────────────────────────────
// RouterThresholds — tunable without recompiling
// ─────────────────────────────────────────────────────────────────────────────
struct RouterThresholds {
    // ATTEND
    float attend_arousal     = 0.55f;   // arousal above this → ATTEND
    float attend_neg_valence = -0.15f;  // valence below this (with EMOTION win) → ATTEND

    // REASON
    float reason_wm_load     = 0.70f;   // WM load above this → REASON
    float reason_low_error   = 0.25f;   // error below this (predictor confident)
    float reason_min_wm      = 0.25f;   // WM must have some content to REASON

    // IMAGINE
    float imagine_salience   = 0.45f;   // salience above this (with approach) → IMAGINE
    float imagine_max_error  = 0.45f;   // error must be low-ish for imagination

    // CONSOLIDATE
    float consol_max_load    = 0.15f;   // WM must be quiet
    float consol_max_error   = 0.20f;   // low surprise
    float consol_max_arousal = 0.25f;   // calm state
    float consol_max_sal     = 0.30f;   // low salience

    // IDLE
    float idle_max_salience  = 0.10f;   // everything at this → IDLE
    float idle_max_error     = 0.15f;
    float idle_max_arousal   = 0.15f;

    // PERCEIVE gain boost
    float novelty_high_gain  = 0.45f;   // error above this → perception_gain > 1
};

// ─────────────────────────────────────────────────────────────────────────────
// InternalRouter
// ─────────────────────────────────────────────────────────────────────────────
class InternalRouter {
public:
    RouterThresholds thresholds;

    InternalRouter() = default;
    explicit InternalRouter(RouterThresholds t) : thresholds(std::move(t)) {}

    /**
     * decide() — core routing decision (pure function, no side-effects).
     *
     * @param error          predictor prediction_error  [0, ∞)
     * @param arousal        emotion arousal             [0, 1]
     * @param valence        emotion valence             [-1, 1]
     * @param salience       attention salience score    [0, 1]
     * @param wm_load        working-memory utilisation  [0, 1]
     * @param approach       emotion approach mode flag
     * @param episodic_stored was an episode committed this step?
     * @param gw_winner      GlobalWorkspace winner id  (cast from GWModule)
     *
     * @return RoutingDecision  — mode + gain knobs + action flags
     */
    RoutingDecision decide(
            float error,
            float arousal,
            float valence,
            float salience,
            float wm_load,
            bool  approach,
            bool  episodic_stored,
            int   gw_winner) const noexcept
    {
        const auto& T = thresholds;
        RoutingDecision d;

        // ── ATTEND (highest priority) ─────────────────────────────────────
        //   Case A: a surprising event was committed to episodic memory AND
        //           the brain is aroused (it noticed this mattered).
        //   Case B: strong negative valence while EMOTION holds the workspace
        //           (threat / painful surprise).
        const bool attend_a = episodic_stored && (arousal > T.attend_arousal);
        const bool attend_b = (valence < T.attend_neg_valence) &&
                              (gw_winner == static_cast<int>(GWModule::EMOTION));
        if (attend_a || attend_b) {
            d.mode             = RouteMode::ATTEND;
            d.perception_gain  = 1.5f;
            d.reasoning_gain   = 0.3f;  // light reasoning on salient events
            d.sync_symbols     = true;  // pull symbolic context in
            d.trigger_episodic = true;
            d.label = attend_a ? "ATTEND(episodic+arousal)"
                               : "ATTEND(neg-emotion)";
            return d;
        }

        // ── REASON ───────────────────────────────────────────────────────
        //   Case A: WM is heavily loaded — brain is actively working a problem.
        //   Case B: Predictor is confident (low error, PREDICT wins the GW)
        //           AND WM has some context — time to deduce, not perceive more.
        const bool reason_a = wm_load > T.reason_wm_load;
        const bool reason_b = (gw_winner == static_cast<int>(GWModule::PREDICT)) &&
                              (error < T.reason_low_error) &&
                              (wm_load > T.reason_min_wm);
        if (reason_a || reason_b) {
            d.mode           = RouteMode::REASON;
            d.reasoning_gain = 1.f;
            d.sync_symbols   = true;  // BG needs symbolic context
            d.label = reason_a ? "REASON(wm-load)" : "REASON(confident-predict)";
            return d;
        }

        // ── IMAGINE ──────────────────────────────────────────────────────
        //   Approach mode + salience + low-ish error: the brain sees something
        //   interesting and wants to simulate continuations rather than wait
        //   for more real input.  imagination_gain scales with salience so
        //   very interesting signals get longer / stronger simulations.
        const bool imagine_ok = approach &&
                                (salience > T.imagine_salience) &&
                                (error    < T.imagine_max_error);
        if (imagine_ok) {
            d.mode             = RouteMode::IMAGINE;
            d.imagination_gain = std::min(salience, 1.f);
            d.label = "IMAGINE(approach+salience)";
            return d;
        }

        // ── IDLE ───────────────────────────────────────────────────────
        //   Checked BEFORE CONSOLIDATE: a fully dormant brain (all signals near
        //   zero) is IDLE, not resting/consolidating — consolidation requires
        //   at least a trace of prior activity (some episodic / WM history).
        //   Tighter thresholds than CONSOLIDATE intentionally.
        const bool idle_ok = (salience < T.idle_max_salience) &&
                             (error    < T.idle_max_error)    &&
                             (arousal  < T.idle_max_arousal);
        if (idle_ok) {
            d.mode            = RouteMode::IDLE;
            d.perception_gain = 0.5f;
            d.label = "IDLE(all-low)";
            return d;
        }

        // ── CONSOLIDATE ───────────────────────────────────────────────────
        //   Rest-like state: WM quiet, error low, arousal low.
        //   Fire dream-replay so episodic → predictive patterns harden.
        //   Explicitly check salience to ensure we aren't just in IDLE.
        const bool consol_ok = (wm_load  < T.consol_max_load)    &&
                               (error    < T.consol_max_error)    &&
                               (arousal  < T.consol_max_arousal)  &&
                               (salience < T.consol_max_sal)      &&
                               (salience > 0.05f);
        if (consol_ok) {
            d.mode               = RouteMode::CONSOLIDATE;
            d.consolidation_gain = 1.f;
            d.trigger_replay     = true;
            d.label = "CONSOLIDATE(rest-phase)";
            return d;
        }

        // ── PERCEIVE (default) ───────────────────────────────────────────
        //   Normal waking perception.  High novelty → boost perception_gain
        //   so SOM + WM gate more aggressively on this surprising input.
        d.mode = RouteMode::PERCEIVE;
        if (error > T.novelty_high_gain) {
            // Clamp boost to [1.0, 1.5]
            d.perception_gain = 1.f + std::min(error, 1.f) * 0.5f;
            d.label = "PERCEIVE(novel-high-gain)";
        } else {
            d.perception_gain = 1.f;
            d.label = "PERCEIVE(normal)";
        }
        return d;
    }
};

} // namespace brain2
