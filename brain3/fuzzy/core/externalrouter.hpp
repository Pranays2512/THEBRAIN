#pragma once
/*
 * externalrouter.hpp — Fuzzy↔Crisp Membrane Controller
 *
 * Every signal that crosses the C++↔Python boundary passes through the
 * ExternalRouter.  The membrane is ASYMMETRIC by design:
 *
 *   Outbound (C++ → Python / crisp layer)
 *     pack() bundles PerceiveResult scalars + RouteMode + domain_hint into an
 *     OutboundSignal.  Python reads this to decide which crisp engine to call
 *     and how confident to be in the request.
 *
 *   Inbound (Python / crisp layer → C++)
 *     accept_fact()   — strict gate: ONLY verified=true facts cross.
 *                       Unverified or conflicting values are rejected / logged.
 *     accept_policy() — strict gate: ONLY verified=true policies cross.
 *
 *   Brain wires two callbacks (FactWriter / PolicyWriter) in its constructor
 *   so ExternalRouter never needs to #include brain.hpp — no circular deps.
 *
 * ── Domain hints ─────────────────────────────────────────────────────────────
 *   self_concept (0..N-1) from SelfModel partitions the concept space into
 *   domain buckets.  Initially this is a modulo mapping; as the brain trains
 *   the SelfSOM clusters will align with real domains and the mapping can be
 *   updated via set_domain_map().
 *
 * ── Dependency policy ────────────────────────────────────────────────────────
 *   Only includes: internalrouter.hpp, policy_engine.hpp, standard headers.
 *   Does NOT include brain.hpp.  Brain wires callbacks at construction time.
 */

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "internalrouter.hpp"   // RouteMode
#include "crisp/core/policy_engine.hpp"    // ExprPtr, PolicyMemory (for InboundPolicy type)

namespace brain2 {

// ─────────────────────────────────────────────────────────────────────────────
// OutboundSignal — what the C++ brain exports to the Python crisp layer
// ─────────────────────────────────────────────────────────────────────────────
struct OutboundSignal {
    // ── Neural state ────────────────────────────────────────────────────────
    float  novelty        = 0.f;  // prediction_error (how surprised the fuzzy brain is)
    float  valence        = 0.f;  // emotion valence [-1, 1]
    float  arousal        = 0.f;  // emotion arousal [0, 1]
    float  salience       = 0.f;  // attention salience [0, 1]
    float  wm_load        = 0.f;  // working memory utilisation [0, 1]

    // ── Cognitive state ─────────────────────────────────────────────────────
    int       bmu          = -1;              // SOM best-matching unit index
    int       self_concept = -1;             // SelfModel cluster (domain identity)
    int       gw_winner    = -1;             // GlobalWorkspace winner module id
    RouteMode internal_mode = RouteMode::PERCEIVE; // what InternalRouter decided

    // ── Routing hints for the crisp layer ───────────────────────────────────
    std::string domain_hint;   // "MATH" | "PHYSICS" | "CODE" | "LANGUAGE" | "UNKNOWN"
    bool        gate_open = false; // true → crisp layer should run reasoning

    // ── Confidence estimate ─────────────────────────────────────────────────
    // [0, 1]: how reliable is the fuzzy signal?
    //   High = brain is confident it knows this domain (low error, good WM state)
    //   Low  = novel territory — crisp should be careful / use conjecture path
    float confidence = 0.f;
};

// ─────────────────────────────────────────────────────────────────────────────
// InboundFact — a crisp fact proposed from Python → C++ brain
// ─────────────────────────────────────────────────────────────────────────────
struct InboundFact {
    std::string entity;
    std::string relation;
    double      value     = 0.0;
    bool        verified  = false; // GATE: only true passes the membrane
    std::string source;            // audit trail ("conjecture_sandbox", "policy_induction", ...)
};

// ─────────────────────────────────────────────────────────────────────────────
// InboundPolicy — a crisp policy proposed from Python → C++ brain
// ─────────────────────────────────────────────────────────────────────────────
struct InboundPolicy {
    std::string              target;
    std::vector<std::string> inputs;
    ExprPtr                  expr;    // formula AST (from policy_engine.hpp)
    bool                     verified = false; // GATE: only true passes the membrane
    std::string              source;
};

// ─────────────────────────────────────────────────────────────────────────────
// InboundDecision — gate result returned to caller
// ─────────────────────────────────────────────────────────────────────────────
struct InboundDecision {
    bool        accepted = false;
    std::string reason;   // "ok" | "unverified" | "null_writer" | "null_expr"
};

// ─────────────────────────────────────────────────────────────────────────────
// ExternalRouter
// ─────────────────────────────────────────────────────────────────────────────
class ExternalRouter {
public:
    // Callbacks wired by Brain in its constructor — avoids #including brain.hpp
    using FactWriter   = std::function<void(const std::string&,
                                            const std::string&,
                                            double)>;
    using PolicyWriter = std::function<void(const std::string&,
                                            const std::vector<std::string>&,
                                            const ExprPtr&)>;

    ExternalRouter() = default;

    void set_fact_writer(FactWriter fw)     { fact_writer_   = std::move(fw); }
    void set_policy_writer(PolicyWriter pw) { policy_writer_ = std::move(pw); }

    // ── Domain map ───────────────────────────────────────────────────────────
    // Maps self_concept index → domain name.
    // Default: modulo-4 bucketing (LANGUAGE / MATH / PHYSICS / CODE).
    // Once the SelfSOM trains, the caller can replace this with a learned map.
    void set_domain_map(std::map<int, std::string> m) { domain_map_ = std::move(m); }

    // ── pack() — build an OutboundSignal from raw perceive scalars ───────────
    /**
     * @param error          predictor prediction_error
     * @param valence        emotion valence
     * @param arousal        emotion arousal
     * @param salience       attention salience
     * @param wm_load        working-memory utilisation
     * @param bmu            SOM best-matching unit
     * @param self_concept   SelfModel cluster index
     * @param gw_winner      GlobalWorkspace winner id
     * @param episodic_stored was an episode committed?
     * @param mode           RoutingDecision.mode from InternalRouter
     */
    OutboundSignal pack(
            float     error,
            float     valence,
            float     arousal,
            float     salience,
            float     wm_load,
            int       bmu,
            int       self_concept,
            int       gw_winner,
            bool      episodic_stored,
            RouteMode mode) const noexcept
    {
        OutboundSignal s;
        s.novelty       = error;
        s.valence       = valence;
        s.arousal       = arousal;
        s.salience      = salience;
        s.wm_load       = wm_load;
        s.bmu           = bmu;
        s.self_concept  = self_concept;
        s.gw_winner     = gw_winner;
        s.internal_mode = mode;
        s.domain_hint   = resolve_domain(self_concept, gw_winner);

        // Gate: open if attention passed (salience > 0) AND brain is not idle
        s.gate_open = (salience > 0.05f) && (mode != RouteMode::IDLE);

        // Confidence: high when error is low + WM has context + not in novel territory
        // Clamped to [0, 1].
        float base_conf = 1.f - std::min(error, 1.f);      // low error → high conf
        float wm_bonus  = wm_load * 0.3f;                  // WM context boosts conf
        float novel_pen = episodic_stored ? -0.15f : 0.f;  // novel event → less conf
        s.confidence = std::max(0.f, std::min(1.f, base_conf + wm_bonus + novel_pen));

        return s;
    }

    // ── accept_fact() — inbound gate for crisp facts ─────────────────────────
    /**
     * GATE: only verified=true facts cross the membrane.
     * On pass: calls the Brain's teach_fact() callback.
     * On reject: returns an InboundDecision with the rejection reason.
     */
    InboundDecision accept_fact(const InboundFact& f) const {
        if (!f.verified) {
            facts_rejected_++;
            return {false, "unverified"};
        }
        if (!fact_writer_) {
            facts_rejected_++;
            return {false, "null_writer"};
        }
        fact_writer_(f.entity, f.relation, f.value);
        facts_accepted_++;
        return {true, "ok"};
    }

    // ── accept_policy() — inbound gate for crisp policies ────────────────────
    /**
     * GATE: only verified=true policies cross the membrane.
     * On pass: calls the Brain's policy_add() callback.
     */
    InboundDecision accept_policy(const InboundPolicy& p) const {
        if (!p.verified) {
            policies_rejected_++;
            return {false, "unverified"};
        }
        if (!policy_writer_) {
            policies_rejected_++;
            return {false, "null_writer"};
        }
        if (!p.expr) {
            policies_rejected_++;
            return {false, "null_expr"};
        }
        policy_writer_(p.target, p.inputs, p.expr);
        policies_accepted_++;
        return {true, "ok"};
    }

    // ── audit helpers ────────────────────────────────────────────────────────
    size_t facts_accepted()    const noexcept { return facts_accepted_; }
    size_t facts_rejected()    const noexcept { return facts_rejected_; }
    size_t policies_accepted() const noexcept { return policies_accepted_; }
    size_t policies_rejected() const noexcept { return policies_rejected_; }

    void reset_counters() noexcept {
        facts_accepted_ = facts_rejected_ = 0;
        policies_accepted_ = policies_rejected_ = 0;
    }

private:
    FactWriter   fact_writer_;
    PolicyWriter policy_writer_;

    std::map<int, std::string> domain_map_;  // custom self_concept → domain name

    mutable size_t facts_accepted_    = 0;
    mutable size_t facts_rejected_    = 0;
    mutable size_t policies_accepted_ = 0;
    mutable size_t policies_rejected_ = 0;

    // Default domain resolution: custom map first, then modulo-4 bucketing.
    std::string resolve_domain(int self_concept, int gw_winner) const {
        // 1. custom map (set after training)
        if (!domain_map_.empty()) {
            auto it = domain_map_.find(self_concept);
            if (it != domain_map_.end()) return it->second;
        }

        // 2. GW winner override for well-known modules
        //    PREDICT winning with a known concept → lean on LANGUAGE
        if (gw_winner == static_cast<int>(GWModule::LANGUAGE)) return "LANGUAGE";

        // 3. Modulo-4 default (updates automatically as self-SOM trains)
        if (self_concept < 0) return "UNKNOWN";
        static const char* DOMAINS[] = {"LANGUAGE", "MATH", "PHYSICS", "CODE"};
        return DOMAINS[self_concept % 4];
    }
};

} // namespace brain2
