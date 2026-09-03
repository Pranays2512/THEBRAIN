#pragma once
/**
 * brain3/core/metacognition.hpp
 *
 * METACOGNITION — the brain watching itself think.
 *
 * Captures reasoning traces (which engines fired, what they produced,
 * whether claims were verified), then runs pattern detectors over those
 * traces to catch structural reasoning failures:
 *   - CONTRADICTION: two trace steps assert incompatible facts
 *   - UNSUPPORTED:   conclusion drawn with no verified premise chain
 *   - CIRCULAR:      step N depends on step M which depends on step N
 *   - REGRESSION:    post-consolidation quality dropped vs pre-consolidation
 *
 * When metacognitive flags fire, confidence in the reasoning chain drops
 * and the rollback gate tightens.
 */
#include <string>
#include <vector>
#include <map>
#include <set>

namespace brain3 {
namespace core {

struct TraceStep {
    std::string engine;          // which engine produced this step
    std::string operation;       // what was asked
    std::string subject, relation, object;
    bool verified = false;
    double confidence = 1.0;
};

class MetacognitionEngine {
public:
    void begin_trace(const std::string& context) {
        current_context_ = context;
        trace_.clear();
    }

    void add_step(const TraceStep& step) { trace_.push_back(step); }

    // ── pattern detectors ────────────────────────────────────────────────
    struct Findings {
        bool has_contradiction = false;
        bool has_unsupported = false;
        bool has_circular = false;
        std::vector<std::string> details;
        double overall_confidence = 1.0;

        bool clean() const {
            return !has_contradiction && !has_unsupported && !has_circular;
        }
    };

    // Check for direct contradictions within the trace:
    // same subject+relation asserted with different objects AND both verified
    Findings check_contradictions() const {
        Findings f;
        for (size_t i = 0; i < trace_.size(); ++i) {
            for (size_t j = i + 1; j < trace_.size(); ++j) {
                if (trace_[i].subject == trace_[j].subject &&
                    trace_[i].relation == trace_[j].relation &&
                    trace_[i].object != trace_[j].object &&
                    trace_[i].verified && trace_[j].verified) {
                    f.has_contradiction = true;
                    f.details.push_back(
                        "CONTRADICTION: '" + trace_[i].subject + "' asserted as '"
                        + trace_[i].object + "' and '" + trace_[j].object + "'");
                }
            }
        }
        return f;
    }

    // Check for unsupported conclusions:
    // final step is marked verified but has no verified premises before it
    Findings check_unsupported() const {
        Findings f;
        if (trace_.empty()) return f;
        const auto& last = trace_.back();
        if (!last.verified || last.engine == "long_term_memory" || last.engine == "axiom") return f;  // axioms / direct teachings need no prior premise
        bool has_premise = false;
        for (const auto& s : trace_) {
            if (&s == &trace_.back()) continue;
            if (s.verified && s.object == last.subject) { has_premise = true; break; }
        }
        if (!has_premise && !last.subject.empty() && !last.relation.empty()) {
            f.has_unsupported = true;
            f.details.push_back("UNSUPPORTED: '" + last.subject + " "
                               + last.relation + " " + last.object
                               + "' has no verified premise chain");
        }
        return f;
    }

    // Check for circular dependencies: step A's output feeds step B,
    // whose output feeds back to step A
    Findings check_circular() const {
        Findings f;
        for (size_t i = 0; i < trace_.size(); ++i)
            for (size_t j = i + 1; j < trace_.size(); ++j) {
                if (trace_[i].subject == trace_[j].object &&
                    trace_[j].subject == trace_[i].object &&
                    trace_[i].verified && trace_[j].verified) {
                    f.has_circular = true;
                    f.details.push_back("CIRCULAR: '" + trace_[i].object
                                       + "' ↔ '" + trace_[j].object + "'");
                }
            }
        return f;
    }

    Findings full_audit() const {
        auto c = check_contradictions();
        auto u = check_unsupported();
        auto ci = check_circular();
        Findings all;
        all.has_contradiction = c.has_contradiction;
        all.has_unsupported = u.has_unsupported;
        all.has_circular = ci.has_circular;
        all.details = c.details;
        for (auto& d : u.details) all.details.push_back(d);
        for (auto& d : ci.details) all.details.push_back(d);
        all.overall_confidence =
            (all.has_contradiction || all.has_circular) ? 0.3 :
            all.has_unsupported ? 0.6 : 1.0;
        return all;
    }

    const std::vector<TraceStep>& trace() const { return trace_; }
    const std::string& context() const { return current_context_; }

    // Cross-turn memory: the orchestrator keeps a rolling window of recent
    // steps so detectors can fire ACROSS turns (e.g. a fact taught on Monday
    // contradicted on Tuesday). Load before auditing; dump after appending.
    void load_history(const std::vector<TraceStep>& prior) {
        history_ = prior;
    }

    std::vector<TraceStep> combined_trace() const {
        std::vector<TraceStep> all = history_;
        all.insert(all.end(), trace_.begin(), trace_.end());
        return all;
    }

private:
    std::string current_context_;
    std::vector<TraceStep> trace_;
    std::vector<TraceStep> history_;

public:
    // Audit over the COMBINED (history + current) trace.
    Findings full_audit_cross_turn() const {
        MetacognitionEngine tmp;
        tmp.trace_ = combined_trace();
        return tmp.full_audit();
    }
};

// ── SENTIMENT PERCEPTION ─────────────────────────────────────────────────────
// Lightweight lexicon-based sentiment classifier. Feeds valence/arousal
// updates to the Emotion system when the brain reads text.
class SentimentPerceptor {
public:
    struct Sentiment { float valence = 0.f; float arousal = 0.f; };

    SentimentPerceptor() {
        positive_ = {"good","great","happy","excellent","love","wonderful",
                     "amazing","best","beautiful","success","win","optimal",
                     "positive","friendly","warm","welcome","joy"};
        negative_ = {"bad","terrible","awful","hate","worst","horrible",
                     "failure","loss","pain","negative","hostile","cold",
                     "angry","fear","danger","broken","wrong"};
        high_arousal_ = {"!", "urgent", "critical", "emergency", "immediately",
                         "shocked", "screamed", "exploded", "crashed"};
    }

    Sentiment perceive(const std::string& text) const {
        Sentiment s;
        std::string lower;
        for (char c : text) lower += (char)std::tolower((unsigned char)c);
        int pos = 0, neg = 0, arous = 0;
        for (const auto& w : positive_)
            if (lower.find(w) != std::string::npos) ++pos;
        for (const auto& w : negative_)
            if (lower.find(w) != std::string::npos) ++neg;
        for (const auto& w : high_arousal_)
            if (lower.find(w) != std::string::npos) ++arous;
        s.valence = std::max(-1.f, std::min(1.f, (float)(pos - neg) / 3.f));
        s.arousal = std::max(0.f, std::min(1.f, 0.3f + 0.2f * arous));
        return s;
    }

private:
    std::vector<std::string> positive_, negative_, high_arousal_;
};

} // namespace core
} // namespace brain3
