// test_metacognition.cpp — the brain watching itself think.
//   1. Sentiment perceptor detects positive/negative/neutral text
//   2. Metacognition catches contradictions in reasoning traces
//   3. Metacognition flags unsupported conclusions
//   4. Clean traces pass audit
#include <iostream>
#include "core/metacognition.hpp"

using namespace brain3::core;
static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    if (ok) { g_pass++; std::cout << "  [PASS] " << name << "\n"; }
    else    { g_fail++; std::cout << "  [FAIL] " << name << "\n"; }
}

int main() {
    std::cout << "=== metacognition + sentiment ===\n";

    // ── 1. sentiment perceptor ──
    SentimentPerceptor sp;
    auto pos = sp.perceive("this is great and wonderful news");
    auto neg = sp.perceive("this is terrible and wrong");
    auto neutral = sp.perceive("the file contains data");

    check(pos.valence > 0, "positive text detected");
    check(neg.valence < 0, "negative text detected");
    check(neutral.valence == 0 && !neutral.valence,
          "neutral text scores zero valence");

    // ── 2. metacognition: contradiction detection ──
    {
        MetacognitionEngine meta;
        meta.begin_trace("test_contradiction");
        meta.add_step({"math", "solve", "sky", "color", "blue", true, 0.9});
        meta.add_step({"kb", "lookup", "sky", "color", "green", true, 0.8});
        auto f = meta.check_contradictions();
        check(f.has_contradiction, "contradiction detected when two verified steps disagree");
    }

    // ── 3. clean trace passes ──
    {
        MetacognitionEngine meta;
        meta.begin_trace("clean");
        meta.add_step({"math", "verify", "x", "=", "5", true, 1.0});
        meta.add_step({"speech", "render", "x = 5", "", "", true, 0.9});
        auto f = meta.full_audit();
        check(f.clean(), "clean trace has no findings");
    }

    // ── 4. unsupported conclusion flagged ──
    {
        MetacognitionEngine meta;
        meta.begin_trace("unsupported");
        // no verified premise before this step
        meta.add_step({"intuition", "assert", "theorem", "proved", "true", true, 0.8});
        auto f = meta.check_unsupported();
        check(f.has_unsupported, "unsupported conclusion detected");
    }

    // ── 5. circular dependency detected ──
    {
        MetacognitionEngine meta;
        meta.begin_trace("circular");
        meta.add_step({"engine_a", "derive", "concept_b", "from", "concept_a", true, 1.0});
        meta.add_step({"engine_b", "derive", "concept_a", "from", "concept_b", true, 1.0});
        auto f = meta.full_audit();
        check(!f.details.empty(), "circular pattern caught by audit");
    }

    std::cout << "=== passed " << g_pass << ", failed " << g_fail << " ===\n";
    return g_fail == 0 ? 0 : 1;
}
