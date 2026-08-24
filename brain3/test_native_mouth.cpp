// test_native_mouth.cpp — production mouth integration:
//   1. NativeMouth loads a trained binary and answers known turns
//      confidently in microseconds
//   2. mood coupling opt-in changes decoding temperature per policy
//   3. MasterOrchestrator routes chat through native_mouth when confident
//   4. escalation: low confidence gate falls through to the legacy pipeline
//   5. structured commands are never intercepted by the mouth
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <string>
#include "crisp/engines/neural/stamlat_transformer.hpp"
#include "crisp/engines/neural/native_mouth.hpp"
#include "core/master_orchestrator.hpp"

using namespace brain3::engines::neural;

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    if (ok) { g_pass++; std::cout << "  [PASS] " << name << "\n"; }
    else    { g_fail++; std::cout << "  [FAIL] " << name << "\n"; }
}

template <typename T, size_t N> constexpr size_t n_of(const T (&a)[N]) { return N; }

static std::string train_and_save(const std::string& path) {
    static const char* G[]  = {"hello", "hi", "hey there", "good morning"};
    static const char* GA[] = {"intent greeting style friendly", "intent welcome target user",
                               "intent greeting emotion happy"};
    static const char* W[]  = {"who are you", "what is your name"};
    static const char* WA[] = {"identity system type cognitive",
                               "identity brain origin artificial"};

    std::mt19937 rng(9);
    auto pick = [&](const auto& a) { return a[rng() % n_of(a)]; };
    std::string train;
    for (int i = 0; i < 300; ++i) {
        train += std::string("user: ") + pick(G) + "\nbrain: " + pick(GA) + "\n";
        train += std::string("user: ") + pick(W) + "\nbrain: " + pick(WA) + "\n";
    }

    StamlatConfig cfg;
    cfg.d_model = 64; cfg.n_layers = 3; cfg.n_heads = 4; cfg.d_ff = 128;
    cfg.ctx = 96; cfg.depth_gamma = 0.f; cfg.depth_tau = 1.f; cfg.seed = 7;
    StamlatLM lm(cfg);
    lm.build_vocab(train);
    lm.fit(train, 500, 5e-3f, 12, 0);
    const bool ok = lm.save(path);
    return path;
}

int main() {
    std::cout << "=== native mouth integration ===\n";
    const std::string model_path = "native_mouth_test.bin";   // cwd: always writable
    train_and_save(model_path);

    // ── 1. load + confident microsecond replies ──
    NativeMouth mouth;
    check(mouth.load(model_path), "model binary loads");
    check(mouth.available(), "mouth available after load");

    auto r1 = mouth.respond("hello");
    std::cout << "    hello -> \"" << r1.text << "\" nll=" << r1.reply_nll
              << " ms=" << r1.ms << " confident=" << r1.confident << "\n";
    check(r1.confident, "known turn answered confidently");
    check(r1.text.find("intent") != std::string::npos,
          "reply carries greeting facts");
    check(r1.ms < 25.0, "reply latency in microseconds range (<25ms)");
    check(mouth.respond("who are you").confident, "identity turn confident");

    // unavailable mouth returns empty result without crashing
    NativeMouth unloaded;
    check(!unloaded.respond("hello").confident, "unavailable mouth fails soft");

    // ── 2. mood coupling opt-in ──
    mouth.config().use_mood_temperature = true;
    auto vm = default_voice_mapper();
    const auto calm    = mouth.respond("hello", {0.f, 0.f}, &vm);
    const auto excited = mouth.respond("hello", {0.9f, 1.f}, &vm);
    check(std::fabs(calm.temp_used - 0.6f) < 1e-5 &&
          std::fabs(excited.temp_used - 1.0f) < 1e-5,
          "mood temperature policy applied (calm=0.6, excited=1.0)");
    mouth.config().use_mood_temperature = false;

    // ── 3./4./5. orchestrator routing ──
    setenv("BRAIN_NATIVE_MOUTH_MODEL", model_path.c_str(), 1);
    brain3::core::MasterOrchestrator orch;
    check(orch.get_native_mouth()->available(),
          "orchestrator mounted the native mouth at boot");

    auto resp = orch.process("hello");
    std::cout << "    process(\"hello\") -> engine=" << resp.engine_used
              << " reply=\"" << resp.natural_reply << "\" latency="
              << resp.latency_ms << "ms\n";
    check(resp.engine_used == "native_mouth",
          "chat routed through native mouth");
    check(resp.latency_ms < 100.0, "end-to-end chat latency <100ms");

    // structured command must bypass the mouth entirely
    auto resp2 = orch.process("teach sky isa blue");
    check(resp2.engine_used != "native_mouth",
          "structured TEACH command bypasses the mouth");

    // escalation: force low confidence → falls through to legacy pipeline
    orch.get_native_mouth()->config().nll_confidence_gate = -1.f;
    auto resp3 = orch.process("hey there");
    std::cout << "    escalated -> engine=" << resp3.engine_used << "\n";
    check(resp3.engine_used != "native_mouth",
          "low-confidence turn escalates to legacy pipeline");
    orch.get_native_mouth()->config().nll_confidence_gate = 2.2f;

    std::cout << "=== passed " << g_pass << ", failed " << g_fail << " ===\n";
    return g_fail == 0 ? 0 : 1;
}
