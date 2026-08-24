// train_native_mouth.cpp — trains and saves the production mouth binary
//
// usage: train_native_mouth [out_path] [--quick]
// default output: mouth_native.bin (picked up by MasterOrchestrator at boot)
#include <cstdio>
#include <string>
#include "crisp/engines/neural/stamlat_transformer.hpp"

using namespace brain3::engines::neural;

template <typename T, size_t N> constexpr size_t n_of(const T (&a)[N]) { return N; }

int main(int argc, char** argv) {
    std::string out = "mouth_native.bin";
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quick") quick = true;
        else out = a;
    }

    static const char* G[]  = {"hello", "hi", "hey there", "good morning",
                               "greetings", "good evening"};
    static const char* GA[] = {"intent greeting style friendly", "intent welcome target user",
                               "intent salutation status ready", "intent greeting emotion happy",
                               "intent greeting emotion positive", "intent welcome style warm"};
    static const char* W[]  = {"who are you", "what is your name", "what are you",
                               "tell me about yourself"};
    static const char* WA[] = {"identity system type cognitive", "identity brain origin artificial",
                               "name brain type ai", "self network type neural",
                               "identity brain type cognitive neural"};
    static const char* S[]  = {"how are you", "how do you feel", "what is your state"};
    static const char* SA[] = {"status good energy high", "state positive mode ready",
                               "feeling great condition excellent", "status optimal emotion happy"};

    std::mt19937 rng(42);
    auto pick = [&](const auto& a) { return a[rng() % n_of(a)]; };
    std::string train;
    const int reps = quick ? 250 : 700;
    for (int i = 0; i < reps; ++i) {
        switch (rng() % 3) {
            case 0: train += std::string("user: ") + pick(G) + "\nbrain: " + pick(GA) + "\n"; break;
            case 1: train += std::string("user: ") + pick(W) + "\nbrain: " + pick(WA) + "\n"; break;
            default: train += std::string("user: ") + pick(S) + "\nbrain: " + pick(SA) + "\n"; break;
        }
    }

    StamlatConfig cfg;
    cfg.d_model = quick ? 64 : 96;
    cfg.n_layers = 3; cfg.n_heads = 6;
    cfg.d_ff = quick ? 128 : 256;
    cfg.ctx = 96; cfg.depth_gamma = 0.f; cfg.depth_tau = 1.f; cfg.seed = 42;

    StamlatLM lm(cfg);
    lm.build_vocab(train);
    std::printf("params=%zu vocab=%d(%d words) ctx=%d\n", lm.param_count(),
                lm.total_vocab_size(), lm.word_vocab_size(), cfg.ctx);
    lm.fit(train, quick ? 400 : 1200, 4e-3f, 12, quick ? 200 : 300);

    // sanity probes before shipping the voice
    struct Probe { const char* q; const char* must; };
    static const Probe probes[] = {
        {"hello", "intent"}, {"who are you", "identity"},
        {"how are you", "status"}, {"what do you do", nullptr},
    };
    int ok = 0, total = 0;
    for (const auto& p : probes) {
        if (!p.must) continue;
        ++total;
        const std::string r = lm.stream_complete_ids(
            lm.encode(std::string("user: ") + p.q + "\nbrain: "), 32, 0.f);
        const bool good = r.find(p.must) != std::string::npos;
        ok += good;
        std::printf("  %-16s -> %-40s [%s]\n", p.q, r.c_str(), good ? "ok" : "MISS");
    }
    if (!lm.save(out)) { std::printf("save FAILED -> %s\n", out.c_str()); return 1; }
    std::printf("saved %s (probe floor %d/%d)\n", out.c_str(), ok, total);
    return ok == total ? 0 : 2;
}
