// train_native_mouth.cpp — trains and saves the production mouth binary
//
// usage: train_native_mouth [out_path] [--quick]
// default output: mouth_native.bin (picked up by MasterOrchestrator at boot)
#include <cstdio>
#include <random>
#include <algorithm>
#include <string>
#include <iostream>
#include <cstdlib>
#include "crisp/engines/neural/stamlat_transformer.hpp"
#include "crisp/engines/neural/utterance_plan.hpp"

using namespace brain3::engines::neural;

template <typename T, size_t N> constexpr size_t n_of(const T (&a)[N]) { return N; }

// ── plan-curriculum mode: train the mouth as a pure renderer ─────────────
// Targets are RANDOM own-class subsequences per act (unmemorizable), plan
// carries exactly those surfaces + scaffold. Produces an amnesia-proof
// plan-conditioned binary consumable via respond_plan().
static int train_plan_mode(const std::string& out, bool quick) {
    using namespace brain3::engines::neural;
    struct Dom { const char* act; std::vector<std::string> clazz; };
    static const Dom doms[] = {
        {"greeting",{"intent","greeting","welcome","salutation","style",
                    "friendly","emotion","happy","target","user"}},
        {"identity",{"identity","name","self","system","brain","network",
                    "type","cognitive","origin","artificial","ai","neural"}},
        {"status",  {"status","state","feeling","good","great","positive",
                    "optimal","energy","high","mode","ready","condition"}},
    };
    std::mt19937 rng(31);
    auto pick = [&](auto& v){ return v[rng() % v.size()]; };

    std::string train;
    for (auto& d : doms)
        for (int rep = 0; rep < (quick ? 160 : 320); ++rep) {
            std::vector<std::string> truth = d.clazz;
            std::shuffle(truth.begin(), truth.end(), rng);
            truth.resize(3 + rng() % 2);
            UtterancePlan p; p.act = d.act;
            p.reg = (rep % 2 == 0) ? "warm" : "neutral";
            p.facts = truth;
            train += p.linearize() + " ";
            for (size_t k = 0; k < truth.size(); ++k)
                train += truth[k] + (k + 1 < truth.size() ? " " : "\n");
        }

    StamlatConfig cfg;
    cfg.d_model = quick ? 64 : 96;
    cfg.n_layers = 3; cfg.n_heads = 6;
    cfg.d_ff = quick ? 128 : 256;
    cfg.ctx = 64; cfg.depth_gamma = 0.f; cfg.depth_tau = 1.f; cfg.seed = 42;
    StamlatLM lm(cfg);
    lm.build_vocab(train);
    std::printf("[plan] vocab=%d (%d words)\n", lm.total_vocab_size(),
                lm.word_vocab_size());
    lm.fit(train, quick ? 2600 : 6000, 4e-3f, 16, quick ? 650 : 1500);

    // sanity: held-out random sequences render exactly
    int ok = 0, tot = 0;
    for (auto& d : doms)
        for (int t2 = 0; t2 < 4; ++t2) {
            std::vector<std::string> truth = d.clazz;
            std::shuffle(truth.begin(), truth.end(), rng);
            truth.resize(3 + rng() % 2);
            UtterancePlan p; p.act = d.act; p.reg = "neutral";
            p.facts = truth;
            auto allowed = p.content_lock_ids(lm);
            auto said = lm.stream_complete_ids(lm.encode(p.linearize()),
                                               20, 0.f, true, &allowed);
            if (std::getenv("PLAN_DEBUG") && tot < 3) {
                std::cout << "PLAN '" << p.linearize() << "'\nSAID '" << said
                          << "' WANT '";
                for (size_t k2=0;k2<truth.size();++k2) std::cout<<truth[k2]<<(k2+1<truth.size()?" ":"");
                std::cout << "'\n";
            }
            ++tot;
            auto trim = [](std::string s){
                size_t a = s.find_first_not_of(' ');
                size_t b = s.find_last_not_of(' ');
                return a == std::string::npos ? "" : s.substr(a, b - a + 1);
            };
            std::string want;
            for (size_t k2 = 0; k2 < truth.size(); ++k2)
                want += truth[k2] + (k2 + 1 < truth.size() ? " " : "");
            if (trim(said) == trim(want)) ++ok;
        }
    std::printf("[plan] held-out exact %d/%d\n", ok, tot);
    if (!lm.save(out)) { std::printf("save FAILED\n"); return 1; }
    std::printf("saved %s\n", out.c_str());
    return ok >= (tot * 3) / 4 ? 0 : 2;
}

int main(int argc, char** argv) {
    std::string out = "mouth_native.bin";
    bool quick = false, plan = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quick") quick = true;
        else if (a == "--plan") plan = true;
        else out = a;
    }
    if (plan) return train_plan_mode(out, quick);

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
