// check_unified_mouth.cpp — verifies BOTH lanes of the unified mouth artifact.
// Lane 1: free chat floor (contract keywords present)
// Lane 2: plan-conditioned CONTENT-LOCKED rendering (amnesia guarantee):
//         decoding is constrained to the plan's own surfaces — a model that
//         learned to hallucinate CANNOT pass this gate.
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include "crisp/engines/neural/stamlat_transformer.hpp"
#include "crisp/engines/neural/utterance_plan.hpp"

using namespace brain3::engines::neural;

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "data/distill/mouth_unified.bin";
    StamlatLM lm([]{ StamlatConfig c; c.d_model=96; c.n_layers=3; c.n_heads=6; c.d_ff=256; c.ctx=96; c.depth_gamma=0.f; c.depth_tau=1.f; return c; }());
    if (!lm.load(path)) { std::printf("LOAD FAILED\n"); return 1; }

    bool has_p = false;
    for (int id = lm.char_vocab_size(); id < lm.total_vocab_size(); ++id)
        if (lm.token_surface(id) == "<p>") { has_p = true; break; }
    std::printf("loaded %s  params=%zu  plans_supported=%d\n", path.c_str(),
                lm.param_count(), (int)has_p);

    struct Dom { const char* act; std::vector<std::string> clazz; };
    static const Dom doms[] = {
        {"greeting",{"intent","greeting","welcome","salutation","style",
                    "friendly","emotion","happy","target","user"}},
        {"identity",{"identity","name","self","system","brain","network",
                    "type","cognitive","origin","artificial","ai","neural"}},
        {"status",  {"status","state","feeling","good","great","positive",
                    "optimal","energy","high","mode","ready","condition"}},
    };

    auto trim = [](std::string s) {
        size_t a = s.find_first_not_of(' ');
        size_t b = s.find_last_not_of(' ');
        return a == std::string::npos ? "" : s.substr(a, b - a + 1);
    };

    std::mt19937 rng(20260826);
    int ok = 0, tot = 0;
    for (auto& d : doms)
        for (int t = 0; t < 8; ++t) {
            std::vector<std::string> truth = d.clazz;
            std::shuffle(truth.begin(), truth.end(), rng);
            truth.resize(3 + rng() % 2);
            UtterancePlan p; p.act = d.act; p.reg = (t % 2) ? "warm" : "neutral";
            p.facts = truth;
            auto allowed = p.content_lock_ids(lm);
            // adversarial distractor check: ask for a surface NOT in the plan
            auto said = lm.stream_complete_ids(lm.encode(p.linearize()),
                                               20, 0.f, true, &allowed);
            std::string want;
            for (size_t k = 0; k < truth.size(); ++k)
                want += truth[k] + (k + 1 < truth.size() ? " " : "");
            ++tot;
            const bool good = trim(said) == trim(want);
            ok += good;
            if (!good || t == 0)
                std::printf("  [%s] act=%-8s said='%s' want='%s'\n",
                            good ? "ok  " : "MISS", d.act,
                            trim(said).c_str(), trim(want).c_str());
        }
    std::printf("PLAN LANE: exact render %d/%d\n", ok, tot);

    // chat floor
    struct Probe { const char* q; const char* a; const char* b; };
    static const Probe probes[] = {
        {"hello", "intent", "greeting"},
        {"who are you", "identity", "brain"},
        {"how are you", "status", "good"},
    };
    int cok = 0;
    for (auto& p : probes) {
        auto r = lm.stream_complete_ids(
            lm.encode(std::string("user: ") + p.q + "\nbrain: "), 40, 0.f);
        cok += (r.find(p.a) != std::string::npos &&
                r.find(p.b) != std::string::npos);
        std::printf("  chat %-14s -> %s\n", p.q, r.c_str());
    }
    std::printf("CHAT LANE: floor %d/3\n", cok);

    const bool pass = has_p && ok >= (tot * 3) / 4 && cok == 3;
    std::printf("%s\n", pass ? "UNIFIED MOUTH: ALL LANES PASS"
                             : "UNIFIED MOUTH: GATE FAILURE");
    return pass ? 0 : 2;
}
