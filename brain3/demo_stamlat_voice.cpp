// demo_stamlat_voice.cpp — the Mouth as expression of internal state
//
//   [A] three moods, one question: arousal sets temperature, valence colors
//       the style lexicon (warm vs guarded)
//   [B] style loop evolution: propose → verify → absorb → retrain, tracked
//       across generations (acceptance rate climbs, reply NLL drops)
//   [C] post-evolution voices: same moods, consolidated style
//
// usage: demo_stamlat_voice [--quick]
#include <iostream>
#include <string>
#include <vector>
#include "fuzzy/core/emotion.hpp"
#include "crisp/engines/neural/stamlat_transformer.hpp"
#include "crisp/engines/neural/mouth_voice.hpp"
#include "crisp/engines/neural/mouth_style_loop.hpp"

using namespace brain3::engines::neural;

template <typename T, size_t N> constexpr size_t n_of(const T (&)[N]) { return N; }

static const char* kQ[] = {"hello", "hi", "hey there", "good morning"};
static const char* kQA[] = {"intent greeting style friendly", "intent welcome target user",
                            "intent salutation status ready", "intent greeting emotion happy"};
static const char* kW[] = {"who are you", "what is your name"};
static const char* kWA[] = {"identity system type cognitive", "identity brain origin artificial",
                            "name brain type ai", "self network type neural"};

static std::string corpus(int reps) {
    std::mt19937 rng(11);
    auto pick = [&](const auto& arr) { return arr[rng() % n_of(arr)]; };
    std::string out;
    for (int i = 0; i < reps; ++i) {
        out += std::string("user: ") + pick(kQ) + "\nbrain: " + pick(kQA) + "\n";
        out += std::string("user: ") + pick(kW) + "\nbrain: " + pick(kWA) + "\n";
    }
    return out;
}

int main(int argc, char** argv) {
    bool quick = argc > 1 && std::string(argv[1]) == "--quick";

    StamlatConfig cfg;
    cfg.d_model = 48; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 96; cfg.ctx = 64;
    if (quick) { cfg.d_model = 32; cfg.d_ff = 48; }

    StamlatLM lm(cfg);
    const std::string train = corpus(250);
    lm.build_vocab(train);
    // deliberately soft training: forks stay uncertain so mood bias and
    // style-loop absorption have visible room to steer the voice
    lm.fit(train, quick ? 90 : 300, 5e-3f, 12, quick ? 45 : 150);

    auto vm = default_voice_mapper();

    // ── A. one question, three inner states ──────────────────────────────
    std::cout << "\n[A] mood-coupled voice (same question, different states)\n";
    struct Mood { const char* name; brain2::EmotionState e; };
    const Mood moods[] = {
        {"calm            ", {0.f, 0.f}},
        {"excited+positive", {0.8f, 1.f}},
        {"alert+negative  ", {-0.7f, 0.8f}},
    };
    for (const auto& m : moods) {
        const auto pol = vm.policy(lm, m.e);
        std::cout << "  " << m.name << " T=" << pol.temperature
                  << " bias=" << pol.bias.size() << "tok | ";
        for (int t = 0; t < 3; ++t)
            std::cout << "\"" << lm.stream_complete_ids(
                lm.encode("user: hello\nbrain: "), 16, pol.temperature,
                true, nullptr, &pol.bias) << "\" ";
        std::cout << "\n";
    }

    // ── B. style loop evolution ──────────────────────────────────────────
    std::cout << "\n[B] style loop (Discovery for voice)\n";
    std::vector<MouthStyleLoop::Probe> probes = {
        {"user: hello\nbrain: ",        {{"intent"}, {"friendly", "happy"}},     "greeting:warm"},
        {"user: who are you\nbrain: ",  {{"identity", "name", "self"}, {"cognitive", "neural", "network"}}, "identity:technical"},
    };

    StyleLoopConfig scfg;
    scfg.candidates_per_probe = quick ? 8 : 12;
    scfg.sft_lr = 1e-3f;
    scfg.sft_epochs = 2;

    MouthStyleLoop loop(lm, probes, scfg);
    const brain2::EmotionState gen_mood[6] = {
        {0.6f, 0.4f}, {0.8f, 0.9f}, {0.5f, 0.6f}, {0.9f, 1.f}, {0.2f, 0.7f}, {0.7f, 0.8f},
    };
    for (int g = 0; g < 6; ++g) {
        const auto& st = loop.evolve(gen_mood[g], &vm);
        std::cout << "  gen" << st.generation << ": "
                  << st.accepted << "/" << st.proposed << " accepted ("
                  << int(st.acceptance_rate * 100) << "%), unique="
                  << st.unique_accepted
                  << ", nll " << st.mean_nll_pre << "→" << st.mean_nll_post
                  << (st.rolled_back ? " [ROLLED BACK]" : "") << "\n";
    }
    std::cout << "  absorbed corpus: " << loop.corpus().size()
              << " weighted examples, " << loop.replies().size() << " distinct replies\n";

    // ── C. consolidated voices ───────────────────────────────────────────
    std::cout << "\n[C] post-evolution voices\n";
    for (const auto& m : moods) {
        const auto pol = vm.policy(lm, m.e);
        std::cout << "  " << m.name << "-> \""
                  << lm.stream_complete_ids(lm.encode("user: who are you\nbrain: "),
                                            18, pol.temperature, true, nullptr, &pol.bias)
                  << "\"\n";
    }

    lm.save("stamlat_voice.bin");
    std::cout << "\nsaved evolved model to stamlat_voice.bin\n";
    return 0;
}
