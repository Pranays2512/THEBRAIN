// demo_stamlat_mouth.cpp — the Mouth as expressive channel (v3 trio demo)
//
//   1. word-tokenizer context compression (chars → multi-turn window)
//   2. KV-cache streaming REPL with per-token cost vs batch recompute
//   3. style-slot constrained sampling (content locked, style sampled)
//   4. verify-and-retry: fuzzy proposes @T≈0.8, crisp checker disposes,
//      bounded retries, guaranteed template fallback
//
// usage: demo_stamlat_mouth [--quick] [--steps N]
// interactive mode: pipe stdin lines, e.g. echo "hello" | ./demo_stamlat_mouth --quick
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <random>
#include "crisp/engines/neural/stamlat_transformer.hpp"

using namespace brain3::engines::neural;

// Array-size helper: taking the parameter as a reference-to-array preserves
// the extent (a plain `auto arr` lambda parameter decays to a pointer, making
// sizeof(arr)/sizeof(arr[0]) == 1 — which silently collapsed every random
// pick to index 0 and shrank the training corpus to one template per intent).
template <typename T, size_t N>
constexpr size_t n_of(const T (&)[N]) { return N; }

static const char* kFallback =
    "user: hello\nbrain: intent greeting style friendly\n"
    "user: who are you\nbrain: identity system type cognitive\n";

static std::string build_corpus() {
    static const char* G[] = {"hello", "hi", "hey there", "greetings", "good morning", "good evening"};
    static const char* GA[] = {"intent greeting style friendly", "intent salutation status ready",
                               "intent greeting emotion positive", "intent welcome target user"};
    static const char* W[] = {"who are you", "what is your name", "what are you", "tell me about yourself"};
    static const char* WA[] = {"identity system type cognitive", "identity brain origin artificial",
                               "self network type neural", "name brain type ai"};
    static const char* S[] = {"how are you", "how is your day", "how do you feel", "what is your state"};
    static const char* SA[] = {"status good energy high", "state positive mode ready",
                               "feeling great condition excellent", "status optimal emotion happy"};
    static const char* D[] = {"what do you do", "what are you doing", "what is your purpose"};
    static const char* DA[] = {"action learning goal communication", "task processing goal understanding",
                               "purpose learning objective thinking", "activity reading state processing"};

    std::mt19937 rng(7);
    auto pick = [&](const auto& arr) { return arr[rng() % n_of(arr)]; };
    std::string out;
    for (int i = 0; i < 900; ++i) {
        switch (rng() % 4) {
            case 0: out += std::string("user: ") + pick(G) + "\nbrain: " + pick(GA) + "\n"; break;
            case 1: out += std::string("user: ") + pick(W) + "\nbrain: " + pick(WA) + "\n"; break;
            case 2: out += std::string("user: ") + pick(S) + "\nbrain: " + pick(SA) + "\n"; break;
            default: out += std::string("user: ") + pick(D) + "\nbrain: " + pick(DA) + "\n"; break;
        }
    }
    return out;
}

// crisp content contract per intent: one alternative from EACH group must
// survive (facts have multiple valid surface forms — the checker verifies the
// equivalence class, not one memorized string)
static const std::map<std::string, std::vector<std::vector<std::string>>> kContent = {
    {"greeting", {{"intent"}, {"greeting", "salutation", "welcome"}}},
    {"identity", {{"identity", "name"}, {"system", "brain", "ai", "network"}}},
    {"status",   {{"status", "state", "feeling"},
                  {"good", "positive", "great", "optimal", "excellent"}}},
    {"action",   {{"action", "task", "purpose", "activity"},
                  {"learning", "processing", "thinking", "understanding"}}},
};
// free style vocabulary the mouth may sample from
static const std::vector<std::string> kStylePool = {
    "style", "friendly", "warm", "positive", "salutation", "ready", "welcome",
    "type", "cognitive", "brain", "ai", "origin", "artificial", "neural", "self",
    "energy", "high", "emotion", "happy", "optimal", "feeling", "great",
    "goal", "communication", "understanding", "processing",
};

static bool preserves_content(const std::string& reply,
                              const std::vector<std::vector<std::string>>& groups) {
    for (const auto& g : groups) {
        bool hit = false;
        for (const auto& r : g)
            if (reply.find(r) != std::string::npos) { hit = true; break; }
        if (!hit) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    bool quick = false;
    int steps = 2500;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quick") quick = true;
        else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
    }
    if (quick && steps == 2500) steps = 500;

    StamlatConfig cfg;
    cfg.d_model = 96; cfg.n_layers = 3; cfg.n_heads = 6; cfg.d_ff = 256;
    cfg.ctx = 96; cfg.depth_gamma = 0.0f; cfg.depth_tau = 1.0f; cfg.seed = 42;
    if (quick) {                       // compact profile: ~1 minute instead of ~15
        cfg.d_model = 64; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 128;
    }
    StamlatLM lm(cfg);

    const std::string train = build_corpus();
    std::cout << "== STAMLAT v3 mouth demo ==\n";
    std::cout << "params: " << lm.param_count() << " | ctx: " << cfg.ctx
              << " tokens | tokenizer: words+char-fallback\n";
    lm.build_vocab(train);
    std::cout << "vocab: " << lm.char_vocab_size() << " chars + "
              << lm.word_vocab_size() << " words = " << lm.total_vocab_size() << " tokens\n";
    lm.fit(train, steps, 4e-3f, 12, steps / 4);
    std::cout << "final eval loss: " << lm.eval_loss(train) << "\n";

    // ── 1. context compression ────────────────────────────────────────────────
    std::cout << "\n[1] context compression\n";
    const std::string history =
        "user: hello\nbrain: intent greeting style friendly\n"
        "user: who are you\nbrain: identity system type cognitive\n";
    const size_t n_chars = history.size();
    const size_t n_word = lm.encode(history).size();
    std::cout << "    two-turn history: " << n_chars << " chars -> "
              << n_word << " tokens ("
              << (double)n_chars / (double)n_word << "x denser); ctx="
              << cfg.ctx << " now spans ~" << cfg.ctx * 5 / n_word
              << " such turns instead of " << cfg.ctx / n_chars << "\n";

    // ── 2. streaming REPL cost profile ───────────────────────────────────────
    std::cout << "\n[2] streaming decode vs batch recompute\n";
    {
        const auto prompt_ids = lm.encode("user: how are you\nbrain: ");
        const int N = 40;   // stays inside ctx: pre-eviction regime, bit-exact
        auto t0 = std::chrono::steady_clock::now();
        const std::string b = lm.complete_ids(prompt_ids, N, 0.f, false);
        auto t1 = std::chrono::steady_clock::now();
        const std::string s = lm.stream_complete_ids(prompt_ids, N, 0.f, false);
        auto t2 = std::chrono::steady_clock::now();
        const double mb = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "    " << N << " tokens (window grows " << prompt_ids.size()
                  << "->" << prompt_ids.size() + N << " < ctx=" << cfg.ctx << "): batch "
                  << mb << " ms vs stream " << ms << " ms (" << mb / ms
                  << "x), bit-exact=" << (b == s) << "\n"
                  << "    (long replies slide the cache: amortized O(1)/tok,\n"
                  << "     see test_stamlat_streaming TEST 8 for the speed profile)\n";
    }

    // ── 3. style-slot locking ────────────────────────────────────────────────
    std::cout << "\n[3] constrained sampling: content slots forced, style sampled\n";
    {
        const auto& groups = kContent.at("identity");
        std::vector<int> content_ids, style_ids;
        std::set<int> seen;
        auto push_surface = [&](std::vector<int>& dst, const std::string& s) {
            for (int id = 0; id < lm.total_vocab_size(); ++id)
                if (lm.token_surface(id) == s && seen.insert(id).second) { dst.push_back(id); return; }
        };
        for (const auto& g : groups)
            for (const auto& r : g) push_surface(content_ids, r);
        for (const auto& st : kStylePool) push_surface(style_ids, st);

        std::vector<int> ws;                       // whitespace glue
        for (int id = 0; id < lm.total_vocab_size(); ++id)
            if (lm.token_surface(id) == " " || lm.token_surface(id) == "\n") ws.push_back(id);

        // phase A: allowed = required facts + glue → some fact token MUST emit
        // phase B: full allow-set → style varies freely around the locked facts
        std::vector<int> all = style_ids;
        all.insert(all.end(), content_ids.begin(), content_ids.end());
        all.insert(all.end(), ws.begin(), ws.end());
        std::vector<int> lock = content_ids;
        lock.insert(lock.end(), ws.begin(), ws.end());

        std::cout << "    allow-set: " << all.size() << " tokens (" << content_ids.size()
                  << " content + " << style_ids.size() << " style + glue)\n";
        for (int trial = 0; trial < 3; ++trial) {
            StamlatLM::StreamCache sc;
            lm.stream_start(lm.encode("user: who are you\nbrain: "), sc);
            std::string out;
            bool locked_done = false;
            for (int n = 0; n < 16 && out.find('\n') == std::string::npos; ++n) {
                const int tok = locked_done ? lm.stream_sample(sc, 0.9f, &all)
                                            : lm.stream_sample(sc, 0.9f, &lock);
                out += lm.token_surface(tok);
                lm.stream_step(tok, sc);
                if (!locked_done && preserves_content(out, groups)) {
                    locked_done = true;
                    out += "|";
                }
            }
            std::cout << "    T=0.9 locked: \"" << out << "\"  guaranteed="
                      << (locked_done ? "true" : "false") << "\n";
        }
        std::cout << "    T=0 template: \""
                  << lm.stream_complete_ids(lm.encode("user: who are you\nbrain: "), 30, 0.f)
                  << "\"\n";
    }

    // ── 4. verify-and-retry ──────────────────────────────────────────────────
    std::cout << "\n[4] verify-and-retry (fuzzy proposes, crisp disposes)\n";
    struct Probe { const char* q; const char* intent; };
    static const Probe probes[] = {
        {"hello", "greeting"}, {"who are you", "identity"},
        {"how are you", "status"}, {"what do you do", "action"},
        {"hey there", "greeting"},
    };
    int accepted = 0, retried_in = 0, fell_back = 0;
    for (const auto& p : probes) {
        const auto& required = kContent.at(p.intent);
        const std::string prompt = std::string("user: ") + p.q + "\nbrain: ";
        const auto ids = lm.encode(prompt);
        std::string reply, verdict;
        for (int attempt = 1; attempt <= 3; ++attempt) {
            reply = lm.stream_complete_ids(ids, 28, 0.8f);
            if (preserves_content(reply, required)) {
                verdict = (attempt == 1) ? "accepted" : "accepted@retry";
                if (attempt == 1) ++accepted; else ++retried_in;
                break;
            }
            verdict = "";
        }
        if (verdict.empty()) {
            reply = lm.complete_ids(ids, 28, 0.f);          // guaranteed render
            verdict = "fallback(T=0)";
            ++fell_back;
        }
        std::cout << "    user: " << p.q << "\n    brain: " << reply
                  << "   [" << verdict << "]\n";
    }
    std::cout << "    tally: accepted=" << accepted
              << " accepted_after_retry=" << retried_in
              << " fallback=" << fell_back << "\n";

    // ── interactive mode ─────────────────────────────────────────────────────
    std::cout << "\n-- interactive REPL (empty line exits) --\n";
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) break;
        for (auto& c : line) c = (char)std::tolower((unsigned char)c);
        const auto ids = lm.encode("user: " + line + "\nbrain: ");

        auto t0 = std::chrono::steady_clock::now();
        StamlatLM::StreamCache sc;
        lm.stream_start(ids, sc);
        std::string reply;
        for (int n = 0; n < 48; ++n) {
            const int tok = lm.stream_sample(sc, 0.f);   // duality switch: T=0 test mode
            const std::string s = lm.token_surface(tok);
            lm.stream_step(tok, sc);
            if (s == "\n") break;
            reply += s;
        }
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "brain: " << reply << "   ["
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() / 48.0
                  << " ms/tok streamed]\n";
    }

    lm.save("stamlat_mouth_v3.bin");
    std::cout << "saved model to stamlat_mouth_v3.bin\n";
    return 0;
}
