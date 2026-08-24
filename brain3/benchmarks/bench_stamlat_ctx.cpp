// bench_stamlat_ctx.cpp — mouth scaling study
//
// For each context length: trains a small dialogue model, then measures
//   1. prefill cost   (stream_start over a full window)
//   2. stream decode  (amortized-O(1) KV-cache path, steady-state sliding)
//   3. batch decode   (full-window recompute per token — legacy baseline)
//   4. pre-eviction bit-exactness at that ctx (correctness guard)
//
// usage: bench_stamlat_ctx [--quick]
#include <cstdio>
#include <chrono>
#include <string>
#include <vector>
#include "crisp/engines/neural/stamlat_transformer.hpp"

using namespace brain3::engines::neural;

template <typename T, size_t N> constexpr size_t n_of(const T (&a)[N]) { return N; }

static const char* kQ[] = {"hello", "hi", "hey there", "good morning", "greetings"};
static const char* kQA[] = {"intent greeting style friendly", "intent welcome target user",
                            "intent salutation status ready", "intent greeting emotion happy",
                            "intent greeting emotion positive"};
static const char* kW[] = {"who are you", "what is your name"};
static const char* kWA[] = {"identity system type cognitive", "identity brain origin artificial"};

static std::string corpus(int reps) {
    std::mt19937 rng(5);
    auto pick = [&](const auto& a) { return a[rng() % n_of(a)]; };
    std::string out;
    for (int i = 0; i < reps; ++i) {
        out += std::string("user: ") + pick(kQ) + "\nbrain: " + pick(kQA) + "\n";
        if (i % 3 == 0) out += std::string("user: ") + pick(kW) + "\nbrain: " + pick(kWA) + "\n";
    }
    return out;
}

struct CtxResult { int ctx; double prefill_ms, stream_ms_per_tok, batch_ms_per_tok; double speedup; bool exact; };

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    const std::vector<int> ctx_sizes = quick ? std::vector<int>{48, 128}
                                             : std::vector<int>{48, 96, 192, 384, 512};
    // constant TOKEN budget per config: steps scale inversely with ctx so
    // every row sees the same amount of training signal
    const int token_budget = quick ? 3000 : 6000;
    auto steps_for = [&](int ctx) { return std::max(24, token_budget / ctx); };
    const int gen_tokens = quick ? 32 : 48;

    StamlatConfig cfg;
    // compact profile: the scaling SHAPE is the result; mouth-grade widths
    // make scalar-backend training minutes-per-row without changing it
    cfg.d_model = 64; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 128;
    cfg.depth_gamma = 0.f; cfg.depth_tau = 1.f; cfg.seed = 42;

    const std::string train = corpus(400);

    std::printf("%6s %12s %16s %16s %10s %8s\n",
                "ctx", "prefill_ms", "stream_ms/tok", "batch_ms/tok", "speedup", "exact");
    for (int ctx : ctx_sizes) {
        cfg.ctx = ctx;
        StamlatLM lm(cfg);
        lm.build_vocab(train);
        lm.fit(train, steps_for(ctx), 5e-3f, 12, 0);

        auto ids = lm.encode("user: hello\nbrain: ");
        while ((int)ids.size() < std::max(1, ctx - gen_tokens))
            for (const char* q : kQ) {
                auto tail = lm.encode(std::string("user: ") + q + "\n");
                ids.insert(ids.end(), tail.begin(), tail.end());
            }
        ids.resize(std::min<size_t>(ids.size(), (size_t)ctx));

        // prefill cost (full-window ingestion)
        const auto t0 = std::chrono::steady_clock::now();
        {
            StamlatLM::StreamCache sc;
            lm.stream_start(ids, sc);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        // streaming decode, steady state (cache full → sliding)
        StamlatLM::StreamCache sc;
        lm.stream_start(ids, sc);
        // warm to full occupancy so measurement covers steady-state slides
        int warm = (int)ids.size();
        while (warm++ < ctx) lm.stream_step(lm.stream_sample(sc, 0.f), sc);
        const int n_stream = std::min(gen_tokens, 24);
        const auto t2 = std::chrono::steady_clock::now();
        for (int n = 0; n < n_stream; ++n)
            lm.stream_step(lm.stream_sample(sc, 0.f), sc);
        const auto t3 = std::chrono::steady_clock::now();
        const double stream_mt =
            std::chrono::duration<double, std::milli>(t3 - t2).count() / n_stream;

        // batch decode baseline (few tokens suffice to estimate the slope)
        const int n_batch = std::min(gen_tokens, 12);
        const auto t4 = std::chrono::steady_clock::now();
        lm.complete_ids(ids, n_batch, 0.f, false);
        const auto t5 = std::chrono::steady_clock::now();
        const double batch_mt =
            std::chrono::duration<double, std::milli>(t5 - t4).count() / n_batch;

        // correctness guard: pre-eviction generation must be bit-exact
        auto probe = lm.encode("user: hello\nbrain: ");
        const bool exact =
            lm.complete_ids(probe, 12, 0.f) == lm.stream_complete_ids(probe, 12, 0.f);

        std::printf("%6d %12.3f %16.4f %16.4f %9.1fx %8s\n",
                    ctx, prefill_ms, stream_mt, batch_mt,
                    batch_mt / stream_mt, exact ? "yes" : "NO");
        std::fflush(stdout);
    }
    return 0;
}
