#pragma once
/**
 * brain3/core/intent_router.hpp
 *
 * LEARNED INTENT ROUTER — replaces regex brittleness at the front door.
 *
 * Classification is learned (char-trigram hashing -> linear softmax over
 * op families, trained lazily on a synthetic paraphrase corpus); slot
 * EXTRACTION stays rule-based (deterministic regexes mirrored from
 * master_orchestrator). Division of labor:
 *   - router decides WHICH family the utterance belongs to
 *   - per-family extractors pull arguments; on failure we fall back to the
 *     legacy parser chain unchanged
 *
 * Safety properties:
 *   - exact BQL commands pass through BEFORE the router ever runs
 *   - pronoun stoplist prevents chatty turns ("who are you") from becoming
 *     knowledge lookups — those belong to the native mouth
 */
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>

namespace brain3 {
namespace core {

class IntentRouter {
public:
    struct Verdict {
        std::string family;
        float confidence = 0.f;
    };

    // Thread-safe lazy singleton: trains once (~ms), serves forever.
    static const IntentRouter& instance() {
        static const IntentRouter router;
        return router;
    }

    // NOTE: deliberately bias-free. An unseen utterance produces near-zero
    // logits -> uniform distribution -> low confidence -> legacy fallback.
    Verdict classify(const std::string& text) const {
        auto feats = featurize(text);
        Verdict v;
        double best = -1e30;
        std::vector<double> p(families_.size());
        for (size_t k = 0; k < families_.size(); ++k) {
            double z = 0.0;
            for (int f : feats) z += W_[k][f];
            p[k] = z;
        }
        double Z = 0.;
        for (auto& z : p) { z = std::exp(z); Z += z; }
        for (size_t k = 0; k < families_.size(); ++k) {
            double pr = p[k] / Z;
            if (pr > best) { best = pr; v.family = families_[k]; }
        }
        v.confidence = (float)best;
        return v;
    }

    const std::vector<std::string>& families() const { return families_; }

    // ── training corpus ─────────────────────────────────────────────────────
    struct FamilySpec {
        const char* name;
        std::vector<const char*> seeds;      // {A}/{B} templated utterances
        bool two_slot;
    };

private:
    static constexpr size_t kDims = 1536;

    IntentRouter() {
        static const FamilySpec specs[] = {
            {"WHAT_IF", {
                "what if {A} causes {B}", "suppose {A} leads to {B}",
                "what happens if {A} causes {B}", "counterfactual {A} results in {B}",
                "if {A} changes what happens to {B}", "imagine {A} leading to {B}",
                "would {B} change if {A} happened", "predict outcome when {A} drives {B}",
                "simulate scenario where {A} affects {B}", "model consequence of {A} on {B}",
                "hypothetically {A} produces {B}", "explore effect of {A} upon {B}",
            }, true},
            {"TEACH", {
                "teach that {A} is a {B}", "remember {A} is an {B}",
                "store that {A} has {B}", "learn that {A} can {B}",
                "commit to memory {A} is a {B}", "note that {A} causes {B}",
                "record {A} as having {B}", "save fact {A} is an {B}",
                "add knowledge {A} can {B}", "absorb {A} is a {B}",
                "keep in mind {A} has {B}", "ingest fact {A} causes {B}",
            }, true},
            {"LOOKUP", {
                "what is {A}", "who is {A}", "what are {A}",
                "tell me about {A}", "look up {A}", "query {A}",
                "info on {A}", "describe {A}", "describe {A} in detail",
                "define {A}", "clarify {A}", "explain simply {A}", "details of {A}",
                "definition of {A}", "meaning of {A}", "facts about {A}",
                "describe {A} in detail",
                "search for {A}", "find {A} in memory",
            }, false},
            {"ANALOGY", {
                "compare {A} to {B}", "analogy between {A} and {B}",
                "map {A} onto {B}", "isomorphism between {A} and {B}",
                "relate {A} structurally to {B}", "{A} versus {B} structural match",
                "draw parallel between {A} and {B}", "align concepts {A} and {B}",
                "transfer structure from {A} to {B}", "correspondence {A} {B}",
            }, true},
            {"REFUTE", {
                "refute that all {A} are {B}", "is it true that all {A} are {B}",
                "disprove claim {A} is {B}", "challenge statement all {A} are {B}",
                "find counterexample to {A} being {B}", "attack premise {A} equals {B}",
                "show {A} is not always {B}", "falsify every {A} is {B}",
            }, true},
            {"EXPLAIN", {
                "how to {A}", "explain {A} step by step", "plan for {A}",
                "outline {A}", "strategy for {A}", "walk me through {A}",
                "brief on {A}", "summarize approach to {A}",
            }, false},
        };
        for (const auto& s : specs)
            families_.push_back(s.name);

        build_corpus(specs, 6);
        train(22, 0.30);
    }

    void build_corpus(const FamilySpec* specs, size_t n_specs) {
        static const char* fills[] = {
            "", "please ", "could you ", "can you ", "i want to ",
            "hey brain ", "quickly ", "for me ", "right now ", "brain, ",
        };
        static const char* tails[] = {"", " please", " now", " for me", " thanks"};
        static const char* As[] = {"smoking", "gravity", "cache misses", "inflation",
                                   "exercise", "rainfall", "supply shocks", "mutation"};
        static const char* Bs[] = {"cancer", "falling apples", "latency spikes",
                                   "recession", "fitness", "floods", "price hikes",
                                   "resistance"};
        std::mt19937 rng(4242);
        for (size_t si = 0; si < n_specs; ++si) {
            const auto& spec = specs[si];
            const int label = (int)si;              // index == families_[si]
            for (const char* tmpl : spec.seeds)
                for (const char* pre : fills)
                    for (const char* tail : tails) {
                        std::string A = As[rng() % (sizeof(As)/sizeof(*As))];
                        std::string B = Bs[rng() % (sizeof(Bs)/sizeof(*Bs))];
                        std::string t = tmpl;
                        size_t pos;
                        while ((pos = t.find("{A}")) != std::string::npos)
                            t.replace(pos, 3, A);
                        while ((pos = t.find("{B}")) != std::string::npos)
                            t.replace(pos, 3, B);
                        std::string line = std::string(pre) + t + tail;
                        corpus_.push_back({featurize(line), label});
                    }
        }
    }

    void train(int epochs, double lr) {
        const size_t K = families_.size();
        W_.assign(K, std::vector<double>(kDims, 0.0));
        bias_.assign(K, 0.0);
        for (int ep = 0; ep < epochs; ++ep) {
            double loss = 0.;
            for (const auto& sample : corpus_) {
                const auto& feats = sample.first;
                const int y = sample.second;
                std::vector<double> z(K);
                for (size_t k = 0; k < K; ++k) {
                    double acc = bias_[k];
                    for (int f : feats) acc += W_[k][f];
                    z[k] = acc;
                }
                double mx = *std::max_element(z.begin(), z.end());
                double Zs = 0.;
                for (auto& v : z) { v = std::exp(v - mx); Zs += v; }
                for (size_t k = 0; k < K; ++k) {
                    double pk = z[k] / Zs;
                    double g = pk - (k == (size_t)y ? 1.0 : 0.0);
                    loss -= (k == (size_t)y ? std::log(pk + 1e-12) : 0.);
                    for (int f : feats) W_[k][f] -= lr * g;
                }
            }
            lr *= 0.9;
        }
    }

    std::vector<int> featurize(const std::string& raw) const {
        std::string s;
        for (char c : raw)
            s += std::isspace((unsigned char)c) ? ' '
                 : (char)std::tolower((unsigned char)c);
        s = "  " + s + "  ";
        std::unordered_map<size_t, int> counts;
        auto add = [&](std::string gram) {
            size_t h = 1469598103934665603ULL;
            for (char c : gram) { h ^= (unsigned char)c; h *= 1099511628211ULL; }
            counts[h % kDims]++;
        };
        for (size_t i = 0; i + 3 <= s.size(); ++i) add(s.substr(i, 3));
        for (size_t i = 0; i + 5 <= s.size(); ++i) add(s.substr(i, 5));
        double norm = 0.;
        for (auto& [h, c] : counts) norm += (double)c * c;
        norm = std::sqrt(norm);
        std::vector<std::pair<size_t, double>> weighted;
        for (auto& [h, c] : counts) weighted.push_back({h, c / norm});
        // quantized fixed-point keeps the int-feature interface honest
        std::vector<int> out;
        for (auto& [h, w] : weighted)
            for (int q = 0; q < (int)(w * 32); ++q) out.push_back((int)h);
        return out;
    }

    std::vector<std::string> families_;
    std::vector<std::pair<std::vector<int>, int>> corpus_;
    std::vector<std::vector<double>> W_;
    std::vector<double> bias_;
};

} // namespace core
} // namespace brain3
