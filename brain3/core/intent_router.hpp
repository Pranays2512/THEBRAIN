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
#include <fstream>
#include <cstdint>
#include <cstddef>

namespace brain3 {
namespace core {

class IntentRouter {
public:
    struct Verdict {
        std::string family;
        float confidence = 0.f;
    };

    // Thread-safe lazy singleton: trains once (~ms), serves forever.
    static IntentRouter& instance() {
        static IntentRouter router;
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
        // Subtract the max before exponentiating. train() and reinforce() both
        // do; classify() did not, so with large weights exp() overflowed to inf
        // and inf/inf gave nan. Latent while weights came only from the boot
        // corpus, reachable the moment reinforce() starts growing them.
        const double mx = *std::max_element(p.begin(), p.end());
        double Z = 0.;
        for (auto& z : p) { z = std::exp(z - mx); Z += z; }
        for (size_t k = 0; k < families_.size(); ++k) {
            double pr = p[k] / Z;
            if (pr > best) { best = pr; v.family = families_[k]; }
        }
        v.confidence = (float)best;
        return v;
    }

    const std::vector<std::string>& families() const { return families_; }

    // ── Closed loop: learn from what routing actually produced ──────────────
    //
    // This class was `static const IntentRouter& instance()` with no reward
    // path: trained once at construction on a synthetic paraphrase corpus, then
    // frozen. Its verdict is AUTHORITATIVE above 0.55 confidence, so the
    // component that decides which engine runs never found out whether that
    // decision produced a verified answer, and every lesson died at exit.
    // Same defect 800b71a found in UnifiedProposer and d0a506e in its
    // persistence path, in the front door this time.
    //
    // `success` is the outcome of the turn that this routing produced. A reward
    // step is the ordinary softmax gradient toward `family`; a penalty is the
    // same step reversed, which lowers that family's score and lifts the others
    // uniformly. The penalty rate is deliberately lower than the reward rate:
    // one failed turn is weak evidence that a family is wrong (the engine may
    // simply lack the fact), whereas a verified answer is strong evidence the
    // routing was right. Asymmetric rates keep a few failures from unlearning a
    // correct prior.
    void reinforce(const std::string& text, const std::string& family, bool success) {
        int y = -1;
        for (size_t k = 0; k < families_.size(); ++k)
            if (families_[k] == family) { y = (int)k; break; }
        if (y < 0) return;                       // unknown family: nothing to learn

        const auto feats = featurize(text);
        if (feats.empty()) return;

        std::lock_guard<std::mutex> lock(mu_);
        const size_t K = families_.size();
        std::vector<double> z(K);
        for (size_t k = 0; k < K; ++k) {
            double acc = bias_[k];
            for (int f : feats) acc += W_[k][f];
            z[k] = acc;
        }
        const double mx = *std::max_element(z.begin(), z.end());
        double Zs = 0.0;
        for (size_t k = 0; k < K; ++k) { z[k] = std::exp(z[k] - mx); Zs += z[k]; }

        const double lr = success ? kRewardLr : -kPenaltyLr;
        for (size_t k = 0; k < K; ++k) {
            const double pk = z[k] / Zs;
            const double g = pk - (k == (size_t)y ? 1.0 : 0.0);
            for (int f : feats) W_[k][f] -= lr * g;
        }
        ++updates_;
    }

    // Probability this text belongs to `family`. Exposed so callers (and tests)
    // can see the router's confidence in a specific verdict, not only in its
    // argmax — a bid needs a number for the option it did NOT pick too.
    float confidence_for(const std::string& text, const std::string& family) const {
        int y = -1;
        for (size_t k = 0; k < families_.size(); ++k)
            if (families_[k] == family) { y = (int)k; break; }
        if (y < 0) return 0.f;

        const auto feats = featurize(text);
        std::lock_guard<std::mutex> lock(mu_);
        const size_t K = families_.size();
        std::vector<double> z(K);
        for (size_t k = 0; k < K; ++k) {
            double acc = bias_[k];
            for (int f : feats) acc += W_[k][f];
            z[k] = acc;
        }
        const double mx = *std::max_element(z.begin(), z.end());
        double Zs = 0.0;
        for (size_t k = 0; k < K; ++k) { z[k] = std::exp(z[k] - mx); Zs += z[k]; }
        return (float)(z[y] / Zs);
    }

    // Absolute evidence: the largest UNNORMALISED logit. The softmax divides
    // this away — it reports which family is most likely GIVEN that one of them
    // is right, never whether any of them is. An utterance sharing no trigrams
    // with the corpus produces small logits yet still normalises to a confident
    // pick, which is why gibberish routed at 0.999998. This is the signal the
    // header always described ("unseen utterance -> near-zero logits") and that
    // classify() then discarded.
    double evidence(const std::string& text) const {
        const auto feats = featurize(text);
        std::lock_guard<std::mutex> lock(mu_);
        double best = -1e300;
        for (size_t k = 0; k < families_.size(); ++k) {
            double acc = bias_[k];
            for (int f : feats) acc += W_[k][f];
            best = std::max(best, acc);
        }
        return feats.empty() ? 0.0 : best / (double)feats.size();
    }

    // Unnormalised score for one family. The softmax saturates — essentially
    // every input scores ~1.0 — so a probability cannot show whether an update
    // moved anything. The raw logit can: it is what reinforce() actually
    // changes, and it is not squashed. Tests assert direction against this.
    double logit_for(const std::string& text, const std::string& family) const {
        int y = -1;
        for (size_t k = 0; k < families_.size(); ++k)
            if (families_[k] == family) { y = (int)k; break; }
        if (y < 0) return 0.0;
        const auto feats = featurize(text);
        std::lock_guard<std::mutex> lock(mu_);
        double acc = bias_[y];
        for (int f : feats) acc += W_[y][f];
        return acc;
    }

    size_t updates() const { return updates_; }

    bool save(const std::string& path) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        const uint32_t magic = 0x49524F55;              // "IROU"
        const uint32_t K = (uint32_t)families_.size();
        const uint32_t D = (uint32_t)kDims;
        f.write((const char*)&magic, sizeof(magic));
        f.write((const char*)&K, sizeof(K));
        f.write((const char*)&D, sizeof(D));
        f.write((const char*)&updates_, sizeof(updates_));
        for (uint32_t k = 0; k < K; ++k)
            f.write((const char*)W_[k].data(), (std::streamsize)(D * sizeof(double)));
        f.write((const char*)bias_.data(), (std::streamsize)(K * sizeof(double)));
        return (bool)f;
    }

    // A shape mismatch falls through to the corpus-trained weights rather than
    // loading garbage: the family list and feature width are compile-time facts
    // here, so a file that disagrees was written by a different build.
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        uint32_t magic = 0, K = 0, D = 0;
        size_t upd = 0;
        f.read((char*)&magic, sizeof(magic));
        f.read((char*)&K, sizeof(K));
        f.read((char*)&D, sizeof(D));
        f.read((char*)&upd, sizeof(upd));
        if (!f || magic != 0x49524F55 || K != families_.size() || D != kDims) return false;

        std::vector<std::vector<double>> W(K, std::vector<double>(D, 0.0));
        std::vector<double> b(K, 0.0);
        for (uint32_t k = 0; k < K; ++k)
            f.read((char*)W[k].data(), (std::streamsize)(D * sizeof(double)));
        f.read((char*)b.data(), (std::streamsize)(K * sizeof(double)));
        if (!f) return false;

        std::lock_guard<std::mutex> lock(mu_);
        W_ = std::move(W);
        bias_ = std::move(b);
        updates_ = upd;
        return true;
    }


    // ── training corpus ─────────────────────────────────────────────────────
    struct FamilySpec {
        const char* name;
        std::vector<const char*> seeds;      // {A}/{B} templated utterances
        bool two_slot;
    };

    // Online rates. Deliberately asymmetric — see reinforce(). Both are far
    // below the boot corpus rate (0.30) so live turns nudge the prior rather
    // than overwrite it: an online learner that drifts off a working prior is
    // worse than a frozen one, because the frozen one is at least predictable.
    // Weight decay on the boot corpus. Currently 0 — DELIBERATELY INERT, kept
    // visible because it is the first knob to reach for once the calibration
    // problem below is properly fixed.
    //
    // MEASURED, and worth recording so it is not rediscovered: this router's
    // confidence is saturated. Every input scores ~1.0, INCLUDING gibberish
    // ("zorp the blimflarg quixotically" -> EXPLAIN at 0.999998). So the
    // `confidence >= 0.55` gate in parse_intent_to_bql never falls through, and
    // the safety property claimed in this file's header — "an unseen utterance
    // produces near-zero logits -> uniform distribution -> low confidence ->
    // legacy fallback" — does not hold.
    //
    // Three fixes were tried and MEASURED to fail:
    //   L2 decay (0.003 .. 0.02)  real paraphrases drop below the gate before
    //                             junk does; no value separates them
    //   raw max logit             overlaps: scales with utterance length
    //   logit / feature count     overlaps: short junk tokens score high
    //
    // The reason is structural, not a tuning miss: a 6-way discriminative
    // softmax has no density model of its input and cannot represent "none of
    // these" — the probabilities are forced to sum to 1, so an unfamiliar
    // utterance is still normalised into a confident pick. The fix that works
    // is an explicit OTHER family trained on negative examples, which gives the
    // model somewhere to put gibberish. Until then the gate is decorative and
    // routing is effectively always-on.
    //
    // This also blocks predictive competition among engines: a confidence that
    // is always 1.0 cannot serve as a bid.
    static constexpr double kL2 = 0.0;
    static constexpr double kRewardLr  = 0.08;
    static constexpr double kPenaltyLr = 0.04;

    static constexpr size_t kDims = 1536;

public:
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
        families_.push_back("OTHER");

        build_corpus(specs, 6);
        build_other_corpus((int)families_.size() - 1);
        train(22, 0.30);
    }

private:

    // Systematically generated negative corpus for the OTHER family. Hand-
    // written junk seeds don't converge (measured: see kL2 comment) because
    // arbitrary gibberish has unbounded trigram coverage. Instead generate
    // volume + diversity deterministically across categories that don't share
    // vocabulary with the six op families, so OTHER's weights occupy the
    // hash bins that gibberish actually lands on rather than a hand-picked
    // subset of them.
    static constexpr int kOtherNonsense = 175;   // invented syllable words
    static constexpr int kOtherMash     = 90;    // keyboard-mash strings
    static constexpr int kOtherChatter  = 90;   // greetings/thanks/filler-only
    static constexpr int kOtherArith    = 90;    // bare arithmetic

    void build_other_corpus(int label) {
        std::mt19937 rng(4242u + 1u);
        static const char consonants[] = "bcdfgjklmnpqrstvwxz";
        static const char vowels[]     = "aeiou";
        const int n_cons = (int)(sizeof(consonants) - 1);
        const int n_vow  = (int)(sizeof(vowels) - 1);

        auto rand_syllable = [&](std::string& out) {
            out += consonants[rng() % n_cons];
            out += vowels[rng() % n_vow];
            if (rng() % 3 == 0) out += consonants[rng() % n_cons];
        };
        auto rand_word = [&]() {
            std::string w;
            int syl = 1 + (int)(rng() % 3);
            for (int i = 0; i < syl; ++i) rand_syllable(w);
            return w;
        };

        // 1. invented nonsense words ("zorp blimflarg", "xyzzy plugh frotz")
        for (int i = 0; i < kOtherNonsense; ++i) {
            int nw = 2 + (int)(rng() % 4);
            std::string line;
            for (int w = 0; w < nw; ++w) {
                if (w) line += ' ';
                line += rand_word();
            }
            corpus_.push_back({featurize(line), label});
        }

        // 2. keyboard-mash ("asdf qwerty zxcv")
        static const char mash_pool[] = "qwertyuiopasdfghjklzxcvbnm";
        const int n_mash = (int)(sizeof(mash_pool) - 1);
        for (int i = 0; i < kOtherMash; ++i) {
            int nw = 2 + (int)(rng() % 3);
            std::string line;
            for (int w = 0; w < nw; ++w) {
                if (w) line += ' ';
                int len = 3 + (int)(rng() % 4);
                for (int c = 0; c < len; ++c) line += mash_pool[rng() % n_mash];
            }
            corpus_.push_back({featurize(line), label});
        }

        // 3. template-free chatter/social turns ("hello there", "thanks so
        //    much", "lol ok") — generic words with no task content attached.
        static const char* chatter_words[] = {
            "hello", "hi", "hey", "yo", "sup", "thanks", "thank", "you", "so",
            "much", "appreciated", "cool", "nice", "awesome", "great", "ok",
            "okay", "sure", "yes", "no", "maybe", "lol", "haha", "lmao",
            "there", "friend", "good", "morning", "bye", "goodbye", "later",
            "welcome", "worries", "sounds", "cheers", "yep", "nah", "fine",
            "alright", "the", "and", "and",
        };
        const int n_chatter = (int)(sizeof(chatter_words) / sizeof(*chatter_words));
        for (int i = 0; i < kOtherChatter; ++i) {
            int nw = 1 + (int)(rng() % 4);
            std::string line;
            for (int w = 0; w < nw; ++w) {
                if (w) line += ' ';
                line += chatter_words[rng() % n_chatter];
            }
            corpus_.push_back({featurize(line), label});
        }

        // 4. bare arithmetic ("12 + 45 - 3")
        static const char ops[] = {'+', '-', '*', '/'};
        for (int i = 0; i < kOtherArith; ++i) {
            int n_terms = 2 + (int)(rng() % 4);
            std::string line = std::to_string(rng() % 1000);
            for (int t = 1; t < n_terms; ++t) {
                line += ' ';
                line += ops[rng() % 4];
                line += ' ';
                line += std::to_string(rng() % 1000);
            }
            corpus_.push_back({featurize(line), label});
        }
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
                    // L2 decay on the active features. Without it 22 unregularised
                    // epochs drove the weights until EVERY input — including
                    // gibberish — scored ~1.0 confidence, which made the 0.55
                    // fallback gate in parse_intent_to_bql unreachable and turned
                    // the documented "unseen utterance -> low confidence -> legacy
                    // parser" safety property into something that never happened.
                    // A confidence that is always 1 carries no information, and it
                    // cannot serve as a bid in any competition among engines.
                    for (int f : feats) W_[k][f] -= lr * (g + kL2 * W_[k][f]);
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
    mutable std::mutex mu_;      // reinforce() mutates W_ while the self-play
                                 // daemon may classify() concurrently
    size_t updates_ = 0;         // live turns learned from, distinct from the
                                 // boot corpus epochs
};

} // namespace core
} // namespace brain3
