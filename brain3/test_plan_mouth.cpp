// test_plan_mouth.cpp — Sprint 1: the plan-conditioned mouth and its falsifiers.
//
// Thesis under test: a mouth trained ONLY on (plan → sentence) pairs holds
// zero content memories. Therefore:
//   A. EXTENSIBILITY   new facts (post-training) are instantly speakable
//   B. RECOMBINATION   held-out fact/register combinations generalize
//   C. AMNESIA         deleted facts cannot be spoken — mechanically, and
//                      (measured honestly) even without the content lock
#include <iostream>
#include <random>
#include <cmath>
#include <set>
#include <string>
#include <vector>
#include "crisp/engines/neural/stamlat_transformer.hpp"
#include "crisp/engines/neural/utterance_plan.hpp"

using namespace brain3::engines::neural;

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    if (ok) { g_pass++; std::cout << "  [PASS] " << name << "\n"; }
    else    { g_fail++; std::cout << "  [FAIL] " << name << "\n"; }
}

template <typename T, size_t N> constexpr size_t n_of(const T (&a)[N]) { return N; }

// ── domain tables ────────────────────────────────────────────────────────────
struct Domain {
    const char* act;
    std::vector<std::vector<std::string>> answers;   // canonical slot sequences
    std::vector<std::string> clazz;                  // membership for inversion
};
static const Domain kDomains[] = {
    {"greeting",
     {{"intent","greeting","style","friendly"}, {"intent","greeting","emotion","happy"},
      {"intent","welcome","target","user"},     {"intent","salutation","status","ready"}},
     {"intent","greeting","welcome","salutation","style","friendly","emotion","happy",
      "target","user","status","ready"}},
    {"identity",
     {{"identity","system","type","cognitive"}, {"identity","brain","origin","artificial"},
      {"name","brain","type","ai"},              {"self","network","type","neural"}},
     {"identity","name","self","system","brain","network","type","cognitive",
      "origin","artificial","ai","neural"}},
    {"status",
     {{"status","good","energy","high"},        {"state","positive","mode","ready"},
      {"feeling","great","condition","excellent"},{"status","optimal","emotion","calm"}},
     {"status","state","feeling","good","great","positive","optimal","energy","high",
      "mode","ready","condition","excellent","emotion","calm"}},
};

static const Domain& domain_of(const std::string& act) {
    for (const auto& d : kDomains) if (d.act == act) return d;
    return kDomains[0];
}
static const std::vector<std::string>& answer_of(const Domain& dom, size_t idx) {
    return dom.answers[idx % dom.answers.size()];
}

// distractor pool for an act: words from OTHER domains ONLY.
// (Same-class distractors would make relevance undecidable from the plan.)
static std::vector<std::string> distractor_pool_for(const std::string& act) {
    std::set<std::string> s;
    for (const auto& d : kDomains) {
        if (d.act == act) continue;
        for (const auto& a : d.clazz) s.insert(a);
    }
    return {s.begin(), s.end()};
}

// build one training text: plan linearization + target sentence
static std::string render_sample(const UtterancePlan& p,
                                 const std::vector<std::string>& target) {
    std::string s = p.linearize() + " ";
    for (size_t i = 0; i < target.size(); ++i)
        s += target[i] + (i + 1 < target.size() ? " " : "\n");
    return s;
}

static bool sentence_ok(const std::vector<std::string>& said,
                        const std::vector<std::string>& truth) {
    if (said.size() != truth.size()) return false;
    for (size_t i = 0; i < truth.size(); ++i) if (said[i] != truth[i]) return false;
    return true;
}

int main() {
    std::cout << "=== plan-conditioned mouth: falsifier suite ===\n";
    std::mt19937 rng(3);
    auto pick = [&](auto&& v) { return v[rng() % v.size()]; };

    // ── dataset v3: rule-forcing curriculum ────────────────────────────────
    // Targets are RANDOM orderings of random class-word subsets — far too
    // many permutations to memorize — so the ONLY way to fit training data
    // is the rule itself: "emit own-class plan tokens in plan order."
    // Held-out cells are fresh random sequences; success there can come
    // from nothing except the learned rule.
    std::string train;
    size_t samples = 0;
    auto insert_distractors = [&](UtterancePlan& p, int k, auto& rngv) {
        auto cross = distractor_pool_for(p.act);
        for (int i = 0; i < k; ++i) {
            std::string d = pick(cross);
            bool dup = false;
            for (const auto& f : p.facts) if (f == d) dup = true;
            if (!dup && !d.empty()) {
                size_t pos = rngv() % (p.facts.size() + 1);
                p.facts.insert(p.facts.begin() + (long)pos, d);
            }
        }
    };
    auto random_truth = [&](const Domain& dom, auto& rngv) {
        std::vector<std::string> poolv = dom.clazz;
        std::shuffle(poolv.begin(), poolv.end(), rngv);
        poolv.resize(3 + rngv() % 2);                 // 3-4 word utterances
        return poolv;
    };
    for (const auto& dom : kDomains)
        for (int rep = 0; rep < 260; ++rep) {
            const char* reg = (rep % 2 == 0) ? "warm" : "neutral";
            auto truth = random_truth(dom, rng);
            UtterancePlan p;
            p.act = dom.act; p.reg = reg;
            p.facts = truth;
            train += render_sample(p, truth);
            ++samples;
        }

    StamlatConfig cfg;
    cfg.d_model = 64; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 160;
    cfg.ctx = 48; cfg.depth_gamma = 0.f; cfg.depth_tau = 1.f; cfg.seed = 11;
    StamlatLM lm(cfg);
    lm.build_vocab(train);
    lm.fit(train, 3200, 5e-3f, 16, 800);
    std::cout << "vocab=" << lm.total_vocab_size()
              << " (" << lm.word_vocab_size() << " words) train_samples="
              << samples << "\n";

    // Few-shot consolidation: teach ONE (plan→sentence) association in
    // milliseconds via weighted SFT — no corpus rebuild, no full retrain.
    auto absorb = [&](const UtterancePlan& p,
                      const std::vector<std::string>& target,
                      int steps = 40) {
        std::string text = render_sample(p, target);
        auto ids = lm.encode(text);
        if ((int)ids.size() > lm.config().ctx + 1)
            ids.resize(lm.config().ctx + 1);
        SftExample e;
        e.x.assign(ids.begin(), ids.end() - 1);
        e.y.assign(ids.begin() + 1, ids.end());
        const size_t prompt_len = lm.encode(p.linearize()).size();
        e.w.assign(e.x.size(), 0.f);
        for (size_t t = 0; t < e.x.size(); ++t)
            if (t + 1 >= prompt_len) e.w[t] = 1.f;
        for (int s = 0; s < steps; ++s) lm.sft_step({e}, 4e-3f);
    };

    // speak(plan): greedy decode under the mechanical content lock
    auto speak = [&](const UtterancePlan& p) {
        auto allowed = p.content_lock_ids(lm);
        auto ids = lm.encode(p.linearize());
        return lm.stream_complete_ids(ids, 24, 0.f, true, &allowed);
    };
    // unconstrained twin: same prompt, free decoding (honesty measurement)
    auto speak_free = [&](const UtterancePlan& p) {
        return lm.stream_complete_ids(lm.encode(p.linearize()), 24, 0.f);
    };
    auto words_of = [&](const std::string& s) {
        std::vector<std::string> out; size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && s[i] == ' ') ++i;
            size_t j = i; while (j < s.size() && s[j] != ' ') ++j;
            if (j > i) out.push_back(s.substr(i, j - i));
            i = j;
        }
        return out;
    };

    // ── sanity: trained cell reproduces its template ────────────────────────
    {
        const Domain& d = domain_of("greeting");
        UtterancePlan p; p.act = "greeting"; p.reg = "warm";
        p.facts = d.answers[0];
        const std::string said = speak(p);
        check(sentence_ok(words_of(said), d.answers[0]),
              "trained cell renders exactly (got \"" + said + "\")");
    }

    // ── EXPERIMENT B: held-out recombination — fresh random sequences ───────
    {
        int ok = 0, total = 0;
        std::mt19937 fresh(777);                  // disjoint draw stream
        auto random_truth_fresh = [&](const Domain& dom) {
            std::vector<std::string> poolv = dom.clazz;
            std::shuffle(poolv.begin(), poolv.end(), fresh);
            poolv.resize(3 + fresh() % 2);
            return poolv;
        };
        for (const auto& dom : kDomains)
            for (int cell = 0; cell < 2; ++cell) {
                const char* reg = (cell == 0) ? "warm" : "neutral";
                UtterancePlan p; p.act = dom.act; p.reg = reg;
                auto truth = random_truth_fresh(dom);
                p.facts = truth;
                ++total;
                if (sentence_ok(words_of(speak(p)), truth)) ++ok;
                else {
                    std::string want; for (auto& w : truth) want += w + " ";
                    std::cout << "    miss [" << dom.act << "/" << reg << "]: \""
                              << speak(p) << "\"  want: " << want << "\n";
                }
            }
        std::cout << "    recombination exact-match: " << ok << "/" << total << "\n";
        check((double)ok >= std::ceil(total * 0.75),
              "held-out sequences generalize under rule-forcing curriculum (>=75%)");
    }

    // ── EXPERIMENT A: extensibility — brand-new facts post-training ─────────
    {
        lm.extend_vocab_words({"quantum", "flurbix", "sky"});
        UtterancePlan p; p.act = "identity"; p.reg = "neutral";
        const std::vector<std::string> target =
            {"identity", "system", "type", "quantum"};
        p.facts = target;
        absorb(p, target, 40);                    // seconds-of-experience learning
        const std::string said = speak(p);        // content lock includes new word
        const auto w = words_of(said);
        bool has_new = false;
        for (const auto& x : w) if (x == "quantum") has_new = true;
        std::cout << "    novel-fact reply: \"" << said << "\"\n";
        check(has_new, "post-training fact is instantly speakable (few-shot absorbed)");
        check(sentence_ok(w, target),
              "novel fact lands in the correct structural slot");
    }

    // ── EXPERIMENT C: amnesia — deleted facts must be unspeakable ───────────
    {
        // "artificial" was heavily trained. Build plans WITHOUT it.
        UtterancePlan p; p.act = "identity"; p.reg = "warm";
        for (const auto& a : domain_of("identity").answers[0]) p.facts.push_back(a);

        int leak_locked = 0, leak_free = 0, N = 30;
        for (int i = 0; i < N; ++i) {
            if (speak(p).find("artificial") != std::string::npos) ++leak_locked;
            if (speak_free(p).find("artificial") != std::string::npos) ++leak_free;
        }
        std::cout << "    deleted-fact leakage: locked=" << leak_locked
                  << "/" << N << "  unconstrained=" << leak_free << "/" << N << "\n";
        check(leak_locked == 0,
              "content lock makes deleted memories unspeakable (0%)");
        std::cout << "    [info] unconstrained leakage " << leak_free
                  << "/" << N << " — the honest number for the no-lock ablation\n";

        // full amnesia: ask about an act whose every surface is absent
        UtterancePlan empty_p; empty_p.act = "identity"; empty_p.reg = "warm";
        empty_p.facts = {};                    // memory wiped
        auto lock_nothing = empty_p.content_lock_ids(lm);
        // only whitespace glue remains admissible
        bool only_glue = true;
        for (int id : lock_nothing) {
            const std::string s = lm.token_surface(id);
            if (!(s == " " || s == "\n")) only_glue = false;
        }
        check(only_glue, "wiped memory leaves nothing but silence to say");
    }

    // persistence: extended vocab + weights survive save/load
    {
        check(lm.save("plan_mouth.bin"), "save plan-mouth ok");
        StamlatLM lm2(cfg);
        check(lm2.load("plan_mouth.bin"), "reload ok");
        UtterancePlan p; p.act = "identity"; p.reg = "neutral";
        p.facts = {"identity", "system", "type", "quantum"};
        auto allowed = p.content_lock_ids(lm2);
        const std::string said =
            lm2.stream_complete_ids(lm2.encode(p.linearize()), 24, 0.f, true, &allowed);
        check(said.find("quantum") != std::string::npos,
              "post-reload mouth still speaks the post-training fact");
    }

    std::cout << "=== passed " << g_pass << ", failed " << g_fail << " ===\n";
    return g_fail == 0 ? 0 : 1;
}
