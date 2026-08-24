// test_intent_router.cpp — Sprint 4a: learned intent routing.
//   1. held-out synthetic paraphrase accuracy >= 90%
//   2. pronoun stoplist: "who are you" must NOT become a knowledge lookup
//   3. literal BQL commands pass through untouched (legacy-first)
//   4. end-to-end: previously-missed eval paraphrases now route correctly
#include <iostream>
#include <algorithm>
#include <string>
#include <random>
#include <vector>
#include "core/master_orchestrator.hpp"

using namespace brain3::core;

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    if (ok) { g_pass++; std::cout << "  [PASS] " << name << "\n"; }
    else    { g_fail++; std::cout << "  [FAIL] " << name << "\n"; }
}

static std::string family_of(const std::string& bql) {
    size_t sp = bql.find(' ');
    std::string f = sp == std::string::npos ? bql : bql.substr(0, sp);
    for (auto& c : f) c = (char)std::tolower((unsigned char)c);
    return f;
}

int main() {
    std::cout << "=== learned intent router ===\n";
    const auto& router = IntentRouter::instance();

    // ── 1. held-out paraphrase accuracy (fresh slot nouns + fillers) ────────
    struct Case { const char* utt; const char* fam; };
    static const Case heldout[] = {
        // unseen filler + unseen nouns vs the training corpus
        {"silently wonder if deforestation causes droughts",          "WHAT_IF"},
        {"please remember that whales are mammals",                   "TEACH"},
        {"can you describe photosynthesis",                           "LOOKUP"},
        {"kindly map neurons to transistors",                         "ANALOGY"},
        {"maybe falsify every prime is odd",                          "REFUTE"},
        {"outline strategy for market entry",                         "EXPLAIN"},
    };
    int ok = 0, n = 0;
    for (const auto& c : heldout) {
        auto v = router.classify(c.utt);
        ++n;
        bool hit = v.family == c.fam && v.confidence >= 0.5f;
        ok += hit;
        if (!hit)
            std::cout << "    miss \"" << c.utt << "\" -> "
                      << v.family << " (" << v.confidence << ")\n";
    }
    std::cout << "    held-out accuracy: " << ok << "/" << n << "\n";
    check((double)ok / n >= 0.9, "held-out routing accuracy >= 90%");

    // ── 2. stoplist: personal turns stay out of LOOKUP ──────────────────────
    {
        std::string out;
        const bool became_lookup =
            route_extract("who are you", "LOOKUP", out);
        check(!became_lookup,
              "'who are you' cannot become LOOKUP you (stoplist)");
        check(is_personal_subject("you") && is_personal_subject("yourself"),
              "pronoun stoplist covers personal subjects");
    }

    // ── 3. literal commands unchanged (legacy-first ordering) ───────────────
    {
        struct L { const char* in; const char* want_prefix; };
        static const L lits[] = {
            {"TEACH sky is_a blue", "TEACH"},
            {"COMPUTE 2^10", "COMPUTE"},
            {"LOOKUP sky isa blue", "LOOKUP"},
        };
        bool all = true;
        for (auto& l : lits) {
            auto got = MasterOrchestrator::parse_intent_to_bql(l.in);
            if (got.rfind(l.want_prefix, 0) != 0) { all = false; break; }
        }
        check(all, "literal BQL commands pass through untouched");
    }

    // ── 4. end-to-end upgrades on the eval suite's misses ───────────────────
    {
        struct E { const char* in; const char* fam; };
        static const E es[] = {
            {"i wonder what happens if rain causes flooding", "what_if"},
            {"please store that sky has color blue",          "teach"},
            {"define an analogy between cpu and brain",       "analogy"},
        };
        bool all = true;
        for (const auto& e : es) {
            auto fam = family_of(MasterOrchestrator::parse_intent_to_bql(e.in));
            if (fam != e.fam) {
                all = false;
                std::cout << "    e2e miss \"" << e.in << "\" -> " << fam << "\n";
            }
        }
        check(all, "eval-suite paraphrases upgraded end-to-end");
        // chat stays chat (native mouth territory)
        check(family_of(MasterOrchestrator::parse_intent_to_bql("hello there friend")) == "instinct"
              && family_of(MasterOrchestrator::parse_intent_to_bql("who are you")) == "instinct",
              "chat turns still fall through to INSTINCT/mouth");
    }

    std::cout << "=== passed " << g_pass << ", failed " << g_fail << " ===\n";
    return g_fail == 0 ? 0 : 1;
}
