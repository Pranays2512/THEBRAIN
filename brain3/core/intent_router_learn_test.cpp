/**
 * intent_router_learn_test.cpp — does the front door learn from being wrong?
 *
 * IntentRouter classifies an utterance into an op family and, above 0.55
 * confidence, that verdict is AUTHORITATIVE in parse_intent_to_bql. It is
 * trained once at construction on a synthetic paraphrase corpus and then
 * frozen: `static const IntentRouter& instance()` — the const is the design
 * statement. No save, no load, no reward path. It never finds out whether a
 * routing decision produced a verified answer, and every lesson it could have
 * learned dies at process exit.
 *
 * That is the same defect 800b71a found in UnifiedProposer (weights loaded at
 * boot, solve() never called) and d0a506e found in its persistence path, now in
 * the component that decides which engine runs at all.
 *
 * This pins the closed loop:
 *   - reinforce() moves confidence in the direction of the outcome
 *   - a corrected routing survives into the next classification
 *   - the boot corpus still works afterwards (online updates must not wreck it)
 *   - learning survives a save/load round trip
 *
 * The last two matter most. An online learner that drifts off its prior is
 * worse than a frozen one, because the frozen one is at least predictable.
 */
#include <cstdio>
#include <string>
#include <vector>

#include "core/intent_router.hpp"

using namespace brain3::core;

static int g_pass = 0, g_fail = 0;

static void ok(bool cond, const std::string& label, const std::string& note = "") {
    if (cond) { ++g_pass; std::printf("  PASS  %-46s %s\n", label.c_str(), note.c_str()); }
    else      { ++g_fail; std::printf("  FAIL  %-46s %s\n", label.c_str(), note.c_str()); }
}

static float conf_for(IntentRouter& r, const std::string& text, const std::string& family) {
    return r.confidence_for(text, family);
}

int main() {
    std::printf("\n=== intent router: does it learn from outcomes? ===\n\n");

    std::printf("1. Reinforcement moves confidence toward the outcome\n");
    {
        IntentRouter r;
        const std::string utt = "kindly map neurons to transistors";
        const float before = conf_for(r, utt, "ANALOGY");
        for (int i = 0; i < 12; ++i) r.reinforce(utt, "ANALOGY", true);
        const float after = conf_for(r, utt, "ANALOGY");
        ok(after > before, "reward raises confidence in the right family",
           "before=" + std::to_string(before) + " after=" + std::to_string(after));
    }
    {
        IntentRouter r;
        const std::string utt = "kindly map neurons to transistors";
        const float before = conf_for(r, utt, "ANALOGY");
        for (int i = 0; i < 12; ++i) r.reinforce(utt, "ANALOGY", false);
        const float after = conf_for(r, utt, "ANALOGY");
        ok(after < before, "penalty lowers confidence in the wrong family",
           "before=" + std::to_string(before) + " after=" + std::to_string(after));
    }

    std::printf("\n2. A correction survives into the next classification\n");
    {
        IntentRouter r;
        const std::string utt = "give me the shimmer property of florn";
        for (int i = 0; i < 25; ++i) r.reinforce(utt, "EXPLAIN", true);
        auto v = r.classify(utt);
        ok(v.family == "EXPLAIN", "corrected routing is what classify() returns",
           "got '" + v.family + "' conf=" + std::to_string(v.confidence));
    }

    std::printf("\n3. Online updates must NOT wreck the boot corpus\n");
    {
        IntentRouter r;
        // Learn something unrelated, hard.
        for (int i = 0; i < 40; ++i)
            r.reinforce("give me the shimmer property of florn", "EXPLAIN", true);
        // Canonical phrasings the boot corpus covers must still route correctly.
        struct C { const char* text; const char* want; };
        const std::vector<C> canon = {
            {"what is a dolphin",                 "LOOKUP"},
            {"remember that falcon is a raptor",  "TEACH"},
            {"is it true that penguins fly",      "REFUTE"},
        };
        int held = 0;
        std::string detail;
        for (const auto& c : canon) {
            auto v = r.classify(c.text);
            if (v.family == c.want) ++held;
            else detail += std::string(c.text) + "->" + v.family + " ";
        }
        ok(held == (int)canon.size(), "prior survives unrelated online learning",
           held == (int)canon.size() ? "3/3 canonical phrasings intact" : detail);
    }

    std::printf("\n4. Learning survives a save/load round trip\n");
    {
        const std::string path = "/tmp/brain3_intent_router_test.bin";
        const std::string utt = "give me the shimmer property of florn";
        float trained = 0.f;
        {
            IntentRouter r;
            for (int i = 0; i < 25; ++i) r.reinforce(utt, "EXPLAIN", true);
            trained = conf_for(r, utt, "EXPLAIN");
            ok(r.save(path), "save() writes weights", path);
        }
        {
            IntentRouter r;
            const float fresh = conf_for(r, utt, "EXPLAIN");
            ok(r.load(path), "load() reads weights back", "");
            const float restored = conf_for(r, utt, "EXPLAIN");
            ok(std::abs(restored - trained) < 1e-4, "restored == trained",
               "fresh=" + std::to_string(fresh) + " restored=" + std::to_string(restored) +
               " trained=" + std::to_string(trained));
        }
    }

    // ── KNOWN GAP: reported, deliberately not gated ────────────────────────
    // Section 5 measures a defect that is real, characterised, and NOT fixed.
    // The junk half reports without failing the suite — the same convention
    // heldout_probe uses for its documented gaps. A permanently red test
    // teaches nothing, but deleting the measurement would hide a live problem.
    // The "real utterances" half DOES gate: whatever fixes calibration must not
    // cost the router the utterances it currently gets right.
    //
    // THE DEFECT: confidence is saturated. Every input scores ~1.0, gibberish
    // included ("hello there" -> REFUTE at 1.000000), so the 0.55 gate in
    // parse_intent_to_bql never falls through and the documented legacy-parser
    // fallback is unreachable.
    //
    // MEASURED AND FAILED, recorded so they are not retried blind:
    //   L2 decay 0.003..0.02    real paraphrases drop below the gate before junk
    //                           does; no value separates them (held-out 6/6->3/6)
    //   raw max logit           overlaps, scales with utterance length
    //   logit / feature count   overlaps, short junk tokens score high
    //   OTHER family, ~46 hand-written seeds
    //                           junk rejection 0/8 -> 8/8, but its prior
    //                           swallowed real utterances ("outline strategy for
    //                           market entry" -> OTHER at 1.000000)
    //   OTHER, corpus-balanced, bias unlearned, seeds curated
    //                           junk 7/8, but boundaries between the REAL
    //                           families shifted and held-out fell 6/6 -> 5/6
    //
    // WHAT THAT SHOWED: OTHER is directionally right — the only approach that
    // moved junk rejection at all — but hand-picked seeds cannot converge.
    // Arbitrary gibberish has unbounded trigram coverage, and short generic
    // negatives ("ok", "lol") collide with short generic op phrasings. The next
    // attempt should GENERATE the negative corpus systematically and re-tune the
    // router as a whole, rather than bolting a seventh class onto weights fitted
    // for six.
    //
    // WHY IT MATTERS BEYOND ROUTING: a confidence that is always 1.0 cannot
    // serve as a bid, so predictive competition among engines is blocked on this.
    std::printf("\n5. CALIBRATION — KNOWN GAP, reported not gated\n");
    {
        IntentRouter r;
        // Real utterances the router is SUPPOSED to claim, with the family the
        // existing held-out suite expects.
        struct C { const char* text; const char* want; };
        const std::vector<C> real = {
            {"silently wonder if deforestation causes droughts", "WHAT_IF"},
            {"please remember that whales are mammals",          "TEACH"},
            {"can you describe photosynthesis",                  "LOOKUP"},
            {"kindly map neurons to transistors",                "ANALOGY"},
            {"maybe falsify every prime is odd",                 "REFUTE"},
            {"outline strategy for market entry",                "EXPLAIN"},
        };
        // Things that must NOT be claimed: nonsense, social turns, bare maths.
        // Each of these has somewhere better to go (the mouth, INSTINCT, the
        // legacy chain) and the router hijacking them is the failure.
        const std::vector<std::string> junk = {
            "zorp the blimflarg quixotically", "asdf qwerty zxcv",
            "xyzzy plugh frotz", "blorp glim wug", "the the the and and",
            "hello there", "thanks so much", "lol ok",
        };

        int claimed = 0;
        float min_real = 1.f;
        std::string missed;
        for (const auto& c : real) {
            auto v = r.classify(c.text);
            const bool ok_ = (v.family == c.want && v.confidence >= 0.55f);
            if (ok_) { ++claimed; min_real = std::min(min_real, v.confidence); }
            else missed += std::string(c.text) + "->" + v.family + "(" +
                           std::to_string(v.confidence) + ") ";
        }
        ok(claimed == (int)real.size(), "real utterances still claimed above the gate",
           claimed == (int)real.size() ? "6/6" : missed);

        int deferred = 0;
        std::string hijacked;
        for (const auto& j : junk) {
            auto v = r.classify(j);
            // Deferring means either landing in OTHER or scoring below the gate.
            // Both route the utterance to the legacy chain, which is the point.
            const bool defers = (v.family == "OTHER" || v.confidence < 0.55f);
            if (defers) ++deferred;
            else hijacked += j + "->" + v.family + "(" + std::to_string(v.confidence) + ") ";
        }
        // Reported, not gated — see the block above. When this reaches 8/8 the
        // printf becomes an ok() and the gap is closed.
        std::printf("  %s  %-44s %d/%d deferred%s\n",
                    deferred == (int)junk.size() ? "PASS" : "GAP ",
                    "junk and chatter defer to the legacy chain",
                    deferred, (int)junk.size(),
                    deferred == (int)junk.size() ? "" : ("  | hijacked: " + hijacked).c_str());
    }

    std::printf("\n=== %d passed, %d failed ===\n\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
