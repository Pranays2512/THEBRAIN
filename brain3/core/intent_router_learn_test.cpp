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

// Direction of learning is asserted against the UNNORMALISED logit, not the
// softmax probability. The router's confidence is saturated — nearly every
// input scores ~1.0 — so `after > before` on a probability is unmeasurable at
// float precision even when the update landed correctly. The logit is what
// reinforce() actually moves.
static double logit_of(IntentRouter& r, const std::string& text, const std::string& family) {
    return r.logit_for(text, family);
}

int main() {
    std::printf("\n=== intent router: does it learn from outcomes? ===\n\n");

    std::printf("1. Reinforcement moves confidence toward the outcome\n");
    {
        IntentRouter r;
        // An utterance the router is genuinely unsure about. Direction is only
        // measurable here — see section 1b for why.
        const std::string utt = "wubble frotzle nimwit";
        const double before = logit_of(r, utt, "LOOKUP");
        for (int i = 0; i < 12; ++i) r.reinforce(utt, "LOOKUP", true);
        const double after = logit_of(r, utt, "LOOKUP");
        ok(after > before + 1.0, "reward raises the target family's score",
           "before=" + std::to_string(before) + " after=" + std::to_string(after));
    }
    {
        IntentRouter r;
        const std::string utt = "wubble frotzle nimwit";
        const double before = logit_of(r, utt, "LOOKUP");
        for (int i = 0; i < 12; ++i) r.reinforce(utt, "LOOKUP", false);
        const double after = logit_of(r, utt, "LOOKUP");
        ok(after < before - 1.0, "penalty lowers the penalised family's score",
           "before=" + std::to_string(before) + " after=" + std::to_string(after));
    }

    // ── 1b. THE LIMIT OF THE LOOP, measured and reported ───────────────────
    // reinforce() uses the softmax gradient g = p_k - [k==y]. Where the router
    // is ALREADY CERTAIN, p_k is ~1.0 for the chosen family, so g is ~0 and the
    // update vanishes. Measured over 12 steps on the same phrase:
    //
    //     uncertain input ("wubble frotzle nimwit")   delta = +6.4e+01
    //     saturated input ("kindly map neurons ...")  delta = +3.3e-10
    //
    // Since confidence is saturated for nearly every input, the closed loop is
    // largely INERT in production: it can learn where it is unsure, and it is
    // unsure of almost nothing. Worse, the case you most want it to learn from —
    // a CONFIDENTLY WRONG routing — produces the smallest gradient of all.
    //
    // This is reported, not gated. Fixing it means fixing saturation itself
    // (temperature scaling on the training objective, or a calibrated loss),
    // not patching reinforce(). Recorded here because the loop LOOKS closed in
    // the code and is mostly not, which is precisely the kind of thing this
    // codebase has been burned by before.
    {
        IntentRouter r;
        const std::string sat = "kindly map neurons to transistors";
        const double b = logit_of(r, sat, "ANALOGY");
        for (int i = 0; i < 12; ++i) r.reinforce(sat, "ANALOGY", true);
        const double d = logit_of(r, sat, "ANALOGY") - b;
        std::printf("  %s  %-46s delta=%.3e over 12 steps\n",
                    std::abs(d) > 1.0 ? "PASS" : "GAP ",
                    "learning on an ALREADY-CONFIDENT input", d);
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

    // ── CALIBRATION: fixed, and now gated ─────────────────────────────────
    // Both halves gate. Junk rejection went 0/8 -> 8/8 without costing the
    // router any of the six real utterances it already claimed.
    //
    // THE DEFECT WAS: a 6-way discriminative softmax has no density model of its
    // input and cannot represent "none of these" — probabilities are forced to
    // sum to 1, so an unfamiliar utterance was still normalised into a confident
    // pick ("hello there" -> REFUTE at 1.000000). The 0.55 gate in
    // parse_intent_to_bql therefore never fell through.
    //
    // THE FIX: a 7th OTHER family whose negative corpus is GENERATED
    // procedurally and deterministically (mt19937 4243) across four categories
    // that share no vocabulary with the six op families — invented syllable
    // words, keyboard mash, bare social chatter, and arithmetic strings.
    //
    // WHAT MADE IT WORK WHERE FIVE EARLIER ATTEMPTS FAILED — recorded because
    // the difference is one number, not one idea:
    //   L2 decay 0.003..0.02   real paraphrases fell below the gate before junk
    //                          did; no value separated them (held-out 6/6->3/6)
    //   raw max logit          overlapped, scales with utterance length
    //   logit / feature count  overlapped, short junk tokens score high
    //   margin (top1 - top2)   useless: 1.000000 for real AND junk, since the
    //                          distribution is already one-hot
    //   OTHER, hand-written seeds, ~2100 samples
    //                          junk 8/8 but its prior swallowed real utterances
    //   OTHER, generated, ~530 samples
    //                          junk 8/8 AND real 6/6
    // The failing and passing OTHER attempts differ almost only in CORPUS SIZE.
    // Oversized, OTHER's prior dominates ties and eats real families; sized to a
    // typical op family (~530 vs 400-950) it does not. Class imbalance was the
    // whole problem.
    //
    // STILL TRUE AND NOT FIXED: confidence remains saturated at ~1.0 for
    // everything, junk included. Deferral works because junk lands in OTHER and
    // route_extract has no OTHER case, NOT because the 0.55 gate fires. That
    // gate is still decorative, and a confidence that is always 1.0 still cannot
    // serve as a bid — predictive competition among engines remains blocked on
    // calibrating the magnitude, not just the argmax.
    //
    // GENERALISATION, measured on phrases in neither test file: unseen junk
    // deferred 6/8 against a baseline of 0/8. Unseen real classified correctly
    // 3/6 against a baseline of 4/6 — but for both cases where the answer
    // changed, the END-TO-END output was identical, because the baseline's pick
    // failed slot extraction and fell through to the same legacy chain.
    std::printf("\n5. CALIBRATION — the router can say 'none of these'\n");
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
        ok(deferred == (int)junk.size(), "junk and chatter defer to the legacy chain",
           deferred == (int)junk.size() ? "8/8 deferred" : hijacked);
    }

    std::printf("\n=== %d passed, %d failed ===\n\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
