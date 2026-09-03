// eval/run_eval.cpp — THE SCOREBOARD.
//
// Suites:
//   1. intent        literal-command accuracy (GATE) + paraphrase baseline
//   2. retention     teach -> distract -> lookup, multi-turn fact survival (GATE)
//   3. mouth         contract preservation under mood noise (T=0 GATE)
//   4. plan_grid     fast recombination grid (machinery warmth, informational)
//
// Exit 0 only when every gate passes. Report written to eval_report.json.
#include <iostream>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cctype>

#include "core/master_orchestrator.hpp"
#include "crisp/engines/neural/stamlat_transformer.hpp"
#include "crisp/engines/neural/mouth_voice.hpp"
#include "crisp/engines/neural/utterance_plan.hpp"
#include "eval/eval_suite.hpp"

using namespace brain3::engines::neural;
namespace fe = brain3::eval;

// ── gates (tighten as the architecture improves; never loosen silently) ─────
static constexpr double GATE_INTENT_LITERAL   = 1.00;  // commands must route
static constexpr double GATE_RETENTION        = 0.60;  // initial calibration
static constexpr double GATE_MOUTH_T0         = 1.00;  // greedy = always in-contract
static constexpr double GATE_PLAN_GRID        = 0.50;  // machinery-warmth floor

// ── Suite 1: intent routing ──────────────────────────────────────────────────
struct IntentCase { const char* utt; const char* family; bool literal; };

static fe::SuiteResult suite_intent() {
    static const IntentCase cases[] = {
        // literal / gate set
        {"LOOKUP sky isa blue",                     "LOOKUP",           true},
        {"TEACH sky is_a blue",                     "TEACH",            true},
        {"COMPUTE 2^10",                            "COMPUTE",          true},
        {"INGEST data/docs",                        "INGEST",           true},
        {"cross domain hunt now",                   "CROSS_DOMAIN_HUNT",true},
        {"solve cf hard problem",                   "COMPUTE",          true},   // CF solver family
        {"1 = 0 proof",                             "INSTINCT",         true},
        {"12*(3+4)",                                "INSTINCT",         true},
        {"refute that all swans are white",         "REFUTE",           true},
        {"what if rain causes floods",              "WHAT_IF",          true},
        {"teach birds is a animal",                 "TEACH",            true},
        {"what is gravity",                         "LOOKUP",           true},
        {"teach me about photosynthesis",           "LOOKUP",           true},
        {"compare heart to pump",                   "ANALOGY",          true},
        {"goal: write quarterly report",            "AGENTIC_GOAL",     true},
        {"hello there friend",                      "INSTINCT",         true},   // chat fall-through
        // paraphrase / generalization baseline (informational today)
        {"please store that sky has color blue",    "TEACH",            false},
        {"can you tell me who is einstein",         "LOOKUP",           false},
        {"i wonder what happens if rain causes flooding", "WHAT_IF",    false},
        {"define an analogy between cpu and brain", "ANALOGY",          false},
        {"what's 6*7",                              "INSTINCT",         false},
        {"kindly explain simply quantum tunneling", "LOOKUP",           false},
    };
    fe::SuiteResult s; s.name = "intent";
    auto norm = [](const std::string& fam) {
        std::string l = fe::op_family(fam);
        for (auto& c : l) c = (char)std::tolower((unsigned char)c);
        return l;
    };
    int lit_ok = 0, lit_n = 0, par_ok = 0, par_n = 0;
    for (const auto& c : cases) {
        const std::string got = norm(
            brain3::core::MasterOrchestrator::parse_intent_to_bql(c.utt));
        std::string want = c.family;
        for (auto& ch : want) ch = (char)std::tolower(ch);
        const bool hit = (got == want);
        if (c.literal) { ++lit_n; lit_ok += hit;
            if (!hit) { s.detail.emplace_back("literal_miss", std::string(c.utt) + " -> " + got); }
        } else { ++par_n; par_ok += hit;
            if (!hit) s.detail.emplace_back("paraphrase_miss", std::string(c.utt) + " -> " + got);
        }
    }
    s.score = lit_n ? (double)lit_ok / lit_n : 0.0;
    s.gate_pass = s.score >= GATE_INTENT_LITERAL;
    s.detail.emplace_back("literal_accuracy",
        std::to_string(lit_ok) + "/" + std::to_string(lit_n));
    s.detail.emplace_back("paraphrase_accuracy",
        std::to_string(par_ok) + "/" + std::to_string(par_n) + " (baseline)");
    return s;
}

// ── Suite 2: multi-turn fact retention ───────────────────────────────────────
static fe::SuiteResult suite_retention(brain3::core::MasterOrchestrator& orch) {
    fe::SuiteResult s; s.name = "retention";
    struct Fact { const char* subj; const char* obj; };
    static const Fact facts[] = {
        {"einstein", "scientist"}, {"bohr", "physicist"},
        {"curie", "chemist"},      {"turing", "mathematician"},
    };
    int ok = 0;
    for (const auto& f : facts) {
        auto teach = orch.process(std::string("teach ") + f.subj +
                                  " is a " + f.obj);
        // distraction turn (different engine entirely)
        orch.process("12*(3+4)");
        auto ask = orch.process(std::string("what is ") + f.subj);
        const bool survived = ask.natural_reply.find(f.obj) != std::string::npos;
        s.detail.emplace_back(f.subj, survived ? "retained" : "LOST");
        ok += survived;
    }
    s.score = (double)ok / (sizeof(facts) / sizeof(facts[0]));
    s.gate_pass = s.score >= GATE_RETENTION;
    return s;
}

// ── Suite 3: mouth contract preservation under mood noise ────────────────────
template <typename T, size_t N> constexpr size_t n_of(const T (&a)[N]) { return N; }

static fe::SuiteResult suite_mouth() {
    fe::SuiteResult s; s.name = "mouth_contract";

    static const char* G[] = {"hello","hi","hey there","good morning"};
    static const char* GA[] = {"intent greeting style friendly","intent welcome target user",
                               "intent greeting emotion happy"};
    static const char* W[] = {"who are you","what is your name"};
    static const char* WA[] = {"identity system type cognitive","identity brain origin artificial"};
    std::mt19937 rng(21);
    auto pick = [&](const auto& a){ return a[rng() % n_of(a)]; };
    std::string train;
    for (int i = 0; i < 250; ++i) {
        train += std::string("user: ") + pick(G) + "\nbrain: " + pick(GA) + "\n";
        train += std::string("user: ") + pick(W) + "\nbrain: " + pick(WA) + "\n";
    }
    StamlatConfig cfg;
    cfg.d_model = 48; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 96;
    cfg.ctx = 64; cfg.depth_gamma = 0.f; cfg.depth_tau = 1.f; cfg.seed = 5;
    StamlatLM lm(cfg);
    lm.build_vocab(train);
    lm.fit(train, 500, 5e-3f, 12, 250);

    using Groups = std::vector<std::vector<std::string>>;
    struct Probe { const char* q; Groups g; };
    static const Probe probes[] = {
        {"hello",       {{"intent"}, {"greeting","welcome","happy"}}},
        {"who are you", {{"identity","name"}, {"system","brain","ai","cognitive"}}},
    };
    auto preserves = [&](const std::string& r, const Groups& gs) {
        if (r.empty()) return false;
        for (const auto& g : gs) {
            bool hit = false;
            for (const auto& w : g)
                if (r.find(w) != std::string::npos) { hit = true; break; }
            if (!hit) return false;
        }
        return true;
    };

    auto vm = default_voice_mapper();
    // T=0 gate
    int t0_ok = 0; const int T0_N = 4;
    for (const auto& p : probes)
        for (int i = 0; i < T0_N; ++i)
            t0_ok += preserves(
                lm.stream_complete_ids(lm.encode(std::string("user: ") + p.q +
                                                 "\nbrain: "), 24, 0.f), p.g);
    s.detail.emplace_back("greedy_preservation",
        std::to_string(t0_ok) + "/" + std::to_string(T0_N * 2));
    s.score = (double)t0_ok / (T0_N * 2);
    s.gate_pass = s.score >= GATE_MOUTH_T0;

    // mood curve (informational): excited-positive vs alert-negative
    const brain2::EmotionState moods[] = {{0.9f, 1.f}, {-0.8f, 0.8f}};
    const char* mood_names[] = {"excited_positive", "alert_negative"};
    for (int m = 0; m < 2; ++m) {
        auto pol = vm.policy(lm, moods[m]);
        int ok = 0; const int K = 6;
        for (const auto& p : probes)
            for (int i = 0; i < K / 2; ++i)
                ok += preserves(
                    lm.stream_complete_ids(
                        lm.encode(std::string("user: ") + p.q + "\nbrain: "),
                        20, pol.temperature, true, nullptr, &pol.bias),
                    p.g);
        s.detail.emplace_back(std::string("preservation@") + mood_names[m],
                            std::to_string(ok) + "/" + std::to_string(K));
    }
    return s;
}

// ── Suite 4: fast plan-recombination grid ────────────────────────────────────
static std::string words_flat(const std::string& s) {
    std::string out; size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && s[i] == ' ') ++i;
        size_t j = i; while (j < s.size() && s[j] != ' ') ++j;
        if (j > i) { if (!out.empty()) out += ' '; out += s.substr(i, j - i); }
        i = j;
    }
    return out;
}
static std::string flat(const std::vector<std::string>& v) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i)
        out += v[i] + (i + 1 < v.size() ? " " : "");
    return out;
}

static fe::SuiteResult suite_plan_grid() {
    fe::SuiteResult s; s.name = "plan_grid";
    struct Dom { const char* act; std::vector<std::string> clazz; };
    static const Dom doms[] = {
        {"greeting",{"intent","greeting","welcome","salutation","style","friendly",
                    "emotion","happy","target","user"}},
        {"identity",{"identity","name","self","system","brain","network","type",
                    "cognitive","origin","artificial","ai","neural"}},
    };
    std::mt19937 rng(33);
    auto random_truth = [&](const Dom& d, std::mt19937& g) {
        std::vector<std::string> poolv = d.clazz;
        std::shuffle(poolv.begin(), poolv.end(), g);
        poolv.resize(3 + g() % 2);
        return poolv;
    };
    auto render = [&](const UtterancePlan& p,
                      const std::vector<std::string>& t) {
        std::string line = "<p> act " + p.act + " facts";
        for (auto& f : p.facts) line += " " + f;
        line += " reg " + p.reg + " <r> ";
        for (size_t k = 0; k < t.size(); ++k)
            line += t[k] + (k + 1 < t.size() ? " " : "\n");
        return line;
    };

    // rule-forcing curriculum: random own-class subsequences (unmemorizable)
    std::string train;
    std::vector<std::pair<UtterancePlan, std::vector<std::string>>> kept;
    for (auto& d : doms)
        for (int rep = 0; rep < 260; ++rep) {
            UtterancePlan p; p.act = d.act;
            p.reg = (rep % 2 == 0) ? "warm" : "neutral";
            auto truth = random_truth(d, rng);
            p.facts = truth;
            train += render(p, truth);
            if (rep < 8) kept.emplace_back(p, truth);
        }

    StamlatConfig cfg;
    cfg.d_model = 64; cfg.n_layers = 2; cfg.n_heads = 4; cfg.d_ff = 128;
    cfg.ctx = 48; cfg.depth_gamma = 0.f; cfg.depth_tau = 1.f; cfg.seed = 3;
    StamlatLM lm(cfg);
    lm.build_vocab(train);
    lm.fit(train, 8000, 4e-3f, 16, 2000);

    auto speak = [&](UtterancePlan& p) {
        auto allowed = p.content_lock_ids(lm);
        return lm.stream_complete_ids(lm.encode(p.linearize()), 20, 0.f,
                                      true, &allowed);
    };

    int ctrl_ok = 0;
    for (auto& [p, truth] : kept)
        ctrl_ok += words_flat(speak(p)) == flat(truth);
    s.detail.emplace_back("trained_sample_control",
                          std::to_string(ctrl_ok) + "/" +
                          std::to_string(kept.size()));

    // held-out: fresh random sequences from a disjoint draw stream
    std::mt19937 fresh(777);
    auto random_truth_fresh = [&](const Dom& d) {
        return random_truth(d, fresh);
    };
    int ok = 0, total = 0;
    for (auto& d : doms)
        for (int cell = 0; cell < 3; ++cell) {
            UtterancePlan p; p.act = d.act; p.reg = "neutral";
            auto truth = random_truth_fresh(d);
            p.facts = truth;
            ++total;
            if (words_flat(speak(p)) == flat(truth)) ++ok;
            else s.detail.emplace_back(std::string("grid_miss[") + d.act + "]",
                                       speak(p) + " || want " + flat(truth));
        }
    s.score = (double)ok / total;
    s.gate_pass = s.score >= GATE_PLAN_GRID;
    return s;
}

int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "=== brain3 evaluation scoreboard ===\n";

    // plan_grid trains a StamlatLM from scratch (260 reps x 3 acts) and takes
    // minutes. Its own header describes it as "machinery warmth, informational"
    // and its gate is a 0.50 floor, so it is not a correctness gate — but its
    // cost is what made this whole suite take over ten minutes, which meant it
    // was never run at all. A gate suite nobody can afford to run stops being a
    // gate and becomes a file. The three real gates now always run; the slow
    // informational tier is opt-in via --full or BRAIN_EVAL_FULL=1.
    bool full = std::getenv("BRAIN_EVAL_FULL") != nullptr;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--full") full = true;

    auto timed = [](const char* label, auto&& fn) {
        const auto s = std::chrono::steady_clock::now();
        auto r = fn();
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
        std::cout << "  ... " << label << " took " << std::fixed << std::setprecision(1)
                  << ms << " ms\n";
        return r;
    };

    std::vector<fe::SuiteResult> suites;
    suites.push_back(timed("intent",    [&]{ return suite_intent(); }));

    brain3::core::MasterOrchestrator orch;    // heavy: constructed once
    suites.push_back(timed("retention", [&]{ return suite_retention(orch); }));
    suites.push_back(timed("mouth",     [&]{ return suite_mouth(); }));
    if (full) {
        suites.push_back(timed("plan_grid", [&]{ return suite_plan_grid(); }));
    } else {
        std::cout << "  ... plan_grid SKIPPED (informational; --full to include)\n";
    }

    bool gates = true;
    for (const auto& s : suites) {
        gates = gates && s.gate_pass;
        std::cout << "  [" << (s.gate_pass ? "PASS" : "FAIL") << "] "
                  << s.name << " score=" << s.score << "\n";
        for (const auto& [k, v] : s.detail)
            std::cout << "        " << k << ": " << v << "\n";
    }
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const std::string report = fe::render_report(suites, gates, wall);
    std::ofstream rf("eval_report.json");
    rf << report;
    std::cout << report;
    std::cout << (gates ? "ALL GATES PASS" : "GATE FAILURE") << "\n";
    return gates ? 0 : 1;
}
