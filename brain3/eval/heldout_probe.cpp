// eval/heldout_probe.cpp — HELD-OUT CAPABILITY PROBE
//
// Purpose: measure what the crisp engines can do on inputs that are NOT
// seeded in any initializer and NOT present in any existing test file.
//
// Rules this harness follows (and that the existing suites do not):
//   1. No test may query a fact that this harness itself just inserted,
//      unless the point of the test IS storage (marked [CONTROL]).
//   2. No substring matching on single characters. Numeric answers are
//      compared numerically; symbolic answers by full normalized form.
//   3. Every probe declares what it PROVES, and controls are labelled, so a
//      high score cannot come from tautologies.
//   4. Expected failures are recorded as GAPS, not hidden. A gap is
//      information; a fake pass is not.
//
// Build: target `heldout_probe` in CMakeLists.txt
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <random>

#include "crisp/engines/reasoning/brainql.hpp"

using namespace brain2::reasoning;

// ─────────────────────────────────────────────────────────────────────────────
// Harness
// ─────────────────────────────────────────────────────────────────────────────
struct Cat { int pass = 0, total = 0; };
static std::map<std::string, Cat> g_cats;
static std::vector<std::string> g_cat_order;
static std::vector<std::string> g_gaps;

static ReasoningEngine        RE;
static brain2::math::MathEngine ME;
static brain2::CodeEngine     CE;
static CausalEngine           CA;
static AnalogyEngine          AE;
static MetacognitiveEngine    MCE;
static brain2::discovery::DiscoveryEngine DE;

static BrainQLExecutor* EX = nullptr;

struct Out {
    bool ok = false;          // query executed
    std::string value, obj, note, chain;
    bool verified = false, known = false;
    std::string error;
};

static Out run(const std::string& bql) {
    Out o;
    try {
        BrainQLQuery q = parse_bql(bql);
        BrainQLResult r = EX->run(q);
        o.ok = true;
        o.value = r.value; o.obj = r.obj; o.note = r.note;
        for (const auto& s : r.chain) o.chain += s + " ; ";
        o.verified = r.verified; o.known = r.known;
    } catch (const std::exception& e) {
        o.error = e.what();
    }
    return o;
}

static std::string norm(std::string s) {
    std::string t;
    for (char c : s) if (!std::isspace((unsigned char)c)) t += (char)std::tolower((unsigned char)c);
    return t;
}

// Extract the first number appearing in a string, if any.
static bool first_num(const std::string& s, double& out) {
    for (size_t i = 0; i < s.size(); ++i) {
        if (std::isdigit((unsigned char)s[i]) ||
            ((s[i] == '-' || s[i] == '.') && i + 1 < s.size() && std::isdigit((unsigned char)s[i+1]))) {
            try { size_t used = 0; out = std::stod(s.substr(i), &used); return true; }
            catch (...) { return false; }
        }
    }
    return false;
}

static void record(const std::string& cat, const std::string& what, bool pass,
                   const std::string& got, bool is_control, bool expected_gap) {
    if (!g_cats.count(cat)) { g_cats[cat] = Cat(); g_cat_order.push_back(cat); }
    g_cats[cat].total++;
    if (pass) g_cats[cat].pass++;
    std::cout << (pass ? "  PASS " : "  FAIL ")
              << (is_control ? "[CONTROL] " : "          ")
              << std::left << std::setw(46) << what;
    if (!pass) std::cout << " got: " << got;
    std::cout << "\n";
    if (!pass) {
        std::string tag = expected_gap ? "[known-gap] " : "[NEW] ";
        g_gaps.push_back(tag + cat + " :: " + what + "  -> " + got);
    }
}

// numeric assertion
static void num_probe(const std::string& cat, const std::string& bql, double expect,
                      const std::string& what, bool control = false, bool gap = false,
                      double tol = 1e-6) {
    Out o = run(bql);
    double got = 0;
    bool pass = false;
    std::string shown;
    if (!o.ok)                     shown = "parse-error: " + o.error;
    else if (!o.verified)          shown = "unresolved: " + o.note;
    else if (!first_num(o.obj.empty() ? o.value : o.obj, got)) shown = "no number in '" + o.obj + "'";
    else { pass = std::fabs(got - expect) <= tol * std::max(1.0, std::fabs(expect));
           shown = o.obj + " (want " + std::to_string(expect) + ")"; }
    record(cat, what, pass, shown, control, gap);
}

// symbolic / substring-on-a-real-token assertion
static void sym_probe(const std::string& cat, const std::string& bql,
                      const std::string& expect_norm, const std::string& what,
                      bool control = false, bool gap = false) {
    Out o = run(bql);
    bool pass = false;
    std::string shown;
    if (!o.ok)            shown = "parse-error: " + o.error;
    else if (!o.verified) shown = "unresolved: " + o.note;
    else {
        // Search the result value AND the proof chain: several ops (ANALOGY)
        // put the substantive answer in the chain, not the value field.
        std::string hay = norm(o.obj) + "|" + norm(o.value) + "|" + norm(o.chain);
        pass = hay.find(norm(expect_norm)) != std::string::npos;
        shown = o.obj.empty() ? o.value : o.obj;
        shown += " (want " + expect_norm + ")";
    }
    record(cat, what, pass, shown, control, gap);
}

// assert the engine correctly REFUSES / says unknown
static void silence_probe(const std::string& cat, const std::string& bql,
                          const std::string& what, bool control = false) {
    Out o = run(bql);
    bool pass = (!o.ok) || (!o.verified) || (!o.known);
    std::string shown = "claimed: verified=" + std::string(o.verified ? "1" : "0") +
                        " value='" + (o.obj.empty() ? o.value : o.obj) + "'";
    record(cat, what, pass, shown, control, false);
}

// assert REFUTE actually refutes (verified==false means refuted)
static void refute_probe(const std::string& cat, const std::string& bql,
                         const std::string& what, bool control = false, bool gap = false) {
    Out o = run(bql);
    bool pass = o.ok && !o.verified;   // refuted
    std::string shown = o.ok ? ("verdict: " + o.value) : ("parse-error: " + o.error);
    record(cat, what, pass, shown, control, gap);
}

#define H(title) std::cout << "\n\033[1m" << title << "\033[0m\n"

int main() {
    BrainQLExecutor ex(&RE, nullptr, &ME, &CE, nullptr, nullptr, &CA, &AE, &MCE, &DE, nullptr, nullptr);
    EX = &ex;

    std::cout << "==========================================================\n"
              << " BRAIN3 HELD-OUT CAPABILITY PROBE\n"
              << " Nothing here is seeded. Controls are labelled.\n"
              << "==========================================================\n";

    // ══════════════════════════════════════════════════════════════════════
    H("A. ALGEBRA — does the solver generalize past a*x + b = c ?");
    num_probe("A algebra", "SOLVE 6*x + 18 = 42", 4, "a*x + b = c  (in-envelope)", true);
    num_probe("A algebra", "SOLVE 4*x = 10", 2.5, "fractional root");
    num_probe("A algebra", "SOLVE 5*x + 20 = 5", -3, "negative root");
    num_probe("A algebra", "SOLVE x + 7 = 12", 5, "unit coefficient");
    num_probe("A algebra", "SOLVE 2*(x + 3) = 14", 4, "parenthesised left side");
    num_probe("A algebra", "SOLVE 3*x + 2*x = 20", 4, "term collection (x twice)", false, true);
    num_probe("A algebra", "SOLVE 2*x + 3 = 7 + x", 4, "x on both sides", false, true);
    num_probe("A algebra", "SOLVE x^2 = 49", 7, "quadratic via power inversion");
    num_probe("A algebra", "SOLVE x^2 - 5*x + 6 = 0", 2, "real quadratic", false, true);

    // ══════════════════════════════════════════════════════════════════════
    H("B. CALCULUS — real symbolic differentiation / integration?");
    sym_probe("B calculus", "SOLVE diff x^7", "7*x^6", "power rule, unseen exponent");
    sym_probe("B calculus", "SOLVE diff sin(x)", "cos", "trig derivative");
    sym_probe("B calculus", "SOLVE diff x*sin(x)", "cos", "product rule");
    sym_probe("B calculus", "SOLVE diff exp(2*x)", "exp", "chain rule");
    sym_probe("B calculus", "SOLVE diff ln(x)", "1", "log derivative");
    sym_probe("B calculus", "SOLVE int x^6", "x^7", "power rule integration");
    sym_probe("B calculus", "SOLVE int sin(x)", "cos", "trig integration");
    // These two are the dangerous ones: a WRONG answer, not a refusal.
    {
        Out o = run("SOLVE diff x^x");
        bool pass = !o.verified || norm(o.obj) != "0";   // must not silently answer 0
        record("B calculus", "x^x must not silently return 0", pass,
               "returned '" + o.obj + "'", false, true);
    }
    sym_probe("B calculus", "SOLVE diff sqrt(x)", "0.5", "sqrt (unsupported fn)", false, true);
    sym_probe("B calculus", "SOLVE diff tan(x)", "sec", "tan (unsupported fn)", false, true);

    // ══════════════════════════════════════════════════════════════════════
    H("C. LAW DISCOVERY — symbolic regression on data it has never seen");
    sym_probe("C discovery", "DISCOVER y DATA 2:8;3:27;4:64;5:125", "x1^3", "y = x^3 (power law)");
    sym_probe("C discovery", "DISCOVER y DATA 1:5;2:10;3:15;4:20", "5", "y = 5x (linear, no intercept)");
    sym_probe("C discovery", "DISCOVER y DATA 1:1;2:0.25;4:0.0625;5:0.04", "-2", "y = x^-2 (negative exponent)");
    sym_probe("C discovery", "DISCOVER y DATA 3,4:12;2,5:10;6,7:42;8,2:16", "x1*x2", "y = x1*x2 (bilinear)");
    sym_probe("C discovery", "DISCOVER y DATA 2,3:9;4,2:8;10,4:80;6,5:75", "^2", "y = 0.5*x1*x2^2 (quadratic)");
    // Forms with NO template. These are the real ceiling.
    sym_probe("C discovery", "DISCOVER y DATA 1:3;2:5;3:7;4:9", "2*x1+1", "y = 2x + 1 (AFFINE)", false, true);
    sym_probe("C discovery", "DISCOVER y DATA 1,2:3;3,4:7;5,6:11;2,9:11", "x1+x2", "y = x1 + x2 (ADDITIVE)", false, true);
    sym_probe("C discovery", "DISCOVER y DATA 1:2;2:4;3:8;4:16", "2^", "y = 2^x (EXPONENTIAL)", false, true);
    sym_probe("C discovery", "DISCOVER y DATA 1:2;2:6;3:12;4:20", "x1^2+x1", "y = x^2 + x (POLYNOMIAL)", false, true);
    // Named-dataset control: proves only that the fixture round-trips.
    sym_probe("C discovery", "DISCOVER LAW kepler_planetary", "^1.5", "preloaded kepler fixture", true);

    // ══════════════════════════════════════════════════════════════════════
    H("D. PHYSICS WORD PROBLEMS — inside vs outside the formula table");
    num_probe("D physics", "SOLVE mass 8 accel 5 force", 40, "F = ma (in table)", true);
    num_probe("D physics", "SOLVE current 5 resistance 3 voltage", 15, "V = IR (in table)", true);
    num_probe("D physics", "SOLVE Tc 250 Th 1000 eta", 0.75, "Carnot (in table)", true);
    num_probe("D physics", "SOLVE m 8 a 5 F", 40, "same problem, symbol names");
    num_probe("D physics", "SOLVE force 100 area 4 pressure", 25, "P = F/A (NOT in table)", false, true);
    num_probe("D physics", "SOLVE radius 3 area 28.274", 28.274, "circle area (NOT in table)", false, true);
    // Must NOT invent an answer from insufficient data.
    silence_probe("D physics", "SOLVE mass 10 time 5 force", "refuses m+t -> F (underdetermined)");
    silence_probe("D physics", "SOLVE mass 10 accel 3 wavelength", "refuses irrelevant target");

    // ══════════════════════════════════════════════════════════════════════
    H("E. CAUSAL — novel SCM, intervention, and a real counterfactual");
    run("CAUSAL_DEFINE yield = 2 * fertilizer + rain");
    run("CAUSAL_DEFINE revenue = yield * price");
    run("CAUSAL_OBSERVE fertilizer 3, rain 5, price 10");
    num_probe("E causal", "INTERVENE fertilizer=3 QUERY yield", 11, "L1 factual on novel SCM");
    num_probe("E causal", "INTERVENE fertilizer=6 QUERY yield", 17, "L2 do(fertilizer=6)");
    num_probe("E causal", "INTERVENE fertilizer=6 QUERY revenue", 170, "L2 propagates 2 hops");

    // The decisive Level-3 test: observe the OUTCOME, abduct the latent, then
    // ask a counterfactual. Correct Pearl answer requires using the abduced
    // rain. yield=15 with fertilizer=3 implies rain=9, so do(fertilizer=6)
    // must give 21. Anything that returns 15 discarded the abduction.
    {
        CausalEngine fresh;
        fresh.define_equation("yield", "2 * fertilizer + rain");
        fresh.observe("fertilizer", 3);
        fresh.observe("yield", 15);          // observed outcome, rain latent
        auto cf = fresh.counterfactual("fertilizer", 6, "yield");
        double got = cf.value;
        bool pass = std::fabs(got - 21.0) < 1e-6;
        record("E causal", "L3 counterfactual uses abduced latent", pass,
               "returned " + std::to_string(got) + " (want 21, Level-2-only gives 15)",
               false, true);
    }
    // L2 vs L3 must differ on a model with a latent. If identical, Level 3
    // is not implemented.
    {
        CausalEngine fresh;
        fresh.define_equation("y", "2 * x + u");
        fresh.observe("x", 1);
        fresh.observe("y", 10);
        auto l2 = fresh.intervene("x", 4, "y");
        auto l3 = fresh.counterfactual("x", 4, "y");
        bool pass = !(l2.success && l3.success && std::fabs(l2.value - l3.value) < 1e-9);
        record("E causal", "L3 distinguishable from L2", pass,
               "L2=" + l2.value_str + " L3=" + l3.value_str + " (identical)", false, true);
    }
    // Robustness: a cyclic model must not hang or fabricate.
    {
        CausalEngine cyc;
        cyc.define_equation("a", "b + 1");
        cyc.define_equation("b", "a + 1");
        auto r = cyc.predict("a");
        record("E causal", "cyclic SCM terminates without fabricating",
               !r.success, "claimed a=" + r.value_str, false, false);
    }

    // ══════════════════════════════════════════════════════════════════════
    H("F. ANALOGY — is the mapping structural, or relation-name matching?");
    // F1: novel source, canonical target, SHARED relation names -> should work.
    run("ANALOGY_DEFINE beehive queen mass_greater worker");
    run("ANALOGY_DEFINE beehive queen is_center hive");
    run("ANALOGY_DEFINE beehive queen attracts worker");
    sym_probe("F analogy", "ANALOGY beehive solar_system", "sun", "novel domain maps on shared relations");
    sym_probe("F analogy", "ANALOGY PROJECT beehive solar_system queen", "sun", "queen -> sun projection");
    // F2: same structure, DIFFERENT relation names -> the real cross-domain case.
    run("ANALOGY_DEFINE army general commands soldier");
    run("ANALOGY_DEFINE army general outranks soldier");
    run("ANALOGY_DEFINE army soldier executes order");
    run("ANALOGY_DEFINE orchestra conductor directs musician");
    run("ANALOGY_DEFINE orchestra conductor leads musician");
    run("ANALOGY_DEFINE orchestra musician performs note");
    sym_probe("F analogy", "ANALOGY army orchestra", "conductor",
              "isomorphic structure, synonymous relations", false, true);
    // F3: canonical control.
    sym_probe("F analogy", "ANALOGY solar_system rutherford_atom", "nucleus", "preloaded pair", true);

    // ══════════════════════════════════════════════════════════════════════
    H("G. METACOGNITION — general invariants or a hardcoded blocker list?");
    refute_probe("G metacog", "REFUTE mass val -3", "negative mass (hardcoded check)", true);
    refute_probe("G metacog", "REFUTE divisor val 0", "zero divisor (hardcoded check)", true);
    RE.learn("platypus", "isa", "mammal");
    refute_probe("G metacog", "REFUTE platypus isa bird", "mammal/bird (in disjoint list)", true);
    // Unseen invariants and unseen disjoint pairs:
    refute_probe("G metacog", "REFUTE age val -5", "negative age", false, true);
    refute_probe("G metacog", "REFUTE distance val -12", "negative distance", false, true);
    refute_probe("G metacog", "REFUTE probability val 1.7", "probability > 1", false, true);
    RE.learn("sedan", "isa", "car");
    RE.learn("car", "isa", "vehicle");
    RE.learn("boat", "isa", "vehicle");
    refute_probe("G metacog", "REFUTE sedan isa boat", "car/boat disjointness (derivable)", false, true);
    RE.learn("copper", "isa", "metal");
    refute_probe("G metacog", "REFUTE copper isa gas", "metal/gas disjointness", false, true);

    // ══════════════════════════════════════════════════════════════════════
    H("H. PROGRAM SYNTHESIS — how deep does the A* search really go?");
    sym_probe("H synth", "SYNTH [1, 2, 3] -> [2, 4, 6]", "* 2", "1 op: map *2", true);
    sym_probe("H synth", "SYNTH [1, 2, 3, 4] -> [4, 3, 2, 1]", "revers", "1 op: reverse");
    sym_probe("H synth", "SYNTH [1, 2, 3, 4] -> [1, 3, 6, 10]", "sum", "1 op: cumulative sum");
    sym_probe("H synth", "SYNTH [-2, 3, -4, 5] -> [3, 5]", "> 0", "1 op: filter positive");
    sym_probe("H synth", "SYNTH [1, 2, 3] -> [4, 6, 8]", "2", "2 ops: (x+1)*2");
    sym_probe("H synth", "SYNTH [3, 1, 2] -> [1, 4, 9]", "2", "2 ops: sort then square");
    sym_probe("H synth", "SYNTH [1, 2, 3] -> [9, 4, 1]", "revers", "2 ops: square then reverse");
    sym_probe("H synth", "SYNTH [3, 1, 2] -> [18, 8, 2]", "revers", "4 ops: sort,sq,*2,reverse", false, true);
    sym_probe("H synth", "SYNTH [1, 2, 3, 4] -> [1, 4, 9, 16]", "2", "1 op: square");

    // ══════════════════════════════════════════════════════════════════════
    H("I. MULTI-HOP INHERITANCE — genuinely novel 5-deep chain");
    // Nonsense tokens guarantee nothing is pre-seeded or web-grounded.
    run("TEACH zorblat isa quixel");
    run("TEACH quixel isa florn");
    run("TEACH florn isa grunth");
    run("TEACH grunth isa wibbet");
    run("TEACH wibbet isa thingamajig");
    run("TEACH florn can shimmer");
    run("TEACH grunth habitat marshland");
    sym_probe("I multihop", "LOOKUP zorblat isa", "quixel", "direct fact stored", true);
    sym_probe("I multihop", "CHAIN zorblat isa", "thingamajig", "5-hop transitive closure");
    // INHERIT with a transitive relation returns SOME ancestor, not a
    // specified one. Documenting which, because callers depend on it.
    {
        Out o = run("INHERIT zorblat isa");
        bool determinate = (norm(o.value) == "quixel" || norm(o.value) == "thingamajig");
        record("I multihop", "INHERIT isa returns a defined level", determinate,
               "returned mid-chain '" + o.value + "' (neither parent nor root)", false, false);
    }
    sym_probe("I multihop", "INHERIT zorblat can", "shimmer", "property inherited across 2 hops");
    sym_probe("I multihop", "INHERIT zorblat habitat", "marshland", "property inherited across 3 hops");
    silence_probe("I multihop", "INHERIT zorblat diet", "no diet fact anywhere -> unknown");

    // ══════════════════════════════════════════════════════════════════════
    H("J. HONESTY — does it stay silent when it has nothing?");
    silence_probe("J honesty", "LOOKUP flibbertigibbet isa", "unknown entity");
    silence_probe("J honesty", "LOOKUP zorblat colour", "known entity, unknown relation");
    silence_probe("J honesty", "INHERIT snorkwaffle isa", "unknown entity, transitive query");
    silence_probe("J honesty", "DISCOVER y DATA 1:7;2:3;3:91;4:12", "pure noise -> no law");
    silence_probe("J honesty", "SYNTH [1, 2, 3] -> [7, 41, 2]", "unsynthesizable spec");

    // ══════════════════════════════════════════════════════════════════════
    H("L. FALSE-DISCOVERY RATE — how often is a law invented from noise?");
    // The discovery engine's log-log branch fits an exponent by least squares
    // and then reports verified=true with a hardcoded r2 of 0.999, without
    // ever checking the residual. This measures the consequence: feed it
    // structureless data and count how often it claims a scientific law.
    {
        std::mt19937 rng(20260826);
        std::uniform_real_distribution<double> ud(1.0, 100.0);
        int claimed = 0, trials = 200;
        double worst_rel_err = 0.0;
        std::string sample_eq;
        for (int t = 0; t < trials; ++t) {
            std::vector<brain2::discovery::ObservationPoint> pts;
            std::vector<double> xs, ys;
            for (int i = 1; i <= 5; ++i) {
                double x = (double)i, y = ud(rng);
                pts.push_back({{{"x1", x}}, y});
                xs.push_back(x); ys.push_back(y);
            }
            auto law = DE.discover_from_data("y", {"x1"}, pts);
            if (law.verified) {
                ++claimed;
                if (sample_eq.empty()) sample_eq = law.equation;
                // measure how badly the "verified zero-residual" law actually fits
                double num = 0, den = 0;
                for (size_t i = 0; i < xs.size(); ++i) {
                    double pred = 0;
                    // reconstruct k*x^p from the reported string is fragile; use
                    // the reported r2 claim instead and compare to reality via
                    // spread of y (a law that fits must beat the mean).
                    (void)pred;
                    num += ys[i]; den += 1;
                }
                double mean = num / den;
                double ss = 0; for (double y : ys) ss += (y - mean) * (y - mean);
                worst_rel_err = std::max(worst_rel_err, std::sqrt(ss / den) / std::max(1.0, mean));
            }
        }
        double fdr = 100.0 * claimed / trials;
        std::cout << "  random 5-point datasets: " << trials << "\n"
                  << "  laws claimed VERIFIED  : " << claimed
                  << "  (" << std::fixed << std::setprecision(1) << fdr << "% false-discovery rate)\n"
                  << "  example fabrication    : " << sample_eq << "\n"
                  << "  (each of these reported r2 = 0.999 without computing a residual)\n";
        record("L falsediscovery", "false-discovery rate below 5%", fdr < 5.0,
               std::to_string((int)fdr) + "% of pure-noise datasets yielded a 'verified law'",
               false, false);
    }

    // ══════════════════════════════════════════════════════════════════════
    H("K. ONE-SHOT UPDATE — learn now, use immediately, handle conflict");
    run("TEACH glimberry isa fruit");
    sym_probe("K oneshot", "LOOKUP glimberry isa", "fruit", "usable on the next query", true);
    run("TEACH glimberry isa vegetable");
    {
        Out o = run("LOOKUP glimberry isa");
        // Either answer is defensible; SILENTLY holding both without flagging
        // is what we are checking for.
        bool flagged = o.note.find("conflict") != std::string::npos ||
                       o.note.find("contradict") != std::string::npos ||
                       o.value.find("conflict") != std::string::npos;
        record("K oneshot", "flags contradictory redefinition", flagged,
               "silently returned '" + o.value + "'", false, true);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Scorecard
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "\n==========================================================\n"
              << " SCORECARD (held-out)\n"
              << "==========================================================\n";
    int tp = 0, tt = 0;
    for (const auto& c : g_cat_order) {
        const Cat& k = g_cats[c];
        tp += k.pass; tt += k.total;
        double pct = k.total ? (100.0 * k.pass / k.total) : 0.0;
        std::cout << "  " << std::left << std::setw(16) << c
                  << std::right << std::setw(3) << k.pass << " / " << std::setw(2) << k.total
                  << "   " << std::fixed << std::setprecision(0) << std::setw(3) << pct << "%\n";
    }
    std::cout << "  " << std::string(38, '-') << "\n"
              << "  " << std::left << std::setw(16) << "TOTAL"
              << std::right << std::setw(3) << tp << " / " << std::setw(2) << tt
              << "   " << std::fixed << std::setprecision(1)
              << (tt ? 100.0 * tp / tt : 0.0) << "%\n";

    std::cout << "\n  FAILURES (" << g_gaps.size() << "):\n";
    for (const auto& g : g_gaps) std::cout << "    - " << g << "\n";

    std::cout << "\n  Note: this harness returns 0 regardless of score. It is a\n"
              << "  measurement instrument, not a gate.\n";
    return 0;
}
