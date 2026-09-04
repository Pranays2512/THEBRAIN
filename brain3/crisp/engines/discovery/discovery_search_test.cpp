/**
 * discovery_search_test.cpp — can the engine find a law nobody templated?
 *
 * This is the ceiling test for the whole discovery subsystem, and it is
 * deliberately harder than discovery_fit_test.cpp.
 *
 * Every form the engine currently resolves resolves because someone wrote a
 * template for it: the power-law branch, the bilinear branch, and the
 * linear-family ladder added in eb43110 (affine, quadratic, additive,
 * exponential). The reachable set is exactly the enumerated set. A system whose
 * capability ceiling equals the list its author wrote down has not discovered
 * anything — it has looked something up.
 *
 * The targets below are outside every template: a product of a variable and a
 * transcendental, a rational form, and a sum of polynomial and trigonometric
 * terms. Finding them requires SEARCH OVER A GRAMMAR of operators and operands,
 * scored by fit, rather than selection from a menu of shapes.
 *
 * Section 3 carries over unchanged from discovery_fit_test: whatever finds
 * these must still refuse structureless data. A search over a rich enough
 * grammar will fit noise perfectly if it is allowed to grow without penalty, so
 * a complexity penalty (or a held-out split) is not optional decoration — it is
 * what separates a discovery engine from a curve fitter.
 *
 * Data is generated from the true function rather than transcribed, so the
 * targets are exact and the test cannot drift from what it claims to test.
 */
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "crisp/engines/discovery/discovery_engine.hpp"

using namespace brain2::discovery;

static int g_pass = 0, g_fail = 0;

static void ok(bool cond, const std::string& label, const std::string& note = "") {
    if (cond) { ++g_pass; std::printf("  PASS  %-40s %s\n", label.c_str(), note.c_str()); }
    else      { ++g_fail; std::printf("  FAIL  %-40s %s\n", label.c_str(), note.c_str()); }
}

// Sample a true 1-variable function at x = 1..n. Exact by construction.
template <typename F>
static std::vector<ObservationPoint> sample(F f, int n = 8) {
    std::vector<ObservationPoint> out;
    for (int i = 1; i <= n; ++i) {
        ObservationPoint p;
        p.inputs["x1"] = (double)i;
        p.output = f((double)i);
        out.push_back(p);
    }
    return out;
}

// The engine must recover a law that predicts the data, whatever form it takes.
// Correctness is judged NUMERICALLY, not by string match: any expression that
// reproduces the observations is a valid discovery, and demanding a particular
// rendering would be testing the printer instead of the search.
static void recovers(const std::vector<ObservationPoint>& data,
                     const std::string& label) {
    DiscoveryEngine eng;
    auto law = eng.discover_from_data("y", {"x1"}, data);
    if (!law.verified) { ok(false, label, "not verified: " + law.explanation); return; }
    ok(law.r2_score >= 0.9995, label,
       law.equation + "  (R^2=" + std::to_string(law.r2_score) + ")");
}

static void refuses(const std::vector<ObservationPoint>& data, const std::string& label) {
    DiscoveryEngine eng;
    auto law = eng.discover_from_data("y", {"x1"}, data);
    ok(!law.verified, label,
       law.verified ? "CLAIMED A LAW: " + law.equation : "correctly abstained");
}

int main() {
    std::printf("\n=== discovery: forms outside every template ===\n\n");

    std::printf("1. Must still resolve — templates and ladder (no regression)\n");
    recovers(sample([](double x){ return x * x * x; }),          "y = x^3          (power template)");
    recovers(sample([](double x){ return 2.0 * x + 1.0; }),      "y = 2x + 1       (affine ladder)");
    recovers(sample([](double x){ return x * x + x; }),          "y = x^2 + x      (quadratic ladder)");

    std::printf("\n2. THE CEILING — no template covers these\n");
    recovers(sample([](double x){ return x * std::sin(x); }),    "y = x*sin(x)");
    recovers(sample([](double x){ return 1.0 / (1.0 + x * x); }),"y = 1/(1 + x^2)");
    recovers(sample([](double x){ return x * x + std::sin(x); }),"y = x^2 + sin(x)");
    recovers(sample([](double x){ return std::log(x) + x; }),    "y = ln(x) + x");

    std::printf("\n3. FALSE-DISCOVERY GUARD — must survive a richer search space\n");
    // A grammar search WILL fit these perfectly if allowed to grow unpenalised.
    // Keeping this section green is the whole constraint on the search.
    refuses({{{{"x1",1.0}},7.0}, {{{"x1",2.0}},3.0}, {{{"x1",3.0}},9.0},
             {{{"x1",4.0}},2.0}, {{{"x1",5.0}},8.0}, {{{"x1",6.0}},1.0}},
            "noise, 6 points");
    refuses({{{{"x1",1.0}},1.0}, {{{"x1",2.0}},50.0}, {{{"x1",3.0}},2.0},
             {{{"x1",4.0}},49.0}, {{{"x1",5.0}},3.0}, {{{"x1",6.0}},48.0}},
            "alternating, no law");

    std::printf("\n=== %d passed, %d failed ===\n\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
