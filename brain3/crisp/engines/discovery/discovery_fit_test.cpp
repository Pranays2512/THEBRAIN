/**
 * discovery_fit_test.cpp — linear-family law induction, and its false-discovery rate.
 *
 * The discovery engine had templates for multiplicative and power forms only, so
 * y = 2x+1, y = x1+x2, y = 2^x and y = x^2+x were all "could not converge"
 * (heldout_probe section C). Adding fitters for them is easy. Adding them
 * WITHOUT reintroducing the defect commit 838880e removed is the actual job.
 *
 * That defect: r2_score/mse were assigned as literal constants, so any dataset
 * was reported as a verified scientific law — measured false-discovery rate on
 * structureless data, 200/200. The fix gated the power-law branch on real
 * residuals. A careless linear fitter walks straight back into it, because an
 * affine model has two free parameters and reproduces ANY two points exactly;
 * a quadratic reproduces any three. Enough parameters and noise fits perfectly.
 *
 * Two defences, both tested here:
 *   1. Every model gates on the existing fit_acceptable (R^2 >= 0.9995,
 *      max relative residual <= 0.02) computed from real residuals.
 *   2. A model is only attempted when n >= params + 2, so there are always at
 *      least two degrees of freedom left for the fit to fail on.
 *
 * Section 3 is the one that matters. If it ever goes red, the engine is
 * inventing laws again.
 */
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "crisp/engines/discovery/discovery_engine.hpp"

using namespace brain2::discovery;

static int g_pass = 0, g_fail = 0;

static void ok(bool cond, const std::string& label, const std::string& note = "") {
    if (cond) { ++g_pass; std::printf("  PASS  %-44s %s\n", label.c_str(), note.c_str()); }
    else      { ++g_fail; std::printf("  FAIL  %-44s %s\n", label.c_str(), note.c_str()); }
}

static ObservationPoint pt1(double x, double y) {
    ObservationPoint p; p.inputs["x1"] = x; p.output = y; return p;
}
static ObservationPoint pt2(double a, double b, double y) {
    ObservationPoint p; p.inputs["x1"] = a; p.inputs["x2"] = b; p.output = y; return p;
}

// The engine must find a law whose rendered equation contains `needle`.
static void discovers(const std::vector<ObservationPoint>& data,
                      const std::vector<std::string>& vars,
                      const std::string& needle, const std::string& label) {
    DiscoveryEngine eng;
    auto law = eng.discover_from_data("y", vars, data);
    if (!law.verified) { ok(false, label, "not verified: " + law.explanation); return; }
    const bool hit = law.equation.find(needle) != std::string::npos;
    ok(hit, label, hit ? law.equation
                       : "got '" + law.equation + "', wanted substring '" + needle + "'");
}

// The engine must REFUSE. This is the important direction.
static void refuses(const std::vector<ObservationPoint>& data,
                    const std::vector<std::string>& vars, const std::string& label) {
    DiscoveryEngine eng;
    auto law = eng.discover_from_data("y", vars, data);
    ok(!law.verified, label,
       law.verified ? "CLAIMED A LAW: " + law.equation : "correctly abstained");
}

int main() {
    std::printf("\n=== discovery: linear-family induction ===\n\n");

    std::printf("1. The four forms with no template (heldout section C)\n");
    discovers({pt1(1,3), pt1(2,5), pt1(3,7), pt1(4,9)},        {"x1"}, "x1", "affine  y = 2x + 1");
    discovers({pt2(1,2,3), pt2(3,4,7), pt2(5,6,11), pt2(2,9,11)}, {"x1","x2"}, "x1", "additive  y = x1 + x2");
    discovers({pt1(1,2), pt1(2,4), pt1(3,8), pt1(4,16)},       {"x1"}, "2", "exponential  y = 2^x");
    discovers({pt1(1,2), pt1(2,6), pt1(3,12), pt1(4,20)},      {"x1"}, "x1", "polynomial  y = x^2 + x");

    std::printf("\n2. Existing forms must not regress\n");
    discovers({pt1(2,8), pt1(3,27), pt1(4,64), pt1(5,125)},    {"x1"}, "3", "power  y = x^3");
    discovers({pt1(1,5), pt1(2,10), pt1(3,15), pt1(4,20)},     {"x1"}, "5", "linear  y = 5x");
    discovers({pt2(3,4,12), pt2(2,5,10), pt2(6,7,42), pt2(8,2,16)}, {"x1","x2"}, "x1", "bilinear  y = x1*x2");

    std::printf("\n3. FALSE-DISCOVERY GUARD — structureless data must be refused\n");
    refuses({pt1(1,7), pt1(2,3), pt1(3,9), pt1(4,2), pt1(5,8)},   {"x1"}, "noise, 5 points, 1 var");
    refuses({pt1(1,4), pt1(2,4.7), pt1(3,1.2), pt1(4,9.9), pt1(5,3.1)}, {"x1"}, "noise, non-integer");
    refuses({pt2(1,2,7), pt2(3,4,2), pt2(5,6,9), pt2(2,9,1), pt2(4,4,6)}, {"x1","x2"}, "noise, 5 points, 2 vars");
    refuses({pt1(1,1), pt1(2,50), pt1(3,2), pt1(4,49), pt1(5,3)}, {"x1"}, "alternating, no law");

    std::printf("\n4. UNDERDETERMINED — too few points to earn a fit\n");
    // An affine model has 2 free parameters and reproduces ANY 2 points exactly.
    // Fitting one here would be arithmetic, not discovery.
    refuses({pt1(1,3), pt1(2,5)},           {"x1"}, "2 points cannot establish an affine law");
    refuses({pt1(1,3), pt1(2,5), pt1(3,99)}, {"x1"}, "3 points, and the third disagrees");

    std::printf("\n=== %d passed, %d failed ===\n\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
