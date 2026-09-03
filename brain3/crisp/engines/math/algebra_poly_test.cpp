/**
 * algebra_poly_test.cpp — term collection & polynomial solving.
 *
 * The three held-out probe gaps (heldout_probe.cpp:179,180,182) plus the edge
 * cases the probe does NOT cover. The property that matters most here is the
 * negative one: the engine must never return a confident wrong number. An
 * honest throw is a pass; a plausible-looking wrong root is the failure mode
 * this whole project exists to avoid.
 */
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "crisp/engines/math/math_parser.hpp"
#include "crisp/engines/math/algebra_engine.hpp"

using namespace brain2::math;

static int g_pass = 0, g_fail = 0;

static void ok(bool cond, const std::string& label, const std::string& note = "") {
    if (cond) { ++g_pass; std::printf("  PASS  %-46s %s\n", label.c_str(), note.c_str()); }
    else      { ++g_fail; std::printf("  FAIL  %-46s %s\n", label.c_str(), note.c_str()); }
}

// Equation solves to `want`.
static void solves(const std::string& eq, double want, const std::string& label) {
    AlgebraEngine engine;
    try {
        auto [value, steps] = engine.solve(parse(eq), "x");
        ok(std::fabs(value - want) < 1e-6, label,
           "got " + std::to_string(value) + " want " + std::to_string(want));
    } catch (const std::exception& e) {
        ok(false, label, std::string("threw: ") + e.what());
    }
}

// Equation must NOT produce a number. Throwing is the correct behaviour.
static void refuses(const std::string& eq, const std::string& label) {
    AlgebraEngine engine;
    try {
        auto [value, steps] = engine.solve(parse(eq), "x");
        ok(false, label, "returned " + std::to_string(value) + " instead of throwing");
    } catch (const std::exception& e) {
        ok(true, label, std::string("threw: ") + e.what());
    }
}

// A solved equation must report every real root it found in its step trace.
static void steps_mention(const std::string& eq, const std::string& needle,
                          const std::string& label) {
    AlgebraEngine engine;
    try {
        auto [value, steps] = engine.solve(parse(eq), "x");
        bool found = false;
        for (const auto& s : steps) if (s.find(needle) != std::string::npos) found = true;
        ok(found, label, found ? "" : "no step contains '" + needle + "'");
    } catch (const std::exception& e) {
        ok(false, label, std::string("threw: ") + e.what());
    }
}

int main() {
    std::printf("\n=== algebra: term collection & polynomial solving ===\n\n");

    std::printf("1. The three held-out gaps\n");
    solves("3*x + 2*x = 20",      4.0, "term collection, same side");
    solves("2*x + 3 = 7 + x",     4.0, "x on both sides");
    solves("x^2 - 5*x + 6 = 0",   2.0, "real quadratic (smallest root)");
    steps_mention("x^2 - 5*x + 6 = 0", "3", "quadratic reports BOTH roots in steps");

    std::printf("\n2. Must not regress (these pass today via isolate)\n");
    solves("6*x + 18 = 42",       4.0, "a*x + b = c");
    solves("4*x = 10",            2.5, "fractional root");
    solves("5*x + 20 = 5",       -3.0, "negative root");
    solves("x + 7 = 12",          5.0, "unit coefficient");
    solves("2*(x + 3) = 14",      4.0, "parenthesised left side");
    solves("x^2 = 49",            7.0, "quadratic via power inversion");

    std::printf("\n3. Non-polynomial — must fall back to isolate(), not misreport\n");
    solves("12 / x = 4",          3.0, "variable in denominator");

    std::printf("\n4. Degenerate — an honest throw is the correct answer\n");
    refuses("x^2 + 1 = 0",             "no real roots (negative discriminant)");
    refuses("x - x = 5",               "degree 0 contradiction");
    refuses("x^3 - 6*x^2 + 11*x - 6 = 0", "degree 3 unsupported, must not guess");
    refuses("2*y + 1 = 7",             "target variable absent");

    std::printf("\n=== %d passed, %d failed ===\n\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
