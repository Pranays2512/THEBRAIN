/**
 * calculus_diff_test.cpp — symbolic differentiation against a numerical oracle.
 *
 * Two defects this pins down:
 *
 *  1. DIVERGENCE. math_engine.hpp:54 defines a free diff(); calculus_engine.hpp:131
 *     defines CalculusEngine::diff. Both are live, both operate on ExprNode, and
 *     they are NOT equivalent — math_engine's power rule omits the chain-rule
 *     factor entirely, so d/dx((2x)^3) loses the factor 2. MathEngine::solve_derivative
 *     calls the wrong one. Same defect class as perceive_text vs
 *     train_lm_sequence_fused (commit 800b71a): two entry points computing
 *     different things.
 *
 *  2. SILENT ZERO. Both fall through to `return make_num(0)` for any form they
 *     cannot differentiate (x^x, tan, sqrt). A derivative it cannot compute is
 *     reported as the derivative being zero. An honest failure is required —
 *     this is the project's own convention and heldout_probe section B tests it.
 *
 * The oracle is central differences on the ORIGINAL expression. That is
 * independent of whichever symbolic path produced the answer, so it cannot be
 * satisfied by agreeing with a wrong implementation.
 */
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "crisp/engines/math/math_parser.hpp"
#include "crisp/engines/math/math_engine.hpp"
#include "crisp/engines/math/calculus_engine.hpp"

using namespace brain2::math;

static int g_pass = 0, g_fail = 0;

static void ok(bool cond, const std::string& label, const std::string& note = "") {
    if (cond) { ++g_pass; std::printf("  PASS  %-48s %s\n", label.c_str(), note.c_str()); }
    else      { ++g_fail; std::printf("  FAIL  %-48s %s\n", label.c_str(), note.c_str()); }
}

// Central-difference derivative of the ORIGINAL expression — the independent oracle.
static double numeric_oracle(const ExprPtr& e, double x, double h = 1e-5) {
    std::map<std::string, double> hi = {{"x", x + h}};
    std::map<std::string, double> lo = {{"x", x - h}};
    return (ev(e, hi) - ev(e, lo)) / (2.0 * h);
}

static double eval_at(const ExprPtr& e, double x) {
    std::map<std::string, double> env = {{"x", x}};
    return ev(e, env);
}

// A symbolic derivative must agree with central differences at several points.
static void matches_oracle(const std::string& expr_str, const std::string& label) {
    const std::vector<double> pts = {0.7, 1.3, 2.1};
    try {
        auto e = parse(expr_str);
        auto d = CalculusEngine::diff(e, "x");
        if (!d) { ok(false, label, "diff returned null"); return; }
        for (double x : pts) {
            const double sym = eval_at(d, x);
            const double num = numeric_oracle(e, x);
            if (std::fabs(sym - num) > 1e-3 * std::max(1.0, std::fabs(num))) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "at x=%.2f symbolic=%.6f oracle=%.6f  [%s]",
                              x, sym, num, render(d).c_str());
                ok(false, label, buf);
                return;
            }
        }
        ok(true, label, render(d));
    } catch (const std::exception& ex) {
        ok(false, label, std::string("threw: ") + ex.what());
    }
}

// The two entry points must produce the same derivative.
static void entry_points_agree(const std::string& expr_str, const std::string& label) {
    MathEngine engine;
    try {
        auto via_calculus = CalculusEngine::diff(parse(expr_str), "x");
        auto res = engine.solve_derivative(expr_str, "x");
        if (!res.success) { ok(false, label, "solve_derivative failed: " + res.explanation); return; }
        if (!via_calculus) { ok(false, label, "CalculusEngine::diff returned null"); return; }

        // Compare numerically rather than by rendered string — different but
        // equivalent forms are fine; different VALUES are the bug.
        auto reparsed = parse(res.symbolic_val);
        for (double x : {0.7, 1.3, 2.1}) {
            const double a = eval_at(reparsed, x);
            const double b = eval_at(via_calculus, x);
            if (std::fabs(a - b) > 1e-6 * std::max(1.0, std::fabs(b))) {
                char buf[200];
                std::snprintf(buf, sizeof(buf),
                              "at x=%.2f MathEngine=%.6f CalculusEngine=%.6f  [%s vs %s]",
                              x, a, b, res.symbolic_val.c_str(), render(via_calculus).c_str());
                ok(false, label, buf);
                return;
            }
        }
        ok(true, label, res.symbolic_val);
    } catch (const std::exception& ex) {
        ok(false, label, std::string("threw: ") + ex.what());
    }
}

// An input the engine cannot differentiate must NOT come back as a derivative of 0.
static void refuses_silently_zero(const std::string& expr_str, const std::string& label) {
    MathEngine engine;
    try {
        auto res = engine.solve_derivative(expr_str, "x");
        if (!res.success) { ok(true, label, "reported failure: " + res.explanation); return; }
        const std::string v = res.symbolic_val;
        const bool is_zero = (v == "0" || v == "0.0" || v == "(0)");
        ok(!is_zero, label, is_zero ? "silently returned '" + v + "'" : "returned " + v);
    } catch (const std::exception& ex) {
        ok(true, label, std::string("threw: ") + ex.what());
    }
}

int main() {
    std::printf("\n=== calculus: differentiation vs numerical oracle ===\n\n");

    std::printf("1. Symbolic derivative must match central differences\n");
    matches_oracle("x^3",          "power rule, bare variable base");
    matches_oracle("(2*x)^3",      "power rule + CHAIN RULE (composite base)");
    matches_oracle("(3*x + 1)^2",  "chain rule over a sum");
    matches_oracle("sin(2*x)",     "chain rule through sin");
    matches_oracle("exp(2*x)",     "chain rule through exp");
    matches_oracle("x*sin(x)",     "product rule");
    matches_oracle("x^2 / (x+1)",  "quotient rule");

    std::printf("\n2. The two entry points must not diverge\n");
    entry_points_agree("x^3",         "MathEngine == CalculusEngine  (x^3)");
    entry_points_agree("(2*x)^3",     "MathEngine == CalculusEngine  ((2x)^3)");
    entry_points_agree("(3*x + 1)^2", "MathEngine == CalculusEngine  ((3x+1)^2)");
    entry_points_agree("sin(2*x)",    "MathEngine == CalculusEngine  (sin 2x)");

    std::printf("\n3. Undifferentiable input must fail honestly, not return 0\n");
    refuses_silently_zero("x^x", "x^x must not silently return 0");

    std::printf("\n=== %d passed, %d failed ===\n\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
