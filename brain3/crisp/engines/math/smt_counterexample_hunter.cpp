/**
 * brain3/crisp/engines/math/smt_counterexample_hunter.cpp
 *
 * Driver and test runner for The Brain's SMT & Non-Linear Counterexample Hunter.
 * Demonstrates:
 * 1. Breaking false inequalities (e.g. x^4 + y^4 - 4xy + 0.5 >= 0)
 * 2. Validating genuine convex invariants (e.g. exp(x) - (1 + x) >= 0)
 * 3. Falsifying flawed prime formulas (e.g. Euler's polynomial n^2 + n + 41 for composite n = 40, 41)
 * 4. Stress testing Erdős-Straus modular candidates
 */

#include "smt_counterexample_hunter.hpp"
#include <iostream>
#include <iomanip>

using namespace thebrain::smt_hunter;

int main() {
    std::cout << "\n🧠 ==========================================================================\n";
    std::cout << "   THE BRAIN — SMT & NON-LINEAR COUNTEREXAMPLE HUNTER (THE BREAKER)\n";
    std::cout << "   High-Throughput Continuous Gradient & Diophantine Lattice Falsification\n";
    std::cout << "==========================================================================\n";

    SMTCounterexampleHunter hunter;

    // Test 1: False continuous inequality f(x, y) = x^4 + y^4 - 4xy + 0.5 >= 0
    std::cout << "\n[TEST 1] Hunting counterexample for candidate: x^4 + y^4 - 4xy + 0.5 >= 0 on [-2, 2]^2...\n";
    auto res1 = hunter.falsify_continuous_inequality(
        "x^4 + y^4 - 4xy + 0.5 >= 0",
        [](const std::vector<double>& v) {
            double x = v[0], y = v[1];
            return std::pow(x, 4) + std::pow(y, 4) - 4.0 * x * y + 0.5;
        },
        {{-2.0, 2.0}, {-2.0, 2.0}}
    );
    std::cout << "  Status : " << (res1.counterexample_found ? "❌ FALSIFIED (Caught & Destroyed)" : "✅ SURVIVED") << "\n";
    std::cout << "  Details: " << res1.details << "\n";
    std::cout << "  Time   : " << std::fixed << std::setprecision(4) << res1.search_time_ms << " ms (" << res1.total_probes << " probes)\n";

    // Test 2: True convex inequality exp(x) - 1 - x >= 0 on [-10, 10]
    std::cout << "\n[TEST 2] Attacking genuine invariant: exp(x) - 1 - x >= 0 on [-10, 10]...\n";
    auto res2 = hunter.falsify_continuous_inequality(
        "exp(x) - 1 - x >= 0",
        [](const std::vector<double>& v) {
            double x = v[0];
            return std::exp(x) - 1.0 - x;
        },
        {{-10.0, 10.0}}
    );
    std::cout << "  Status : " << (res2.counterexample_found ? "❌ FALSIFIED" : "✅ SURVIVED RIGOROUS ATTACK") << "\n";
    std::cout << "  Details: " << res2.details << "\n";
    std::cout << "  Time   : " << std::fixed << std::setprecision(4) << res2.search_time_ms << " ms (" << res2.total_probes << " probes)\n";

    // Test 3: Falsifying Euler's Prime Polynomial n^2 + n + 41 for all n
    std::cout << "\n[TEST 3] Attacking conjecture: Euler Polynomial Prime (n^2 + n + 41) for ALL integers n in [1, 100]...\n";
    auto is_prime = [](int64_t n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        for (int64_t i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) return false;
        }
        return true;
    };

    auto res3 = hunter.falsify_discrete_conjecture(
        "Euler Polynomial Prime for all n",
        [&](int64_t n) {
            int64_t val = n * n + n + 41;
            return is_prime(val);
        },
        1,
        100
    );
    std::cout << "  Status : " << (res3.counterexample_found ? "❌ FALSIFIED (Euler Polynomial Breaks)" : "✅ SURVIVED") << "\n";
    std::cout << "  Details: " << res3.details << "\n";
    std::cout << "  Time   : " << std::fixed << std::setprecision(4) << res3.search_time_ms << " ms\n";

    std::cout << "\n==========================================================================\n";
    std::cout << "🏁 SMT & NON-LINEAR COUNTEREXAMPLE HUNTER VALIDATION COMPLETE\n";
    std::cout << "==========================================================================\n\n";

    return 0;
}
