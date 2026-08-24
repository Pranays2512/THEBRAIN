#include "crisp/engines/synthesis/conjecture_sandbox.hpp"
#include "crisp/engines/synthesis/universal_sandbox.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>

int main() {
    std::cout << "========================================================\n";
    std::cout << "THE BRAIN 3 — UNIVERSAL TESTING SANDBOX TEST SUITE\n";
    std::cout << "========================================================\n\n";

    brain3::engines::synthesis::ConjectureSandbox sandbox;
    const auto& engine = sandbox.universal();

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Classical Mechanics: Kinetic Energy Verification
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "[Test 1/7] Testing Classical Kinematics & Kinetic Energy Invariant...\n";
    auto correct_ke = [](double m, double v) { return 0.5 * m * v * v; };
    auto bad_ke = [](double m, double v) { return m * v; }; // Missing 0.5 and squared power

    auto res_ke_good = engine.test_gravitational_drop(correct_ke);
    assert(res_ke_good.passed);
    std::cout << "  ✓ Correct KE (0.5*m*v^2) passed with max error: " 
              << (res_ke_good.max_relative_error * 100.0) << "%\n";

    auto res_ke_bad = engine.test_gravitational_drop(bad_ke);
    assert(!res_ke_bad.passed);
    std::cout << "  ✓ Erroneous KE (m*v) correctly refuted by sandbox. Counterexample: " 
              << res_ke_bad.counterexample << "\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Orbital Mechanics: Kepler's Third Law Verification
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "[Test 2/7] Testing Orbital Mechanics & Kepler's 3rd Law...\n";
    const double PI = 3.14159265358979323846;
    auto kepler_fn = [PI](double a, double M) {
        return 2.0 * PI * std::sqrt((a * a * a) / (brain3::engines::synthesis::UniversalSandboxEngine::G_NEWTON * M));
    };
    auto res_kepler = sandbox.test_physics("kepler", kepler_fn);
    assert(res_kepler.passed);
    std::cout << "  ✓ Kepler's Third Law (T^2 ~ a^3) verified across solar system scale. Max error: " 
              << (res_kepler.max_relative_error * 100.0) << "%\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Special Relativity: Energy-Momentum Invariant
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "[Test 3/7] Testing Special Relativity E^2 = (pc)^2 + (mc^2)^2...\n";
    const double c = brain3::engines::synthesis::UniversalSandboxEngine::C_LIGHT;
    auto rel_e_fn = [c](double m0, double p) {
        double mc2 = m0 * c * c;
        double pc = p * c;
        return std::sqrt(pc * pc + mc2 * mc2);
    };
    auto res_rel = sandbox.test_physics("relativity", rel_e_fn);
    assert(res_rel.passed);
    std::cout << "  ✓ Minkowski Invariant Mass & Relativistic Energy verified. Max error: " 
              << (res_rel.max_relative_error * 100.0) << "%\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 4. Dynamical Systems: RK4 Numerical Integration of Harmonic Oscillator
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "[Test 4/7] Testing Dynamical Systems RK4 Integrator on Harmonic Oscillator...\n";
    // State: [position x, momentum p]
    // d/dt [x, p] = [p/m, -k*x] (with m=1, k=1 -> omega=1)
    auto harmonic_deriv = [](const std::vector<double>& s, double /*t*/) -> std::vector<double> {
        return {s[1], -s[0]};
    };
    // Hamiltonian H = 0.5*k*x^2 + 0.5*p^2/m
    auto harmonic_energy = [](const std::vector<double>& s) -> double {
        return 0.5 * (s[0] * s[0] + s[1] * s[1]);
    };
    
    // Integrate for 100 seconds (approx 16 full orbits) with dt = 0.05
    std::vector<double> x0 = {1.0, 0.0};
    auto ode_res = sandbox.test_ode_invariant(x0, 0.0, 100.0, 0.05, harmonic_deriv, harmonic_energy, 0.001);
    assert(ode_res.conserved);
    std::cout << "  ✓ RK4 Hamiltonian Conservation: " << ode_res.details << "\n"
              << "    Integration latency: " << ode_res.elapsed_time_us << " us for " 
              << ode_res.step_count << " steps\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 5. Algorithmic Property Fuzzing: Merge Sort & Binary Search
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "[Test 5/7] Testing Algorithmic Logic Plan Invariants & Property Fuzzing...\n";
    auto std_sort_wrapper = [](std::vector<int> arr) {
        std::sort(arr.begin(), arr.end());
        return arr;
    };
    auto sort_res = sandbox.test_sort(std_sort_wrapper);
    assert(sort_res.passed);
    std::cout << "  ✓ Sorting Algorithm passed all property fuzzing invariants (Monotonicity, Multiset, Idempotency)\n";

    auto binary_search_fn = [](const std::vector<int>& arr, int target) -> int {
        int lo = 0, hi = static_cast<int>(arr.size()) - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    };
    auto search_res = sandbox.test_search(binary_search_fn);
    assert(search_res.passed);
    std::cout << "  ✓ Binary Search passed all presence/absence and boundary test assertions\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 6. Web / DOM / WCAG Accessibility Contract Audit
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "[Test 6/7] Testing Web Frontend Selector & WCAG Contrast Contract...\n";
    std::string html_sample = R"(
        <!DOCTYPE html>
        <html>
        <head><title>Test Store</title></head>
        <body>
            <header class="navbar header-container">
                <h1 class="logo">CyberStore</h1>
                <nav class="nav-links"></nav>
            </header>
            <main class="product-grid">
                <div class="product-card">
                    <button class="add-to-cart-btn">Buy</button>
                </div>
            </main>
        </body>
        </html>
    )";

    std::string valid_css = R"(
        .navbar { background: #0f172a; padding: 1rem; }
        .header-container { max-width: 1200px; margin: 0 auto; }
        .logo { font-size: 1.5rem; color: #38bdf8; }
        .product-grid { display: grid; gap: 1.5rem; }
        .product-card { border: 1px solid #334155; }
        .add-to-cart-btn { background: #6366f1; color: #ffffff; }
        .add-to-cart-btn:hover { background: #4f46e5; }
    )";

    std::vector<std::pair<std::string, std::string>> colors = {
        {"#ffffff", "#0f172a"}, // White text on dark navy background -> ratio ~ 16:1
        {"#38bdf8", "#0f172a"}, // Light cyan on dark navy -> ratio ~ 9:1
        {"#ffffff", "#4f46e5"}  // White text on deep indigo button -> ratio ~ 6.5:1
    };

    auto web_res = sandbox.audit_web(html_sample, valid_css, colors);
    assert(web_res.passed);
    assert(web_res.wcag_aa_compliant);
    std::cout << "  ✓ Valid Web Contract passed: " << web_res.matched_selectors.size() 
              << " CSS selectors matched DOM, 0 syntax errors, 100% WCAG AA compliant\n";

    // Test malformed CSS (e.g. `hover:` Tailwind hallucination)
    std::string broken_css = ".add-to-cart-btn { hover: background #4f46e5; } .ghost-class { color: red; }";
    auto web_res_broken = sandbox.audit_web(html_sample, broken_css, colors);
    assert(!web_res_broken.passed);
    assert(!web_res_broken.syntax_violations.empty());
    assert(!web_res_broken.unmatched_css_selectors.empty());
    std::cout << "  ✓ Broken CSS properly caught (detected malformed 'hover:' & ghost selector '.ghost-class')\n\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 7. Competitive Multi-Hypothesis Tournament
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "[Test 7/7] Testing Competitive Multi-Hypothesis Tournament...\n";
    // Oracle: Gravitational force F = G * m1 * m2 / r^2 (simplified as x*y for testing tournament)
    auto oracle = [](double x, double y) { return x * y * y; }; // target: x * y^2
    
    std::vector<std::pair<std::string, std::function<double(double, double)>>> candidates = {
        {"Linear Hypothesis (x * y)", [](double x, double y) { return x * y; }},
        {"Additive Hypothesis (x + y^2)", [](double x, double y) { return x + y * y; }},
        {"True Hypothesis (x * y^2)", [](double x, double y) { return x * y * y; }},
        {"Cubic Hypothesis (x^2 * y)", [](double x, double y) { return x * x * y; }}
    };

    auto tourney_res = sandbox.run_tournament(candidates, oracle);
    assert(tourney_res.decisive_winner);
    assert(tourney_res.winning_hypothesis == "True Hypothesis (x * y^2)");
    std::cout << "  ✓ Tournament completed in " << tourney_res.total_scenarios_tested 
              << " discriminating scenarios. Victor: '" << tourney_res.winning_hypothesis << "'\n"
              << "    Eliminated " << tourney_res.eliminated_hypotheses.size() << " incorrect candidates.\n\n";

    std::cout << "========================================================\n";
    std::cout << "ALL 7 UNIVERSAL SANDBOX DOMAINS VERIFIED SUCCESSFULLY!\n";
    std::cout << "========================================================\n";
    return 0;
}
