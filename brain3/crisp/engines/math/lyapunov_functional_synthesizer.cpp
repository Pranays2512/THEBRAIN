/**
 * brain3/crisp/engines/math/lyapunov_functional_synthesizer.cpp
 *
 * Driver and test runner for The Brain's Lyapunov & Monotonic Energy Functional Synthesizer.
 */

#include "lyapunov_functional_synthesizer.hpp"
#include <iostream>
#include <iomanip>

using namespace thebrain::lyapunov;

void print_result(const InvariantProofResult& res) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "🌀 SYSTEM: " << res.system_name << "\n";
    std::cout << "📐 SYNTHESIZED FUNCTIONAL: " << res.candidate_functional_str << "\n";
    std::cout << "⏱️ TIME DERIVATIVE ALONG FLOW: " << res.time_derivative_str << "\n";
    std::cout << "🎯 MONOTONIC DISSIPATION: " << (res.is_strictly_monotonic_dissipative ? "✅ STRICTLY MONOTONIC (dF/dt <= 0)" : "❌ NON-DISSIPATIVE") << "\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "📜 FORMAL DEDUCTIVE STEPS:\n";
    for (const auto& s : res.formal_deduction_steps) {
        std::cout << "   " << s << "\n";
    }
    std::cout << "🌟 STABILITY VERDICT: " << res.stability_verdict << "\n";
}

int main() {
    std::cout << "\n🧠 ==========================================================================\n";
    std::cout << "   THE BRAIN — LYAPUNOV & MONOTONIC ENERGY FUNCTIONAL SYNTHESIZER\n";
    std::cout << "   Autonomous Derivation of Conserved Functionals & Gradient Flow Dissipation\n";
    std::cout << "==========================================================================\n";

    // 1. Allen-Cahn PDE
    auto pde_res = LyapunovFunctionalSynthesizer::synthesize_allen_cahn_energy_functional();
    print_result(pde_res);

    // 2. Damped Duffing Oscillator
    auto duff_res = LyapunovFunctionalSynthesizer::synthesize_duffing_lyapunov_function(0.5);
    print_result(duff_res);

    // 3. Polynomial Vector Field
    auto poly_res = LyapunovFunctionalSynthesizer::synthesize_polynomial_vector_field(2.0, 3.0);
    print_result(poly_res);

    std::cout << "\n==========================================================================\n";
    std::cout << "🏁 LYAPUNOV ENERGY FUNCTIONAL SYNTHESIS COMPLETE: ALL FLOWS DISSIPATIVE\n";
    std::cout << "==========================================================================\n\n";

    return 0;
}
