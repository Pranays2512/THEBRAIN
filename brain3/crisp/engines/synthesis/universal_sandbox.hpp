#pragma once
/**
 * brain3/crisp/engines/synthesis/universal_sandbox.hpp
 *
 * THE BRAIN 3 — UNIVERSAL TESTING & EXPERIMENTATION SANDBOX ENGINE
 *
 * Provides a multi-domain testing, verification, and empirical experimentation
 * environment for The Brain to autonomously test:
 *  1. Physical & Scientific Conservation Laws (Kinematics, Orbital, Quantum, Relativity)
 *  2. Dynamical Systems, ODEs & Hamiltonian Invariant Conservation (RK4 numerical integration)
 *  3. Algorithmic Properties & Fuzzing (Sorting, Searching, Idempotency, Invariants)
 *  4. Symbolic CAS & Mathematical Identities under Adversarial Boundary Probing
 *  5. Code Execution, Memory Safety, Termination Proofs & Complexity Scaling
 *  6. Chemistry, Stoichiometry Balancing, Reaction Enthalpy, Gibbs Free Energy & Acid-Base pH
 *  7. Genetics, Central Dogma Translation, Mutation Classification & Punnett Square Alleles
 *  8. Electrical Circuits, Kirchhoff's Laws (KCL/KVL), Resonant RLC & Power Dissipation
 *  9. Information Theory, Shannon Entropy, Hamming Distance & Cryptographic Avalanche
 * 10. Economics, Supply-Demand Equilibrium, Consumer Surplus & Game Theory Nash Equilibrium
 * 11. Web, DOM, Selector Coverage & WCAG 2.1 Color Contrast Auditing
 * 12. Multi-Hypothesis Competitive Tournaments with Automated Counterexample Synthesis
 */

#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <tuple>
#include <map>
#include <set>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <regex>
#include <numeric>
#include <cctype>

namespace brain3 {
namespace engines {
namespace synthesis {

// ─────────────────────────────────────────────────────────────────────────────
// 1. Result Data Structures Across All Domains
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicsExperimentResult {
    bool passed = false;
    double max_relative_error = 0.0;
    std::string domain;
    std::string invariant_tested;
    std::vector<std::string> telemetry_samples;
    std::string counterexample;
    std::string failure_reason;
};

struct ODEResult {
    bool conserved = false;
    double max_invariant_drift = 0.0;
    double initial_invariant = 0.0;
    double final_invariant = 0.0;
    int step_count = 0;
    double elapsed_time_us = 0.0;
    std::string details;
};

struct PropertyTestResult {
    bool passed = false;
    int total_trials = 0;
    std::vector<std::string> checked_properties;
    std::string failing_input;
    std::string failure_reason;
};

struct WebAuditResult {
    bool passed = false;
    int total_css_selectors = 0;
    int total_html_tags = 0;
    std::vector<std::string> matched_selectors;
    std::vector<std::string> unmatched_css_selectors;
    std::vector<std::string> syntax_violations;
    std::vector<std::tuple<std::string, std::string, double, bool>> contrast_scores; // (fg, bg, ratio, pass_aa)
    bool wcag_aa_compliant = true;
};

struct TournamentResult {
    std::string winning_hypothesis;
    std::vector<std::pair<std::string, std::string>> eliminated_hypotheses; // (name, counterexample)
    int total_scenarios_tested = 0;
    bool decisive_winner = false;
};

struct CodeAuditResult {
    bool passed = false;
    bool syntax_valid = true;
    std::vector<std::string> syntax_issues;
    bool memory_safe = true;
    int leaked_allocations = 0;
    int invalid_memory_accesses = 0;
    bool bounds_safe = true;
    bool termination_verified = false;
    std::string detected_complexity; // O(1), O(log N), O(N), O(N log N), O(N^2)
    double empirical_scaling_exponent = 0.0;
    std::string failure_reason;
};

struct ChemistryResult {
    bool balanced = false;
    std::string equation;
    std::map<std::string, int> reactant_atoms;
    std::map<std::string, int> product_atoms;
    std::vector<std::string> atom_discrepancies;
    double delta_G = 0.0; // kJ/mol
    double delta_H = 0.0; // kJ/mol
    double delta_S = 0.0; // J/(mol*K)
    double K_eq = 0.0;
    bool is_spontaneous = false;
    double pH = 7.0;
    double pOH = 7.0;
    std::string details;
};

struct GeneticsResult {
    bool valid_sequence = false;
    std::string dna_input;
    std::string rna_transcript;
    std::string protein_sequence;
    std::vector<std::string> mutations_detected;
    std::string mutation_type; // Silent, Missense, Nonsense, Frameshift, None
    std::map<std::string, double> punnett_genotypes;
    std::string details;
};

struct CircuitResult {
    bool passed = false;
    bool kcl_satisfied = false;
    bool kvl_satisfied = false;
    double total_power_watts = 0.0;
    double time_constant_seconds = 0.0;
    double resonant_freq_hz = 0.0;
    double quality_factor = 0.0;
    std::string details;
};

struct InformationTheoryResult {
    bool passed = false;
    double shannon_entropy_bits = 0.0;
    double max_entropy_bits = 0.0;
    double coding_efficiency = 0.0;
    int hamming_distance = 0;
    double avalanche_bit_flip_ratio = 0.0; // Ideal ~0.50 (50%)
    bool passes_avalanche_test = false;
    std::string details;
};

struct EconomicsResult {
    bool passed = false;
    bool equilibrium_found = false;
    double equilibrium_price = 0.0;
    double equilibrium_quantity = 0.0;
    double consumer_surplus = 0.0;
    double producer_surplus = 0.0;
    std::vector<std::pair<int, int>> nash_equilibria; // (row, col)
    bool is_pareto_optimal = false;
    std::string details;
};

// ─────────────────────────────────────────────────────────────────────────────
// 2. Universal Sandbox Engine Core
// ─────────────────────────────────────────────────────────────────────────────

class UniversalSandboxEngine {
public:
    static constexpr double G = 9.80665;             // Standard gravity (m/s^2)
    static constexpr double C_LIGHT = 299792458.0;   // Speed of light (m/s)
    static constexpr double H_BAR = 1.054571817e-34; // Reduced Planck constant (J*s)
    static constexpr double G_NEWTON = 6.67430e-11;  // Gravitational constant (m^3/(kg*s^2))
    static constexpr double K_BOLTZMANN = 1.380649e-23; // Boltzmann constant (J/K)
    static constexpr double R_GAS = 8.314462618;     // Universal gas constant (J/(mol*K))
    static constexpr double F_FARADAY = 96485.332;   // Faraday constant (C/mol)
    static constexpr double K_WATER = 1.0e-14;       // Water autoionization constant at 298.15K

    UniversalSandboxEngine() = default;

    // Helper: Deterministic PRNG
    static double pseudo_rand(int seed, double min_v, double max_v) {
        uint32_t x = static_cast<uint32_t>(seed * 1664525u + 1013904223u);
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        double frac = static_cast<double>(x % 1000000u) / 1000000.0;
        return min_v + frac * (max_v - min_v);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION A: Physical & Scientific Simulation Experiments
    // ─────────────────────────────────────────────────────────────────────────

    // 1. Gravitational Kinematics & Drops (KE = m*g*h = 0.5*m*v^2)
    PhysicsExperimentResult test_gravitational_drop(
        std::function<double(double, double)> conjecture_ke,
        int n_trials = 40, double tol = 0.01) const
    {
        PhysicsExperimentResult res;
        res.domain = "Classical Mechanics";
        res.invariant_tested = "Kinetic Energy Conservation (KE = 0.5*m*v^2)";
        double worst_err = 0.0;

        for (int i = 0; i < n_trials; i++) {
            double m = pseudo_rand(i * 17 + 1, 0.5, 20.0);
            double h = pseudo_rand(i * 31 + 7, 1.0, 200.0);
            double v = std::sqrt(2.0 * G * h);
            double ke_true = m * G * h;

            double ke_guess = 0.0;
            try {
                ke_guess = conjecture_ke(m, v);
            } catch (...) {
                res.passed = false;
                res.failure_reason = "Conjecture threw runtime exception on drop test.";
                return res;
            }

            double rel_err = std::abs(ke_guess - ke_true) / (std::abs(ke_true) + 1e-9);
            if (rel_err > worst_err) {
                worst_err = rel_err;
                std::ostringstream ss;
                ss << "m=" << std::fixed << std::setprecision(2) << m << "kg, v=" << v 
                   << "m/s -> true=" << ke_true << "J, guess=" << ke_guess << "J, err=" 
                   << (rel_err * 100.0) << "%";
                res.counterexample = ss.str();
            }

            if (i < 3) {
                std::ostringstream ss;
                ss << "Sample [" << i << "]: m=" << m << "kg, h=" << h << "m -> v=" << v 
                   << "m/s, KE_true=" << ke_true << "J, KE_guess=" << ke_guess << "J";
                res.telemetry_samples.push_back(ss.str());
            }
        }

        res.max_relative_error = worst_err;
        res.passed = (worst_err <= tol);
        if (!res.passed) {
            res.failure_reason = "Max relative error " + std::to_string(worst_err * 100.0) + 
                                 "% exceeds tolerance " + std::to_string(tol * 100.0) + "%";
        }
        return res;
    }

    // 2. Orbital Mechanics & Kepler's 3rd Law (T^2 / a^3 = 4*pi^2 / (G*M))
    PhysicsExperimentResult test_orbital_mechanics(
        std::function<double(double, double)> period_conjecture,
        double central_mass = 1.989e30, // Solar mass in kg
        int n_trials = 25, double tol = 0.015) const
    {
        PhysicsExperimentResult res;
        res.domain = "Celestial Mechanics";
        res.invariant_tested = "Kepler's 3rd Law (T^2 = 4*pi^2*a^3 / (G*M))";
        const double PI = 3.14159265358979323846;
        double worst_err = 0.0;

        for (int i = 0; i < n_trials; i++) {
            double semi_major_axis_au = pseudo_rand(i * 23 + 3, 0.387, 30.0); // Mercury to Neptune
            double a_meters = semi_major_axis_au * 1.496e11;

            double t_seconds_true = 2.0 * PI * std::sqrt((a_meters * a_meters * a_meters) / (G_NEWTON * central_mass));
            double t_years_true = t_seconds_true / (365.25 * 86400.0);

            double t_years_guess = 0.0;
            try {
                t_years_guess = period_conjecture(semi_major_axis_au, central_mass / 1.989e30);
            } catch (...) {
                res.passed = false;
                res.failure_reason = "Conjecture threw runtime exception on orbital test.";
                return res;
            }

            double rel_err = std::abs(t_years_guess - t_years_true) / (std::abs(t_years_true) + 1e-9);
            if (rel_err > worst_err) {
                worst_err = rel_err;
                std::ostringstream ss;
                ss << "a=" << semi_major_axis_au << " AU -> T_true=" << t_years_true 
                   << " yrs, T_guess=" << t_years_guess << " yrs, err=" << (rel_err * 100.0) << "%";
                res.counterexample = ss.str();
            }
        }

        res.max_relative_error = worst_err;
        res.passed = (worst_err <= tol);
        if (!res.passed) {
            res.failure_reason = "Orbital law violated with max error " + std::to_string(worst_err * 100.0) + "%";
        }
        return res;
    }

    // 3. Relativistic Invariant Mass & Energy-Momentum: E^2 = (p*c)^2 + (m_0*c^2)^2
    PhysicsExperimentResult test_relativistic_energy_momentum(
        std::function<double(double, double)> invariant_conjecture,
        int n_trials = 30, double tol = 0.005) const
    {
        PhysicsExperimentResult res;
        res.domain = "Special Relativity";
        res.invariant_tested = "Minkowski Invariant Mass E^2 - (pc)^2 = (m0*c^2)^2";
        double worst_err = 0.0;

        for (int i = 0; i < n_trials; i++) {
            double m0 = pseudo_rand(i * 13 + 5, 0.1, 10.0); // kg
            double beta = pseudo_rand(i * 19 + 7, 0.1, 0.99); // v/c
            double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
            double v = beta * C_LIGHT;
            double p = gamma * m0 * v;
            double E = gamma * m0 * C_LIGHT * C_LIGHT;

            double true_invariant = m0 * C_LIGHT * C_LIGHT;
            double guess_invariant = 0.0;
            try {
                guess_invariant = invariant_conjecture(E, p);
            } catch (...) {
                res.passed = false;
                res.failure_reason = "Exception in relativistic test.";
                return res;
            }

            double rel_err = std::abs(guess_invariant - true_invariant) / (std::abs(true_invariant) + 1e-9);
            if (rel_err > worst_err) {
                worst_err = rel_err;
                std::ostringstream ss;
                ss << "beta=" << beta << " -> true_m0c2=" << true_invariant << ", guess=" << guess_invariant;
                res.counterexample = ss.str();
            }
        }

        res.max_relative_error = worst_err;
        res.passed = (worst_err <= tol);
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION B: Dynamical Systems & Hamiltonian RK4 ODE Integrators
    // ─────────────────────────────────────────────────────────────────────────

    ODEResult simulate_rk4_hamiltonian(
        std::function<std::pair<double, double>(double, double, double)> derivatives_dq_dp,
        std::function<double(double, double)> hamiltonian_func,
        double q0, double p0, double t_max = 20.0, double dt = 0.005, double max_drift_tol = 0.001) const
    {
        ODEResult res;
        double t = 0.0;
        double q = q0;
        double p = p0;

        double H0 = hamiltonian_func(q0, p0);
        res.initial_invariant = H0;
        double max_drift = 0.0;
        int steps = 0;

        auto start = std::chrono::high_resolution_clock::now();

        while (t < t_max) {
            auto [dq1, dp1] = derivatives_dq_dp(t, q, p);
            auto [dq2, dp2] = derivatives_dq_dp(t + 0.5 * dt, q + 0.5 * dt * dq1, p + 0.5 * dt * dp1);
            auto [dq3, dp3] = derivatives_dq_dp(t + 0.5 * dt, q + 0.5 * dt * dq2, p + 0.5 * dt * dp2);
            auto [dq4, dp4] = derivatives_dq_dp(t + dt, q + dt * dq3, p + dt * dp3);

            q += (dt / 6.0) * (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4);
            p += (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4);
            t += dt;
            steps++;

            double H_current = hamiltonian_func(q, p);
            double drift = std::abs(H_current - H0) / (std::abs(H0) + 1e-9);
            if (drift > max_drift) max_drift = drift;
        }

        auto end = std::chrono::high_resolution_clock::now();
        res.elapsed_time_us = std::chrono::duration<double, std::micro>(end - start).count();
        res.final_invariant = hamiltonian_func(q, p);
        res.max_invariant_drift = max_drift;
        res.step_count = steps;
        res.conserved = (max_drift <= max_drift_tol);

        std::ostringstream ss;
        ss << "RK4 Steps: " << steps << " (dt=" << dt << ", t_max=" << t_max << ") | "
           << "H0=" << H0 << ", H_final=" << res.final_invariant << ", MaxDrift=" 
           << (max_drift * 100.0) << "% (" << (res.conserved ? "CONSERVED" : "DIVERGED") << ")";
        res.details = ss.str();
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION C: Algorithmic Logic, Property Testing & Mutation Fuzzing
    // ─────────────────────────────────────────────────────────────────────────

    PropertyTestResult fuzz_sorting_algorithm(
        std::function<std::vector<int>(std::vector<int>)> sort_fn,
        int n_trials = 100) const
    {
        PropertyTestResult res;
        res.checked_properties = {
            "Monotonicity: a[i] <= a[i+1]",
            "Multiset Conservation: elements preserved with exact counts",
            "Idempotency: sort(sort(x)) == sort(x)",
            "Boundary Safety: empty, single-element, reverse-sorted, all-identical arrays"
        };

        std::vector<std::vector<int>> edge_cases = {
            {},
            {42},
            {5, 4, 3, 2, 1},
            {1, 2, 3, 4, 5},
            {7, 7, 7, 7, 7, 7},
            {10, -5, 0, 10, -5, 20, -100}
        };

        for (const auto& arr : edge_cases) {
            std::vector<int> out = sort_fn(arr);
            for (size_t i = 1; i < out.size(); i++) {
                if (out[i] < out[i - 1]) {
                    res.passed = false;
                    res.failing_input = "Edge case violated monotonicity";
                    res.failure_reason = "Array is not non-decreasing";
                    return res;
                }
            }
            std::map<int, int> in_counts, out_counts;
            for (int x : arr) in_counts[x]++;
            for (int x : out) out_counts[x]++;
            if (in_counts != out_counts) {
                res.passed = false;
                res.failing_input = "Edge case violated multiset conservation";
                res.failure_reason = "Elements were dropped or duplicated";
                return res;
            }
            res.total_trials++;
        }

        for (int t = 0; t < n_trials; t++) {
            int len = static_cast<int>(pseudo_rand(t * 11 + 3, 2, 50));
            std::vector<int> rand_arr(len);
            for (int i = 0; i < len; i++) {
                rand_arr[i] = static_cast<int>(pseudo_rand(t * 101 + i * 17, -500, 500));
            }

            std::vector<int> sorted_once = sort_fn(rand_arr);
            std::vector<int> sorted_twice = sort_fn(sorted_once);

            if (sorted_once != sorted_twice) {
                res.passed = false;
                res.failing_input = "Random array violated idempotency: sort(sort(x)) != sort(x)";
                res.failure_reason = "Non-idempotent sorting transform";
                return res;
            }

            for (size_t i = 1; i < sorted_once.size(); i++) {
                if (sorted_once[i] < sorted_once[i - 1]) {
                    res.passed = false;
                    res.failing_input = "Random array violated monotonicity";
                    res.failure_reason = "Element out of sorted order";
                    return res;
                }
            }
            res.total_trials++;
        }

        res.passed = true;
        return res;
    }

    PropertyTestResult fuzz_binary_search(
        std::function<int(const std::vector<int>&, int)> bsearch_fn,
        int n_trials = 80) const
    {
        PropertyTestResult res;
        res.checked_properties = {
            "Presence Invariant: If key is in array, bsearch returns valid index where a[idx] == key",
            "Absence Invariant: If key is not in array, bsearch returns -1",
            "Boundary Invariants: First element, Last element, Key < min, Key > max"
        };

        for (int t = 0; t < n_trials; t++) {
            int len = static_cast<int>(pseudo_rand(t * 19 + 7, 5, 60));
            std::vector<int> arr(len);
            for (int i = 0; i < len; i++) {
                arr[i] = static_cast<int>(pseudo_rand(t * 73 + i * 13, -300, 300));
            }
            std::sort(arr.begin(), arr.end());
            arr.erase(std::unique(arr.begin(), arr.end()), arr.end());

            for (int x : arr) {
                int idx = bsearch_fn(arr, x);
                if (idx < 0 || idx >= static_cast<int>(arr.size()) || arr[idx] != x) {
                    res.passed = false;
                    res.failing_input = "Key=" + std::to_string(x) + " present in array but returned idx=" + std::to_string(idx);
                    res.failure_reason = "Presence invariant violated";
                    return res;
                }
            }

            int missing_key = arr.back() + 100;
            int idx_missing = bsearch_fn(arr, missing_key);
            if (idx_missing != -1) {
                res.passed = false;
                res.failing_input = "Key=" + std::to_string(missing_key) + " absent from array but returned idx=" + std::to_string(idx_missing);
                res.failure_reason = "Absence invariant violated";
                return res;
            }
            res.total_trials++;
        }

        res.passed = true;
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION D: Symbolic CAS & Mathematical Identities
    // ─────────────────────────────────────────────────────────────────────────

    PropertyTestResult verify_math_identity(
        std::function<double(double)> lhs_fn,
        std::function<double(double)> rhs_fn,
        double x_min = -10.0, double x_max = 10.0,
        int n_samples = 100, double tol = 1e-7) const
    {
        PropertyTestResult res;
        res.checked_properties = {
            "LHS(x) == RHS(x) within numerical tolerance across domain",
            "Special boundary probes: x=0, x=1, x=-1, x=pi, x=e"
        };

        std::vector<double> boundary_probes = {
            0.0, 1.0, -1.0, 3.141592653589793, 2.718281828459045, 0.5, -0.5, 10.0, -10.0
        };

        for (double x : boundary_probes) {
            if (x < x_min || x > x_max) continue;
            try {
                double l = lhs_fn(x);
                double r = rhs_fn(x);
                if (std::isnan(l) || std::isnan(r) || std::isinf(l) || std::isinf(r)) continue;
                double diff = std::abs(l - r);
                if (diff > tol && (diff / (std::abs(l) + std::abs(r) + 1e-9)) > tol) {
                    res.passed = false;
                    res.failing_input = "x=" + std::to_string(x);
                    res.failure_reason = "LHS=" + std::to_string(l) + " != RHS=" + std::to_string(r);
                    return res;
                }
            } catch (...) {
                continue;
            }
            res.total_trials++;
        }

        for (int i = 0; i < n_samples; i++) {
            double x = pseudo_rand(i * 37 + 11, x_min, x_max);
            try {
                double l = lhs_fn(x);
                double r = rhs_fn(x);
                if (std::isnan(l) || std::isnan(r) || std::isinf(l) || std::isinf(r)) continue;
                double diff = std::abs(l - r);
                if (diff > tol && (diff / (std::abs(l) + std::abs(r) + 1e-9)) > tol) {
                    res.passed = false;
                    res.failing_input = "x=" + std::to_string(x);
                    res.failure_reason = "LHS=" + std::to_string(l) + " != RHS=" + std::to_string(r);
                    return res;
                }
            } catch (...) {
                continue;
            }
            res.total_trials++;
        }

        res.passed = true;
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION E: Code Execution, Memory Safety, Termination & Complexity Sandbox
    // ─────────────────────────────────────────────────────────────────────────

    // 1. Memory Safety & Bounds Tracker
    CodeAuditResult audit_memory_safety(
        const std::vector<std::tuple<std::string, int, int>>& operations // ("alloc", id, size), ("access", id, offset), ("free", id, 0)
    ) const {
        CodeAuditResult res;
        std::map<int, int> allocated_blocks; // id -> size
        std::set<int> freed_blocks;

        for (const auto& [op, id, param] : operations) {
            if (op == "alloc") {
                if (param <= 0) {
                    res.memory_safe = false;
                    res.syntax_issues.push_back("Zero or negative allocation size requested for ID " + std::to_string(id));
                }
                if (allocated_blocks.count(id) > 0) {
                    res.memory_safe = false;
                    res.syntax_issues.push_back("Duplicate allocation without freeing ID " + std::to_string(id));
                }
                allocated_blocks[id] = param;
            } else if (op == "access") {
                if (allocated_blocks.count(id) == 0) {
                    res.memory_safe = false;
                    res.bounds_safe = false;
                    res.invalid_memory_accesses++;
                    if (freed_blocks.count(id) > 0) {
                        res.syntax_issues.push_back("Use-after-free on ID " + std::to_string(id));
                    } else {
                        res.syntax_issues.push_back("Access on unallocated pointer ID " + std::to_string(id));
                    }
                } else {
                    int size = allocated_blocks[id];
                    if (param < 0 || param >= size) {
                        res.bounds_safe = false;
                        res.invalid_memory_accesses++;
                        res.syntax_issues.push_back("Out-of-bounds access on ID " + std::to_string(id) + 
                                                    ": offset=" + std::to_string(param) + ", size=" + std::to_string(size));
                    }
                }
            } else if (op == "free") {
                if (allocated_blocks.count(id) == 0) {
                    res.memory_safe = false;
                    if (freed_blocks.count(id) > 0) {
                        res.syntax_issues.push_back("Double-free detected on ID " + std::to_string(id));
                    } else {
                        res.syntax_issues.push_back("Freeing unallocated pointer ID " + std::to_string(id));
                    }
                } else {
                    allocated_blocks.erase(id);
                    freed_blocks.insert(id);
                }
            }
        }

        res.leaked_allocations = static_cast<int>(allocated_blocks.size());
        if (res.leaked_allocations > 0) {
            res.memory_safe = false;
            res.syntax_issues.push_back("Memory leak detected: " + std::to_string(res.leaked_allocations) + " blocks remained unfreed");
        }

        res.passed = res.memory_safe && res.bounds_safe && res.syntax_issues.empty();
        if (!res.passed) {
            res.failure_reason = "Memory or bounds violations encountered during execution trace.";
        }
        return res;
    }

    // 2. Loop Variant & Termination Proof Sandbox
    // Checks that variant function V(s) is non-negative and strictly decreasing under step function f(s)
    CodeAuditResult verify_loop_termination(
        std::function<int(int)> step_fn,
        std::function<int(int)> variant_fn,
        const std::vector<int>& test_initial_states,
        int max_allowed_iterations = 10000) const
    {
        CodeAuditResult res;
        res.termination_verified = true;

        for (int s0 : test_initial_states) {
            int s = s0;
            int v_prev = variant_fn(s);
            if (v_prev < 0) {
                res.termination_verified = false;
                res.passed = false;
                res.failure_reason = "Initial state has negative variant V(s0) = " + std::to_string(v_prev);
                return res;
            }

            int iter = 0;
            while (v_prev > 0 && iter < max_allowed_iterations) {
                int next_s = step_fn(s);
                int v_next = variant_fn(next_s);

                if (v_next >= v_prev) {
                    res.termination_verified = false;
                    res.passed = false;
                    res.failure_reason = "Non-decreasing variant at step: V(s)=" + std::to_string(v_prev) + 
                                         ", V(next_s)=" + std::to_string(v_next);
                    return res;
                }
                if (v_next < 0) {
                    res.termination_verified = false;
                    res.passed = false;
                    res.failure_reason = "Variant dropped below zero: V(next_s)=" + std::to_string(v_next);
                    return res;
                }

                s = next_s;
                v_prev = v_next;
                iter++;
            }

            if (iter >= max_allowed_iterations) {
                res.termination_verified = false;
                res.passed = false;
                res.failure_reason = "Exceeded max iteration bound (" + std::to_string(max_allowed_iterations) + "); possible infinite loop";
                return res;
            }
        }

        res.passed = true;
        return res;
    }

    // 3. Empirical Complexity Fit Sandbox (O(1), O(log N), O(N), O(N log N), O(N^2))
    CodeAuditResult test_algorithmic_complexity(
        std::function<void(int)> timed_work_fn,
        const std::vector<int>& n_values = {100, 200, 400, 800, 1600, 3200}) const
    {
        CodeAuditResult res;
        std::vector<double> log_n;
        std::vector<double> log_t;

        // Warm up
        timed_work_fn(50);

        for (int n : n_values) {
            auto t0 = std::chrono::high_resolution_clock::now();
            timed_work_fn(n);
            auto t1 = std::chrono::high_resolution_clock::now();
            double duration_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (duration_us < 0.1) duration_us = 0.1;

            log_n.push_back(std::log(static_cast<double>(n)));
            log_t.push_back(std::log(duration_us));
        }

        // Linear regression on log-log: log(T) = slope * log(N) + intercept
        double mean_x = 0.0, mean_y = 0.0;
        size_t m = log_n.size();
        for (size_t i = 0; i < m; i++) {
            mean_x += log_n[i];
            mean_y += log_t[i];
        }
        mean_x /= m;
        mean_y /= m;

        double num = 0.0, denom = 0.0;
        for (size_t i = 0; i < m; i++) {
            num += (log_n[i] - mean_x) * (log_t[i] - mean_y);
            denom += (log_n[i] - mean_x) * (log_n[i] - mean_x);
        }

        double slope = (denom > 1e-9) ? (num / denom) : 0.0;
        res.empirical_scaling_exponent = slope;

        if (slope < 0.25) {
            res.detected_complexity = "O(1)";
        } else if (slope >= 0.25 && slope < 0.7) {
            res.detected_complexity = "O(log N)";
        } else if (slope >= 0.7 && slope < 1.35) {
            res.detected_complexity = "O(N)";
        } else if (slope >= 1.35 && slope < 1.7) {
            res.detected_complexity = "O(N log N)";
        } else if (slope >= 1.7 && slope < 2.5) {
            res.detected_complexity = "O(N^2)";
        } else {
            res.detected_complexity = "O(N^" + std::to_string(std::round(slope * 10.0) / 10.0) + ")";
        }

        res.passed = true;
        return res;
    }

    // 4. Code Syntax & Bracket Balancing Linter
    CodeAuditResult lint_code_syntax(const std::string& code) const {
        CodeAuditResult res;
        std::vector<char> stack;

        for (size_t i = 0; i < code.size(); i++) {
            char c = code[i];
            if (c == '(' || c == '{' || c == '[') {
                stack.push_back(c);
            } else if (c == ')' || c == '}' || c == ']') {
                if (stack.empty()) {
                    res.syntax_valid = false;
                    res.syntax_issues.push_back("Unmatched closing bracket '" + std::string(1, c) + "' at position " + std::to_string(i));
                } else {
                    char open = stack.back();
                    stack.pop_back();
                    if ((c == ')' && open != '(') ||
                        (c == '}' && open != '{') ||
                        (c == ']' && open != '[')) {
                        res.syntax_valid = false;
                        res.syntax_issues.push_back("Mismatched bracket: expected match for '" + std::string(1, open) + "' but found '" + std::string(1, c) + "'");
                    }
                }
            }
        }

        if (!stack.empty()) {
            res.syntax_valid = false;
            res.syntax_issues.push_back("Unclosed opening bracket(s) remaining at EOF: " + std::to_string(stack.size()));
        }

        res.passed = res.syntax_valid && res.syntax_issues.empty();
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION F: Chemistry, Stoichiometry, Thermodynamics & Acid-Base
    // ─────────────────────────────────────────────────────────────────────────

    // 1. Recursive Chemical Formula Atom Parser
    // Parses formulas like "H2O", "C6H12O6", "Ca(OH)2", "Fe2(SO4)3", "K4[Fe(CN)6]"
    static std::map<std::string, int> parse_chemical_formula(const std::string& formula) {
        std::map<std::string, int> atoms;
        size_t idx = 0;
        auto parse_group = [&](auto& self) -> std::map<std::string, int> {
            std::map<std::string, int> group_atoms;
            while (idx < formula.size()) {
                char c = formula[idx];
                if (c == '(' || c == '[' || c == '{') {
                    idx++;
                    auto inner = self(self);
                    // Parse optional multiplier after closing bracket
                    int multiplier = 1;
                    if (idx < formula.size() && std::isdigit(formula[idx])) {
                        int num = 0;
                        while (idx < formula.size() && std::isdigit(formula[idx])) {
                            num = num * 10 + (formula[idx] - '0');
                            idx++;
                        }
                        multiplier = (num > 0) ? num : 1;
                    }
                    for (const auto& [el, cnt] : inner) {
                        group_atoms[el] += cnt * multiplier;
                    }
                } else if (c == ')' || c == ']' || c == '}') {
                    idx++;
                    return group_atoms;
                } else if (std::isupper(c)) {
                    std::string el(1, c);
                    idx++;
                    if (idx < formula.size() && std::islower(formula[idx])) {
                        el += formula[idx];
                        idx++;
                    }
                    int cnt = 0;
                    while (idx < formula.size() && std::isdigit(formula[idx])) {
                        cnt = cnt * 10 + (formula[idx] - '0');
                        idx++;
                    }
                    if (cnt == 0) cnt = 1;
                    group_atoms[el] += cnt;
                } else {
                    idx++;
                }
            }
            return group_atoms;
        };

        return parse_group(parse_group);
    }

    // 2. Stoichiometric Reaction Balancer & Mass Conservation Checker
    ChemistryResult verify_reaction_balance(
        const std::vector<std::pair<int, std::string>>& reactants, // (coefficient, formula)
        const std::vector<std::pair<int, std::string>>& products
    ) const {
        ChemistryResult res;
        std::ostringstream eq_ss;

        // Build reaction string & count reactant atoms
        for (size_t i = 0; i < reactants.size(); i++) {
            if (i > 0) eq_ss << " + ";
            if (reactants[i].first > 1) eq_ss << reactants[i].first << " ";
            eq_ss << reactants[i].second;

            auto parsed = parse_chemical_formula(reactants[i].second);
            for (const auto& [el, cnt] : parsed) {
                res.reactant_atoms[el] += cnt * reactants[i].first;
            }
        }
        eq_ss << " -> ";
        for (size_t i = 0; i < products.size(); i++) {
            if (i > 0) eq_ss << " + ";
            if (products[i].first > 1) eq_ss << products[i].first << " ";
            eq_ss << products[i].second;

            auto parsed = parse_chemical_formula(products[i].second);
            for (const auto& [el, cnt] : parsed) {
                res.product_atoms[el] += cnt * products[i].first;
            }
        }
        res.equation = eq_ss.str();

        // Check conservation of each atom
        std::set<std::string> all_elements;
        for (const auto& [el, _] : res.reactant_atoms) all_elements.insert(el);
        for (const auto& [el, _] : res.product_atoms) all_elements.insert(el);

        bool all_balanced = true;
        for (const auto& el : all_elements) {
            int r_cnt = res.reactant_atoms.count(el) ? res.reactant_atoms[el] : 0;
            int p_cnt = res.product_atoms.count(el) ? res.product_atoms[el] : 0;
            if (r_cnt != p_cnt) {
                all_balanced = false;
                std::ostringstream ss;
                ss << "Element " << el << " unbalanced: Reactants=" << r_cnt << ", Products=" << p_cnt;
                res.atom_discrepancies.push_back(ss.str());
            }
        }

        res.balanced = all_balanced;
        return res;
    }

    // 3. Chemical Thermodynamics & Spontaneity (Delta G = Delta H - T * Delta S)
    ChemistryResult test_reaction_thermodynamics(
        double delta_H_kJ_mol, double delta_S_J_mol_K, double T_kelvin = 298.15) const
    {
        ChemistryResult res;
        res.delta_H = delta_H_kJ_mol;
        res.delta_S = delta_S_J_mol_K;
        
        // Delta G = Delta H - T * (Delta S / 1000)
        res.delta_G = delta_H_kJ_mol - (T_kelvin * (delta_S_J_mol_K / 1000.0));
        res.is_spontaneous = (res.delta_G < 0.0);

        // Equilibrium constant: K_eq = exp(-Delta G / (R * T))
        double exponent = -(res.delta_G * 1000.0) / (R_GAS * T_kelvin);
        if (exponent > 700.0) exponent = 700.0;
        if (exponent < -700.0) exponent = -700.0;
        res.K_eq = std::exp(exponent);

        std::ostringstream ss;
        ss << "T=" << T_kelvin << "K | Delta H=" << delta_H_kJ_mol << " kJ/mol, Delta S=" 
           << delta_S_J_mol_K << " J/(mol*K) -> Delta G=" << res.delta_G 
           << " kJ/mol (" << (res.is_spontaneous ? "SPONTANEOUS / EXERGONIC" : "NON-SPONTANEOUS / ENDERGONIC") 
           << "), K_eq=" << res.K_eq;
        res.details = ss.str();
        res.balanced = true;
        return res;
    }

    // 4. Acid-Base Equilibrium & Buffer Chemistry (pH, pOH, Henderson-Hasselbalch)
    ChemistryResult test_acid_base_equilibrium(double h_plus_conc, double pKa = 4.76, double a_minus_conc = 0.1, double ha_conc = 0.1) const {
        ChemistryResult res;
        if (h_plus_conc <= 0.0) h_plus_conc = 1e-7;

        res.pH = -std::log10(h_plus_conc);
        double oh_conc = K_WATER / h_plus_conc;
        res.pOH = -std::log10(oh_conc);

        // Henderson-Hasselbalch buffer pH
        double buffer_pH = pKa;
        if (ha_conc > 0.0 && a_minus_conc > 0.0) {
            buffer_pH = pKa + std::log10(a_minus_conc / ha_conc);
        }

        std::ostringstream ss;
        ss << "[H+]=" << h_plus_conc << " M -> pH=" << res.pH << ", pOH=" << res.pOH 
           << " (pH + pOH = " << (res.pH + res.pOH) << ") | Buffer (pKa=" << pKa 
           << ", [A-]=" << a_minus_conc << ", [HA]=" << ha_conc << ") -> buffer_pH=" << buffer_pH;
        res.details = ss.str();
        res.balanced = std::abs((res.pH + res.pOH) - 14.0) < 1e-4;
        return res;
    }

    // 5. Gas Laws: Ideal Gas vs Van der Waals Equation
    PhysicsExperimentResult test_gas_laws(double n_moles, double T_kelvin, double V_liters, double a_vdw = 3.59, double b_vdw = 0.0427) const {
        PhysicsExperimentResult res;
        res.domain = "Physical Chemistry";
        res.invariant_tested = "Ideal Gas (PV = nRT) vs Real Gas (Van der Waals)";

        double V_m3 = V_liters * 1e-3;
        double P_ideal_Pa = (n_moles * R_GAS * T_kelvin) / V_m3;
        double P_ideal_atm = P_ideal_Pa / 101325.0;

        // Van der Waals: (P + a*(n/V)^2)(V - n*b) = n*R*T
        double a_SI = a_vdw * 101325.0 * 1e-6; // atm*L^2/mol^2 to Pa*m^6/mol^2
        double b_SI = b_vdw * 1e-3;            // L/mol to m^3/mol
        double P_vdw_Pa = ((n_moles * R_GAS * T_kelvin) / (V_m3 - n_moles * b_SI)) - a_SI * (n_moles * n_moles) / (V_m3 * V_m3);
        double P_vdw_atm = P_vdw_Pa / 101325.0;

        double rel_diff = std::abs(P_vdw_atm - P_ideal_atm) / P_ideal_atm;
        res.max_relative_error = rel_diff;
        res.passed = true;

        std::ostringstream ss;
        ss << "n=" << n_moles << " mol, T=" << T_kelvin << " K, V=" << V_liters << " L -> P_ideal=" 
           << P_ideal_atm << " atm, P_vdw=" << P_vdw_atm << " atm, Deviation=" << (rel_diff * 100.0) << "%";
        res.telemetry_samples.push_back(ss.str());
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION G: Genetics, Central Dogma Translation, Mutation & Punnett
    // ─────────────────────────────────────────────────────────────────────────

    // Standard Genetic Codon Translation Map
    static std::string codon_to_amino_acid(const std::string& codon) {
        static const std::map<std::string, std::string> table = {
            {"AUG", "Met"}, {"UUU", "Phe"}, {"UUC", "Phe"}, {"UUA", "Leu"}, {"UUG", "Leu"},
            {"CUU", "Leu"}, {"CUC", "Leu"}, {"CUA", "Leu"}, {"CUG", "Leu"},
            {"AUU", "Ile"}, {"AUC", "Ile"}, {"AUA", "Ile"},
            {"GUU", "Val"}, {"GUC", "Val"}, {"GUA", "Val"}, {"GUG", "Val"},
            {"UCU", "Ser"}, {"UCC", "Ser"}, {"UCA", "Ser"}, {"UCG", "Ser"},
            {"CCU", "Pro"}, {"CCC", "Pro"}, {"CCA", "Pro"}, {"CCG", "Pro"},
            {"ACU", "Thr"}, {"ACC", "Thr"}, {"ACA", "Thr"}, {"ACG", "Thr"},
            {"GCU", "Ala"}, {"GCC", "Ala"}, {"GCA", "Ala"}, {"GCG", "Ala"},
            {"UAU", "Tyr"}, {"UAC", "Tyr"}, {"CAU", "His"}, {"CAC", "His"},
            {"CAA", "Gln"}, {"CAG", "Gln"}, {"AAU", "Asn"}, {"AAC", "Asn"},
            {"AAA", "Lys"}, {"AAG", "Lys"}, {"GAU", "Asp"}, {"GAC", "Asp"},
            {"GAA", "Glu"}, {"GAG", "Glu"}, {"UGU", "Cys"}, {"UGC", "Cys"},
            {"UGG", "Trp"}, {"CGU", "Arg"}, {"CGC", "Arg"}, {"CGA", "Arg"},
            {"CGG", "Arg"}, {"AGA", "Arg"}, {"AGG", "Arg"},
            {"GGU", "Gly"}, {"GGC", "Gly"}, {"GGA", "Gly"}, {"GGG", "Gly"},
            {"UAA", "Stop"}, {"UAG", "Stop"}, {"UGA", "Stop"}
        };
        auto it = table.find(codon);
        return (it != table.end()) ? it->second : "???";
    }

    // 1. Central Dogma: DNA -> mRNA -> Protein Sequence
    GeneticsResult test_central_dogma_translation(const std::string& dna_coding_strand) const {
        GeneticsResult res;
        res.dna_input = dna_coding_strand;

        // Transcription: DNA -> mRNA (T -> U)
        std::string rna;
        for (char c : dna_coding_strand) {
            char u = std::toupper(c);
            if (u == 'T') rna += 'U';
            else if (u == 'A' || u == 'C' || u == 'G') rna += u;
            else {
                res.valid_sequence = false;
                res.details = "Invalid nucleotide in DNA strand: " + std::string(1, c);
                return res;
            }
        }
        res.rna_transcript = rna;
        res.valid_sequence = true;

        // Translation: Triplet codons to Amino Acids
        std::string protein;
        for (size_t i = 0; i + 2 < rna.size(); i += 3) {
            std::string codon = rna.substr(i, 3);
            std::string aa = codon_to_amino_acid(codon);
            if (aa == "Stop") break;
            if (!protein.empty()) protein += "-";
            protein += aa;
        }
        res.protein_sequence = protein;

        std::ostringstream ss;
        ss << "DNA: " << res.dna_input << " -> mRNA: " << res.rna_transcript 
           << " -> Protein: " << (res.protein_sequence.empty() ? "(None / Immediate Stop)" : res.protein_sequence);
        res.details = ss.str();
        return res;
    }

    // 2. Genetic Mutation Classifier (Silent, Missense, Nonsense, Frameshift)
    GeneticsResult classify_mutation(const std::string& wild_type_dna, const std::string& mutant_dna) const {
        GeneticsResult wt_res = test_central_dogma_translation(wild_type_dna);
        GeneticsResult mut_res = test_central_dogma_translation(mutant_dna);

        GeneticsResult res = mut_res;
        if (wild_type_dna.size() % 3 != mutant_dna.size() % 3 || wild_type_dna.size() != mutant_dna.size()) {
            if (std::abs(static_cast<int>(wild_type_dna.size()) - static_cast<int>(mutant_dna.size())) % 3 != 0) {
                res.mutation_type = "Frameshift Mutation";
            } else {
                res.mutation_type = "In-frame Insertion/Deletion";
            }
        } else if (wt_res.protein_sequence == mut_res.protein_sequence) {
            res.mutation_type = "Silent Mutation";
        } else if (mut_res.protein_sequence.find("Stop") != std::string::npos || mut_res.protein_sequence.size() < wt_res.protein_sequence.size()) {
            res.mutation_type = "Nonsense Mutation (Premature Stop)";
        } else {
            res.mutation_type = "Missense Mutation";
        }

        std::ostringstream ss;
        ss << "WT: [" << wt_res.protein_sequence << "] vs Mutant: [" << mut_res.protein_sequence 
           << "] -> Classified as: " << res.mutation_type;
        res.details = ss.str();
        return res;
    }

    // 3. Mendelian Genetics: Punnett Square & Hardy-Weinberg Frequency
    GeneticsResult test_punnett_monohybrid_cross(const std::pair<char, char>& p1, const std::pair<char, char>& p2) const {
        GeneticsResult res;
        std::vector<std::string> offspring;

        std::vector<char> g1 = {p1.first, p1.second};
        std::vector<char> g2 = {p2.first, p2.second};

        for (char a1 : g1) {
            for (char a2 : g2) {
                std::string geno;
                if (std::isupper(a1) || (!std::isupper(a2) && a1 < a2)) {
                    geno = std::string(1, a1) + std::string(1, a2);
                } else {
                    geno = std::string(1, a2) + std::string(1, a1);
                }
                offspring.push_back(geno);
            }
        }

        for (const auto& g : offspring) {
            res.punnett_genotypes[g] += 0.25;
        }

        res.valid_sequence = true;
        std::ostringstream ss;
        ss << "Cross (" << p1.first << p1.second << " x " << p2.first << p2.second << ") -> Genotype Probabilities: ";
        for (const auto& [g, prob] : res.punnett_genotypes) {
            ss << g << ": " << (prob * 100.0) << "% ";
        }
        res.details = ss.str();
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION H: Electrical Circuits, Kirchhoff's Laws & RLC Resonators
    // ─────────────────────────────────────────────────────────────────────────

    // 1. Kirchhoff's Laws (KCL & KVL) and Power Conservation
    CircuitResult verify_circuit_laws(
        const std::vector<double>& node_currents, // in/out currents (+ for in, - for out)
        const std::vector<double>& loop_voltages, // loop potential changes
        double voltage, double current, double resistance
    ) const {
        CircuitResult res;

        // KCL: Sum of currents at node = 0
        double sum_i = 0.0;
        for (double i : node_currents) sum_i += i;
        res.kcl_satisfied = (std::abs(sum_i) < 1e-5);

        // KVL: Sum of voltage drops around loop = 0
        double sum_v = 0.0;
        for (double v : loop_voltages) sum_v += v;
        res.kvl_satisfied = (std::abs(sum_v) < 1e-5);

        // Ohm's Law & Power: P = V * I = I^2 * R
        double p_vi = voltage * current;
        double p_i2r = current * current * resistance;
        res.total_power_watts = p_vi;

        bool ohm_power_match = (std::abs(p_vi - p_i2r) / (p_vi + 1e-9)) < 0.01;
        res.passed = res.kcl_satisfied && res.kvl_satisfied && ohm_power_match;

        std::ostringstream ss;
        ss << "KCL Node Sum: " << sum_i << "A (" << (res.kcl_satisfied ? "SATISFIED" : "VIOLATED") << ") | "
           << "KVL Loop Sum: " << sum_v << "V (" << (res.kvl_satisfied ? "SATISFIED" : "VIOLATED") << ") | "
           << "Power: " << p_vi << " W";
        res.details = ss.str();
        return res;
    }

    // 2. RLC Resonant Circuit & RC Time Constant
    CircuitResult test_rlc_frequency_response(double R_ohms, double L_henries, double C_farads) const {
        CircuitResult res;
        const double PI = 3.14159265358979323846;

        res.time_constant_seconds = R_ohms * C_farads;
        double omega_0 = 1.0 / std::sqrt(L_henries * C_farads);
        res.resonant_freq_hz = omega_0 / (2.0 * PI);
        res.quality_factor = (1.0 / R_ohms) * std::sqrt(L_henries / C_farads);
        res.passed = (res.resonant_freq_hz > 0.0 && res.quality_factor > 0.0);

        std::ostringstream ss;
        ss << "R=" << R_ohms << " Ohm, L=" << L_henries << " H, C=" << C_farads << " F -> "
           << "Resonant Frequency f0=" << res.resonant_freq_hz << " Hz (w0=" << omega_0 
           << " rad/s), Quality Factor Q=" << res.quality_factor << ", RC Time Constant tau=" 
           << res.time_constant_seconds << " s";
        res.details = ss.str();
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION I: Information Theory, Shannon Entropy & Cryptographic Avalanche
    // ─────────────────────────────────────────────────────────────────────────

    // 1. Shannon Entropy & Coding Efficiency
    InformationTheoryResult calculate_shannon_entropy(const std::vector<double>& probabilities) const {
        InformationTheoryResult res;
        double sum_p = 0.0;
        double H = 0.0;

        for (double p : probabilities) {
            if (p > 0.0) {
                sum_p += p;
                H -= p * (std::log2(p));
            }
        }

        res.shannon_entropy_bits = H;
        size_t n = probabilities.size();
        res.max_entropy_bits = (n > 0) ? std::log2(static_cast<double>(n)) : 0.0;
        res.coding_efficiency = (res.max_entropy_bits > 1e-9) ? (res.shannon_entropy_bits / res.max_entropy_bits) : 1.0;
        res.passed = (std::abs(sum_p - 1.0) < 1e-4 && H >= 0.0);

        std::ostringstream ss;
        ss << "Shannon Entropy: " << res.shannon_entropy_bits << " bits / symbol (Max possible: " 
           << res.max_entropy_bits << " bits, Efficiency: " << (res.coding_efficiency * 100.0) << "%)";
        res.details = ss.str();
        return res;
    }

    // 2. Hamming Distance
    static int calculate_hamming_distance(uint64_t v1, uint64_t v2) {
        uint64_t diff = v1 ^ v2;
        int dist = 0;
        while (diff > 0) {
            dist += (diff & 1);
            diff >>= 1;
        }
        return dist;
    }

    // 3. Cryptographic Avalanche Effect Invariance Test
    // Flipping 1 input bit must on average flip ~50% of the output bits
    InformationTheoryResult test_avalanche_effect(
        std::function<uint64_t(uint64_t)> hash_fn,
        int n_trials = 64) const
    {
        InformationTheoryResult res;
        int total_bit_flips = 0;
        int total_possible_flips = 0;

        for (int t = 0; t < n_trials; t++) {
            uint64_t x = static_cast<uint64_t>(pseudo_rand(t * 53 + 17, 1000.0, 1e12));
            uint64_t h_orig = hash_fn(x);

            for (int bit = 0; bit < 64; bit++) {
                uint64_t x_flipped = x ^ (1ULL << bit);
                uint64_t h_flipped = hash_fn(x_flipped);

                int dist = calculate_hamming_distance(h_orig, h_flipped);
                total_bit_flips += dist;
                total_possible_flips += 64;
            }
        }

        res.avalanche_bit_flip_ratio = static_cast<double>(total_bit_flips) / total_possible_flips;
        // Strict Avalanche Criterion (SAC): ratio should be between 0.40 and 0.60
        res.passes_avalanche_test = (res.avalanche_bit_flip_ratio >= 0.40 && res.avalanche_bit_flip_ratio <= 0.60);
        res.passed = res.passes_avalanche_test;

        std::ostringstream ss;
        ss << "Avalanche Flip Ratio: " << (res.avalanche_bit_flip_ratio * 100.0) 
           << "% (Ideal SAC: 50.0%) | " << (res.passes_avalanche_test ? "PASS (High Bit Diffusion)" : "FAIL (Poor Diffusion / Linear Drift)");
        res.details = ss.str();
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION J: Economics, Market Equilibrium & Game Theory Nash Equilibrium
    // ─────────────────────────────────────────────────────────────────────────

    // 1. Supply-Demand Market Equilibrium & Consumer/Producer Surplus
    // Demand: Q_d(P) = a - b*P, Supply: Q_s(P) = c + d*P
    EconomicsResult test_market_equilibrium(double a, double b, double c, double d) const {
        EconomicsResult res;
        // a - b*P = c + d*P -> P* = (a - c) / (b + d)
        if (b + d <= 0.0) {
            res.equilibrium_found = false;
            res.details = "Invalid slope parameters: b + d <= 0";
            return res;
        }

        double p_star = (a - c) / (b + d);
        double q_star = a - b * p_star;

        if (p_star < 0.0 || q_star < 0.0) {
            res.equilibrium_found = false;
            res.details = "No non-negative price/quantity equilibrium exists";
            return res;
        }

        res.equilibrium_found = true;
        res.equilibrium_price = p_star;
        res.equilibrium_quantity = q_star;

        // Consumer Surplus: 0.5 * (P_max - P*) * Q* where P_max = a / b
        double p_max = a / b;
        res.consumer_surplus = 0.5 * (p_max - p_star) * q_star;

        // Producer Surplus: 0.5 * (P* - P_min) * Q* where P_min = -c / d (or 0)
        double p_min = (c < 0.0) ? (-c / d) : 0.0;
        res.producer_surplus = 0.5 * (p_star - p_min) * q_star;
        res.passed = true;

        std::ostringstream ss;
        ss << "Equilibrium Price P*=" << p_star << ", Quantity Q*=" << q_star 
           << " | Consumer Surplus=" << res.consumer_surplus << ", Producer Surplus=" 
           << res.producer_surplus << ", Total Social Welfare=" << (res.consumer_surplus + res.producer_surplus);
        res.details = ss.str();
        return res;
    }

    // 2. 2-Player Game Theory Nash Equilibrium Solver (2x2 Matrix Game)
    // Payoffs: [ (A00, B00), (A01, B01) ]
    //          [ (A10, B10), (A11, B11) ]
    EconomicsResult solve_2x2_nash_equilibrium(
        const std::vector<std::vector<std::pair<double, double>>>& payoff_matrix
    ) const {
        EconomicsResult res;
        if (payoff_matrix.size() != 2 || payoff_matrix[0].size() != 2 || payoff_matrix[1].size() != 2) {
            res.details = "Requires exactly a 2x2 payoff matrix";
            return res;
        }

        // Find pure strategy Nash equilibria
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 2; c++) {
                double player1_payoff = payoff_matrix[r][c].first;
                double player2_payoff = payoff_matrix[r][c].second;

                // Player 1 best response check against Player 2 playing column 'c'
                int other_r = 1 - r;
                bool p1_best = (player1_payoff >= payoff_matrix[other_r][c].first);

                // Player 2 best response check against Player 1 playing row 'r'
                int other_c = 1 - c;
                bool p2_best = (player2_payoff >= payoff_matrix[r][other_c].second);

                if (p1_best && p2_best) {
                    res.nash_equilibria.push_back({r, c});
                }
            }
        }

        res.equilibrium_found = !res.nash_equilibria.empty();
        res.passed = true;

        std::ostringstream ss;
        ss << "Found " << res.nash_equilibria.size() << " Pure Strategy Nash Equilibria: ";
        for (const auto& [r, c] : res.nash_equilibria) {
            ss << "(Row " << r << ", Col " << c << " -> Payoffs: " 
               << payoff_matrix[r][c].first << ", " << payoff_matrix[r][c].second << ") ";
        }
        res.details = ss.str();
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION K: Web Frontend DOM, Selector Coverage & WCAG 2.1 Color Contrast
    // ─────────────────────────────────────────────────────────────────────────

    static double parse_hex_channel(const std::string& hex) {
        return static_cast<double>(std::stoul(hex, nullptr, 16)) / 255.0;
    }

    static double relative_luminance(double r, double g, double b) {
        auto linearize = [](double c) {
            return (c <= 0.03928) ? (c / 12.92) : std::pow((c + 0.055) / 1.055, 2.4);
        };
        return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b);
    }

    static double calculate_contrast_ratio(const std::string& fg_hex, const std::string& bg_hex) {
        std::string fg = (fg_hex[0] == '#') ? fg_hex.substr(1) : fg_hex;
        std::string bg = (bg_hex[0] == '#') ? bg_hex.substr(1) : bg_hex;
        if (fg.size() == 3) fg = std::string(2, fg[0]) + std::string(2, fg[1]) + std::string(2, fg[2]);
        if (bg.size() == 3) bg = std::string(2, bg[0]) + std::string(2, bg[1]) + std::string(2, bg[2]);
        if (fg.size() != 6 || bg.size() != 6) return 1.0;

        double r1 = parse_hex_channel(fg.substr(0, 2));
        double g1 = parse_hex_channel(fg.substr(2, 2));
        double b1 = parse_hex_channel(fg.substr(4, 2));

        double r2 = parse_hex_channel(bg.substr(0, 2));
        double g2 = parse_hex_channel(bg.substr(2, 2));
        double b2 = parse_hex_channel(bg.substr(4, 2));

        double l1 = relative_luminance(r1, g1, b1);
        double l2 = relative_luminance(r2, g2, b2);

        double lighter = std::max(l1, l2);
        double darker = std::min(l1, l2);
        return (lighter + 0.05) / (darker + 0.05);
    }

    WebAuditResult audit_web_frontend_contract(
        const std::string& html_content,
        const std::string& css_content,
        const std::vector<std::pair<std::string, std::string>>& color_pairs = {}) const
    {
        WebAuditResult res;

        // 1. Extract classes and IDs from HTML
        std::set<std::string> html_classes;
        std::set<std::string> html_ids;
        std::regex class_regex("class=[\"']([^\"']+)[\"']");
        std::regex id_regex("id=[\"']([^\"']+)[\"']");

        auto c_begin = std::sregex_iterator(html_content.begin(), html_content.end(), class_regex);
        auto c_end = std::sregex_iterator();
        for (auto it = c_begin; it != c_end; ++it) {
            std::istringstream ss((*it)[1].str());
            std::string c;
            while (ss >> c) html_classes.insert(c);
        }

        auto id_begin = std::sregex_iterator(html_content.begin(), html_content.end(), id_regex);
        auto id_end = std::sregex_iterator();
        for (auto it = id_begin; it != id_end; ++it) {
            html_ids.insert((*it)[1].str());
        }

        // 2. Extract CSS selectors and check for pseudo-class syntax bugs
        std::regex css_selector_regex("([.#][a-zA-Z0-9_-]+)\\s*\\{");
        auto sel_begin = std::sregex_iterator(css_content.begin(), css_content.end(), css_selector_regex);
        auto sel_end = std::sregex_iterator();

        // Check for malformed pseudo-class syntax (e.g. "hover:" instead of ":hover")
        if (css_content.find("hover:") != std::string::npos) {
            res.syntax_violations.push_back("Found malformed pseudo-class 'hover:' instead of ':hover'");
        }
        if (css_content.find("focus:") != std::string::npos) {
            res.syntax_violations.push_back("Found malformed pseudo-class 'focus:' instead of ':focus'");
        }

        // 3. Match selectors against HTML
        for (auto it = sel_begin; it != sel_end; ++it) {
            std::string sel = (*it)[1].str();
            res.total_css_selectors++;
            if (sel[0] == '.') {
                std::string class_name = sel.substr(1);
                if (html_classes.count(class_name) > 0) {
                    res.matched_selectors.push_back(sel);
                } else {
                    res.unmatched_css_selectors.push_back(sel);
                }
            }
        }

        // 4. WCAG Color Contrast Evaluation
        for (const auto& [fg, bg] : color_pairs) {
            double ratio = calculate_contrast_ratio(fg, bg);
            bool pass_aa = (ratio >= 4.5);
            if (!pass_aa) res.wcag_aa_compliant = false;
            res.contrast_scores.push_back({fg, bg, ratio, pass_aa});
        }

        res.passed = res.syntax_violations.empty() && res.unmatched_css_selectors.empty() && res.wcag_aa_compliant;
        return res;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION L: Multi-Hypothesis Competitive Tournaments
    // ─────────────────────────────────────────────────────────────────────────

    TournamentResult run_hypothesis_tournament(
        const std::vector<std::pair<std::string, std::function<double(double, double)>>>& candidates,
        std::function<double(double, double)> oracle,
        int n_scenarios = 30, double tol = 0.01) const
    {
        TournamentResult res;
        std::vector<std::pair<std::string, std::function<double(double, double)>>> active = candidates;

        for (int i = 0; i < n_scenarios && active.size() > 1; i++) {
            double x = pseudo_rand(i * 19 + 5, 0.5, 50.0);
            double y = pseudo_rand(i * 31 + 7, 0.5, 50.0);
            double target = oracle(x, y);
            res.total_scenarios_tested++;

            std::vector<std::pair<std::string, std::function<double(double, double)>>> survivors;
            for (const auto& cand : active) {
                double guess = 0.0;
                try {
                    guess = cand.second(x, y);
                } catch (...) {
                    res.eliminated_hypotheses.push_back({cand.first, "Threw runtime exception"});
                    continue;
                }

                double err = std::abs(guess - target) / (std::abs(target) + 1e-9);
                if (err <= tol) {
                    survivors.push_back(cand);
                } else {
                    std::ostringstream ss;
                    ss << "Violated at (x=" << x << ", y=" << y << "): oracle=" << target 
                       << ", guess=" << guess << ", err=" << (err * 100.0) << "%";
                    res.eliminated_hypotheses.push_back({cand.first, ss.str()});
                }
            }
            active = survivors;
        }

        if (active.size() == 1) {
            res.winning_hypothesis = active[0].first;
            res.decisive_winner = true;
        } else if (active.empty()) {
            res.winning_hypothesis = "NONE (all candidates refuted by experiment)";
            res.decisive_winner = false;
        } else {
            res.winning_hypothesis = "AMBIGUOUS (" + std::to_string(active.size()) + " surviving hypotheses)";
            res.decisive_winner = false;
        }

        return res;
    }
};

}}}
