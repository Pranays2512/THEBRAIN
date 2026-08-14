#pragma once
#include <string>
#include <functional>
#include <cmath>
#include <tuple>
#include <vector>

namespace brain3 {
namespace engines {
namespace synthesis {

class ConjectureSandbox {
public:
    static constexpr double G = 9.8;

    // TRUSTED principle: energy conservation (anchor)
    std::pair<double, double> trusted_KE(double m, double h) const {
        double v = std::sqrt(2 * G * h);
        return {v, m * G * h};
    }

    struct TestResult {
        bool survived;
        double worst_err;
        std::tuple<double, double, double, double> counter; // m, v, ke_true, ke_guess
        std::string fail_reason;
    };

    // Design experiments, run conjecture vs trusted anchor, judge it
    TestResult design_and_test(
        std::function<double(double, double)> conjecture,
        int n = 40, double tol = 0.01) const
    {
        double worst = 0.0;
        std::tuple<double, double, double, double> counter;

        // Custom pseudo-random generation to avoid global state issues
        auto rand_uniform = [](int i, double min, double max) {
            double frac = (double)(i % 1000) / 1000.0;
            return min + frac * (max - min);
        };

        for (int i = 0; i < n; i++) {
            double m = rand_uniform(i * 7, 0.5, 10.0);
            double h = rand_uniform(i * 13, 1.0, 100.0);
            auto [v, ke_true] = trusted_KE(m, h);

            double ke_guess = 0.0;
            try {
                ke_guess = conjecture(m, v);
            } catch (...) {
                return {false, 1e9, {m, v, ke_true, 0.0}, "raised exception"};
            }

            double rel = std::abs(ke_guess - ke_true) / (std::abs(ke_true) + 1e-9);
            if (rel > worst) {
                worst = rel;
                counter = {m, v, ke_true, ke_guess};
            }
        }
        
        bool survived = worst <= tol;
        return {survived, worst, counter, ""};
    }
};

}}}
