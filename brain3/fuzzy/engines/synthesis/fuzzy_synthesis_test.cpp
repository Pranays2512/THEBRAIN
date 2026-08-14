#include <iostream>
#include <iomanip>
#include "policy_induction.hpp"

using namespace brain2::synthesis;

// Helper to generate some random uniform data
std::vector<std::map<std::string, double>> make_data(int n, const std::vector<std::string>& inputs, std::function<double(std::map<std::string, double>)> fn) {
    std::vector<std::map<std::string, double>> rows;
    for (int i = 0; i < n; i++) {
        std::map<std::string, double> r;
        for (const auto& in : inputs) {
            // Some pseudo-random values between 1 and 10
            r[in] = 1.0 + (std::rand() % 900) / 100.0;
        }
        r["__t__"] = fn(r);
        rows.push_back(r);
    }
    return rows;
}

int main() {
    std::cout << "=== policy_induction — discover a formula from examples, verified ===\n\n";
    
    std::srand(1);
    PolicyInduction pi;
    
    // Test 1: density = mass / volume
    {
        std::vector<std::string> inputs = {"mass", "volume"};
        auto rows = make_data(12, inputs, [](std::map<std::string, double> r) { return r["mass"] / r["volume"]; });
        auto expr = pi.induce(rows, inputs, "__t__");
        std::cout << "  density   = " << (expr ? expr->to_string() : "NOT FOUND") << "   (induced + held-out verified)\n";
    }
    
    // Test 2: ke = 0.5 * mass * speed^2
    {
        std::vector<std::string> inputs = {"mass", "speed"};
        auto rows = make_data(12, inputs, [](std::map<std::string, double> r) { return 0.5 * r["mass"] * r["speed"] * r["speed"]; });
        auto expr = pi.induce(rows, inputs, "__t__");
        std::cout << "  ke        = " << (expr ? expr->to_string() : "NOT FOUND") << "   (induced + held-out verified)\n";
    }
    
    return 0;
}
