#include <iostream>
#include "crisp_external_router.hpp"
#include "crisp_internal_router.hpp"

using namespace brain2::routing;

int main() {
    std::cout << "=== Crisp Routing Tests ===\n\n";
    
    CrispExternalRouter er;
    CrispFact f1{"rocket", "mass", 1000.0, true, "test", 0.9};
    CrispFact f2{"apple", "color", 1.0, false, "test", 0.9}; // Unverified
    
    auto r1 = er.push_fact(f1);
    std::cout << "Fact 1 push: " << (r1.accepted ? "OK" : "REJECTED") << " (" << r1.reason << ")\n";
    
    auto r2 = er.push_fact(f2);
    std::cout << "Fact 2 push: " << (r2.accepted ? "OK" : "REJECTED") << " (" << r2.reason << ")\n";
    
    CrispInternalRouter ir;
    auto d1 = ir.decide(0.9, 0, 0.1, 0.1, "factual", "statement", true);
    std::cout << "\nInternal decision 1 (high conf, verified): " << d1.label << "\n";
    
    auto d2 = ir.decide(0.5, 0, 0.8, 0.1, "factual", "statement", true);
    std::cout << "Internal decision 2 (high novelty): " << d2.label << "\n";
    
    return 0;
}
