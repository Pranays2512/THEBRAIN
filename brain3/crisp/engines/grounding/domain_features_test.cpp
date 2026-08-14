#include <iostream>
#include "domain_features.hpp"

using namespace brain2::grounding;

void test_domain_features() {
    std::cout << "=== domain_features — dimensional analysis pruning ===\n\n";
    
    std::map<std::string, DimVec> units = {
        {"mass", {1, 0, 0}},     // M
        {"accel", {0, 1, -2}},   // L T^-2
        {"force", {1, 1, -2}},   // M L T^-2
        {"dist", {0, 1, 0}},     // L
        {"energy", {1, 2, -2}}   // M L^2 T^-2
    };
    
    // F = m * a
    auto expr_force = ExprNode::make_op("*", {ExprNode::make_var("mass"), ExprNode::make_var("accel")});
    PolicyDef p_force = {"force", expr_force, {"mass", "accel"}};
    
    auto res1 = dim_consistent(p_force, units);
    std::cout << "Force = mass * accel: " << (res1.value_or(false) ? "Consistent" : "Inconsistent") << "\n";
    
    // F = m + a (Inconsistent)
    auto expr_bad = ExprNode::make_op("+", {ExprNode::make_var("mass"), ExprNode::make_var("accel")});
    PolicyDef p_bad = {"force", expr_bad, {"mass", "accel"}};
    
    auto res2 = dim_consistent(p_bad, units);
    std::cout << "Force = mass + accel: " << (res2.has_value() && res2.value() ? "Consistent" : "Inconsistent") << "\n";

    // E = F * d
    auto expr_energy = ExprNode::make_op("*", {ExprNode::make_var("force"), ExprNode::make_var("dist")});
    PolicyDef p_energy = {"energy", expr_energy, {"force", "dist"}};
    
    auto res3 = dim_consistent(p_energy, units);
    std::cout << "Energy = force * dist: " << (res3.value_or(false) ? "Consistent" : "Inconsistent") << "\n";
}

int main() {
    test_domain_features();
    return 0;
}
