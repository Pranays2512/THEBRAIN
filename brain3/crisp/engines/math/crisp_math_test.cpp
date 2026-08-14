#include <iostream>
#include <iomanip>
#include "calculus_engine.hpp"
#include "dimensional_verify.hpp"
#include "factorizer.hpp"
#include "math_engine.hpp"

using namespace brain2::math;

void test_calculus_engine() {
    std::cout << "=== calculus_engine — exact symbolic manipulation ===\n\n";
    auto expr = ExprNode::make_op("+", {ExprNode::make_num(5.0), ExprNode::make_num(0.0)});
    auto simp = simplify(expr);
    std::cout << "simplify(5 + 0) = " << render(simp) << "\n";
    
    auto comp = ExprNode::make_op("^", {ExprNode::make_var("x"), ExprNode::make_num(3.0)});
    std::cout << "render(x^3) = " << render(comp) << "\n";
}

void test_dimensional_verify() {
    std::cout << "\n=== dimensional_verify — unit gatekeeper ===\n\n";
    std::map<std::string, DimVec> UNITS = {
        {"mass", {1, 0, 0}},
        {"accel", {0, 1, -2}},
        {"speed", {0, 1, -1}}
    };
    
    auto force_units = parse_units("*", UNITS["mass"], UNITS["accel"]);
    std::cout << "units(mass * accel) = (" << force_units[0] << "," << force_units[1] << "," << force_units[2] << ") -> SOUND\n";
    
    try {
        parse_units("+", UNITS["mass"], UNITS["accel"]);
    } catch (const std::exception& e) {
        std::cout << "units(mass + accel) = INVALID (" << e.what() << ") -> REJECTED\n";
    }
}

void test_factorizer() {
    std::cout << "\n=== factorizer — AST anti-unification ===\n\n";
    auto a = ExprNode::make_op("*", {ExprNode::make_var("m"), ExprNode::make_var("v")});
    auto b = ExprNode::make_op("*", {ExprNode::make_var("F"), ExprNode::make_var("d")});
    auto p = factor_au(a, b);
    std::cout << "factor(m*v, F*d) discovers shared primitive: " << render(p) << "\n";
}

void test_math_engine() {
    std::cout << "\n=== math_engine — rule discovery via numerical verification ===\n\n";
    MathEngine me;
    bool found = me.discover_power_rule();
    if (found) {
        std::cout << "SUCCESS: MathEngine discovered rules:\n";
        for (const auto& r : me.discovered_rules) std::cout << "  ✓ " << r << "\n";
    }
}

int main() {
    test_calculus_engine();
    test_dimensional_verify();
    test_factorizer();
    test_math_engine();
    return 0;
}
