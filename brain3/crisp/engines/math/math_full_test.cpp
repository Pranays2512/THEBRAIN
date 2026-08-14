#include "crisp/engines/math/math_parser.hpp"
#include "crisp/engines/math/physics_engine.hpp"
#include "crisp/engines/math/algebra_engine.hpp"
#include "crisp/engines/math/integral_engine.hpp"
#include "crisp/engines/knowledge/nl_query.hpp"
#include "crisp/engines/knowledge/policy_pack.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace brain2::math;
using namespace brain2::knowledge;

int main() {
    std::cout << "=== Math Engines Full Test ===\n\n";

    // ── 1. Math Parser ───────────────────────────────────────────────────────
    std::cout << "--- math_parser ---\n";
    {
        auto e1 = parse("x^3");
        assert(e1->op == "^");
        std::cout << "  parse(x^3) = " << render(e1) << "\n";

        auto e2 = parse("sin(x^2)");
        assert(e2->op == "sin");
        std::cout << "  parse(sin(x^2)) = " << render(e2) << "\n";

        auto e3 = parse("2*x + 3 = 7");
        assert(e3->op == "=");
        std::cout << "  parse(2*x + 3 = 7) = op='" << e3->op << "'\n";

        auto e4 = parse("-cos(2*x)");
        assert(e4->op == "neg");
        std::cout << "  parse(-cos(2*x)) = " << render(e4) << "\n";
    }
    std::cout << "  math_parser OK\n\n";

    // ── 2. Physics Engine ────────────────────────────────────────────────────
    std::cout << "--- physics_engine ---\n";
    {
        PhysicsEngine pe;
        // F = m*a
        auto rhs_fma = ExprNode::make_op("*", {ExprNode::make_var("m"), ExprNode::make_var("a")});
        pe.add_law("newton2", "F", rhs_fma);

        auto [val1, steps1] = pe.solve("newton2", "F", {{"m", 3.0}, {"a", 4.0}});
        assert(std::abs(val1 - 12.0) < 1e-6);
        std::cout << "  F = m*a => F=" << val1 << " (expected 12)\n";

        auto [val2, steps2] = pe.solve("newton2", "a", {{"F", 12.0}, {"m", 3.0}});
        assert(std::abs(val2 - 4.0) < 1e-6);
        std::cout << "  solve for a => a=" << val2 << " (expected 4)\n";

        // v = d/t
        auto rhs_speed = ExprNode::make_op("/", {ExprNode::make_var("d"), ExprNode::make_var("t")});
        pe.add_law("speed", "v", rhs_speed);
        auto [val3, steps3] = pe.solve("speed", "t", {{"d", 100.0}, {"v", 20.0}});
        assert(std::abs(val3 - 5.0) < 1e-6);
        std::cout << "  solve t=d/v => t=" << val3 << " (expected 5)\n";
    }
    std::cout << "  physics_engine OK\n\n";

    // ── 3. Algebra Engine ────────────────────────────────────────────────────
    std::cout << "--- algebra_engine ---\n";
    {
        AlgebraEngine ae;
        // 2*x + 3 = 7  => x=2
        auto eq1 = parse("2*x + 3 = 7");
        auto [v1, s1] = ae.solve(eq1, "x");
        assert(std::abs(v1 - 2.0) < 1e-5);
        std::cout << "  2*x + 3 = 7 => x=" << v1 << " (expected 2)\n";

        // 5*x = 20  => x=4
        auto eq2 = parse("5*x = 20");
        auto [v2, s2] = ae.solve(eq2, "x");
        assert(std::abs(v2 - 4.0) < 1e-5);
        std::cout << "  5*x = 20 => x=" << v2 << " (expected 4)\n";

        // 3*x - 5 = 10  => x=5
        auto eq3 = parse("3*x - 5 = 10");
        auto [v3, s3] = ae.solve(eq3, "x");
        assert(std::abs(v3 - 5.0) < 1e-5);
        std::cout << "  3*x - 5 = 10 => x=" << v3 << " (expected 5)\n";
    }
    std::cout << "  algebra_engine OK\n\n";

    // ── 4. Integral Engine ───────────────────────────────────────────────────
    std::cout << "--- integral_engine ---\n";
    {
        IntegralEngine ie;

        // ∫ x^2 dx = x^3/3
        auto e_x2 = parse("x^2");
        auto F1 = ie.integrate(e_x2);
        assert(F1 != nullptr);
        bool ok1 = ie.verify(e_x2, F1);
        std::cout << "  ∫x^2 dx = " << render(F1) << "  [" << (ok1 ? "checked" : "WRONG") << "]\n";
        assert(ok1);

        // ∫ cos(x) dx = sin(x)
        auto e_cos = parse("cos(x)");
        auto F2 = ie.integrate(e_cos);
        assert(F2 != nullptr);
        bool ok2 = ie.verify(e_cos, F2);
        std::cout << "  ∫cos(x) dx = " << render(F2) << "  [" << (ok2 ? "checked" : "WRONG") << "]\n";
        assert(ok2);

        // ∫ sin(x^2) dx = nullptr (no elementary form)
        auto e_sin2 = parse("sin(x^2)");
        auto F3 = ie.integrate(e_sin2);
        assert(F3 == nullptr);
        std::cout << "  ∫sin(x^2) dx = nullptr (honest fail, expected)\n";
    }
    std::cout << "  integral_engine OK\n\n";

    // ── 5. Policy Pack ───────────────────────────────────────────────────────
    std::cout << "--- policy_pack ---\n";
    {
        auto entries = POLICY_PACK_ENTRIES();
        assert(entries.size() >= 17);
        // Check key entries are present
        bool has_force = false, has_density = false, has_moles = false;
        for (const auto& e : entries) {
            if (e.target == "force")   has_force   = true;
            if (e.target == "density") has_density = true;
            if (e.target == "moles")   has_moles   = true;
        }
        assert(has_force && has_density && has_moles);
        std::cout << "  " << entries.size() << " policies loaded (force, density, moles all present)\n";
    }
    std::cout << "  policy_pack OK\n\n";

    // ── 6. NL Query Parser ───────────────────────────────────────────────────
    std::cout << "--- nl_query ---\n";
    {
        std::set<std::string> entities = {"sample", "rocket", "probe"};
        std::vector<std::string> relations = {"force", "mass", "speed", "density",
                                               "ke", "momentum", "pressure",
                                               "moles", "molarity", "volume", "accel"};
        NLQueryParser parser(entities, relations);

        auto [e1, r1, s1] = parser.parse("how much force does the sample have?");
        assert(e1 == "sample" && r1 == "force");
        std::cout << "  'how much force does the sample have?' => entity=" << e1 << " rel=" << r1 << "\n";

        auto [e2, r2, s2] = parser.parse("what is the velocity of the sample?");
        assert(e2 == "sample" && r2 == "speed");
        std::cout << "  'what is the velocity of the sample?' => entity=" << e2 << " rel=" << r2 << " (velocity→speed)\n";

        auto [e3, r3, s3] = parser.parse("compute the kinetic energy of the sample");
        assert(e3 == "sample" && r3 == "ke");
        std::cout << "  'compute the kinetic energy of the sample' => entity=" << e3 << " rel=" << r3 << "\n";

        auto [e4, r4, s4] = parser.parse("how dense is the sample?");
        assert(e4 == "sample" && r4 == "density");
        std::cout << "  'how dense is the sample?' => entity=" << e4 << " rel=" << r4 << "\n";
    }
    std::cout << "  nl_query OK\n\n";

    std::cout << "=== All Math Engines: PASS ===\n";
    return 0;
}
