#include "crisp/engines/synthesis/logic_plan.hpp"
#include "crisp/engines/synthesis/online_proposer2.hpp"
#include "crisp/engines/synthesis/policy_proposer.hpp"
#include "crisp/engines/synthesis/inductive_engine.hpp"
#include "crisp/engines/synthesis/calculus_synth.hpp"
#include <iostream>
#include <cassert>

int main() {
    // ── 1. LogicPlan: prompt generation ──────────────────────────────────────
    const auto& reg = brain3::engines::synthesis::plan_registry();
    assert(reg.count("binary_search"));
    auto prompt = reg.at("binary_search").to_prompt("cpp");
    assert(prompt.find("binary_search") != std::string::npos);
    std::cout << "LogicPlan OK (" << reg.size() << " plans in registry)\n";

    // ── 2. OnlineProposer2: feature signature + ordering ─────────────────────
    brain3::engines::synthesis::OnlineProposer2 op2;
    brain3::engines::synthesis::TaskSig sig = {"scalar", "out_scalar"};
    auto ordered = op2.order("int1", sig);
    assert(!ordered.empty());
    op2.reward(sig, ordered[0]);
    op2.penalize(sig, ordered[0]);
    std::cout << "OnlineProposer2 OK (ordered " << ordered.size() << " spaces)\n";

    // ── 3. PolicyProposer: groundability scoring ──────────────────────────────
    brain3::engines::synthesis::MultiPolicyMemory mem;
    mem.add({"force", {"mass", "accel"}, "mass*accel",
        [](const std::map<std::string, double>& e){ return e.at("mass") * e.at("accel"); }});
    mem.add({"power", {"force", "speed"}, "force*speed",
        [](const std::map<std::string, double>& e){ return e.at("force") * e.at("speed"); }});

    std::map<std::string, double> facts = {{"mass", 1000}, {"accel", 20}, {"speed", 300}};
    brain3::engines::synthesis::PolicyProposerSolver solver(facts, mem, true);
    auto* ans = solver.solve("power");
    assert(ans != nullptr);
    std::cout << "PolicyProposer OK (power = " << *ans << ", work = " << solver.work << ")\n";

    // ── 4. InductiveLearner: rule mining + promotion ──────────────────────────
    brain3::engines::synthesis::InductiveLearner il;
    std::vector<std::vector<std::string>> train = {
        {"rain", "wet_ground", "puddles"},
        {"rain", "wet_ground", "puddles"},
        {"rain", "wet_ground", "puddles"},
        {"study", "pass"}, {"study", "pass"}, {"study", "pass"},
        {"cat", "rainbow"}, {"cat", "rainbow"},
    };
    std::vector<std::vector<std::string>> test_ep = {
        {"rain", "wet_ground", "puddles"},
        {"rain", "wet_ground", "puddles"},
        {"study", "pass"}, {"study", "pass"},
        {"cat", "cloud"}, {"cat", "wind"},
    };
    auto result = il.mine(train, test_ep);
    std::cout << "InductiveLearner OK: promoted=" << result.promoted.size()
              << " rejected=" << result.rejected.size() << "\n";
    for (auto& r : result.promoted)
        std::cout << "  " << r.a << " -> " << r.b
                  << " (train=" << r.conf_train << " test=" << r.conf_test << ")\n";

    // ── 5. CalculusSynth: discover differentiation rules ─────────────────────
    brain3::engines::synthesis::CalculusSynth cs;
    cs.learn(true);
    assert(cs.learned_rules.count("power"));
    assert(cs.learned_rules.count("sin"));
    assert(cs.learned_rules.count("cos"));
    assert(cs.learned_rules.count("exp"));
    assert(cs.learned_rules.count("ln"));
    std::cout << "\nCalculusSynth OK (" << cs.learned_rules.size() << " rules learned)\n";

    // Verify: d/dx(x^3) at x=2 should be 3*x^2 = 12
    auto x3 = brain3::engines::synthesis::binop("^", brain3::engines::synthesis::var(), brain3::engines::synthesis::lit(3));
    double num_d = brain3::engines::synthesis::numerical_diff(x3, 2.0);
    std::cout << "  numerical d/dx(x^3) at x=2: " << num_d << " (expected ~12)\n";
    assert(std::abs(num_d - 12.0) < 0.001);

    std::cout << "\nPhase 5 Synthesis Compiled and Verified successfully!\n";
    return 0;
}
