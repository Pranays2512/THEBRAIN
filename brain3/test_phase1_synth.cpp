#include "crisp/engines/synthesis/brain_codegen.hpp"
#include "crisp/engines/synthesis/math_synth.hpp"
#include "crisp/engines/synthesis/refuter.hpp"
#include "fuzzy/engines/synthesis/unified_proposer.hpp"
#include <iostream>
#include <cassert>

int main() {
    brain3::engines::synthesis::BrainCodeGen bcg;
    brain3::engines::synthesis::LearnedArithmetic la;
    brain3::engines::synthesis::Refuter refuter;
    brain3::engines::synthesis::UnifiedProposer up;

    brain3::engines::synthesis::Problem good;
    good.type = "conjecture";
    good.variables = {"m", "v"};
    good.law_name = "kinetic_energy";
    good.lhs = "KE";
    good.rhs = "0.5*m*v^2";
    good.test_fn = [](const std::map<std::string, double>& x) {
        return 0.5 * x.at("m") * x.at("v") * x.at("v");
    };
    good.trusted_fn = good.test_fn;
    assert(up.solve(good));

    brain3::engines::synthesis::Problem bad = good;
    bad.law_name = "bad_kinetic_energy";
    bad.test_fn = [](const std::map<std::string, double>& x) {
        return x.at("m") * x.at("v") * x.at("v") * x.at("v");
    };
    assert(!up.solve(bad));
    
    std::cout << "Phase 1 Synthesis Compiled successfully!" << std::endl;
    return 0;
}
