#include "crisp/engines/synthesis/code_gen.hpp"
#include "crisp/engines/synthesis/conjecture_sandbox.hpp"
#include "crisp/engines/synthesis/refuter.hpp"
#include "crisp/engines/synthesis/refute_synth.hpp"
#include "crisp/engines/synthesis/irregularity_detector.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    // 1. code_gen
    brain3::engines::synthesis::ClassSpec spec{"Point", {{"x", "int"}, {"y", "int"}}, {{"distance", {}, "float"}}};
    brain3::engines::synthesis::CodeGen cg;
    std::string cpp_code = cg.generate(spec, "cpp");
    assert(cpp_code.find("class Point") != std::string::npos);
    std::cout << "CodeGen OK (generated C++ stub)\n";

    // 2. conjecture_sandbox
    brain3::engines::synthesis::ConjectureSandbox sandbox;
    auto right_shape = [](double m, double v) { return 0.5 * m * v * v; };
    auto result = sandbox.design_and_test(right_shape);
    assert(result.survived);
    std::cout << "ConjectureSandbox OK (correct shape survived self-designed tests)\n";

    // 3. refuter
    brain3::engines::synthesis::Refuter refuter;
    auto good_max = [](const std::vector<int>& lst) {
        if(lst.empty()) throw std::runtime_error("empty");
        int m = lst[0]; for (int x : lst) if (x > m) m = x; return m;
    };
    auto oracle_max = [](const std::vector<int>& lst) {
        if(lst.empty()) throw std::runtime_error("empty");
        int m = lst[0]; for (int x : lst) if (x > m) m = x; return m;
    };
    auto overfit_max = [](const std::vector<int>& lst) {
        if(lst.empty()) throw std::runtime_error("empty");
        int m = 0; for (int x : lst) if (x > m) m = x; return m;
    };
    auto rep_good = refuter.refute_list(good_max, oracle_max);
    assert(rep_good.robust);
    auto rep_bad = refuter.refute_list(overfit_max, oracle_max);
    assert(!rep_bad.robust);
    assert(rep_bad.scope.find("BREAKS") != std::string::npos);
    std::cout << "Refuter OK (diagnosed overfit vs robust rule)\n";

    // 4. refute_synth
    brain3::engines::synthesis::RefuteSynth rs;
    std::cout << "RefuteSynth OK (compile check)\n";

    // 5. irregularity_detector
    brain3::engines::synthesis::IrregularityDetector det;
    std::vector<std::pair<double, double>> train = {{0.387, 0.241}, {0.723, 0.615}, {1.0, 1.0}};
    std::vector<std::pair<double, double>> holdout = {{1.524, 1.881}};
    auto v = det.assess(train, holdout);
    assert(v.verdict == "REGULAR");
    std::vector<std::pair<double, double>> noise_train = {{1, 74}, {2, 3}};
    std::vector<std::pair<double, double>> noise_holdout = {{3, 99}, {4, 12}};
    auto v2 = det.assess(noise_train, noise_holdout);
    assert(v2.verdict == "IRREGULAR");
    std::cout << "IrregularityDetector OK (distinguished Kepler's law from noise)\n";

    std::cout << "\nPhase 6 Synthesis Compiled and Verified successfully!\n";
    return 0;
}
