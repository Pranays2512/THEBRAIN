#!/usr/bin/env python3
"""
brain3/tests/test_harmonic_analysis_functional_engine.py

Unit tests for The Brain's Continuous Functional & Harmonic Analysis Engine.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"

CPP_TEST_HARNESS = """
#include <iostream>
#include <cassert>
#include <vector>
#include "crisp/engines/math/harmonic_analysis_functional_engine.hpp"

using namespace thebrain::harmonic_analysis;

int main() {
    HarmonicAnalysisFunctionalEngine engine;

    // 1. Critical Sobolev Embedding: H^{1/2}(R^3) ↪ L^3(R^3)
    // 1/p = 1/2 - s/d => 1/3 = 1/2 - 0.5/3 = 1/2 - 1/6 = 2/6 = 1/3
    auto sob1 = engine.verify_sobolev_embedding(3.0, 0.5, 3.0);
    assert(sob1.is_valid_embedding);
    assert(sob1.is_critical_scaling);
    std::cout << "[PASSED] " << sob1.explanation << "\\n";

    // 2. Critical Sobolev Embedding in 2D: H^1(R^2) ↪ L^p for all 2 <= p < inf
    auto sob2 = engine.verify_sobolev_embedding(2.0, 1.0, 4.0);
    assert(sob2.is_valid_embedding);
    std::cout << "[PASSED] " << sob2.explanation << "\\n";

    // 3. Gagliardo-Nirenberg: ‖u‖_{L^4}^2 ≤ C ‖u‖_{L^2} ‖∇u‖_{L^2} (Ladyzhenskaya 2D)
    auto gn = engine.verify_gagliardo_nirenberg(2.0, 4.0, 2.0, 2.0, 1, 0.0);
    assert(gn.is_admissible);
    assert(std::abs(gn.theta - 0.5) < 1e-3);
    std::cout << "[PASSED] " << gn.explanation << "\\n";

    // 4. Besov Space Norm
    std::vector<DyadicFrequencyShell> shells = {
        {0, 1.0, 1.0},
        {1, 0.5, 2.0},
        {2, 0.25, 4.0},
        {3, 0.125, 8.0}
    };
    double besov_norm = engine.compute_besov_norm(shells, 0.5, 2.0, 2.0);
    assert(besov_norm > 0.0);
    std::cout << "[PASSED] Computed Besov B^{1/2}_{2,2} norm: " << besov_norm << "\\n";

    // 5. Beale-Kato-Majda Blowup Evaluator
    std::vector<double> vorticity_smooth = {1.0, 1.2, 1.5, 1.8, 2.0};
    auto bkm_smooth = engine.evaluate_bkm_criterion(vorticity_smooth, 0.1);
    assert(bkm_smooth.global_smoothness_guaranteed);
    assert(!bkm_smooth.enstrophy_diverges);
    std::cout << "[PASSED] " << bkm_smooth.blowup_characterization << "\\n";

    std::cout << "ALL HARMONIC ANALYSIS FUNCTIONAL TESTS PASSED\\n";
    return 0;
}
"""

class TestHarmonicAnalysisFunctionalEngine(unittest.TestCase):
    def test_harmonic_engine_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_file = Path(tmpdir) / "test_harmonic.cpp"
            bin_file = Path(tmpdir) / "test_harmonic_bin"
            cpp_file.write_text(CPP_TEST_HARNESS)

            cmd = [
                "clang++", "-std=c++17", "-O3",
                "-I.", "-I..", "-I../brain2",
                "-Wno-deprecated-declarations",
                "-o", str(bin_file),
                str(cpp_file)
            ]
            res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Compilation failed: {res.stderr}")

            run_res = subprocess.run([str(bin_file)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
            self.assertEqual(run_res.returncode, 0, f"Execution failed: {run_res.stderr}")
            self.assertIn("ALL HARMONIC ANALYSIS FUNCTIONAL TESTS PASSED", run_res.stdout)

if __name__ == "__main__":
    unittest.main()
