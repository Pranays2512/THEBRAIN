#!/usr/bin/env python3
"""
brain3/tests/test_neural_policy_value_prior_engine.py

Unit tests for The Brain's Neural Policy and Value Prior Guidance Engine.
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
#include "crisp/engines/math/neural_policy_value_prior_engine.hpp"

using namespace thebrain::neural_prior;

int main() {
    NeuralPolicyValuePriorEngine prior_engine;

    // 1. Vectorize Proof State AST
    auto emb = prior_engine.embed_proof_state("goal_navier_stokes_L5", 6, 4, 18.5, "MATHEMATICS");
    assert(emb.feature_vector.size() == 8);
    assert(emb.domain_embedding_weight > 1.0);
    std::cout << "[PASSED] Embedded goal AST to 8-dim feature vector\\n";

    // 2. Rank candidate tactics & premises
    std::vector<std::pair<std::string, std::string>> candidates = {
        {"ring", "algebra_identities"},
        {"linarith", "sobolev_poincare_bound"},
        {"apply", "gagliardo_nirenberg_3d"},
        {"exact", "critical_sobolev_embedding_3d"},
        {"simp", "trigonometric_identities"}
    };

    auto ranked = prior_engine.rank_candidate_actions(emb, candidates, 3);
    assert(ranked.size() == 3);
    assert(ranked[0].combined_score >= ranked[1].combined_score);
    assert(ranked[1].combined_score >= ranked[2].combined_score);

    std::cout << "[PASSED] Top-1 Candidate: Tactic [" << ranked[0].tactic_name 
              << "] with Prior P(a|s) = " << ranked[0].policy_prior_prob 
              << ", Value V(s') = " << ranked[0].value_estimate 
              << ", Combined = " << ranked[0].combined_score << "\\n";

    std::cout << "ALL NEURAL POLICY VALUE PRIOR TESTS PASSED\\n";
    return 0;
}
"""

class TestNeuralPolicyValuePriorEngine(unittest.TestCase):
    def test_prior_engine_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_file = Path(tmpdir) / "test_prior.cpp"
            bin_file = Path(tmpdir) / "test_prior_bin"
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
            self.assertIn("ALL NEURAL POLICY VALUE PRIOR TESTS PASSED", run_res.stdout)

if __name__ == "__main__":
    unittest.main()
