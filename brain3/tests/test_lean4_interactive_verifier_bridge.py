#!/usr/bin/env python3
"""
brain3/tests/test_lean4_interactive_verifier_bridge.py

Unit tests for The Brain's Interactive Lean 4 / Formal Verifier IPC Bridge.
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
#include "crisp/engines/math/lean4_interactive_verifier_bridge.hpp"

using namespace thebrain::lean4_bridge;

int main() {
    Lean4InteractiveVerifierBridge bridge;

    // 1. Build a valid Lean 4 proof script
    std::vector<std::string> tactics = {
        "intro x",
        "exact HasFDerivAt.differentiableAt hasFDerivAt_const"
    };
    auto script = bridge.build_proof_script("differentiableAt_const", "{E F : Type*} (x : E)", tactics);
    assert(script.complete_lean_code.find("theorem differentiableAt_const") != std::string::npos);
    std::cout << "[PASSED] Generated Lean 4 code:\\n" << script.complete_lean_code << "\\n";

    // 2. Verify proof script via IPC bridge
    auto resp = bridge.verify_proof_script(script);
    assert(resp.is_valid_proof);
    assert(resp.all_goals_closed);
    assert(resp.open_goals_count == 0);
    std::cout << "[PASSED] Verified proof script in " << resp.verification_time_ms << " ms\\n";

    // 3. Test detection of 'sorry' placeholder
    std::vector<std::string> invalid_tactics = {"sorry"};
    auto invalid_script = bridge.build_proof_script("incomplete_theorem", "∀ (x : Nat), x = x", invalid_tactics);
    auto invalid_resp = bridge.verify_proof_script(invalid_script);
    assert(!invalid_resp.is_valid_proof);
    assert(!invalid_resp.all_goals_closed);
    assert(invalid_resp.open_goals_count == 1);
    std::cout << "[PASSED] Correctly flagged incomplete proof with 'sorry' placeholder\\n";

    std::cout << "ALL LEAN4 INTERACTIVE VERIFIER BRIDGE TESTS PASSED\\n";
    return 0;
}
"""

class TestLean4InteractiveVerifierBridge(unittest.TestCase):
    def test_bridge_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_file = Path(tmpdir) / "test_bridge.cpp"
            bin_file = Path(tmpdir) / "test_bridge_bin"
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
            self.assertIn("ALL LEAN4 INTERACTIVE VERIFIER BRIDGE TESTS PASSED", run_res.stdout)

if __name__ == "__main__":
    unittest.main()
