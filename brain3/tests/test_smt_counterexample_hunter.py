#!/usr/bin/env python3
"""
brain3/tests/test_smt_counterexample_hunter.py

Unit and integration tests for The Brain's SMT & Non-Linear Counterexample Hunter:
1. Fast continuous gradient descent falsification
2. Discrete Diophantine / prime modular falsification
3. Invariant survival verification
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
HUNTER_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "smt_counterexample_hunter"

class TestSMTCounterexampleHunter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(HUNTER_BIN),
            "crisp/engines/math/smt_counterexample_hunter.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_smt_hunter_execution(self):
        res = subprocess.run([str(HUNTER_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Test 1 (False continuous inequality caught)
        self.assertIn("x^4 + y^4 - 4xy + 0.5 >= 0", out)
        self.assertIn("FALSIFIED (Caught & Destroyed)", out)

        # Verify Test 2 (True invariant survives)
        self.assertIn("exp(x) - 1 - x >= 0", out)
        self.assertIn("SURVIVED RIGOROUS ATTACK", out)

        # Verify Test 3 (Euler polynomial false conjecture caught at n=40/41)
        self.assertIn("Euler Polynomial Prime", out)
        self.assertIn("FALSIFIED (Euler Polynomial Breaks)", out)

        # Overall completion
        self.assertIn("SMT & NON-LINEAR COUNTEREXAMPLE HUNTER VALIDATION COMPLETE", out)

if __name__ == "__main__":
    unittest.main()
