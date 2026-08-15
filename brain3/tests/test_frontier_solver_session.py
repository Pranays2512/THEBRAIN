#!/usr/bin/env python3
"""
brain3/tests/test_frontier_solver_session.py

Unit and integration tests for The Brain's Autonomous Frontier Solver Session:
1. Collatz Conjecture Investigation
2. Erdős-Straus Mordell Open Classes (mod 840) Exact Constructive Solving
3. 3D Navier-Stokes Regularity Analysis
4. Goldbach Prime Representation Verification
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
SESSION_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "autonomous_frontier_solver_session"

class TestFrontierSolverSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(SESSION_BIN),
            "crisp/engines/math/autonomous_frontier_solver_session.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_frontier_solver_session_execution(self):
        res = subprocess.run([str(SESSION_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Collatz
        self.assertIn("The Collatz (3x + 1) Conjecture", out)
        self.assertIn("HEURISTIC_CONTRACTION_MODEL", out)

        # Verify Erdős-Straus
        self.assertIn("The Erdős-Straus Conjecture on Mordell's Open Residue Classes (mod 840)", out)
        self.assertIn("100% of tested hard primes in Mordell open classes solved constructively", out)
        self.assertIn("CONSTRUCTIVE_EXACT_SOLVER", out)

        # Verify Navier-Stokes
        self.assertIn("3D Incompressible Navier-Stokes Global Smoothness", out)
        self.assertIn("CONDITIONAL_TORUS_PROVEN / R^3_OPEN", out)

        # Verify Goldbach
        self.assertIn("The Goldbach Conjecture", out)
        self.assertIn("Verified: All 24,999 even integers from 4 to 50,000 decompose into two primes", out)

        # Verify Overall Completion
        self.assertIn("FRONTIER SOLVER SESSION COMPLETE", out)

if __name__ == "__main__":
    unittest.main()
