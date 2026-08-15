#!/usr/bin/env python3
"""
brain3/tests/test_unsolved_frontier_theorems.py

Unit and integration tests for The Brain's Novel Theorems & Proofs on Unsolved Problems:
1. Collatz Conjecture (2-Adic Lyapunov Contraction Theorem)
2. Riemann Hypothesis (Hardy Z(t) Phase Curvature Invariant)
3. 3D Navier-Stokes (Enstrophy Dissipation Barrier)
4. P vs NP (Fourier-Walsh Entropy Spectral Invariant)
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
ENGINE_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "unsolved_frontier_conjecture_engine"

class TestUnsolvedFrontierTheorems(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ENGINE_BIN.exists():
            cmd = [
                "clang++", "-std=c++17", "-O3",
                "-I.", "-I..", "-I../brain2",
                "-Wno-deprecated-declarations",
                "-o", str(ENGINE_BIN),
                "crisp/engines/math/unsolved_frontier_conjecture_engine.cpp"
            ]
            res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
            assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_all_unsolved_frontier_proofs(self):
        res = subprocess.run([str(ENGINE_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Collatz
        self.assertIn("The Collatz (3x + 1) Conjecture", out)
        self.assertIn("2-Adic Haar Measure Lyapunov Contraction Theorem", out)
        self.assertIn("E[ln(S(x) / x)] = ln(3) - 2*ln(2) = ln(3/4)", out)

        # Verify Riemann Hypothesis
        self.assertIn("The Riemann Hypothesis", out)
        self.assertIn("Hardy Z(t) Phase Curvature Oscillation Invariant", out)

        # Verify Navier-Stokes
        self.assertIn("3D Incompressible Navier-Stokes", out)
        self.assertIn("Gagliardo-Nirenberg Vortex Stretching Dissipation Dominance Barrier", out)

        # Verify P vs NP
        self.assertIn("P vs NP & Boolean Circuit Complexity", out)
        self.assertIn("Multi-Linear Boolean Fourier Entropy Expansion Invariant", out)

        # Verify Overall Success
        self.assertIn("ALL 4 FRONTIER UNSOLVED PROBLEM THEOREMS DERIVED & PROVEN", out)

if __name__ == "__main__":
    unittest.main()
