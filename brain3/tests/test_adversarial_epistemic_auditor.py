#!/usr/bin/env python3
"""
brain3/tests/test_adversarial_epistemic_auditor.py

Unit and integration tests for The Brain's Adversarial Epistemic Auditor & Skeptic Gate.
Verifies that all overclaimed and mathematically flawed arguments are successfully refuted.
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
AUDITOR_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "adversarial_epistemic_auditor"

class TestAdversarialEpistemicAuditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(AUDITOR_BIN),
            "crisp/engines/math/adversarial_epistemic_auditor.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_adversarial_auditor_execution(self):
        res = subprocess.run([str(AUDITOR_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # 1. Verify Navier-Stokes audit
        self.assertIn("3D Incompressible Navier-Stokes Enstrophy Dissipation Dominance Claim", out)
        self.assertIn("EXPONENT ARITHMETIC ERROR", out)
        self.assertIn("STRUCTURAL ODE BLOW-UP ERROR", out)
        self.assertIn("DOMAIN POINCARÉ VIOLATION", out)

        # 2. Verify Collatz audit
        self.assertIn("Collatz 2-Adic Haar Measure Convergence Claim", out)
        self.assertIn("MEASURE-ZERO CATEGORY ERROR", out)
        self.assertIn("CORRELATED ORBIT VIOLATION", out)

        # 3. Verify Erdős-Straus audit
        self.assertIn("Erdős-Straus Modulo Residue Basis Classification", out)
        self.assertIn("Mordell (1967) proved the open classes reduce to n ≡ {1, 121, 169, 289, 361, 529} (mod 840)", out)

        # 4. Verify P vs NP audit
        self.assertIn("P vs NP Circuit Complexity Separation Claim", out)
        self.assertIn("NATURAL PROOFS BARRIER", out)
        self.assertIn("ALGEBRIZATION BARRIER", out)

        # 5. Overall verification
        self.assertIn("ADVERSARIAL AUDIT COMPLETE: ALL EPISTEMIC OVERCLAIMS FORMALLY BLOCKED", out)

if __name__ == "__main__":
    unittest.main()
