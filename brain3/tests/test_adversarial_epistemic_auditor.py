#!/usr/bin/env python3
"""
brain3/tests/test_adversarial_epistemic_auditor.py

Unit tests for The Brain's Adversarial Epistemic Auditor & Skeptic Pass:
- Catches Navier-Stokes enstrophy exponent & ODE blow-up errors
- Catches Collatz Haar measure 0 / i.i.d. leap errors
- Catches Erdős-Straus modulo 24 vs 840 errors
- Catches P vs NP Natural Proofs barrier violations
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

        # Verify Navier-Stokes refutation
        self.assertIn("3D Incompressible Navier-Stokes Enstrophy Dissipation Dominance Claim", out)
        self.assertIn("EXPONENT ARITHMETIC ERROR", out)
        self.assertIn("STRUCTURAL ODE BLOW-UP ERROR", out)
        self.assertIn("DOMAIN POINCARÉ VIOLATION", out)

        # Verify Collatz refutation
        self.assertIn("Collatz 2-Adic Haar Measure Convergence Claim", out)
        self.assertIn("MEASURE-ZERO CATEGORY ERROR", out)

        # Verify Erdős-Straus modulo 840 check
        self.assertIn("Erdős-Straus Unresolved Prime Modulo Classification", out)
        self.assertIn("IMPRECISE_MODULO_CLASSIFICATION", out)
        self.assertIn("840", out)

        # Verify P vs NP Natural Proofs Barrier
        self.assertIn("P vs NP Multi-Linear Fourier Entropy Lower Bound Claim", out)
        self.assertIn("NATURAL PROOFS BARRIER", out)

        # Verify Overall Success
        self.assertIn("ADVERSARIAL AUDIT COMPLETE: ALL EPISTEMIC OVERCLAIMS FORMALLY BLOCKED", out)

if __name__ == "__main__":
    unittest.main()
