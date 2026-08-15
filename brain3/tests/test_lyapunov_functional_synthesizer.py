#!/usr/bin/env python3
"""
brain3/tests/test_lyapunov_functional_synthesizer.py

Unit and integration tests for The Brain's Lyapunov & Monotonic Energy Functional Synthesizer:
1. Allen-Cahn PDE energy dissipation
2. Non-linear Duffing oscillator LaSalle invariance proof
3. Polynomial coupled vector field Lyapunov stability
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
SYNTH_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "lyapunov_functional_synthesizer"

class TestLyapunovFunctionalSynthesizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(SYNTH_BIN),
            "crisp/engines/math/lyapunov_functional_synthesizer.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_lyapunov_synthesis_execution(self):
        res = subprocess.run([str(SYNTH_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Allen-Cahn PDE
        self.assertIn("Allen-Cahn Reaction-Diffusion PDE", out)
        self.assertIn("dF/dt = - \\int_\\Omega |u_t|^2 dx = - ||u_t||_{L^2}^2 <= 0", out)
        self.assertIn("GLOBAL GRADIENT FLOW DISSIPATION", out)

        # Verify Duffing
        self.assertIn("Damped Duffing Oscillator", out)
        self.assertIn("GLOBALLY ASYMPTOTICALLY STABLE", out)

        # Verify Overall
        self.assertIn("LYAPUNOV ENERGY FUNCTIONAL SYNTHESIS COMPLETE", out)

if __name__ == "__main__":
    unittest.main()
