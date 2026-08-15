#!/usr/bin/env python3
"""
brain3/tests/test_universal_self_play_bootstrap.py

Unit and integration tests for The Brain's Continuous Universal Self-Play Bootstrap Engine:
1. End-to-end multi-disciplinary discovery cycles
2. Seamless integration of Engines 1-4 + CAS + Adversarial Auditor
3. Policy crystallization without unverified hallucination
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
BOOTSTRAP_BIN = BRAIN3_DIR / "core" / "universal_self_play_bootstrap_engine"

class TestUniversalSelfPlayBootstrapEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(BOOTSTRAP_BIN),
            "core/universal_self_play_bootstrap_engine.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_self_play_bootstrap_execution(self):
        res = subprocess.run([str(BOOTSTRAP_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Cycle 1
        self.assertIn("3D Incompressible Navier-Stokes Global Regularity", out)
        self.assertIn("CRYSTALLIZED_SUB_LEMMAS (Instance & Torus Proven; Open R^3 Disclaimed)", out)

        # Verify Cycle 2
        self.assertIn("Unitary Evaporation & Page Curve in Quantum Black Hole Thermodynamics", out)
        self.assertIn("CRYSTALLIZED_IN_POLICY_STORE", out)

        # Verify Overall
        self.assertIn("UNIVERSAL SELF-PLAY BOOTSTRAP ENGINE READY: 2 DISCOVERY CYCLES COMPLETED", out)

if __name__ == "__main__":
    unittest.main()
