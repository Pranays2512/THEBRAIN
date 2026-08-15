#!/usr/bin/env python3
"""
brain3/tests/test_cross_domain_bridge_builder.py

Unit and integration tests for The Brain's Cross-Domain Isomorphism & Conceptual Bridge Builder:
1. Multi-domain bridge retrieval (Zeta-GUE, Navier-Ricci, Holography, SAT-SpinGlass)
2. Structural concept mapping & bidirectional translation
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
BRIDGE_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "cross_domain_bridge_builder"

class TestCrossDomainBridgeBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(BRIDGE_BIN),
            "crisp/engines/math/cross_domain_bridge_builder.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_bridge_builder_execution(self):
        res = subprocess.run([str(BRIDGE_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Bridges
        self.assertIn("Montgomery-Odlyzko Spectral Pair Correlation Bridge", out)
        self.assertIn("Hydrodynamic Vortex Stretching to Geometric Ricci Curvature", out)
        self.assertIn("Ryu-Takayanagi Holographic Entanglement Entropy Bridge", out)
        self.assertIn("Karp NPC Phase Transitions to Mezard-Parisi Cavity Spin Glass", out)

        # Verify Translation
        self.assertIn("GUE Hermitian matrix eigenvalues lambda_n", out)

        # Verify Overall
        self.assertIn("CROSS-DOMAIN BRIDGE BUILDER READY", out)

if __name__ == "__main__":
    unittest.main()
