#!/usr/bin/env python3
"""
brain3/tests/test_hierarchical_goal_decomposer.py

Unit and integration tests for The Brain's General Hierarchical Goal Decomposer:
1. Navier-Stokes Millennium Lemma DAG decomposition
2. Quantum Black Hole Page Curve Lemma DAG decomposition
3. Critical bottleneck identification & DAG validation
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
DECOMPOSER_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "hierarchical_goal_decomposer"

class TestHierarchicalGoalDecomposer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(DECOMPOSER_BIN),
            "crisp/engines/math/hierarchical_goal_decomposer.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_decomposer_execution(self):
        res = subprocess.run([str(DECOMPOSER_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Navier-Stokes
        self.assertIn("3D Incompressible Navier-Stokes Global Regularity", out)
        self.assertIn("Leray Global Energy Dissipation Inequality", out)
        self.assertIn("Beale-Kato-Majda Finite Time Singularity Criterion", out)
        self.assertIn("Critical Bottleneck: ns_L5_large_data_regularity_R3", out)

        # Verify Black Hole Paradox
        self.assertIn("Unitary Evaporation & Page Curve", out)
        self.assertIn("Quantum Extremal Surface Island", out)

        # Verify Overall
        self.assertIn("GOAL DECOMPOSER READY", out)

if __name__ == "__main__":
    unittest.main()
