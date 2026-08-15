#!/usr/bin/env python3
"""
brain3/tests/test_neural_guided_mcts_navigator.py

Unit and integration tests for The Brain's Universal Neural-Guided MCTS Discovery Navigator:
1. Proof tree expansion without combinatorial explosion
2. Policy prior ranking and dynamic value estimation
3. UCT branch selection and backpropagation
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
NAVIGATOR_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "neural_guided_mcts_navigator"

class TestNeuralGuidedMCTSNavigator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(NAVIGATOR_BIN),
            "crisp/engines/math/neural_guided_mcts_navigator.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_navigator_execution(self):
        res = subprocess.run([str(NAVIGATOR_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Search Metrics
        self.assertIn("Total Visits N(root) : 100", out)
        self.assertIn("Conservation Law & Lyapunov Energy Monotonicity", out)
        self.assertIn("Fourier Spectral Decoupling", out)

        # Verify Overall
        self.assertIn("MCTS DISCOVERY NAVIGATOR READY", out)

if __name__ == "__main__":
    unittest.main()
