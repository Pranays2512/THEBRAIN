#!/usr/bin/env python3
"""
brain3/tests/test_novel_theory_generator.py

Unit and integration tests for The Brain's Autonomous Novel Theory Synthesis Engine.
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
THEORY_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "novel_theory_generator"

class TestNovelTheoryGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(THEORY_BIN),
            "crisp/engines/math/novel_theory_generator.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_novel_theory_synthesis(self):
        res = subprocess.run([str(THEORY_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Theory 1: Information-Theoretic Fluid Regularity
        self.assertIn("Information-Theoretic Fisher Curvature Regularity Invariant for 3D Navier-Stokes", out)
        self.assertIn("Fisher Information Curvature I_F(t)", out)
        self.assertIn("Fisher-Enstrophy Balance Equation", out)

        # Theory 2: Non-Hermitian Topological Memory
        self.assertIn("Non-Hermitian Exceptional-Point Topological Protection for Open Quantum Memory", out)
        self.assertIn("Dissipation-Induced Exceptional Point (EP) Spectral Braiding", out)

        # Theory 3: Holographic Quantum Island Hubble Tension
        self.assertIn("Holographic Quantum Island Backreaction & Early Dark Energy Relaxation", out)
        self.assertIn("The 5-sigma Hubble Tension", out)
        self.assertIn("Cosmological Quantum Extremal Island Horizon Transition", out)

        # Theory 4: Holographic Linear Recurrent Accumulator (H2RL)
        self.assertIn("Holographic Linear Recurrent Accumulator (H2RL)", out)
        self.assertIn("The KV-Cache Memory Bandwidth Wall", out)
        self.assertIn("Circular-Convolution Binding in Decayed State-Space Sequence Modeling", out)

        # Completion Banner
        self.assertIn("AUTONOMOUS THEORY SYNTHESIS COMPLETE: 4 NOVEL THEORIES GENERATED & AUDITED", out)

if __name__ == "__main__":
    unittest.main()
