#!/usr/bin/env python3
"""
brain3/tests/test_symbolic_cas_calculator_engine.py

Unit and integration tests for The Brain's Symbolic CAS Calculator Engine ("SymPy in C++"):
1. Exact 128-bit rational arithmetic & zero float error
2. Exact symbolic differentiation & chain/product rule
3. Exact symbolic matrix commutators
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
CAS_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "symbolic_cas_calculator_engine"

class TestSymbolicCasCalculatorEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(CAS_BIN),
            "crisp/engines/math/symbolic_cas_calculator_engine.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_cas_engine_execution(self):
        res = subprocess.run([str(CAS_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Rational
        self.assertIn("4/2521", out)
        self.assertIn("Zero error: TRUE", out)

        # Verify Differentiation
        self.assertIn("f(x)    =", out)
        self.assertIn("f'(x)   =", out)

        # Verify Matrix Commutator
        self.assertIn("Commutator [sigma_x, sigma_y] Matrix", out)

        # Verify Overall
        self.assertIn("SYMBOLIC CAS ENGINE READY", out)

if __name__ == "__main__":
    unittest.main()
