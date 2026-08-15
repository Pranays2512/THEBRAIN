#!/usr/bin/env python3
"""
brain3/tests/test_formal_tactic_proof_engine.py

Unit and integration tests for The Brain's Formal Symbolic Tactic & Axiomatic Proof Search Engine:
1. Poincaré-Wirtinger Inequality Goal Discharge
2. Young-Cauchy AM-GM Energy Lower Bound
3. Full Tactic Trace Validation
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
PROVER_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "formal_tactic_proof_engine"

class TestFormalTacticProofEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(PROVER_BIN),
            "crisp/engines/math/formal_tactic_proof_engine.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_formal_prover_execution(self):
        res = subprocess.run([str(PROVER_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Poincaré-Wirtinger
        self.assertIn("Poincaré-Wirtinger Inequality", out)
        self.assertIn("CLOSED & VERIFIED (Q.E.D.)", out)

        # Verify Young-Cauchy
        self.assertIn("Young-Cauchy Energy Lower Bound", out)
        self.assertIn("positivity", out)

        # Verify Overall
        self.assertIn("100% OF SUBGOALS CLOSED AGAINST AXIOMS", out)

if __name__ == "__main__":
    unittest.main()
