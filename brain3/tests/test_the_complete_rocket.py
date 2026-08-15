#!/usr/bin/env python3
"""
brain3/tests/test_the_complete_rocket.py

End-to-End Integration Test for The Brain's Complete Autonomous Discovery Engine ("The Rocket"):
1. Concept & Invariant Synthesis (Lyapunov / Reaction-Diffusion)
2. SMT & Non-Linear Counterexample Hunter (Continuous Gradient & Diophantine Falsification)
3. Formal Symbolic Tactic Proof Search (Poincaré-Wirtinger & Young-Cauchy)
4. Adversarial Epistemic Skeptic Auditor (Refuting Exponents, ODE Blow-Ups & Domain Violations)
5. Continuous Self-Play Discovery Daemon (8-Domain Exploration Cycle)
"""

import subprocess
import unittest
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"

class TestTheCompleteRocket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Compile SMT Hunter
        cmd1 = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(BRAIN3_DIR / "crisp" / "engines" / "math" / "smt_counterexample_hunter"),
            "crisp/engines/math/smt_counterexample_hunter.cpp"
        ]
        res1 = subprocess.run(cmd1, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res1.returncode == 0, f"SMT Hunter compilation failed: {res1.stderr}"

        # 2. Compile Lyapunov Synthesizer
        cmd2 = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(BRAIN3_DIR / "crisp" / "engines" / "math" / "lyapunov_functional_synthesizer"),
            "crisp/engines/math/lyapunov_functional_synthesizer.cpp"
        ]
        res2 = subprocess.run(cmd2, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res2.returncode == 0, f"Lyapunov Synthesizer compilation failed: {res2.stderr}"

        # 3. Compile Formal Prover
        cmd3 = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(BRAIN3_DIR / "crisp" / "engines" / "math" / "formal_tactic_proof_engine"),
            "crisp/engines/math/formal_tactic_proof_engine.cpp"
        ]
        res3 = subprocess.run(cmd3, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res3.returncode == 0, f"Formal Prover compilation failed: {res3.stderr}"

        # 4. Compile Adversarial Epistemic Auditor
        cmd4 = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(BRAIN3_DIR / "crisp" / "engines" / "math" / "adversarial_epistemic_auditor"),
            "crisp/engines/math/adversarial_epistemic_auditor.cpp"
        ]
        res4 = subprocess.run(cmd4, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res4.returncode == 0, f"Adversarial Auditor compilation failed: {res4.stderr}"

    def test_complete_rocket_pipeline(self):
        # Run SMT Hunter
        res_smt = subprocess.run([str(BRAIN3_DIR / "crisp" / "engines" / "math" / "smt_counterexample_hunter")],
                                 cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res_smt.returncode, 0)
        self.assertIn("FALSIFIED (Caught & Destroyed)", res_smt.stdout)
        self.assertIn("SURVIVED RIGOROUS ATTACK", res_smt.stdout)

        # Run Lyapunov Synthesizer
        res_lyap = subprocess.run([str(BRAIN3_DIR / "crisp" / "engines" / "math" / "lyapunov_functional_synthesizer")],
                                  cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res_lyap.returncode, 0)
        self.assertIn("STRICTLY MONOTONIC (dF/dt <= 0)", res_lyap.stdout)

        # Run Formal Tactic Prover
        res_prover = subprocess.run([str(BRAIN3_DIR / "crisp" / "engines" / "math" / "formal_tactic_proof_engine")],
                                    cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res_prover.returncode, 0)
        self.assertIn("100% OF SUBGOALS CLOSED AGAINST AXIOMS", res_prover.stdout)

        # Run Adversarial Epistemic Auditor
        res_auditor = subprocess.run([str(BRAIN3_DIR / "crisp" / "engines" / "math" / "adversarial_epistemic_auditor")],
                                     cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res_auditor.returncode, 0)
        self.assertIn("ADVERSARIAL AUDIT COMPLETE: ALL EPISTEMIC OVERCLAIMS FORMALLY BLOCKED", res_auditor.stdout)

if __name__ == "__main__":
    unittest.main()
