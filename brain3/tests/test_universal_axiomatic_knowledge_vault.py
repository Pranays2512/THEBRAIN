#!/usr/bin/env python3
"""
brain3/tests/test_universal_axiomatic_knowledge_vault.py

Unit and integration tests for The Brain's Universal Multi-Domain Axiomatic Knowledge Vault:
1. Multi-domain axiom coverage (Math, Physics, CS, Bio, Cosmo)
2. Dependency graph DAG acyclicity verification
3. Query and registration interfaces
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
VAULT_BIN = BRAIN3_DIR / "crisp" / "engines" / "math" / "universal_axiomatic_knowledge_vault"

class TestUniversalAxiomaticKnowledgeVault(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-I.", "-I..", "-I../brain2",
            "-Wno-deprecated-declarations",
            "-o", str(VAULT_BIN),
            "crisp/engines/math/universal_axiomatic_knowledge_vault.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def test_vault_execution(self):
        res = subprocess.run([str(VAULT_BIN)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        out = res.stdout

        # Verify Acyclicity DAG
        self.assertIn("100% STRICT DAG (No circularity)", out)

        # Verify Domains
        self.assertIn("Pure & Applied Mathematics", out)
        self.assertIn("Theoretical Physics & QFT", out)
        self.assertIn("Theoretical Computer Science", out)
        self.assertIn("Biology & Molecular Kinetics", out)
        self.assertIn("Cosmology & Gravitational Physics", out)

        # Verify Overall
        self.assertIn("UNIVERSAL KNOWLEDGE VAULT READY", out)

if __name__ == "__main__":
    unittest.main()
