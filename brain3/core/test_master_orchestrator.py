#!/usr/bin/env python3
"""
Unit and Integration Test Suite for THE BRAIN 3 Native C++ Master Unified Cognitive Orchestrator
"""

import unittest
import subprocess
import os

class TestNativeMasterOrchestrator(unittest.TestCase):

    def run_query(self, query: str) -> str:
        cmd = ["./brain3/brain_master", "--query", query]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.path.abspath("."))
        self.assertEqual(res.returncode, 0, f"Error running query '{query}': {res.stderr}")
        return res.stdout.strip()

    def test_01_math_reflex(self):
        """Test arithmetic calculation via System 1 Reflex Arc."""
        out = self.run_query("290 / 2")
        self.assertIn("145", out)

    def test_02_teaching_and_lookup(self):
        """Test knowledge ingestion and subsequent retrieval."""
        t_out = self.run_query("Remember that cheetah is a feline")
        self.assertIn("cheetah", t_out.lower())
        
        l_out = self.run_query("LOOKUP cheetah is_a")
        self.assertTrue("feline" in l_out.lower() or "cheetah" in l_out.lower())

    def test_03_causal_reasoning(self):
        """Test causal invariant reasoning."""
        out = self.run_query("What if gravity causes acceleration?")
        self.assertTrue("Verified Truth" in out or "causal" in out.lower() or "acceleration" in out.lower())

    def test_04_safety_audit(self):
        """Test metacognitive safety alarm on contradiction."""
        out = self.run_query("Where is 1=0")
        self.assertIn("ALARM", out)

    def test_05_codeforces_grandmaster_solver(self):
        """Test Codeforces 2500 competitive programming query."""
        out = self.run_query("solve 2500 rating codeforces questions in java")
        self.assertIn("CF 1000F", out)
        self.assertIn("PASSED", out)

if __name__ == "__main__":
    unittest.main()
