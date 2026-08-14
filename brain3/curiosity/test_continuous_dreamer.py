#!/usr/bin/env python3
"""
brain3/curiosity/test_continuous_dreamer.py

Unit tests for Pillar 2: 24/7 Autonomous Epistemic Dreaming & Self-Play Prover.
"""

import unittest
import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from brain3.curiosity.continuous_dreamer import ContinuousEpistemicDreamer

class TestContinuousEpistemicDreamer(unittest.TestCase):

    def test_01_dream_cycle_execution(self):
        """Test a complete 4-phase autonomous epistemic dreaming session."""
        dreamer = ContinuousEpistemicDreamer()
        try:
            res = dreamer.run_dream_cycle(max_theorems=6)
            self.assertIsInstance(res, dict)
            self.assertGreaterEqual(res.get("theorems_synthesized", 0), 1)
            self.assertGreaterEqual(res.get("total_reflexes", 0), 1)
            self.assertEqual(dreamer.stats["dreams_completed"], 1)
        finally:
            dreamer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
