#!/usr/bin/env python3
"""
Unit and Integration Test Suite for THE BRAIN 3 Autonomous Curiosity & Dreaming Engine
"""

import unittest
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from brain3.curiosity.autonomous_dreaming import AutonomousDreamingEngine

class TestAutonomousDreaming(unittest.TestCase):

    def setUp(self):
        self.dreamer = AutonomousDreamingEngine(base_dir=".")

    def tearDown(self):
        self.dreamer.close()

    def test_01_epistemic_gap_detection(self):
        """Test scanning and detection of transitive reasoning gaps in knowledge graphs."""
        gaps = self.dreamer.scan_epistemic_gaps()
        self.assertGreater(len(gaps), 0)
        self.assertIn(("eagle", "has", "wings"), gaps)
        self.assertIn(("heart", "part_of", "organism"), gaps)

    def test_02_hypothesis_audit_and_discovery(self):
        """Test auditing hypotheses through the Metacognitive Refuter Gate."""
        candidate_gaps = [
            ("eagle", "has", "wings"),
            ("chloroplast", "provides", "energy")
        ]
        verified = self.dreamer.formulate_and_audit_hypotheses(candidate_gaps)
        self.assertEqual(len(verified), 2)
        self.assertGreaterEqual(len(self.dreamer.discovered_rules), 2)

    def test_03_autonomous_dream_cycle(self):
        """Test full dream consolidation and curiosity decay cycle."""
        initial_tension = self.dreamer.curiosity_tension
        self.dreamer.run_dream_consolidation_cycle(cycle_num=1)
        self.assertLess(self.dreamer.curiosity_tension, initial_tension)


if __name__ == "__main__":
    unittest.main(verbosity=2)
