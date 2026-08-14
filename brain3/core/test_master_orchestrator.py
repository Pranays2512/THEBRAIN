#!/usr/bin/env python3
"""
Unit and Integration Test Suite for THE BRAIN 3 Master Unified Cognitive Orchestrator
"""

import unittest
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from brain3.core.master_orchestrator import MasterCognitiveOrchestrator

class TestMasterCognitiveOrchestrator(unittest.TestCase):

    def setUp(self):
        self.orchestrator = MasterCognitiveOrchestrator(base_dir=".")

    def tearDown(self):
        self.orchestrator.close()

    def test_01_intent_parsing(self):
        """Test natural language intent classification."""
        p1 = self.orchestrator.parse_user_intent("290 / 2")
        self.assertEqual(p1["intent"], "math_reflex")

        p2 = self.orchestrator.parse_user_intent("What if heat causes expansion?")
        self.assertEqual(p2["intent"], "causal_reasoning")

        p3 = self.orchestrator.parse_user_intent("Compare bird to airplane")
        self.assertEqual(p3["intent"], "analogy_synthesis")

        p4 = self.orchestrator.parse_user_intent("Remember that dolphin is a mammal")
        self.assertEqual(p4["intent"], "teach_fact")

        p5 = self.orchestrator.parse_user_intent("Plan how to synthesize medicine")
        self.assertEqual(p5["intent"], "hierarchical_planning")

    def test_02_end_to_end_cognitive_cycle(self):
        """Test full cognitive loop from user input to fluent response."""
        # 1. Math query
        res1 = self.orchestrator.execute_cognitive_cycle("2+2")
        self.assertIn("4", res1["fluent_response"])
        self.assertLess(res1["latency_ms"], 50.0)

        # 2. Teaching query
        res2 = self.orchestrator.execute_cognitive_cycle("Remember that eagle is a raptor")
        self.assertIn("eagle", res2["fluent_response"])

        # 3. Safety query
        res3 = self.orchestrator.execute_cognitive_cycle("1=0")
        self.assertTrue(res3["is_alarm"])
        self.assertIn("Alarm", res3["fluent_response"])

    def test_03_sleep_consolidation(self):
        """Test autonomous sleep cycle triggering through the orchestrator."""
        sleep_stats = self.orchestrator.run_sleep_cycle()
        self.assertEqual(sleep_stats["status"], "consolidated")
        self.assertLess(sleep_stats["latency_ms"], 500.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
