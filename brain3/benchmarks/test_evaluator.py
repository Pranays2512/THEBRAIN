#!/usr/bin/env python3
"""
Unit and Integration Test Suite for THE BRAIN 3 Benchmark Evaluator
"""

import unittest
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from brain3.benchmarks.benchmark_evaluator import BenchmarkEvaluator

class TestBenchmarkEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = BenchmarkEvaluator(base_dir=".")

    def tearDown(self):
        self.evaluator.close()

    def test_01_contradiction_safety_evaluation(self):
        """Test the Metacognitive Refuter and Contradiction Safety Gate evaluation."""
        stats = self.evaluator.eval_contradiction_safety(num_probes=10)
        self.assertEqual(stats["samples"], 10)
        self.assertGreaterEqual(stats["accuracy_pct"], 90.0)
        self.assertLess(stats["avg_latency_ms"], 50.0)

    def test_02_mini_benchmark_run(self):
        """Test mini evaluation run across SciQ, GSM8K, and SVAMP."""
        sciq_stats = self.evaluator.eval_sciq(num_samples=5)
        self.assertIn("accuracy_pct", sciq_stats)
        self.assertEqual(sciq_stats["samples"], 5)

        gsm_stats = self.evaluator.eval_gsm8k(num_samples=5)
        self.assertIn("accuracy_pct", gsm_stats)
        self.assertEqual(gsm_stats["samples"], 5)

        svamp_stats = self.evaluator.eval_svamp(num_samples=5)
        self.assertIn("accuracy_pct", svamp_stats)
        self.assertEqual(svamp_stats["samples"], 5)

    def test_03_scorecard_generation(self):
        """Test full scorecard computation and aggregation."""
        self.evaluator.eval_contradiction_safety(num_probes=6)
        self.assertIn("safety", self.evaluator.results)
        self.evaluator.print_scorecard()


if __name__ == "__main__":
    unittest.main(verbosity=2)
