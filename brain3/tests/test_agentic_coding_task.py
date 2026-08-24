#!/usr/bin/env python3
"""
brain3/tests/test_agentic_coding_task.py

TESTS THE COMPLETE END-TO-END AUTONOMOUS AGENTIC CODING WORKFLOW:
1. Receives complex coding task.
2. Formulates invariant plan.
3. Synthesizes code in isolated sandbox.
4. Synthesizes test suite.
5. Runs sandbox subprocess execution.
6. Handles reflexion & commits invariant to long-term memory.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.agentic_task_sandbox_runner import AgenticTaskSandboxRunner

class TestAgenticCodingTask(unittest.TestCase):
    def test_end_to_end_agentic_coding_and_verification(self):
        runner = AgenticTaskSandboxRunner()
        result = runner.execute_agentic_coding_task(
            "Synthesize and verify an optimal Topological Sort algorithm with cycle detection",
            "topological_sort.py",
            "test_topological_sort.py"
        )

        print("\n" + "="*80)
        print("🤖 THE BRAIN 3: AUTONOMOUS AGENTIC CODING TASK REPORT")
        print("="*80)
        print(f"Task       : {result['task']}")
        print(f"Status     : {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        print(f"Duration   : {result['duration_ms']:.2f} ms")
        print(f"Target File: {result['target_file']}")
        print(f"Test File  : {result['test_file']}")
        print("="*80 + "\n")

        print("⚡ Full Autonomous ReAct Trajectory:")
        for step in result['trajectory']:
            print(f"Step {step['step']}:")
            print(f"  🤔 {step['thought']}")
            print(f"  ⚡ {step['action']}")
            print(f"  👁️ {step['observation']}\n")

        self.assertTrue(result['success'])
        self.assertEqual(len(result['trajectory']), 5)
        self.assertTrue(os.path.exists(result['target_file']))
        self.assertTrue(os.path.exists(result['test_file']))

if __name__ == "__main__":
    unittest.main()
