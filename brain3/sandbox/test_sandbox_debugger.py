#!/usr/bin/env python3
"""
brain3/sandbox/test_sandbox_debugger.py

Unit tests for Pillar 4: Autonomous Closed-Loop Tool & Code Execution Sandbox.
"""

import unittest
import os
import sys

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from brain3.sandbox.code_execution_sandbox import CodeExecutionSandbox
from brain3.sandbox.autonomous_debugger import AutonomousDebugger

class TestSandboxDebugger(unittest.TestCase):

    def setUp(self):
        self.sandbox = CodeExecutionSandbox(timeout_sec=3.0)
        self.debugger = AutonomousDebugger(sandbox=self.sandbox)

    def test_01_python_sandbox_execution(self):
        """Test basic python execution and output capture."""
        code = "print('HELLO_BRAIN_SANDBOX')"
        res = self.sandbox.execute_python(code)
        self.assertTrue(res["success"])
        self.assertEqual(res["stdout"], "HELLO_BRAIN_SANDBOX")
        self.assertFalse(res["timeout"])

    def test_02_cpp_sandbox_execution(self):
        """Test C++ compilation, execution, and stdout capture."""
        cpp_code = """
        #include <iostream>
        int main() {
            std::cout << "CPP_SANDBOX_SUCCESS" << std::endl;
            return 0;
        }
        """
        res = self.sandbox.execute_cpp(cpp_code)
        self.assertTrue(res["success"])
        self.assertEqual(res["stdout"], "CPP_SANDBOX_SUCCESS")

    def test_03_autonomous_self_debugging_repair(self):
        """Test that missing typing import is automatically repaired and verified."""
        buggy_code = """
def filter_evens(nums: List[int]) -> List[int]:
    return [x for x in nums if x % 2 == 0]
"""
        test_harness = """
assert filter_evens([1, 2, 3, 4, 5, 6]) == [2, 4, 6]
print("ALL_TESTS_PASSED")
"""
        res = self.debugger.solve_and_verify("filter_evens", buggy_code, test_harness)
        self.assertTrue(res["verified"])
        self.assertEqual(res["attempts"], 2)
        self.assertIn("ALL_TESTS_PASSED", res["stdout"])
        self.assertIn("from typing import List", res["final_code"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
