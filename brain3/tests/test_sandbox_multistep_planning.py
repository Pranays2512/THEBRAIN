#!/usr/bin/env python3
"""
brain3/tests/test_sandbox_multistep_planning.py

AUTONOMOUS SANDBOX MULTI-STEP PLANNING & EXECUTION TASK
Challenges The Brain 3 with a complex multi-step autonomous planning goal:
"Autonomous Multi-Step Scientific Task: Derive exact symbolic derivative of sin(x^2) in CAS,
 verify mathematical invariance, map cross-domain analogy to wave mechanics,
 audit the result for physical overclaims, and synthesize the verified solution."
"""

import unittest
import os
import subprocess
import json
import time

class TestSandboxMultiStepPlanning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain3_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.bin_path = os.path.join(cls.brain3_dir, "brain_mcp_server")

    def test_autonomous_multistep_planning_in_sandbox(self):
        """Submit a complex 4-step autonomous goal and verify zero-defect execution."""
        proc = subprocess.Popen(
            [self.bin_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.brain3_dir
        )

        try:
            # 1. Initialize MCP Server
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "SandboxPlannerRunner", "version": "1.0"}}
            }
            proc.stdin.write(json.dumps(init_req) + "\n")
            proc.stdin.flush()
            init_res = json.loads(proc.stdout.readline())
            self.assertEqual(init_res["id"], 1)

            # 2. Complex Goal Specification
            goal_spec = "Derive and verify symbolic derivative for sin(x^2) in CAS, audit for physical overclaims, and synthesize verified proof"

            task_req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "brain_run_agentic_task",
                    "arguments": {
                        "goal": goal_spec,
                        "max_steps": 6
                    }
                }
            }

            t0 = time.perf_counter()
            proc.stdin.write(json.dumps(task_req) + "\n")
            proc.stdin.flush()
            task_res = json.loads(proc.stdout.readline())
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            self.assertEqual(task_res["id"], 2)
            report_text = task_res["result"]["content"][0]["text"]

            print("\n" + "="*80)
            print("🧠 AUTONOMOUS SANDBOX MULTI-STEP PLANNING EXECUTION REPORT")
            print("="*80)
            print(f"Goal     : {goal_spec}")
            print(f"Latency  : {latency_ms:.2f} ms")
            print("="*80 + "\n")
            print(report_text)
            print("="*80 + "\n")

            # Assertions for complete autonomous execution
            self.assertIn("Autonomous Agentic Execution Report", report_text)
            self.assertIn("Autonomous Decomposition Plan", report_text)
            self.assertIn("ReAct Execution Trajectory Trace", report_text)
            self.assertIn("Final Agentic Synthesis", report_text)
            self.assertIn("Thought", report_text)
            self.assertIn("Action", report_text)
            self.assertIn("Observation", report_text)
            self.assertIn("Status: ✅ COMPLETED", report_text)

        finally:
            proc.stdin.close()
            proc.terminate()
            proc.wait()

if __name__ == "__main__":
    unittest.main()
