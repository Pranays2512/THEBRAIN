#!/usr/bin/env python3
"""
brain3/tests/test_agentic_ai.py

Comprehensive Test Suite for The Brain 3 Agentic AI System:
1. Ingestion of Agentic AI Knowledge Base (ReAct, Reflexion, ToT, MemGPT, MCP, DAG planning).
2. Autonomous Goal Planning (DAG subtask decomposition).
3. ReAct (Reason + Act + Observe + Reflect) Multi-Step Execution Loop.
4. Episodic Trajectory Logging and Self-Correction.
5. MCP stdio and TCP socket calls for `brain_run_agentic_task`.
6. Reading `brain://agent_memory` and `brain://agentic_knowledge` resources.
"""

import unittest
import os
import subprocess
import json
import socket
import time

class TestAgenticAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain3_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.repo_root = os.path.abspath(os.path.join(cls.brain3_dir, ".."))
        cls.bin_path = os.path.join(cls.brain3_dir, "brain_mcp_server")

        # Compile MCP Server if not already present
        if not os.path.exists(cls.bin_path):
            print("\n🔨 Compiling brain_mcp_server...")
            cmd = [
                "clang++", "-std=c++17", "-O3", "-pthread",
                "-I", cls.brain3_dir,
                "-I", os.path.join(cls.brain3_dir, "fuzzy"),
                "-I", os.path.join(cls.brain3_dir, "crisp"),
                os.path.join(cls.brain3_dir, "core", "mcp_server_main.cpp"),
                "-o", cls.bin_path
            ]
            res = subprocess.run(cmd, cwd=cls.brain3_dir, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Compilation failed:\n{res.stderr}")

    def test_01_agentic_knowledge_file_exists_and_valid(self):
        """Verify agentic_ai_knowledge.txt exists and contains rich ontology."""
        kb_path = os.path.join(self.repo_root, "brain2", "data", "agentic_ai_knowledge.txt")
        self.assertTrue(os.path.exists(kb_path), f"File missing: {kb_path}")
        with open(kb_path, "r", encoding="utf-8") as fh:
            content = fh.read()
            self.assertGreater(len(content), 1000)
            self.assertIn("react_framework", content)
            self.assertIn("reflexion_framework", content)
            self.assertIn("tree_of_thoughts", content)
            self.assertIn("memgpt_tiered_memory", content)
            self.assertIn("model_context_protocol", content)

    def test_02_stdio_mcp_agentic_goal_execution(self):
        """Test autonomous agentic ReAct loop execution via stdio MCP tool call."""
        proc = subprocess.Popen(
            [self.bin_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.brain3_dir
        )

        try:
            # 1. Initialize
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "AgenticTestClient", "version": "1.0"}}
            }
            proc.stdin.write(json.dumps(init_req) + "\n")
            proc.stdin.flush()
            init_res = json.loads(proc.stdout.readline())
            self.assertEqual(init_res["id"], 1)

            # 2. Call brain_run_agentic_task
            goal_req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "brain_run_agentic_task",
                    "arguments": {
                        "goal": "Derive and verify symbolic derivative for x^2 in CAS",
                        "max_steps": 5
                    }
                }
            }
            proc.stdin.write(json.dumps(goal_req) + "\n")
            proc.stdin.flush()
            goal_res = json.loads(proc.stdout.readline())
            self.assertEqual(goal_res["id"], 2)
            output_text = goal_res["result"]["content"][0]["text"]

            self.assertIn("Autonomous Agentic Execution Report", output_text)
            self.assertIn("Autonomous Decomposition Plan", output_text)
            self.assertIn("ReAct Execution Trajectory Trace", output_text)
            self.assertIn("Thought", output_text)
            self.assertIn("Action", output_text)
            self.assertIn("Observation", output_text)

        finally:
            proc.stdin.close()
            proc.terminate()
            proc.wait()

    def test_03_tcp_socket_agentic_resources(self):
        """Test reading agentic memory and architectures over TCP socket."""
        port = 9777
        server_proc = subprocess.Popen(
            [self.bin_path, "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.brain3_dir
        )
        time.sleep(1.0)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", port))
            f = sock.makefile("rw", encoding="utf-8")

            # 1. Read agentic_knowledge resource
            res_req = {
                "jsonrpc": "2.0",
                "id": 301,
                "method": "resources/read",
                "params": {"uri": "brain://agentic_knowledge"}
            }
            f.write(json.dumps(res_req) + "\n")
            f.flush()
            resp = json.loads(f.readline())
            self.assertEqual(resp["id"], 301)
            content = resp["result"]["contents"][0]["text"]
            self.assertIn("ReAct", content)
            self.assertIn("Reflexion", content)
            self.assertIn("MemGPT", content)

            # 2. Read agent_memory resource
            res_req_2 = {
                "jsonrpc": "2.0",
                "id": 302,
                "method": "resources/read",
                "params": {"uri": "brain://agent_memory"}
            }
            f.write(json.dumps(res_req_2) + "\n")
            f.flush()
            resp_2 = json.loads(f.readline())
            content_2 = resp_2["result"]["contents"][0]["text"]
            self.assertIn("ONLINE_READY", content_2)

            sock.close()
        finally:
            server_proc.terminate()
            server_proc.wait()

    def test_04_natural_language_orchestrator_agentic_command(self):
        """Test issuing an autonomous agent goal directly to MasterOrchestrator."""
        proc = subprocess.Popen(
            [self.bin_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.brain3_dir
        )

        try:
            # Query: "agentic goal: Align ancient Samkhya with modern quantum observer"
            q_req = {
                "jsonrpc": "2.0",
                "id": 401,
                "method": "tools/call",
                "params": {
                    "name": "brain_query",
                    "arguments": {"query": "agentic goal: Align ancient Samkhya with modern quantum observer"}
                }
            }
            proc.stdin.write(json.dumps(q_req) + "\n")
            proc.stdin.flush()
            res = json.loads(proc.stdout.readline())
            text = res["result"]["content"][0]["text"]
            self.assertIn("The Brain Autonomous Agentic Execution Report", text)
            self.assertIn("Autonomous Decomposition Plan", text)
            self.assertIn("ReAct Execution Trajectory Trace", text)

        finally:
            proc.stdin.close()
            proc.terminate()
            proc.wait()

if __name__ == "__main__":
    unittest.main()
