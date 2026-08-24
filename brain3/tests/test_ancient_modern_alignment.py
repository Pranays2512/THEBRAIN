#!/usr/bin/env python3
"""
brain3/tests/test_ancient_modern_alignment.py

Comprehensive Test Suite for The Brain's Ancient-Modern Epistemic Alignment System:
1. Validates Knowledge Ingestion of Ancient Indian Philosophies (Nyaya, Vaisheshika, Samkhya, Yoga, Mimamsa, Vedanta, Jain, Buddhist).
2. Validates Knowledge Ingestion of Vedic Cosmological Texts (Nasadiya Sukta, Purusha Sukta, Mandukya, Chandogya, Katha, Brihadaranyaka).
3. Validates Knowledge Ingestion of Ancient Epics, Mathematics & Sciences (Bhagavad Gita, Mahabharata, Yoga Vasistha, Pingala, Aryabhata, Brahmagupta).
4. Verifies Structural Isomorphisms and Gentner SME Systematicity Scores.
5. Verifies MCP Tool `brain_align_ancient_modern` and Resource endpoints.
"""

import unittest
import os
import subprocess
import json
import socket
import time

class TestAncientModernAlignment(unittest.TestCase):
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

    def test_01_ancient_knowledge_files_exist_and_populated(self):
        """Verify that all three ancient knowledge files exist in brain2/data and contain rich ontology."""
        data_dir = os.path.join(self.repo_root, "brain2", "data")
        files = [
            "ancient_indian_philosophies.txt",
            "ancient_vedic_texts_cosmology.txt",
            "ancient_stories_epics_science.txt"
        ]
        for f in files:
            p = os.path.join(data_dir, f)
            self.assertTrue(os.path.exists(p), f"File missing: {p}")
            with open(p, "r", encoding="utf-8") as fh:
                content = fh.read()
                self.assertGreater(len(content), 1000, f"File {f} is too short")
                self.assertIn("FACT:", content, f"File {f} must have FACT triples")
                self.assertIn("ISA:", content, f"File {f} must have ISA ontologies")

    def test_02_stdio_mcp_align_ancient_modern_tool(self):
        """Test calling brain_align_ancient_modern tool over JSON-RPC 2.0 stdio."""
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
                "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "TestClient", "version": "1.0"}}
            }
            proc.stdin.write(json.dumps(init_req) + "\n")
            proc.stdin.flush()
            init_res = json.loads(proc.stdout.readline())
            self.assertEqual(init_res["id"], 1)

            # 2. Call brain_align_ancient_modern for Samkhya
            call_req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "brain_align_ancient_modern",
                    "arguments": {"topic": "samkhya"}
                }
            }
            proc.stdin.write(json.dumps(call_req) + "\n")
            proc.stdin.flush()
            call_res = json.loads(proc.stdout.readline())
            self.assertEqual(call_res["id"], 2)
            text_out = call_res["result"]["content"][0]["text"]
            
            self.assertIn("Samkhya Purusha-Prakriti", text_out)
            self.assertIn("Quantum State Vector", text_out)
            self.assertIn("Structural Systematicity Score", text_out)
            self.assertIn("Epistemic Boundary", text_out)

            # 3. Call for Nasadiya Sukta
            call_req_2 = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "brain_align_ancient_modern",
                    "arguments": {"topic": "nasadiya"}
                }
            }
            proc.stdin.write(json.dumps(call_req_2) + "\n")
            proc.stdin.flush()
            call_res_2 = json.loads(proc.stdout.readline())
            text_out_2 = call_res_2["result"]["content"][0]["text"]
            self.assertIn("Nasadiya Sukta", text_out_2)
            self.assertIn("Quantum Vacuum Fluctuations", text_out_2)

            # 4. Call for Pingala
            call_req_3 = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "brain_align_ancient_modern",
                    "arguments": {"topic": "pingala"}
                }
            }
            proc.stdin.write(json.dumps(call_req_3) + "\n")
            proc.stdin.flush()
            call_res_3 = json.loads(proc.stdout.readline())
            text_out_3 = call_res_3["result"]["content"][0]["text"]
            self.assertIn("Pingala", text_out_3)
            self.assertIn("Pascal's Triangle", text_out_3)

        finally:
            proc.stdin.close()
            proc.terminate()
            proc.wait()

    def test_03_tcp_socket_ancient_resources(self):
        """Test fetching ancient knowledge resources over TCP socket."""
        port = 9888
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

            # 1. Read ancient_philosophies resource
            res_req = {
                "jsonrpc": "2.0",
                "id": 101,
                "method": "resources/read",
                "params": {"uri": "brain://ancient_philosophies"}
            }
            f.write(json.dumps(res_req) + "\n")
            f.flush()
            resp = json.loads(f.readline())
            self.assertEqual(resp["id"], 101)
            content = resp["result"]["contents"][0]["text"]
            self.assertIn("Nyaya", content)
            self.assertIn("Vaisheshika", content)
            self.assertIn("Samkhya", content)
            self.assertIn("Advaita", content)

            # 2. Read vedic_cosmology resource
            res_req_2 = {
                "jsonrpc": "2.0",
                "id": 102,
                "method": "resources/read",
                "params": {"uri": "brain://vedic_cosmology"}
            }
            f.write(json.dumps(res_req_2) + "\n")
            f.flush()
            resp_2 = json.loads(f.readline())
            content_2 = resp_2["result"]["contents"][0]["text"]
            self.assertIn("Nasadiya Sukta", content_2)
            self.assertIn("Mandukya Upanishad", content_2)

            # 3. Read epics resource
            res_req_3 = {
                "jsonrpc": "2.0",
                "id": 103,
                "method": "resources/read",
                "params": {"uri": "brain://epics_and_ancient_sciences"}
            }
            f.write(json.dumps(res_req_3) + "\n")
            f.flush()
            resp_3 = json.loads(f.readline())
            content_3 = resp_3["result"]["contents"][0]["text"]
            self.assertIn("Bhagavad Gita", content_3)
            self.assertIn("Pingala", content_3)

            sock.close()
        finally:
            server_proc.terminate()
            server_proc.wait()

    def test_04_natural_language_orchestrator_ancient_inquiry(self):
        """Test asking natural language questions about ancient texts via MCP query."""
        proc = subprocess.Popen(
            [self.bin_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.brain3_dir
        )

        try:
            # Query: "connect ancient samkhya and quantum observer"
            q_req = {
                "jsonrpc": "2.0",
                "id": 201,
                "method": "tools/call",
                "params": {
                    "name": "brain_query",
                    "arguments": {"query": "connect ancient samkhya and quantum observer"}
                }
            }
            proc.stdin.write(json.dumps(q_req) + "\n")
            proc.stdin.flush()
            res = json.loads(proc.stdout.readline())
            text = res["result"]["content"][0]["text"]
            self.assertIn("Samkhya", text)
            self.assertIn("Purusha", text)
            self.assertIn("Prakriti", text)

        finally:
            proc.stdin.close()
            proc.terminate()
            proc.wait()

if __name__ == "__main__":
    unittest.main()
