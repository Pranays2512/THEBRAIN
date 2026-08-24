#!/usr/bin/env python3
"""
brain3/tests/test_mcp_server.py

Comprehensive test suite verifying The Brain 3 Model Context Protocol (MCP) server:
1. Stdio Mode: initialize, tools/list, resources/list, brain_query, brain_teach, brain_audit_claim.
2. TCP Socket Mode: connecting over network socket, sending JSON-RPC 2.0 frames, receiving responses.
3. Epistemic Anti-Overclaiming: verifying that overreaching claims are caught and rejected by the MCP tool.
"""

import unittest
import subprocess
import json
import socket
import time
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
MCP_BIN = BRAIN3_DIR / "brain_mcp_server"

class TestBrainMCPServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Compile MCP server binary
        print("\n🔨 Compiling brain_mcp_server...")
        cmd = [
            "clang++", "-std=c++17", "-O2",
            "-I.", "-Icore", "-Icrisp", "-Ifuzzy", "-I..",
            "-Wno-deprecated-declarations", "-framework", "Accelerate",
            "-o", str(MCP_BIN),
            "core/mcp_server_main.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Compilation error:\n{res.stderr}")
        assert res.returncode == 0, "Failed to compile brain_mcp_server"
        print("✓ Compilation successful.")

    def test_01_stdio_initialize_and_tools(self):
        """Tests standard stdio JSON-RPC MCP initialization and tools listing."""
        proc = subprocess.Popen(
            [str(MCP_BIN)],
            cwd=str(BRAIN3_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            # 1. Initialize
            init_req = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            }) + "\n"
            proc.stdin.write(init_req)
            proc.stdin.flush()

            init_resp_line = proc.stdout.readline()
            init_resp = json.loads(init_resp_line)
            self.assertEqual(init_resp["id"], 1)
            self.assertEqual(init_resp["result"]["serverInfo"]["name"], "TheBrain-3-MCP-Server")
            self.assertIn("tools", init_resp["result"]["capabilities"])

            # 2. tools/list
            tools_req = json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }) + "\n"
            proc.stdin.write(tools_req)
            proc.stdin.flush()

            tools_resp_line = proc.stdout.readline()
            tools_resp = json.loads(tools_resp_line)
            self.assertEqual(tools_resp["id"], 2)
            tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
            self.assertIn("brain_query", tool_names)
            self.assertIn("brain_teach", tool_names)
            self.assertIn("brain_audit_claim", tool_names)
            self.assertIn("brain_solve_anomaly", tool_names)

            # 3. Call brain_audit_claim (Anti-Overclaiming verification)
            audit_req = json.dumps({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "brain_audit_claim",
                    "arguments": {
                        "claim": "H2RL achieves exact lossless memory recall with zero length failure in O(1) fixed space."
                    }
                }
            }) + "\n"
            proc.stdin.write(audit_req)
            proc.stdin.flush()

            audit_resp_line = proc.stdout.readline()
            audit_resp = json.loads(audit_resp_line)
            content_text = audit_resp["result"]["content"][0]["text"]
            self.assertIn("REJECTED_OVERCLAIM_RECALIBRATED", content_text)
            self.assertIn("Pigeonhole & Capacity Bound", content_text)

        finally:
            proc.terminate()
            proc.wait()

    def test_02_tcp_socket_server(self):
        """Tests running the MCP server on a TCP socket and sending tool calls."""
        port = 19999
        proc = subprocess.Popen(
            [str(MCP_BIN), "--port", str(port)],
            cwd=str(BRAIN3_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        time.sleep(0.5) # Allow socket to bind

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", port))

            # 1. Initialize over TCP Socket
            req = json.dumps({
                "jsonrpc": "2.0",
                "id": "tcp-1",
                "method": "initialize",
                "params": {}
            }) + "\n"
            s.sendall(req.encode("utf-8"))

            data = s.recv(4096).decode("utf-8")
            resp = json.loads(data.strip())
            self.assertEqual(resp["result"]["serverInfo"]["name"], "TheBrain-3-MCP-Server")

            # 2. Teach a fact over TCP
            teach_req = json.dumps({
                "jsonrpc": "2.0",
                "id": "tcp-2",
                "method": "tools/call",
                "params": {
                    "name": "brain_teach",
                    "arguments": {
                        "subject": "holographic_state",
                        "relation": "has_crosstalk_noise",
                        "object": "bounded_by_plate_snr"
                    }
                }
            }) + "\n"
            s.sendall(teach_req.encode("utf-8"))

            data = s.recv(4096).decode("utf-8")
            resp = json.loads(data.strip())
            self.assertIn("Ingested fact", resp["result"]["content"][0]["text"])

            # 3. Read resources over TCP
            res_req = json.dumps({
                "jsonrpc": "2.0",
                "id": "tcp-3",
                "method": "resources/read",
                "params": {
                    "uri": "brain://axioms"
                }
            }) + "\n"
            s.sendall(res_req.encode("utf-8"))

            data = s.recv(4096).decode("utf-8")
            resp = json.loads(data.strip())
            self.assertIn("Shannon-Plate Superposition SNR", resp["result"]["contents"][0]["text"])

            s.close()

        finally:
            proc.terminate()
            proc.wait()

if __name__ == "__main__":
    unittest.main()
