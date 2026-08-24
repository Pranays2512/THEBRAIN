#!/usr/bin/env python3
"""
brain3/mcp/brain_mcp_client.py

Client library & interactive CLI for connecting to The Brain 3 MCP Server
via TCP Socket or Subprocess Stdio.
"""

import sys
import json
import socket
import subprocess
from typing import Dict, Any, Optional

class BrainMCPClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9999, use_stdio: bool = False, binary_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.use_stdio = use_stdio
        self.binary_path = binary_path
        self.socket = None
        self.proc = None
        self.req_id = 0

    def connect(self):
        if self.use_stdio:
            if not self.binary_path:
                raise ValueError("binary_path required for stdio mode")
            self.proc = subprocess.Popen(
                [self.binary_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))

    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self.req_id,
            "method": method,
            "params": params or {}
        }
        raw_req = json.dumps(payload) + "\n"

        if self.use_stdio:
            self.proc.stdin.write(raw_req)
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            return json.loads(line)
        else:
            self.socket.sendall(raw_req.encode("utf-8"))
            data = self.socket.recv(8192).decode("utf-8")
            return json.loads(data.strip())

    def initialize(self) -> Dict[str, Any]:
        return self.send_request("initialize", {})

    def list_tools(self) -> Dict[str, Any]:
        return self.send_request("tools/list", {})

    def query(self, query_text: str) -> str:
        res = self.send_request("tools/call", {
            "name": "brain_query",
            "arguments": {"query": query_text}
        })
        return res.get("result", {}).get("content", [{}])[0].get("text", "")

    def audit_claim(self, claim_text: str) -> str:
        res = self.send_request("tools/call", {
            "name": "brain_audit_claim",
            "arguments": {"claim": claim_text}
        })
        return res.get("result", {}).get("content", [{}])[0].get("text", "")

    def teach(self, subject: str, relation: str, obj: str) -> str:
        res = self.send_request("tools/call", {
            "name": "brain_teach",
            "arguments": {"subject": subject, "relation": relation, "object": obj}
        })
        return res.get("result", {}).get("content", [{}])[0].get("text", "")

    def close(self):
        if self.socket:
            self.socket.close()
        if self.proc:
            self.proc.terminate()
            self.proc.wait()

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python3 -m brain3.mcp.brain_mcp_client [--port 9999]")
        return

    port = 9999
    if len(sys.argv) > 2 and sys.argv[1] == "--port":
        port = int(sys.argv[2])

    print(f"Connecting to The Brain MCP server on port {port}...")
    client = BrainMCPClient(port=port)
    try:
        client.connect()
        init_res = client.initialize()
        print(f"Connected: {init_res.get('result', {}).get('serverInfo', {})}")
        
        tools = client.list_tools()
        print(f"Available tools: {[t['name'] for t in tools.get('result', {}).get('tools', [])]}")

        print("\nType your query, or 'audit <claim>' to audit, or 'exit' to quit:")
        while True:
            try:
                line = input("brain-mcp> ").strip()
                if not line or line.lower() in ("exit", "quit"):
                    break
                if line.startswith("audit "):
                    print(client.audit_claim(line[6:]))
                else:
                    print(client.query(line))
            except (EOFError, KeyboardInterrupt):
                break
    finally:
        client.close()

if __name__ == "__main__":
    main()
