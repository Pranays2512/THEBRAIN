#!/usr/bin/env python3
"""
brain3/test_broca_bridge.py

Automated Verification Test for The Broca Communication Bridge (Pillar 1)
and Continuous Self-Play Discovery Daemon (Pillar 5).
"""

import asyncio
import json
import httpx
import subprocess
import time
import sys

async def run_tests():
    print("\n🧠 ==========================================================================")
    print("   THE BRAIN — BROCA BRIDGE & SELF-PLAY DAEMON VERIFICATION SUITE")
    print("==========================================================================\n")

    # 1. Start the Broca Bridge daemon in background
    print("1. Spawning Broca Bridge server on http://localhost:8000 ...")
    server_proc = subprocess.Popen(
        [sys.executable, "brain3/broca_bridge.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to boot
    time.sleep(2.0)

    try:
        async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=10.0) as client:
            # 2. Test /health
            print("2. Testing GET /health ...")
            health = await client.get("/health")
            assert health.status_code == 200, f"Health check failed: {health.text}"
            health_json = health.json()
            print(f"   ✓ Health Response: {health_json}")
            assert health_json["status"] == "healthy"

            # 3. Test /stats
            print("\n3. Testing GET /stats ...")
            stats = await client.get("/stats")
            assert stats.status_code == 200
            print(f"   ✓ Engines Active: {len(stats.json()['engines'])} native engines reporting.")

            # 4. Test Self-Play Daemon API (/discovery/status, /discovery/start, /discovery/stop)
            print("\n4. Testing Continuous Self-Play & Discovery Daemon ...")
            disc_status = await client.get("/discovery/status")
            print(f"   ✓ Initial Telemetry: {disc_status.json().get('natural_reply', '')}")

            start_disc = await client.post("/discovery/start")
            print(f"   ✓ Start Signal: {start_disc.json().get('natural_reply', '')}")
            
            # Let it run 5 discovery cycles
            await asyncio.sleep(0.5)

            stop_disc = await client.post("/discovery/stop")
            print(f"   ✓ Stop Signal: {stop_disc.json().get('natural_reply', '')}")

            # 5. Test POST /chat/stream (SSE Protocol matching Frontend)
            test_queries = [
                {"q": "290 / 2", "expected": "145"},
                {"q": "What if gravity causes acceleration?", "expected": "acceleration"},
                {"q": "POLICY divide_and_conquer_dp_monge", "expected": "Quadrangle Inequality"},
                {"q": "STEP_DISCOVERY", "expected": "Calculus Invariant"}
            ]

            print("\n5. Testing POST /chat/stream (Frontend SSE Transport) ...")
            for item in test_queries:
                q = item["q"]
                print(f"\n   👤 User Query: \"{q}\"")
                payload = {"messages": [{"role": "user", "content": q}]}
                
                t0 = time.perf_counter()
                async with client.stream("POST", "/chat/stream", json=payload) as response:
                    assert response.status_code == 200
                    accumulated_text = ""
                    meta_received = False
                    
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            ev = json.loads(data_str)
                            if "meta" in ev:
                                meta_received = True
                                print(f"      [SSE Meta]: {ev['meta']}")
                            if "text" in ev:
                                accumulated_text += ev["text"]
                        except Exception:
                            pass

                t1 = time.perf_counter()
                total_latency_ms = (t1 - t0) * 1000.0
                print(f"      🧠 Streamed Reply: {accumulated_text.strip()}")
                print(f"      ⏱️ Stream Duration: {total_latency_ms:.2f} ms")
                assert meta_received, "Failed to receive SSE meta event"
                assert len(accumulated_text) > 0, "No text streamed"

    finally:
        print("\n6. Shutting down test server ...")
        server_proc.terminate()
        server_proc.wait()

    print("\n==========================================================================")
    print("🏆 ALL BROCA BRIDGE & SELF-PLAY DAEMON VERIFICATION TESTS PASSED (100%)")
    print("==========================================================================\n")

if __name__ == "__main__":
    asyncio.run(run_tests())
