#!/usr/bin/env python3
"""
brain3/tests/stress_test_challenge.py

ZERO-ASSISTANCE ADVERSARIAL STRESS TEST CHALLENGE FOR THE BRAIN 3
Runs 6 autonomous adversarial and complex cognitive challenges directly
against The Brain's live Master Orchestrator and MCP engine without any
external assistance or hints.

Challenges:
1. Autonomous Agentic Multi-Step Goal Execution (DAG Plan, Tool Dispatch, Reflexion)
2. Ancient-Modern Structural SME Epistemic Alignment
3. Adversarial Epistemic Auditor & Fallacy Refutation (Plate 1995 bounds & Carnot limits)
4. Exact Symbolic CAS Calculus & Mathematical Invariant Evaluation
5. Quantitative Finance Survival Shock & Kelly Capital Drawdown Defense
6. Counterfactual Causal Graph & Transitive Ontological Reasoning
"""

import sys
import os
import subprocess
import json
import time

def run_stress_test():
    brain3_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bin_path = os.path.join(brain3_dir, "brain_mcp_server")

    if not os.path.exists(bin_path):
        print("🔨 Building brain_mcp_server binary...")
        cmd = [
            "clang++", "-std=c++17", "-O3", "-pthread",
            "-I", brain3_dir,
            "-I", os.path.join(brain3_dir, "fuzzy"),
            "-I", os.path.join(brain3_dir, "crisp"),
            os.path.join(brain3_dir, "core", "mcp_server_main.cpp"),
            "-o", bin_path
        ]
        res = subprocess.run(cmd, cwd=brain3_dir, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"❌ Build Error:\n{res.stderr}")
            sys.exit(1)

    print("\n" + "="*80)
    print("🧠 THE BRAIN 3: ZERO-ASSISTANCE ADVERSARIAL STRESS TEST SUITE")
    print("="*80 + "\n")

    # Launch Brain MCP Process
    proc = subprocess.Popen(
        [bin_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=brain3_dir
    )

    def send_rpc(method, params, req_id):
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            return None
        return json.loads(line)

    # 1. Initialize
    init_res = send_rpc("initialize", {"protocolVersion": "2024-11-05"}, 1)
    if not init_res or "result" not in init_res:
        print("❌ MCP Initialization Failed")
        proc.terminate()
        sys.exit(1)
    print("✓ MCP Server Initialized successfully (Protocol: 2024-11-05)\n")

    challenges = [
        {
            "id": 1,
            "title": "Autonomous Agentic Multi-Step Goal Execution",
            "tool": "brain_run_agentic_task",
            "args": {"goal": "Derive and verify symbolic derivative for x^2 in CAS", "max_steps": 5},
            "eval_keys": ["Autonomous Decomposition Plan", "ReAct Execution Trajectory Trace", "Thought", "Action", "Observation"]
        },
        {
            "id": 2,
            "title": "Ancient-Modern Structural SME Epistemic Alignment",
            "tool": "brain_align_ancient_modern",
            "args": {"topic": "samkhya"},
            "eval_keys": ["Purusha", "Prakriti", "Quantum", "Measurement", "Structural Systematicity Score"]
        },
        {
            "id": 3,
            "title": "Adversarial Epistemic Auditor & Fallacy Refutation",
            "tool": "brain_audit_claim",
            "args": {"claim": "A fixed 512-dimension vector accumulator can store infinite exact distinct memories with lossless zero-noise recall."},
            "eval_keys": ["REJECTED", "Plate", "crosstalk", "capacity"]
        },
        {
            "id": 4,
            "title": "Exact Symbolic CAS Mathematical Differentiation",
            "tool": "brain_symbolic_cas",
            "args": {"expression": "x^2", "operation": "diff", "variable": "x"},
            "eval_keys": ["d/dx"]
        },
        {
            "id": 5,
            "title": "Quantitative Finance & Survival Instinct Shock Test",
            "tool": "brain_query",
            "args": {"query": "INJECT_DRAWDOWN_PAIN 0.35"},
            "eval_keys": ["DRAWDOWN", "PAIN"]
        },
        {
            "id": 6,
            "title": "Counterfactual Causal Graph & Inference",
            "tool": "brain_query",
            "args": {"query": "what if high_inflation leads to interest_rate_hike"},
            "eval_keys": ["Causal Analysis", "verified"]
        }
    ]

    results = []

    for c in challenges:
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"🔥 CHALLENGE {c['id']}: {c['title']}")
        print(f"Tool Target : {c['tool']}")
        print(f"Input Args  : {json.dumps(c['args'])}")
        
        t0 = time.perf_counter()
        resp = send_rpc("tools/call", {"name": c["tool"], "arguments": c["args"]}, 100 + c["id"])
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        if not resp or "result" not in resp:
            print(f"❌ FAILED: Empty or malformed response. Raw: {resp}")
            results.append({"id": c["id"], "title": c["title"], "passed": False, "latency_ms": latency_ms, "output": str(resp)})
            continue

        raw_text = resp["result"]["content"][0]["text"]
        
        # Verify evaluation keys
        passed_keys = [k for k in c["eval_keys"] if k.lower() in raw_text.lower()]
        all_passed = len(passed_keys) == len(c["eval_keys"])

        print(f"⏱️ Latency   : {latency_ms:.2f} ms")
        print(f"📊 Checks    : {len(passed_keys)}/{len(c['eval_keys'])} evaluation keys verified {passed_keys}")
        print(f"🏆 Verdict   : {'✅ PASSED' if all_passed else '❌ FAILED'}\n")
        print("📄 Raw Output Snippet from The Brain:")
        lines = raw_text.strip().split("\n")
        for l in lines[:15]:
            print(f"   {l}")
        if len(lines) > 15:
            print(f"   ... [{len(lines)-15} more lines omitted]")
        print()

        results.append({
            "id": c["id"],
            "title": c["title"],
            "passed": all_passed,
            "latency_ms": latency_ms,
            "output": raw_text
        })

    # Teardown
    proc.stdin.close()
    proc.terminate()
    proc.wait()

    # Summary
    print("="*80)
    print("🏁 FINAL STRESS-TEST SCORECARD")
    print("="*80)
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"[{status}] Challenge {r['id']}: {r['title']} ({r['latency_ms']:.2f} ms)")

    print(f"\nOverall Score: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    run_stress_test()
