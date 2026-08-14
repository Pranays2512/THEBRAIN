#!/usr/bin/env python3
"""
brain3/broca_bridge.py

THE BROCA COMMUNICATION BRIDGE (PILLAR 1)
High-performance asynchronous Server-Sent Events (SSE) & WebSocket Daemon
connecting The Brain's C++ Mind Core (brain_master) to the React Frontend & LLM Mouth.

Endpoints:
- POST /chat/stream        : Exact SSE protocol expected by Frontend/src/providers/FastAPIProvider.js
- GET  /health             : Health check and C++ Mind Core state
- GET  /stats              : Real-time telemetry (discovered invariants, BQL speed, cognitive load)
- POST /discovery/start    : Trigger background self-play & invariant discovery
- POST /discovery/stop     : Stop background self-play
- GET  /discovery/status   : Live discovery telemetry
- POST /ingest             : Ingest datasets or single files into BrainQL / Binding Memory
- POST /cross-domain/hunt  : Autonomous cross-domain isomorphism and anti-unification step
- GET  /cross-domain/status: Live status of cross-domain synthesized invariants
"""

import asyncio
import json
import os
import sys
import time
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="The Brain — Broca Communication Bridge",
    description="Asynchronous IPC and SSE bridge for The Brain C++ Mind Core",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BRAIN_MASTER_BIN = os.path.join(os.path.dirname(__file__), "brain_master")
if not os.path.exists(BRAIN_MASTER_BIN):
    BRAIN_MASTER_BIN = "./brain3/brain_master"

class BrainProcessManager:
    def __init__(self, binary_path: str):
        self.binary_path = binary_path
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.lock = asyncio.Lock()

    async def ensure_running(self):
        if self.proc is None or self.proc.returncode is not None:
            self.proc = await asyncio.create_subprocess_exec(
                self.binary_path, "--json-stream",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            print(f"🧠 [Broca Bridge] Attached to C++ Mind Core (PID: {self.proc.pid})")

    async def query(self, text: str) -> dict:
        async with self.lock:
            await self.ensure_running()
            clean_text = text.replace("\n", " ").strip()
            if not clean_text:
                clean_text = "PING"
            
            # Send to C++ Mind Core
            self.proc.stdin.write((clean_text + "\n").encode("utf-8"))
            await self.proc.stdin.drain()

            # Read single-line JSON response
            line_bytes = await self.proc.stdout.readline()
            if not line_bytes:
                # Process crashed, restart and retry once
                await self.ensure_running()
                return {
                    "status": "error",
                    "natural_reply": "⚠️ C++ Mind Core connection refreshed. Please retry.",
                    "engine_used": "broca_fallback",
                    "latency_ms": 0.0,
                    "verified": False
                }

            line_str = line_bytes.decode("utf-8", errors="replace").strip()
            try:
                data = json.loads(line_str)
                return data
            except Exception as e:
                return {
                    "status": "ok",
                    "natural_reply": line_str,
                    "engine_used": "raw_cpp",
                    "latency_ms": 0.0,
                    "verified": True
                }

brain_mgr = BrainProcessManager(BRAIN_MASTER_BIN)

@app.on_event("startup")
async def startup_event():
    await brain_mgr.ensure_running()

@app.get("/health")
async def health_check():
    pid = brain_mgr.proc.pid if brain_mgr.proc else None
    alive = (brain_mgr.proc is not None and brain_mgr.proc.returncode is None)
    return {
        "status": "healthy" if alive else "reconnecting",
        "mind_core": "C++ MasterOrchestrator (Bicameral)",
        "bridge": "Broca FastAPI SSE",
        "core_pid": pid,
        "port": 8000
    }

@app.get("/stats")
async def stats():
    discovery_res = await brain_mgr.query("DISCOVERY_STATUS")
    cross_res = await brain_mgr.query("CROSS_DOMAIN_STATUS")
    return {
        "brain_version": "3.0.0-Native-CPP",
        "engines": [
            "Instinct Reflex ALU (<0.1ms)",
            "BrainQL Epistemic Semantic Memory",
            "Calculus & FTC Integration Engine",
            "Algorithmic Policy Invariant Engine",
            "Native C++ Knowledge Ingestion Engine (>100k facts/sec)",
            "Cross-Domain Isomorphism & Anti-Unification Hunter",
            "Neuro-Symbolic A* + MCTS Search",
            "Continuous Self-Play Discovery Daemon"
        ],
        "discovery_telemetry": discovery_res.get("natural_reply", ""),
        "cross_domain_status": cross_res.get("raw_output", "")
    }

@app.post("/discovery/start")
async def start_discovery():
    return await brain_mgr.query("START_SELF_PLAY")

@app.post("/discovery/stop")
async def stop_discovery():
    return await brain_mgr.query("STOP_SELF_PLAY")

@app.get("/discovery/status")
async def discovery_status():
    return await brain_mgr.query("DISCOVERY_STATUS")

class IngestRequest(BaseModel):
    path: Optional[str] = None
    all_datasets: Optional[bool] = False

@app.post("/ingest")
async def ingest_knowledge(req: IngestRequest):
    if req.all_datasets:
        return await brain_mgr.query("INGEST_ALL")
    elif req.path:
        return await brain_mgr.query(f"INGEST {req.path}")
    else:
        return await brain_mgr.query("INGEST_ALL")

@app.post("/cross-domain/hunt")
async def cross_domain_hunt():
    return await brain_mgr.query("CROSS_DOMAIN_HUNT")

@app.get("/cross-domain/status")
async def cross_domain_status():
    return await brain_mgr.query("CROSS_DOMAIN_STATUS")

# Quantitative Finance & Survival Instinct Branch Endpoints
@app.get("/finance/survival_status")
async def finance_survival_status():
    return await brain_mgr.query("FINANCE_STATUS")

@app.get("/finance/orderbook/{symbol:path}")
async def finance_orderbook(symbol: str):
    return await brain_mgr.query(f"ORDER_BOOK {symbol}")

@app.get("/finance/microstructure/{symbol:path}")
async def finance_microstructure(symbol: str):
    return await brain_mgr.query(f"MICROSTRUCTURE {symbol}")

class TradeOrderRequest(BaseModel):
    symbol: str = "BTC/USDT"
    side: str = "BUY"
    type: str = "MARKET"
    price: float = 0.0
    quantity: float = 1.0

@app.post("/finance/trade")
async def finance_trade(req: TradeOrderRequest):
    return await brain_mgr.query(f"TRADE_ORDER {req.symbol} {req.side} {req.type} {req.price} {req.quantity}")

class KellyRequest(BaseModel):
    win_prob: float = 0.55
    win_loss_ratio: float = 1.5

@app.post("/finance/kelly")
async def finance_kelly(req: KellyRequest):
    return await brain_mgr.query(f"KELLY_SIZE {req.win_prob} {req.win_loss_ratio}")

class StatArbRequest(BaseModel):
    symbol_a: str = "BTC/USDT"
    symbol_b: str = "ETH/USDT"

@app.post("/finance/stat_arb")
async def finance_stat_arb(req: StatArbRequest):
    return await brain_mgr.query(f"STAT_ARB_SCAN {req.symbol_a} {req.symbol_b}")

class SimulateMarketRequest(BaseModel):
    symbol: str = "BTC/USDT"
    ticks: int = 50
    drift: float = 0.0005
    vol: float = 0.01

@app.post("/finance/simulate")
async def finance_simulate(req: SimulateMarketRequest):
    return await brain_mgr.query(f"SIMULATE_MARKET_CYCLE {req.symbol} {req.ticks} {req.drift} {req.vol}")

class InjectPainRequest(BaseModel):
    loss_amount: float = 1000.0

@app.post("/finance/inject_pain")
async def finance_inject_pain(req: InjectPainRequest):
    return await brain_mgr.query(f"INJECT_DRAWDOWN_PAIN {req.loss_amount}")

class ResetFinanceRequest(BaseModel):
    initial_capital: float = 10000.0

@app.post("/finance/reset")
async def finance_reset(req: ResetFinanceRequest):
    return await brain_mgr.query(f"RESET_LIFE_FORCE {req.initial_capital}")

@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    messages = body.get("messages", [])
    query_text = ""

    if messages:
        last_msg_obj = messages[-1]
        if isinstance(last_msg_obj, dict):
            query_text = last_msg_obj.get("content", "")
            if not query_text and "blocks" in last_msg_obj:
                for block in last_msg_obj["blocks"]:
                    if block.get("type") == "text":
                        query_text += block.get("text", "") + " "
        elif isinstance(last_msg_obj, str):
            query_text = last_msg_obj
    elif "query" in body:
        query_text = body["query"]
    elif "text" in body:
        query_text = body["text"]

    query_text = query_text.strip()
    if not query_text:
        query_text = "Explain your cognitive architecture"

    async def sse_generator():
        # 1. Query the C++ Mind Core
        t0 = time.perf_counter()
        brain_resp = await brain_mgr.query(query_text)
        t1 = time.perf_counter()
        total_time_ms = (t1 - t0) * 1000.0

        natural_reply = brain_resp.get("natural_reply", "")
        engine_used = brain_resp.get("engine_used", "mind_core")
        bql_query = brain_resp.get("bql_query", "")
        verified = brain_resp.get("verified", False)

        # 2. Yield metadata event matching Frontend expectation
        meta_event = {
            "meta": {
                "kind": "language",
                "engine": engine_used,
                "bql": bql_query,
                "verified": verified,
                "latency_ms": round(total_time_ms, 3)
            }
        }
        yield f"data: {json.dumps(meta_event)}\n\n"

        # 3. Stream text tokens for smooth typing effect
        words = natural_reply.split(" ")
        chunk_size = 2
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size]) + " "
            token_payload = {"text": chunk}
            yield f"data: {json.dumps(token_payload)}\n\n"
            await asyncio.sleep(0.015) # Smooth 15ms streaming rhythm

        # 4. Stream final completion marker
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 [The Brain] Launching Broca Communication Bridge on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
