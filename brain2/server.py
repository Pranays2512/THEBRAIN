import asyncio
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
import os
import json
from fastapi.responses import StreamingResponse
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from faculties.whole_brain import WholeBrain
from adapters.llm_adapter import OllamaClient, LLMEyes, LLMMouth
from engines.reasoning.neuro_bridge import Answer

app = FastAPI(title="Royal Brain Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (e.g., Vite dev server on 5173)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

print("Initializing LLM Eyes & Mouth...")
# Use a cloud teacher LLM for the parsing fallback (Rung 3) and a fast local one for the mouth
teacher_client = OllamaClient("gpt-oss:120b-cloud")
mouth_client = OllamaClient("qwen3:1.7B")
eyes = LLMEyes(teacher_client)
mouth = LLMMouth(mouth_client)

print("Initializing WholeBrain (this may take a moment)...")
brain = WholeBrain(eyes=eyes)

# We need a pubsub mechanism to push state to the websocket
class StateBroadcaster:
    def __init__(self):
        self.connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.connections.remove(websocket)

    async def broadcast(self, data: dict):
        for connection in self.connections:
            try:
                await connection.send_json(data)
            except Exception:
                pass

broadcaster = StateBroadcaster()

@app.get("/")
async def get_index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

class ChatRequest(BaseModel):
    message: str

class TeachRequest(BaseModel):
    subject: str
    relation: str
    object: str

def synthesize_som(bmu_index, text=""):
    # If BMU is static fallback 59, map text hash to a dynamic BMU in 0..63
    if bmu_index == 59 and text:
        bmu_index = abs(hash(text)) % 64
        
    som = [0.0] * 64
    bmu_x = bmu_index % 8
    bmu_y = bmu_index // 8
    for i in range(64):
        x = i % 8
        y = i // 8
        dist = ((x - bmu_x)**2 + (y - bmu_y)**2)**0.5
        val = math.exp(-dist * 1.5)
        som[i] = round(val, 3)
    return som

def build_telemetry_payload(text: str, result: dict):
    perc = result.get("perception", {})
    ans_info = result.get("answer", {})
    neural = perc.get("neural", {}) or {}
    
    bmu = neural.get("bmu", 59)
    surprise_val = perc.get("surprise")
    if surprise_val is None:
        surprise_val = neural.get("surprise", 0.0)
        
    novelty = perc.get("novelty", 0.5)
    kind = ans_info.get("kind", "language")
    verified = ans_info.get("verified", False)
    
    som_activation = synthesize_som(bmu, text)
    
    # Calculate cognitive valence and arousal
    arousal = min(max(novelty * 0.6 + (surprise_val or 0.0) * 0.4, 0.1), 1.0)
    valence = 0.8 if verified else (0.2 if kind != "none" else -0.2)
    
    words = text.split()
    subj = words[0] if words else "--"
    obj = words[-1] if len(words) > 1 else "target"
    
    return {
        "type": "state",
        "som_activation": som_activation,
        "emotion": {
            "arousal": round(arousal, 2),
            "valence": round(valence, 2),
            "novelty": round(novelty, 2),
            "surprise": round(surprise_val if surprise_val is not None else 0.0, 2)
        },
        "working_memory": {
            "subject": subj,
            "relation": kind,
            "object": obj
        },
        "answer_meta": {
            "kind": kind,
            "verified": verified
        }
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, brain.sense, request.message)
    
    state_payload = build_telemetry_payload(request.message, result)
    await broadcaster.broadcast(state_payload)
    
    return {
        "reply": result["answer"]["msg"],
        "kind": result["answer"]["kind"],
        "verified": result["answer"]["verified"]
    }

@app.post("/api/teach")
async def teach(request: TeachRequest):
    try:
        brain.teach(request.subject, request.relation, request.object)
        return {"status": "success", "message": f"Brain learned: {request.subject} {request.relation} {request.object}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat/stream")
async def chat_stream(request: dict):
    messages = request.get("messages", [])
    if not messages:
        return {"error": "No messages"}
        
    last_msg_obj = messages[-1]
    last_msg = last_msg_obj.get("content", "")
    
    # Support for React / Block format
    if not last_msg and "blocks" in last_msg_obj:
        for block in last_msg_obj["blocks"]:
            if block.get("type") == "text":
                last_msg += block.get("text", "") + " "
    
    last_msg = last_msg.strip()
    
    loop = asyncio.get_event_loop()
    
    # 1. Update Brain perception for the Dashboard
    result = await loop.run_in_executor(None, brain.sense, last_msg)
    state_payload = build_telemetry_payload(last_msg, result)
    await broadcaster.broadcast(state_payload)
    
    # 2. Extract Answer from WholeBrain's result
    ans = result["answer"]["msg"]
    kind = result["answer"]["kind"]
    verified = result["answer"]["verified"]
    is_known = (kind != "none")
    
    # 3. Stream through Mouth
    async def event_generator():
        # First send metadata
        meta_event = {
            "meta": {
                "kind": kind,
                "verified": verified,
                "known": is_known
            }
        }
        yield f'data: {json.dumps(meta_event)}\n\n'
        
        answer_obj = Answer(kind=kind if kind != "none" else "language", 
                            known=is_known, value=ans, verified=verified)
        for chunk in mouth.render_stream(answer_obj):
            yield f'data: {json.dumps({"text": chunk})}\n\n'
        yield 'data: [DONE]\n\n'
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


