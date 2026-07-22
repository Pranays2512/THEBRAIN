import asyncio
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from faculties.whole_brain import WholeBrain

app = FastAPI(title="Royal Brain Server")

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

print("Initializing WholeBrain (this may take a moment)...")
brain = WholeBrain()

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

def synthesize_som(bmu_index):
    # 8x8 grid gaussian neighborhood around BMU
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

@app.post("/api/chat")
async def chat(request: ChatRequest):
    loop = asyncio.get_event_loop()
    # Run the brain sense in a threadpool so it doesn't block the async loop
    result = await loop.run_in_executor(None, brain.sense, request.message)
    
    # Broadcast the neural state update
    perc = result.get("perception", {})
    neural = perc.get("neural", {}) or {}
    
    # Defaults if neural layer is uninitialized
    bmu = neural.get("bmu", 36)
    surprise = neural.get("surprise", 0.0)
    valence = neural.get("valence", 0.0)
    
    som_activation = synthesize_som(bmu)
    
    # Push state to websocket
    state_payload = {
        "type": "state",
        "som_activation": som_activation,
        "emotion": {
            "arousal": min(perc.get("novelty", 0.0) + surprise, 1.0),
            "valence": valence
        },
        "working_memory": {
            "subject": request.message.split()[0] if len(request.message.split()) > 0 else "--",
            "relation": result["answer"]["kind"],
            "object": "processing"
        }
    }
    await broadcaster.broadcast(state_payload)
    
    return {"reply": result["answer"]["msg"]}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        while True:
            # We just keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
