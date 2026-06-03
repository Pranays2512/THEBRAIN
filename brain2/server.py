import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import brain2

# Global Brain Instance
b = None
brain_lock = threading.Lock()
clients = set()
daydreaming_active = True

def load_brain():
    global b
    print("Initializing Brain v3 for Scanner...")
    b = brain2.Brain(som_rows=8, som_cols=8, n_dims=16)
    
    # Load components
    ckpt_dir = "checkpoints/stage4_parsing"
    if os.path.exists(ckpt_dir):
        print(f"Loading full Brain architecture from {ckpt_dir}...")
        b.load_components(
            predictor_path=f"{ckpt_dir}/predictor.bin",
            language_path=f"{ckpt_dir}/language.bin",
            som_path=f"{ckpt_dir}/som.bin",
            episodic_path=f"{ckpt_dir}/episodic.bin",
            emotion_path=f"{ckpt_dir}/emotion.bin",
            self_path=f"{ckpt_dir}/self.bin",
            symbolic_path=f"{ckpt_dir}/symbolic.bin",
            binding_path=f"{ckpt_dir}/binding.bin",
            bg_path=f"{ckpt_dir}/bg.bin",
            procedures_path=f"{ckpt_dir}/procedures.bin",
            hpred_path=f"{ckpt_dir}/hpred.bin"
        )
    else:
        print("Warning: No checkpoint found, starting fresh.")

def get_brain_state():
    with brain_lock:
        # We need to extract the raw state.
        # Since we don't have direct python access to som.activation_map,
        # we can reconstruct it from the recent word or rely on the UI to pulse 
        # based on prediction errors.
        # Wait, if we use the scratchpad:
        
        # Actually, let's expose emotion
        # we don't have a direct python binding for emotion.valence yet, but we have get_last_episode!
        
        # We will just expose what we have or mock the activation for the visualizer.
        # Since the visualizer is a grid, we can just send the 'subject', 'relation', 'object' vectors
        # and the UI can map those 16-D vectors to an 8x8 grid (e.g. by duplicating or taking outer products).
        
        subj = b.scratchpad.read("subject")
        rel = b.scratchpad.read("relation")
        obj = b.scratchpad.read("object")
        
        subj_w = b.language.best_word(subj) if len(subj)>0 else ""
        rel_w = b.language.best_word(rel) if len(rel)>0 else ""
        obj_w = b.language.best_word(obj) if len(obj)>0 else ""
        
        # Simulate an 8x8 grid of activations based on the current context vectors
        # We will take the 16-D subject vector, reshape it, and make a 64-D map
        act_map = [0.0] * 64
        if len(subj) == 16:
            for i in range(16):
                act_map[i*4] = abs(subj[i])
                act_map[i*4+1] = abs(subj[i]) * 0.8
        
        return {
            "type": "state",
            "working_memory": {
                "subject": subj_w,
                "relation": rel_w,
                "object": obj_w
            },
            "som_activation": act_map,
            "emotion": {
                "valence": 0.5, # We'll fake it or bind it later
                "arousal": 0.2
            }
        }

async def broadcast_state():
    if not clients:
        return
    state = get_brain_state()
    msg = json.dumps(state)
    disconnected = set()
    for client in clients:
        try:
            await client.send_text(msg)
        except Exception:
            disconnected.add(client)
    clients.difference_update(disconnected)

def daydream_loop(loop):
    while daydreaming_active:
        with brain_lock:
            if b:
                b.daydream()
        # Tell asyncio loop to broadcast
        asyncio.run_coroutine_threadsafe(broadcast_state(), loop)
        import time
        time.sleep(0.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_brain()
    loop = asyncio.get_running_loop()
    thread = threading.Thread(target=daydream_loop, args=(loop,), daemon=True)
    thread.start()
    yield
    global daydreaming_active
    daydreaming_active = False

app = FastAPI(lifespan=lifespan)

# Create static dir if not exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    text = req.message.lower().strip()
    words = text.split()
    if not words:
        return {"reply": "..."}
        
    with brain_lock:
        b.scratchpad.clear()
        b.start_reasoning()
        
        is_query = "?" in text
        parse_words = [w.replace("?", "") for w in words if w.replace("?", "")]
        
        # Handle "what is X" format by converting it to "X is"
        if len(parse_words) >= 2 and parse_words[0] == "what" and parse_words[1] in ["is", "isa"]:
            parse_words = parse_words[2:] + ["is"]
        
        slot_ops = [9, 10, 11]
        slot_idx = 0
        has_not = False
        curiosity_triggered = None
        
        # Parse sentence (word by word parsing)
        for w in parse_words:
            if not w: continue
            if w == "not":
                has_not = True
                continue
            
            # Map 'is' to 'isa' for internal consistency
            if w == "is": w = "isa"
                
            is_novel = not b.symbolic_table.knows(w)
            if is_novel:
                b.learn_word(w)
                if not curiosity_triggered and w not in ["what", "isa", "is", "a", "an", "the"]:
                    curiosity_triggered = w
                
            vec = b.language.encode(w)
            if sum(abs(x) for x in vec) > 0:
                res = b.perceive(vec)
                if is_novel and curiosity_triggered == w:
                    b.force_reason_step(14, "curiosity")
                
                op = slot_ops[min(slot_idx, 2)]
                b.force_reason_step(op, "parse")
                slot_idx += 1
                
        if has_not:
            b.force_reason_step(12, "parse")
            
        if curiosity_triggered:
            reply = f"What is {curiosity_triggered}?"
        elif "say" in words or "remember" in words or "history" in words:
            last_payload = b.get_last_episode()
            if len(last_payload) > 0:
                topic = b.language.best_word(last_payload)
                reply = f"(retrieving from episodic memory...) you were talking about '{topic}'."
            else:
                reply = "I don't remember anything yet."
        else:
            # Full autonomy: Let the Brain reason about what to do next!
            b.scratchpad.write("goal", b.language.encode("reply"), "goal")
            
            # We let it reason. If the BG is not fully trained to pick the exact sequence,
            # we will force the optimal steps so it doesn't get stuck forever, but in theory
            # we would just do b.reason_step("reply", 0.0) here.
            
            # For demonstration of autonomy:
            if is_query:
                b.force_reason_step(5, "reply")  # OP_BIND_QUERY
                b.force_reason_step(15, "reply") # OP_SPEAK
                b.force_reason_step(8, "reply")  # OP_HALT
                spoken = b.get_spoken_words()
                if spoken:
                    reply = " ".join(spoken) + "."
                    if has_not: reply = "not " + reply
                else:
                    reply = "I don't know."
            else:
                b.force_reason_step(13, "reply") # OP_BIND_ISA
                b.force_reason_step(8, "reply")  # OP_HALT
                reply = "Got it."
                
        if reply in ["color", "binding", ""]:
            reply = "..."
            
        subj_vec = b.scratchpad.read("subject")
        if len(subj_vec) > 0 and sum(abs(x) for x in subj_vec) > 1e-6:
            b.commit_episode(1.0, subj_vec[:16])
            
    return {"reply": reply}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
