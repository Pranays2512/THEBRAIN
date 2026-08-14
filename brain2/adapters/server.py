"""
server.py — Brain2 FastAPI Server
Live cognitive engine with chat, causal chain reasoning, confidence-gated answers,
WebSocket state broadcasting, and continuous daydreaming.
"""
import asyncio
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import brain2

# ─── Global State ────────────────────────────────────────────────────────────
b: brain2.Brain = None
bi_instance = None
brain_lock        = threading.Lock()
clients: set      = set()
daydreaming_active = True
CKPT_DIR = "checkpoints/stage1_32d"

# Op codes
OP_MATH_SUB  = 2;  OP_MATH_DIV  = 3;  OP_BIND_QUERY  = 5
OP_HALT      = 8;  OP_STORE_SUBJ= 9;  OP_STORE_REL   = 10
OP_STORE_OBJ = 11; OP_NOT       = 12; OP_BIND_ISA    = 13
OP_SPEAK     = 15; OP_SPEAK_SUBJ= 17; OP_SPEAK_REL   = 18
OP_SPEAK_OBJ = 19; OP_MUL       = 21; OP_PERM_N      = 22
OP_PERM_K    = 23; OP_POWER     = 24; OP_DIV_FLOAT   = 26
OP_CHAIN_FOLLOW = 28

# ─── Brain Loader ─────────────────────────────────────────────────────────────
def load_brain():
    global b
    print(f"Loading Brain from {CKPT_DIR}...", flush=True)
    b = brain2.Brain(som_rows=10, som_cols=10, n_dims=32)
    if os.path.exists(CKPT_DIR):
        b.load_components(
            predictor_path  = f"{CKPT_DIR}/predictor.bin",
            language_path   = f"{CKPT_DIR}/language.bin",
            som_path        = f"{CKPT_DIR}/som.bin",
            episodic_path   = f"{CKPT_DIR}/episodic.bin",
            emotion_path    = f"{CKPT_DIR}/emotion.bin",
            self_path       = f"{CKPT_DIR}/self.bin",
            symbolic_path   = f"{CKPT_DIR}/symbolic.bin",
            binding_path    = f"{CKPT_DIR}/binding.bin",
            bg_path         = f"{CKPT_DIR}/bg.bin",
            procedures_path = f"{CKPT_DIR}/procedures.bin",
            hpred_path      = f"{CKPT_DIR}/hpred.bin",
        )
        b.symbolic_table.seed_math_symbols()
        for i in range(1000):
            b.symbolic_table.bind(str(i))
        for w in ["probability", "permute", "area", "power", "causes", "solve"]:
            b.language.register_word(w)
        print("Brain loaded ✓", flush=True)
    else:
        print(f"WARNING: checkpoint not found at {CKPT_DIR}", flush=True)

# ─── State Broadcaster ────────────────────────────────────────────────────────
def get_brain_state() -> dict:
    subj = b.scratchpad.read("subject")
    rel  = b.scratchpad.read("relation")
    obj  = b.scratchpad.read("object")
    conf_raw = b.scratchpad.read("confidence")
    conf = conf_raw[0] if conf_raw else 0.0

    subj_w = b.language.best_word(subj) if subj else ""
    rel_w  = b.language.best_word(rel)  if rel  else ""
    obj_w  = b.language.best_word(obj)  if obj  else ""

    # Real SOM-like activation map from subject vector
    act_map = [0.0] * 64
    if len(subj) == 16:
        for i in range(16):
            act_map[i * 4    ] = abs(subj[i])
            act_map[i * 4 + 1] = abs(subj[i]) * 0.7
            act_map[i * 4 + 2] = abs(subj[i]) * 0.4
            act_map[i * 4 + 3] = abs(subj[i]) * 0.2

    return {
        "type": "state",
        "working_memory": {"subject": subj_w, "relation": rel_w, "object": obj_w},
        "som_activation": act_map,
        "confidence": round(conf, 3),
        "emotion": {
            "valence": getattr(b, "emotion", None) and b.emotion.valence or 0.5,
            "arousal": getattr(b, "emotion", None) and b.emotion.arousal or 0.2,
        }
    }

async def broadcast_state():
    if not clients:
        return
    state = get_brain_state()
    msg = json.dumps(state)
    dead = set()
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)

def daydream_loop(loop):
    while daydreaming_active:
        with brain_lock:
            if b:
                b.daydream()
        asyncio.run_coroutine_threadsafe(broadcast_state(), loop)
        time.sleep(0.5)

# ─── App Lifecycle ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_brain()
    global bi_instance
    from adapters.brain_interface import BrainInterface
    from adapters.llm_adapter import OllamaClient
    print("Loading BrainInterface (LLM: qwen3:1.7B)...", flush=True)
    bi_instance = BrainInterface(OllamaClient("qwen3:1.7B"))
    print("BrainInterface loaded ✓", flush=True)

    loop = asyncio.get_running_loop()
    t = threading.Thread(target=daydream_loop, args=(loop,), daemon=True)
    t.start()
    yield
    global daydreaming_active
    daydreaming_active = False

app = FastAPI(title="Brain2 Server", lifespan=lifespan)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    text  = req.message.strip()
    lower = text.lower()
    words = lower.split()

    if not words:
        return {"reply": "...", "confidence": 0.0, "mode": "empty"}

    reply     = "..."
    mode      = "parse"
    conf_out  = 0.0

    with brain_lock:
        b.scratchpad.clear()
        b.clear_spoken_words()
        b.start_reasoning()

        # ── DETECT MODE ──────────────────────────────────────────────────
        is_query      = "?" in text
        is_causal     = re.search(r"\bcauses?\b|\bleads? to\b|\bresults? in\b", lower)
        is_algebra    = re.search(r"(\d+)\s*x\s*\+\s*(\d+)\s*=\s*(\d+)", lower)
        is_prob       = re.search(r"probability\s+of\s+(\d+)\s+(?:in|out of)\s+(\d+)", lower)
        is_area       = re.search(r"area\s+of\s+(\d+)\s+(?:and|by)\s+(\d+)", lower)
        is_power      = re.search(r"(\d+)\s+(?:power|\^)\s+(\d+)", lower)
        is_perm       = re.search(r"(\d+)\s+permute\s+(\d+)", lower)
        is_episodic   = any(w in words for w in ["remember", "recall", "history", "memory"])

        # Register novel words
        for w in words:
            w2 = w.replace("?", "").replace(".", "")
            if w2 and not b.language.knows(w2):
                b.language.register_word(w2)
                b.symbolic_table.bind(w2)

        # ── ALGEBRA: ax + b = c ──────────────────────────────────────────
        if is_algebra:
            m = is_algebra
            a_v, b_v, c_v = m.group(1), m.group(2), m.group(3)
            for wv in [a_v, b_v, c_v]: b.symbolic_table.bind(wv)
            b.scratchpad.write("subject",  b.language.encode(c_v), "context")
            b.scratchpad.write("relation", b.language.encode(a_v), "context")
            b.scratchpad.write("object",   b.language.encode(b_v), "context")
            b.force_reason_step(OP_MATH_SUB, "solve")
            b.force_reason_step(OP_MATH_DIV, "solve")
            b.force_reason_step(OP_SPEAK,    "solve")
            spoken = b.get_spoken_words()
            reply  = f"x = {spoken[-1]}" if spoken else "I couldn't solve that."
            mode   = "algebra"; conf_out = 1.0

        # ── PROBABILITY ──────────────────────────────────────────────────
        elif is_prob:
            n, d = is_prob.group(1), is_prob.group(2)
            for wv in [n, d]: b.symbolic_table.bind(wv)
            b.scratchpad.write("subject", b.language.encode(n), "context")
            b.scratchpad.write("object",  b.language.encode(d), "context")
            seq = b.procedures.retrieve(b.language.encode("probability"))
            for op in seq: b.force_reason_step(op, "reply")
            spoken = b.get_spoken_words()
            reply  = f"{spoken[-1]}" if spoken else "?"
            mode   = "probability"; conf_out = 1.0

        # ── AREA ─────────────────────────────────────────────────────────
        elif is_area:
            w_v, h_v = is_area.group(1), is_area.group(2)
            for wv in [w_v, h_v]: b.symbolic_table.bind(wv)
            b.scratchpad.write("subject", b.language.encode(w_v), "context")
            b.scratchpad.write("object",  b.language.encode(h_v), "context")
            seq = b.procedures.retrieve(b.language.encode("area"))
            for op in seq: b.force_reason_step(op, "reply")
            spoken = b.get_spoken_words()
            reply  = f"{spoken[-1]}" if spoken else "?"
            mode   = "area"; conf_out = 1.0

        # ── POWER ────────────────────────────────────────────────────────
        elif is_power:
            base, exp = is_power.group(1), is_power.group(2)
            for wv in [base, exp]: b.symbolic_table.bind(wv)
            b.scratchpad.write("subject", b.language.encode(base), "context")
            b.scratchpad.write("object",  b.language.encode(exp),  "context")
            seq = b.procedures.retrieve(b.language.encode("power"))
            for op in seq: b.force_reason_step(op, "reply")
            spoken = b.get_spoken_words()
            reply  = f"{spoken[-1]}" if spoken else "?"
            mode   = "power"; conf_out = 1.0

        # ── PERMUTATION ──────────────────────────────────────────────────
        elif is_perm:
            n_v, k_v = is_perm.group(1), is_perm.group(2)
            for wv in [n_v, k_v]: b.symbolic_table.bind(wv)
            b.scratchpad.write("subject", b.language.encode(n_v), "context")
            b.scratchpad.write("object",  b.language.encode(k_v), "context")
            seq = b.procedures.retrieve(b.language.encode("permute"))
            for op in seq: b.force_reason_step(op, "reply")
            spoken = b.get_spoken_words()
            reply  = f"{spoken[-1]}" if spoken else "?"
            mode   = "permute"; conf_out = 1.0

        # ── CAUSAL CHAIN: "X causes ?" or "what does X cause?" ───────────
        elif is_causal or ("cause" in lower and is_query):
            cause_match = re.search(r"(\w+)\s+causes?\s+", lower)
            if cause_match:
                start_word = cause_match.group(1)
                b.language.register_word(start_word)
                b.scratchpad.write("subject",  b.language.encode(start_word), "context")
                b.scratchpad.write("relation", b.language.encode("causes"),   "context")
                b.force_reason_step(OP_CHAIN_FOLLOW, "causes")
                result   = b.scratchpad.read("result")
                conf_out = b.get_last_confidence()
                best     = b.language.best_word(result)
                if best and conf_out >= 0.25:
                    reply = f"{best}"
                else:
                    reply = "I don't know what that leads to."
                mode = "causal_chain"
            else:
                reply = "I didn't understand the causal query."; mode = "error"

        # ── EPISODIC RECALL ───────────────────────────────────────────────
        elif is_episodic:
            last = b.get_last_episode()
            if last:
                topic = b.language.best_word(last)
                reply = f"You were talking about '{topic}'."
            else:
                reply = "I don't remember anything yet."
            mode = "episodic"

        # ── SEMANTIC QUERY / GENERAL STATEMENT ───────────────────────────
        else:
            # We defer this outside the lock so we don't block daydreaming while the LLM generates
            pass

        # Episodic commit
        sv = b.scratchpad.read("subject")
        if sv:
            b.commit_episode(1.0, sv[:16])

    # If none of the exact-match math/causal paths handled it, delegate to the unified BrainInterface
    if not is_algebra and not is_prob and not is_area and not is_power and not is_perm and not is_causal and not is_episodic:
        if bi_instance:
            # We are outside the lock, so the LLM generation can take its time
            res = bi_instance.respond(text)
            reply = res["reply"]
            mode = res["kind"]
            conf_out = 1.0 if res["verified"] else 0.0
        else:
            reply = "I don't know."
            mode = "error"
            conf_out = 0.0

    return {"reply": reply, "confidence": round(conf_out, 3), "mode": mode}

class TeachRequest(BaseModel):
    subject: str
    relation: str
    object: str

@app.post("/api/teach")
async def teach_endpoint(req: TeachRequest):
    if bi_instance:
        added = bi_instance.teach(req.subject, req.relation, req.object)
        return {"ok": True, "added": added, "msg": f"Got it: {req.subject} {req.relation} {req.object}"}
    return {"ok": False, "msg": "BrainInterface not loaded"}

class BqlRequest(BaseModel):
    query: str

@app.post("/api/bql")
async def bql_endpoint(req: BqlRequest):
    if bi_instance:
        from engines.reasoning.brainql import parse_bql_block, BrainQLParseError
        try:
            queries = parse_bql_block(req.query)
            # Use internal _execute to get the raw BrainQLResult objects
            results = bi_instance._execute(queries)
            # Serialize the first result
            if results:
                r = results[0]
                return {"ok": True, "known": r.known, "value": r.value, "chain": r.chain}
            return {"ok": True, "results": []}
        except BrainQLParseError as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "msg": "BrainInterface not loaded"}


# ─── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
