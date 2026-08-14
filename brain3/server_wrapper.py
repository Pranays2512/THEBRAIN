import asyncio
import json
import subprocess
import threading
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Brain3 API Wrapper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start the Brain3 C++ REPL as a background process
print("Starting Brain3 C++ Core...")
brain_process = subprocess.Popen(
    ["./build_cmake/demo_native_chat"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# We use a lock to ensure only one chat request talks to stdin at a time
process_lock = asyncio.Lock()

class ChatRequest(BaseModel):
    messages: list

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

    async def event_generator():
        # Yield metadata mimicking the Brain2 format so the UI works
        meta_event = {
            "meta": {
                "kind": "language",
                "verified": False,
                "known": True
            }
        }
        yield f'data: {json.dumps(meta_event)}\n\n'

        # Send input to Brain3 C++ process and read output
        async with process_lock:
            # We are writing to a synchronous pipe in an async function.
            # For a production app we'd use asyncio.subprocess, but this is a quick demo.
            brain_process.stdin.write(last_msg + "\n")
            brain_process.stdin.flush()

            # The Brain3 REPL outputs lines until it hits an empty line or "YOU: " prompt
            response_text = ""
            while True:
                line = brain_process.stdout.readline()
                if not line:
                    break
                
                # Check if it's the prompt waiting for next input
                if "YOU: " in line:
                    break
                
                if "BRAIN THOUGHTS:" in line or "BRAIN:" in line:
                    clean_line = line.replace("🧠 BRAIN THOUGHTS:", "").replace("🧠 BRAIN:", "").strip()
                    if clean_line:
                        response_text += clean_line + "\n"
                        yield f'data: {json.dumps({"text": clean_line + " "})}\n\n'

            # If it didn't output anything special, just echo a default
            if not response_text:
                yield f'data: {json.dumps({"text": "[Processed by Biological Core]"})}\n\n'
                
        yield 'data: [DONE]\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Eat the initial loading text from the C++ REPL
    def eat_startup():
        while True:
            line = brain_process.stdout.readline()
            if "YOU:" in line:
                break
    threading.Thread(target=eat_startup, daemon=True).start()

    print("Brain3 API Wrapper running on http://localhost:8000")
    uvicorn.run("server_wrapper:app", host="0.0.0.0", port=8000, log_level="info")
