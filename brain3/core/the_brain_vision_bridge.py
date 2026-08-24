#!/usr/bin/env python3
"""
brain3/core/the_brain_vision_bridge.py

THE BRAIN NATIVE VISION & PERCEPTION BRIDGE
(The Brain Sees & Grounds Facts -> Qwen Speaks What The Brain Saw)
"""

import os
import sys
import subprocess
import json
import urllib.request
import re

WORKSPACE = "/Users/pranay./Documents/THEBRAIN"
BRAIN3_DIR = os.path.join(WORKSPACE, "brain3")

def perceive_and_reason_visually():
    print("==================================================================")
    print(" 👁️ THE BRAIN: NATIVE C++ VISION PERCEPTION & SPATIAL REASONING")
    print("==================================================================")

    # 1. Run The Brain's Native C++ Vision Engine
    print("\n[Step 1] Initializing The Brain's Native Vision Engine (stb_image + ReasoningEngine)...")
    cmd = ["./demo_vision"]
    proc = subprocess.run(cmd, cwd=BRAIN3_DIR, capture_output=True, text=True)
    vision_output = proc.stdout.strip()
    print(vision_output)

    # 2. Extract The Brain's Grounded Facts
    grounded_facts = []
    for line in vision_output.split("\n"):
        if "Blob" in line or ">" in line:
            grounded_facts.append(line.strip())

    # 3. Direct Qwen 3 (The Mouth) to Verbalize The Brain's Perceptual State
    print("\n[Step 2] Passing The Brain's Grounded Visual Facts to Qwen 3 (The Mouth)...")
    prompt = f"""You are The Mouth for The Brain.
The Brain has physically perceived an image through its native C++ Vision Engine and grounded the following relational facts into its Reasoning Engine:
{chr(10).join(grounded_facts)}

Speak in a concise, authoritative first-person voice ('I perceive...'). Describe the exact spatial layout, colors, and relative sizes of the objects The Brain is looking at. Do NOT invent objects not grounded by The Brain."""

    payload = {
        "model": "qwen3:1.7B",
        "prompt": prompt,
        "think": False,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 300}
    }

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))["response"]

    mouth_speech = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()

    print("\n==================================================================")
    print(" 🗣️ QWEN 3 (THE MOUTH) VERBALIZING THE BRAIN'S PERCEPTION:")
    print("==================================================================")
    print(mouth_speech)
    print("==================================================================\n")

if __name__ == "__main__":
    perceive_and_reason_visually()
