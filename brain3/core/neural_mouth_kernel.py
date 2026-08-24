#!/usr/bin/env python3
"""
brain3/core/neural_mouth_kernel.py

THE BRAIN 3 — NEURAL MOUTH KERNEL (QWEN 3 1.7B VIA OLLAMA)
Provides The Brain with its generative "Mouth" actuator.
The Brain formulates the invariants, plans, logic, and self-corrections;
The Neural Mouth translates The Brain's structured prompts into raw code/tokens.
"""

import json
import re
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

class NeuralMouthKernel:
    def __init__(self, model_name: str = "qwen3:1.7B", endpoint: str = "http://localhost:11434"):
        self.model_name = model_name
        self.endpoint = endpoint.rstrip("/")
        self.generate_url = f"{self.endpoint}/api/generate"

    def is_available(self) -> bool:
        """Checks if the Ollama endpoint is reachable."""
        try:
            req = urllib.request.Request(f"{self.endpoint}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self.model_name in m for m in models) or len(models) > 0
        except Exception:
            return False

    def synthesize_code(self, prompt: str, system_invariants: str = "") -> str:
        """
        Synthesizes code from The Brain's directive.
        Strips markdown wrappers and returns clean executable code.
        """
        full_prompt = prompt
        if system_invariants:
            full_prompt = f"System Invariants & Mathematical Constraints from The Brain:\n{system_invariants}\n\nTask:\n{prompt}\n\nOutput ONLY valid code without conversational filler."

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.85,
                "num_predict": 1536
            }
        }

        try:
            req = urllib.request.Request(
                self.generate_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_response = data.get("response", "")
                return self._extract_clean_code(raw_response)
        except Exception as e:
            return f"# [Neural Mouth Error]: {e}"

    def _extract_clean_code(self, text: str) -> str:
        """Extracts pure code from markdown backticks or raw text."""
        # Find ```lang ... ``` blocks
        code_blocks = re.findall(r"```(?:[a-zA-Z0-9_\-]+)?\n(.*?)```", text, re.DOTALL)
        if code_blocks:
            return "\n\n".join(b.strip() for b in code_blocks)
        
        # If no triple backticks, return clean stripped text
        return text.strip()

if __name__ == "__main__":
    mouth = NeuralMouthKernel()
    print(f"Mouth Available: {mouth.is_available()}")
    code = mouth.synthesize_code("Write a binary search function in Python that returns the index of target in sorted arr.")
    print("Synthesized Code:\n", code)
