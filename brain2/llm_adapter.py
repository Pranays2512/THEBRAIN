#!/usr/bin/env python3
"""
llm_adapter.py — plug a local LLM into the Eyes and Mouth slots.

The brain is the mind; the LLM is only IO. This implements neuro_bridge's
Eyes/Mouth with a local model (Ollama / llama.cpp on a Mac), with NO training —
the model is used off the shelf as a translator:

  Eyes : messy language -> a structured Query the brain can reason over
  Mouth: a verified Answer -> a fluent sentence (constrained to the answer)

Design keeps the exact path exact: math notation still goes through the
deterministic recursive-descent parser; the LLM is only the FALLBACK for phrasing
the parser can't handle. So adding the model never makes a math answer less
reliable — it only widens what the eyes can read and smooths what the mouth says.

    eyes = LLMEyes(OllamaClient("qwen2.5"))
    mouth = LLMMouth(OllamaClient("qwen2.5"))
    Mind(eyes, Brain(), mouth)

Honest limit: prompt-constraining the mouth reduces but does not PROVE no
hallucination — the verified content is always in the Answer for audit, and hard
guarantees need constrained decoding. So verified/unknown answers fall back to the
deterministic grammar mouth; the LLM only polishes language-domain answers.
"""

import json
import os
import re
import sys
from abc import ABC, abstractmethod

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from math_parser import parse, ParseError
from neuro_bridge import Eyes, Mouth, Query, RuleEyes, GrammarMouth


# ── the model client (swap Stub for Ollama) ──────────────────────────────────
class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str: ...


class StubClient(LLMClient):
    """Deterministic client for tests: first matching substring -> response."""
    def __init__(self, table):
        self.table = table

    def complete(self, prompt, system=""):
        for key, resp in self.table.items():
            if key.lower() in prompt.lower():
                return resp
        return ""


class OllamaClient(LLMClient):                      # pragma: no cover (needs a server)
    """Real client: `ollama pull qwen2.5` (or gemma3:4b), then this calls it."""
    def __init__(self, model="qwen2.5", host="http://localhost:11434"):
        self.model, self.host = model, host

    def complete(self, prompt, system=""):
        import urllib.request
        body = json.dumps({"model": self.model, "prompt": prompt, "system": system,
                           "stream": False, "think": False,   # qwen3: skip slow thinking
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read()).get("response", "")
        return re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()


def _first_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ── LLM as Eyes: exact parser first, model only as fallback ──────────────────
EYES_SYSTEM = (
    "You convert a request into JSON. Output ONLY JSON. "
    'Schema: {"kind": "differentiate|integrate|solve|language", '
    '"expr": "<the expression to OPERATE ON, in notation — NEVER the answer>", '
    '"text": "<cleaned text, for language only>"}. '
    "expr is the INPUT expression, never the computed result. "
    'Examples: "slope of x squared" -> {"kind":"differentiate","expr":"x^2"}. '
    '"integral of cosine x" -> {"kind":"integrate","expr":"cos(x)"}. '
    '"what x makes 2x+3=7" -> {"kind":"solve","expr":"2*x+3=7"}. '
    'Anything not math -> {"kind":"language","text":"..."}. '
    "Use * for multiply and ^ for power."
)


class LLMEyes(Eyes):
    def __init__(self, client, rule=None):
        self.client = client
        self.rule = rule or RuleEyes()

    def parse(self, text):
        q = self.rule.parse(text)                   # exact path wins
        if q.kind not in ("language", "error"):
            return q
        data = _first_json(self.client.complete(text, EYES_SYSTEM))
        if not data:
            return q                                # fall back to language
        kind = data.get("kind")
        if kind in ("differentiate", "integrate", "solve"):
            try:
                node = parse(data.get("expr", ""))
            except ParseError:
                return q
            key = "equation" if kind == "solve" else "expr"
            return Query(kind, {key: node}, text)
        return Query("language", {"text": data.get("text", text)}, text)


# ── LLM as Mouth: polish language answers, keep verified ones deterministic ──
MOUTH_SYSTEM = (
    "Render this answer as ONE short natural sentence. Use ONLY the facts given; "
    "add nothing. Keep numbers and symbols exactly. Output only the sentence."
)


class LLMMouth(Mouth):
    def __init__(self, client, grammar=None):
        self.client = client
        self.grammar = grammar or GrammarMouth()

    def render(self, a):
        # verified / unknown answers stay deterministic (no LLM risk on the facts)
        if a.verified or not a.known:
            return self.grammar.render(a)
        payload = json.dumps({"kind": a.kind, "value": a.value, "steps": a.steps})
        out = self.client.complete(payload, MOUTH_SYSTEM).strip()
        return out or self.grammar.render(a)        # fall back if the model is silent


def _demo():
    from neuro_bridge import Mind, Brain
    # a stub standing in for the local model
    stub = StubClient({
        "slope of x squared": '{"kind":"differentiate","expr":"x^2"}',
        "apple":              "The apple is a red fruit.",
    })
    brain = Brain()
    brain.teach("apple", "isa", "fruit")
    brain.teach("apple", "is", "red")
    mind = Mind(LLMEyes(stub), brain, LLMMouth(stub))

    print("=== llm_adapter — LLM as eyes & mouth (stub stands in for Ollama) ===\n")
    for q in ["differentiate sin(x^2)",          # exact path, no LLM
              "what is the slope of x squared?",  # LLM-eyes rescues an odd phrasing
              "solve 2*x + 3 = 7 for x",          # verified -> deterministic mouth
              "what is apple?"]:                  # language -> LLM mouth polishes
        print(f"  > {q}\n    {mind.respond(q)}")
    print("\n(real use: swap StubClient for OllamaClient('qwen2.5') after `ollama pull qwen2.5`)")


if __name__ == "__main__":
    _demo()
