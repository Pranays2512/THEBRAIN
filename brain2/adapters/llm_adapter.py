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
from engines.math.math_parser import parse, ParseError
from engines.reasoning.neuro_bridge import Eyes, Mouth, Query, RuleEyes, GrammarMouth
from engines.reasoning.brainql import (
    BrainQLQuery, BrainQLResult, parse_bql_block,
    EYES_SYSTEM_BQL, MOUTH_SYSTEM_BQL, result_to_mouth_payload,
)


# ── the model client (swap Stub for Ollama) ──────────────────────────────────
class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str: ...


class SafeClient(LLMClient):
    """Wrap any client so a down/unreachable server degrades to '' (-> the caller
    abstains / falls back) instead of crashing. Optional local fallback client tried
    first on failure."""
    def __init__(self, primary, fallback=None):
        self.primary, self.fallback = primary, fallback

    def complete(self, prompt, system=""):
        for c in (self.primary, self.fallback):
            if c is None:
                continue
            try:
                return c.complete(prompt, system)
            except Exception:
                continue
        return ""                                   # all down -> empty -> caller abstains


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
    """Real client: `ollama pull qwen3:1.7B` (or gemma3:4b), then this calls it."""
    def __init__(self, model="qwen3:1.7B", host="http://localhost:11434"):
        self.model, self.host = model, host

    def complete(self, prompt, system=""):
        import urllib.request
        import urllib.error
        body = json.dumps({"model": self.model, "prompt": prompt, "system": system,
                           "stream": False, "think": False,   # qwen3: skip slow thinking
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read()).get("response", "")
            return re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        except urllib.error.URLError:
            return ""  # degrade gracefully when offline

    def stream(self, prompt, system=""):
        import urllib.request
        import urllib.error
        body = json.dumps({"model": self.model, "prompt": prompt, "system": system,
                           "stream": True, "think": False,
                           "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                for line in r:
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
        except urllib.error.URLError:
            return  # degrade gracefully when offline


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
    'Schema: {"kind": "differentiate|integrate|solve|factual|language", '
    '"expr": "<the expression to OPERATE ON, in notation — NEVER the answer>", '
    '"subject": "<the main entity being asked about, for factual>", '
    '"relation": "<the relation or property, for factual>", '
    '"text": "<cleaned text, for language only>"}. '
    "expr is the INPUT expression, never the computed result. "
    'Examples: "slope of x squared" -> {"kind":"differentiate","expr":"x^2"}. '
    '"integral of cosine x" -> {"kind":"integrate","expr":"cos(x)"}. '
    '"what x makes 2x+3=7" -> {"kind":"solve","expr":"2*x+3=7"}. '
    '"is vehicle used for transport?" -> {"kind":"factual","subject":"vehicle","relation":"used_for"}. '
    'Anything not math or factual -> {"kind":"language","text":"..."}. '
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
        try:
            data = _first_json(self.client.complete(text, EYES_SYSTEM))
        except Exception:
            return q                                # fall back to language on connection error
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
        if kind == "factual":
            return Query("factual", {"subject": data.get("subject", ""), "relation": data.get("relation", "")}, text)
        return Query("language", {"text": data.get("text", text)}, text)


# ── LLM as Mouth: polish language answers, keep verified ones deterministic ──
MOUTH_SYSTEM = (
    "You are a strict verbalizer. Render the provided factual answer or calculation "
    "into ONE short, natural, conversational sentence. You MUST use ONLY the facts "
    "given in the JSON. Do NOT add ANY outside information, context, or hallucinated "
    "facts whatsoever. Keep numbers, math, and symbols exactly as provided. "
    "Output ONLY the verbalized sentence."
)


class LLMMouth(Mouth):
    def __init__(self, client, grammar=None):
        self.client = client
        self.grammar = grammar or GrammarMouth()

    def render(self, a):
        # Unknown answers stay deterministic (bypass LLM to avoid confident hallucinations)
        if not a.known:
            return self.grammar.render(a)
        payload = json.dumps({"kind": a.kind, "value": str(a.value), "steps": getattr(a, "steps", [])})
        try:
            out = self.client.complete(payload, MOUTH_SYSTEM).strip()
            return out or self.grammar.render(a)        # fall back if the model is silent
        except Exception:
            return self.grammar.render(a)

    def render_stream(self, a):
        # Unknown or non-language answers stay deterministic
        if not a.known or a.kind in ("code", "compute"):
            yield self.grammar.render(a)
            return
        payload = json.dumps({"kind": a.kind, "value": str(a.value), "steps": getattr(a, "steps", [])})
        if hasattr(self.client, "stream"):
            try:
                for chunk in self.client.stream(payload, MOUTH_SYSTEM):
                    yield chunk
            except Exception:
                yield self.grammar.render(a)
        else:
            try:
                yield self.render(a)
            except Exception:
                yield self.grammar.render(a)




# ── BrainQL Eyes: LLM generates BrainQL queries (for factual/reasoning) ──────
class BrainQLEyes:
    """LLM as Eyes in BrainQL mode: converts natural language → BrainQL instructions.

    Exact math paths (differentiate/integrate/solve) still go through RuleEyes first
    so adding this never makes math less reliable. Only factual/language queries
    route through BrainQL.

    Returns a list[BrainQLQuery] (may be multiple instructions for complex queries),
    or falls back to a single Query(kind='language') if the LLM output can't parse.
    """

    def __init__(self, client, rule=None):
        self.client = client
        self.rule = rule or RuleEyes()

    def parse(self, text: str):
        """Returns either:
          - A list[BrainQLQuery]  (BrainQL path — reasoning/factual)
          - A Query               (math path — differentiate/integrate/solve)
        Callers should isinstance-check the return type.
        """
        # 1. Exact math path wins; never pass math to BrainQL
        q = self.rule.parse(text)
        if q.kind not in ("language", "error"):
            return q   # math Query — existing path unchanged

        # 2. Ask the LLM to emit BrainQL
        try:
            raw_bql = self.client.complete(text, EYES_SYSTEM_BQL).strip()
        except Exception:
            return q   # server down → fall back to language Query

        if not raw_bql:
            return q

        try:
            queries = parse_bql_block(raw_bql)
            if queries:
                return queries   # list[BrainQLQuery]
        except Exception:
            pass

        return q   # parse failed → language Query fall-through


# ── BrainQL Mouth: LLM verbalises BrainQLResult ───────────────────────────────
class BrainQLMouth:
    """LLM as Mouth in BrainQL mode: converts BrainQLResult → fluent English.

    The LLM only reads the brain's structured result — it cannot hallucinate
    facts because all content comes from verified BrainQLResult fields.
    Falls back to a deterministic template renderer when the LLM is down.
    """

    def __init__(self, client, grammar=None):
        self.client = client
        self.grammar = grammar or GrammarMouth()

    def render_result(self, result: BrainQLResult) -> str:
        """Convert a BrainQLResult to a fluent sentence."""
        if not result.known:
            # Deterministic honest abstention — never ask the LLM to make up an answer
            return f"I don't know: {result.note or 'the brain has no answer for that.'}"
        payload = result_to_mouth_payload(result)
        try:
            out = self.client.complete(payload, MOUTH_SYSTEM_BQL).strip()
            return out or self._fallback_render(result)
        except Exception:
            return self._fallback_render(result)

    def render_stream(self, result: BrainQLResult):
        """Streaming version for websocket/server use."""
        if not result.known:
            yield f"I don't know: {result.note or 'the brain has no answer for that.'}"
            return
        payload = result_to_mouth_payload(result)
        if hasattr(self.client, "stream"):
            try:
                for chunk in self.client.stream(payload, MOUTH_SYSTEM_BQL):
                    yield chunk
                return
            except Exception:
                pass
        yield self._fallback_render(result)

    def _fallback_render(self, result: BrainQLResult) -> str:
        """Deterministic template-based rendering when LLM is unavailable."""
        if not result.known:
            return f"I don't know ({result.note})."
        v = result.value
        subj, rel = result.subj, result.rel
        chain = result.chain
        chain_str = " — ".join(chain) if chain else ""

        if result.op in ("LOOKUP", "DERIVE"):
            base = f"{subj} {rel}: {v}."
        elif result.op == "INHERIT":
            base = f"{subj} {rel} {v}" + (f" (inherited: {chain_str})" if chain_str else ".")
        elif result.op == "CHAIN":
            items = ", ".join(v) if isinstance(v, list) else str(v)
            base = f"{subj} is: {items}."
        elif result.op == "COMPUTE":
            base = f"{subj}.{rel} = {v:.4g}" if isinstance(v, float) else f"{subj}.{rel} = {v}"
        elif result.op == "TEACH":
            base = f"Got it: {subj} {rel} {v}."
        elif result.op == "TEACH_RULE":
            base = f"Rule registered: {v}."
        elif result.op == "EXPLAIN":
            base = f"{subj} {rel} {v}" + (f" because: {chain_str}" if chain_str else ".")
        else:
            base = f"{v}"
        return base


def _demo():
    from engines.reasoning.neuro_bridge import Mind, Brain
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
