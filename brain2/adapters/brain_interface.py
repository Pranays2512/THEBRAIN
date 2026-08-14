#!/usr/bin/env python3
"""
brain_interface.py — the single unified entry point for all brain interaction.

Replaces the three scattered pipelines:
  - adapters/chat.py        (C++ direct, bypasses Python reasoning)
  - engines/reasoning/neuro_bridge.Mind  (math only)
  - faculties/whole_brain.WholeBrain.ask() (Python reasoning only)

Architecture:
  User text
    → BrainInterface.respond(text)
        ├─ C++ Brain.perceive_text()     [always: SOM/emotion/episodic]
        ├─ BrainQLEyes.parse(text)       [LLM → BrainQL instructions]
        │     ├─ If BrainQL returned:
        │     │     → WholeBrain.execute_bql(queries)   [brain reasons]
        │     │     → BrainQLMouth.render_result()      [LLM verbalizes]
        │     └─ If math Query returned (LLM offline or math phrasing):
        │           → neuro_bridge.Brain.answer()       [exact math]
        │           → GrammarMouth.render()             [deterministic]
        └─ Falls back to WholeBrain.ask() if all else fails

LLM is STRICTLY eyes and mouth:
  - Eyes: natural language → BrainQL (or math Query for exact math)
  - Mouth: BrainQLResult → fluent English (constrained to brain's answer)
  - Brain: owns all reasoning, no LLM involvement in the middle

Usage:
    from adapters.brain_interface import BrainInterface
    from adapters.llm_adapter import OllamaClient

    client = OllamaClient("qwen2.5")
    brain = BrainInterface(client=client)

    # Teach facts
    brain.teach("acid", "turns_litmus", "red")
    brain.teach("HCL", "isa", "acid")

    # Ask — brain generalizes, LLM only verbalizes
    print(brain.respond("Does HCL turn litmus red?"))
    # → "HCL turns litmus red because it is an acid, and all acids turn litmus red."

    # Offline / no LLM — BrainQL still works via StubClient / deterministic fallback
    offline_brain = BrainInterface(client=None)
    offline_brain.teach("acid", "turns_litmus", "red")
    offline_brain.teach("HCL", "isa", "acid")
    print(offline_brain.respond("INHERIT HCL turns_litmus"))  # direct BrainQL passthrough
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.reasoning.brainql import (
    BrainQLQuery, BrainQLResult, BrainQLExecutor,
    parse_bql_block, BrainQLParseError,
)
from engines.reasoning.neuro_bridge import Brain as NeuroBrain, GrammarMouth
from adapters.llm_adapter import BrainQLEyes, BrainQLMouth, StubClient


class BrainInterface:
    """The unified LLM↔Brain pipeline.

    The brain is the thinker. The LLM is only the translator (eyes/mouth).

    Args:
        client   : any LLMClient with .complete(prompt, system) → str.
                   Use OllamaClient for Ollama, StubClient for tests,
                   or None for fully deterministic offline operation.
        wb       : optional pre-built WholeBrain instance. If None, a fresh
                   WholeBrain is built (this takes a few seconds).
    """

    def __init__(self, client=None, wb=None):
        # Build LLM clients: stub when offline
        if client is None:
            self._client = StubClient({})  # empty stub → always returns '' → fallback renders
        else:
            self._client = client

        # Eyes: BrainQLEyes tries exact math first, then BrainQL via LLM
        self._eyes = BrainQLEyes(self._client)

        # Mouth: BrainQLMouth verbalises BrainQLResult via LLM, falls back deterministically
        self._mouth = BrainQLMouth(self._client)

        # Grammar mouth for math Queries (unchanged exact path)
        self._grammar_mouth = GrammarMouth()

        # Neuro-bridge Brain for math queries
        self._nb_brain = NeuroBrain()

        # The full symbolic WholeBrain (lazy-load to avoid import cost in tests)
        if wb is not None:
            self._wb = wb
        else:
            try:
                from faculties.whole_brain import WholeBrain
                self._wb = WholeBrain(eyes=self._eyes)
                # Ensure isa is always transitive
                self._wb.kre.set_transitive("isa")
            except Exception as e:
                print(f"[BrainInterface] WholeBrain init failed ({e}); using neuro-bridge only")
                self._wb = None

        # Direct BrainQL executor (used when WholeBrain is unavailable)
        self._exec = BrainQLExecutor(
            self._nb_brain.lang.r,
            means_ends_solver=None,
        )

    # ── teaching ──────────────────────────────────────────────────────────────

    def teach(self, subj: str, rel: str, obj: str) -> bool:
        """Teach a fact to the brain. Returns True if the fact was new.

        Uses WholeBrain.teach() (not kre.learn()) so the entity/relation sets
        stay live and the BrainQL router can find newly taught entities immediately.
        """
        if self._wb is not None:
            return self._wb.teach(subj, rel, obj)
        return self._nb_brain.teach(subj, rel, obj)

    def teach_rule(self, prem1: str, prem2: str, concl: str):
        """Teach a composition rule: X prem1 Y AND Y prem2 Z => X concl Z."""
        if self._wb is not None:
            self._wb.kre.add_rule(prem1, prem2, concl)
        self._nb_brain.set_transitive("isa")  # always keep isa transitive

    def set_transitive(self, rel: str):
        """Mark a relation as transitive (X rel Y AND Y rel Z => X rel Z)."""
        if self._wb is not None:
            self._wb.kre.set_transitive(rel)
        self._nb_brain.set_transitive(rel)

    # ── querying ──────────────────────────────────────────────────────────────

    def respond(self, text: str) -> dict:
        """Full pipeline: natural language → brain → structured response.

        Returns a dict::
            {
                "reply":    str,   # fluent English answer
                "kind":     str,   # 'bql', 'math', 'fallback', 'unknown'
                "verified": bool,  # True when the answer is structurally verified
            }

        Route:
          1. If text is already BrainQL (starts with a BrainQL op), execute directly.
          2. BrainQLEyes: try exact math parser first; then ask LLM for BrainQL.
          3. If BrainQL returned: WholeBrain.execute_bql() → BrainQLMouth.render_result()
          4. If math Query returned: NeuroBrain.answer() → GrammarMouth.render()
          5. Fallback: WholeBrain.sense() (existing symbolic pipeline)
        """
        # 1. Direct BrainQL passthrough (REPL / raw API)
        first_word = text.strip().split()[0].upper() if text.strip() else ""
        from engines.reasoning.brainql import VALID_OPS
        if first_word in VALID_OPS:
            try:
                queries = parse_bql_block(text)
                reply = self._run_bql(queries)
                return {"reply": reply, "kind": "bql", "verified": True}
            except BrainQLParseError:
                pass

        # 2. Eyes: exact math first, then LLM→BrainQL
        parsed = self._eyes.parse(text)

        # 3. BrainQL path (LLM translated to BrainQL successfully)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], BrainQLQuery):
            reply = self._run_bql(parsed)
            verified = not reply.lower().startswith("i don't know")
            return {"reply": reply, "kind": "bql", "verified": verified}

        # 4. Math / exact-parse Query path
        ans = self._nb_brain.answer(parsed)
        if ans.known:
            return {"reply": self._grammar_mouth.render(ans), "kind": "math", "verified": True}

        # 5. Fallback: WholeBrain.sense() — full symbolic pipeline
        if self._wb is not None:
            result = self._wb.sense(text)
            a = result.get("answer", {})
            msg = a.get("msg") or "I don't know."
            return {"reply": msg, "kind": "fallback", "verified": bool(a.get("verified"))}

        return {"reply": "I don't know.", "kind": "unknown", "verified": False}

    def respond_str(self, text: str) -> str:
        """Convenience wrapper: respond() → just the reply string."""
        return self.respond(text)["reply"]

    def respond_bql(self, text: str) -> list:
        """Like respond() but returns the raw list[BrainQLResult] instead of text.

        Useful for programmatic callers that want the structured answer.
        """
        first_word = text.strip().split()[0].upper() if text.strip() else ""
        from engines.reasoning.brainql import VALID_OPS
        if first_word in VALID_OPS:
            try:
                queries = parse_bql_block(text)
                return self._execute(queries)
            except BrainQLParseError:
                pass

        parsed = self._eyes.parse(text)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], BrainQLQuery):
            return self._execute(parsed)
        return []

    # ── internal helpers ──────────────────────────────────────────────────────

    def _execute(self, queries: list) -> list:
        """Execute BrainQL queries using WholeBrain if available, else direct executor."""
        if self._wb is not None:
            return self._wb.execute_bql(queries)
        return self._exec.run_block(queries)

    def _run_bql(self, queries: list) -> str:
        """Execute queries and verbalize the results."""
        results = self._execute(queries)
        parts = [self._mouth.render_result(r) for r in results]
        return " ".join(parts) if parts else "I don't know."

    # ── REPL convenience ──────────────────────────────────────────────────────

    def repl(self):
        """Interactive REPL for the BrainInterface.

        Accepts natural language or raw BrainQL. Type 'quit' to exit.
        """
        print("BrainInterface REPL — type natural language or BrainQL ('quit' to exit)")
        print("  TEACH  <subj> <rel> <obj>     — assert a fact")
        print("  INHERIT <subj> <rel>           — infer via isa chain")
        print("  Or just ask in plain English.\n")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            try:
                print(f"Brain: {self.respond_str(text)}\n")
            except Exception as e:
                print(f"[error] {e}\n")


# ── Demo ─────────────────────────────────────────────────────────────────────

def _demo():
    print("=== BrainInterface demo ===\n")

    # Offline (no LLM): use StubClient that maps "turn litmus" → BrainQL
    stub = StubClient({
        "turn litmus": "INHERIT hcl turns_litmus",
        "what is hcl": "CHAIN hcl isa",
    })

    bi = BrainInterface(client=stub)

    # Teach domain facts
    bi.teach("acid", "turns_litmus", "red")
    bi.teach("HCL", "isa", "acid")
    bi.set_transitive("isa")

    print("Teaching: acid turns_litmus red")
    print("Teaching: HCL isa acid")
    print()

    # Direct BrainQL passthrough
    print("Direct BrainQL: INHERIT HCL turns_litmus")
    print(f"  → {bi.respond_str('INHERIT HCL turns_litmus')}\n")

    # Structured result
    results = bi.respond_bql("INHERIT HCL turns_litmus")
    r = results[0] if results else None
    if r:
        print(f"Structured result: known={r.known} value={r.value!r} chain={r.chain}")
        assert r.value == "red", f"FAIL: expected 'red', got {r.value!r}"
        print("  ✓ Generalization works!\n")

    # Natural language (stub translates "turn litmus" → BrainQL)
    print("NL query: 'Does HCL turn litmus red?'")
    print(f"  → {bi.respond_str('Does HCL turn litmus red?')}\n")

    print("=== demo passed ===")


if __name__ == "__main__":
    _demo()
