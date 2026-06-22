#!/usr/bin/env python3
"""
rung3_parser.py — free-form parsing via a LOCAL LLM, verifier-gated.

Rungs 1-2 (lexical) and the distilled student handle direct/short phrasings. The
long tail — convoluted, multi-clause, indirect questions — is where a small LLM
earns its place as the Eyes. This uses the LOCAL qwen3:1.7B (offline, no cloud) to
map a free-form question to a structured Need, CONSTRAINED to the executive's known
entity/relation vocabulary and GATED by the executive: if the parse doesn't resolve
to a verified answer, it's rejected (honest "I don't know"), not hallucinated.

The escalation ladder: lexical -> distilled student -> LLM, each step only accepted
if it produces a solvable Need. The LLM never invents the ANSWER — only the routing
— and the verified executive produces the number.

    LLMParser(OllamaClient('qwen3:1.7B'), entities, relations).parse(q) -> (ent, rel)
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from llm_adapter import OllamaClient, StubClient, _first_json
from nl_front import _build


class LLMParser:
    def __init__(self, client, entities, relations):
        self.client = client
        self.entities = sorted(entities)
        self.relations = sorted(relations)
        self.system = (
            "Extract the entity and the attribute being asked about. "
            f"entity MUST be one of: {self.entities}. "
            f"attribute MUST be one of: {self.relations}. "
            'Output ONLY JSON: {"entity":"<one>","rel":"<one>"}. '
            "Pick the closest listed names; output nothing else.")

    def parse(self, question):
        data = _first_json(self.client.complete(question, self.system))
        if not data:
            return None, None
        ent, rel = data.get("entity"), data.get("rel")
        # GATE: only accept names that exist in the vocabulary
        ent = ent if ent in self.entities else None
        rel = rel if rel in self.relations else None
        return ent, rel


def _demo(use_stub=True):
    front = _build()
    entities = front.entities
    relations = sorted(front.solvable)

    # free-form questions rungs 1-2 / the student tend to miss
    queries = [
        "For the rocket, I'm after the quantity you get by multiplying its mass by its acceleration.",
        "Tell me, regarding that sample of ours, how tightly its matter is packed together.",
        "What kind of push is the rocket capable of producing?",
    ]
    if use_stub:
        # deterministic stand-in so the demo runs without invoking the model
        client = StubClient({
            "multiplying its mass by its acceleration": '{"entity":"rocket","rel":"force"}',
            "how tightly its matter is packed":         '{"entity":"sample","rel":"density"}',
            "push is the rocket capable":               '{"entity":"rocket","rel":"force"}',
        })
    else:
        client = OllamaClient("qwen3:1.7B")        # real local model, offline

    parser = LLMParser(client, entities, relations)
    print(f"=== rung3_parser — free-form via {'STUB' if use_stub else 'qwen3:1.7B'}, "
          f"verifier-gated ===\n")
    from policy_proposer import Solver
    for q in queries:
        ent, rel = parser.parse(q)
        if ent is None or rel is None:
            print(f"  > {q}\n      (LLM parse rejected — not in vocab)\n")
            continue
        val = Solver(front.kb, front.mem, use_proposer=True).solve(ent, rel)
        verdict = "—  (parsed but unknown)" if val is None else f"{val:.4g}"
        print(f"  > {q}")
        print(f"      -> Need({ent}, {rel}) = {verdict}\n")


if __name__ == "__main__":
    use_stub = "--real" not in sys.argv
    _demo(use_stub=use_stub)
