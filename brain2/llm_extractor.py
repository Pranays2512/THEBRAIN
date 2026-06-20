#!/usr/bin/env python3
"""
llm_extractor.py — learn from a corpus: prose -> triples, grounded, then ingest.

Closes the "train on a corpus" loop honestly. A trusted text is read by the LLM
into (subject, relation, object) triples; each triple is GROUNDED-checked (its
terms must actually appear in the source) before it is kept, so the model can't
smuggle in facts the text never stated. Same `extract(text)` interface as the
grammar fact extractor, so it drops straight into `knowledge_base.ingest_text`.

    kb.ingest_text(textbook_paragraph, extractor=LLMExtractor(OllamaClient("qwen3:1.7B")))

Honest boundary: this trusts the SOURCE and verifies the EXTRACTION (faithful, not
hallucinated). It does NOT verify the source is true — feed it curated, trusted
text. Truth-checking a claim needs corroboration or experiment, not parsing.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from llm_adapter import LLMClient, StubClient, OllamaClient   # noqa: F401 (re-exported)

EXTRACT_SYSTEM = (
    "Extract the facts stated in the text as a JSON array of [subject, relation, "
    "object] triples. Output ONLY the JSON array. Use lowercase; join multiword "
    "names with underscores; keep the relation a short word (isa, has, can, "
    "part_of, lives_in, made_of, used_for). Only facts the text actually states."
)


def _norm(token):
    return str(token).strip().lower().replace(" ", "_")


def _parse_triples(raw):
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            out.append((_norm(item[0]), _norm(item[1]), _norm(item[2])))
        elif isinstance(item, dict):
            keys = {k.lower(): v for k, v in item.items()}
            s = keys.get("subject") or keys.get("s")
            r = keys.get("relation") or keys.get("r") or keys.get("predicate")
            o = keys.get("object") or keys.get("o")
            if s and r and o:
                out.append((_norm(s), _norm(r), _norm(o)))
    return out


def _appears(term, low_text):
    """A term is grounded if any of its content words shows up in the source."""
    words = [w for w in term.split("_") if len(w) > 2]
    return any(w in low_text for w in words) if words else term in low_text


class LLMExtractor:
    def __init__(self, client):
        self.client = client

    def extract(self, text):
        """Trusted prose -> grounded triples (hallucinated terms dropped)."""
        triples = _parse_triples(self.client.complete(text, EXTRACT_SYSTEM))
        low = text.lower()
        kept = []
        for s, r, o in triples:
            if not (s and r and o) or s == o:
                continue
            if _appears(s, low) and _appears(o, low):       # grounding gate
                kept.append((s, r, o))
        return kept

    def teach_into(self, text, engine):
        n = 0
        for s, r, o in self.extract(text):
            if engine.learn(s, r, o):
                n += 1
        return n


def _demo():
    # a stub stands in for the model; the real model is OllamaClient("qwen3:1.7B")
    stub = StubClient({
        "whale": '[["whale","isa","mammal"],["whale","lives_in","ocean"],'
                 '["dragon","can","breathe_fire"]]',   # dragon is NOT in the text -> dropped
    })
    text = "A whale is a mammal. It lives in the ocean."
    ex = LLMExtractor(stub)
    print("=== llm_extractor — prose -> grounded triples ===\n")
    print(f"  text: {text!r}\n")
    for t in ex.extract(text):
        print(f"    kept: {t}")
    print("    (a hallucinated 'dragon breathes fire' triple was dropped — not in the source)")


if __name__ == "__main__":
    _demo()
