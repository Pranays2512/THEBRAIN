#!/usr/bin/env python3
"""
benchmark_pro.py — run MMLU-Pro honestly, separating brain from LLM.

These are knowledge benchmarks. The symbolic brain holds ~1600 common-sense
facts, so it covers almost none of MMLU-Pro and CORRECTLY declines (coverage ~0).
The qwen3:1.7B is only the eyes/mouth and adds no knowledge — so to even attempt
the questions, this also measures the 1.7B ALONE (the LLM doing the whole task,
brain bypassed) as the honest baseline. The point isn't a good score; it's an
honest measurement of the coverage gap, not an estimate.

    python3 benchmark_pro.py [n]

Honest framing: brain-only ~0 (it doesn't know grad knowledge, and won't bluff);
qwen3-1.7B-direct is just a weak small model on a hard 10-choice test. Lifting
this needs the knowledge-ingestion grind, not the symbolic core.
"""

import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from neuro_bridge import Brain, RuleEyes
from llm_adapter import OllamaClient

LETTERS = "ABCDEFGHIJ"


def fetch_mmlu_pro(n, offset=0):
    url = ("https://datasets-server.huggingface.co/rows?dataset=TIGER-Lab/MMLU-Pro"
           f"&config=default&split=test&offset={offset}&length={n}")
    with urllib.request.urlopen(url, timeout=30) as r:
        rows = json.loads(r.read())["rows"]
    return [x["row"] for x in rows]


def brain_coverage(questions):
    """How many the brain can answer at all (expected ~0 — honest declines)."""
    brain, eyes = Brain(), RuleEyes()
    known = 0
    for q in questions:
        a = brain.answer(eyes.parse(q["question"]))
        known += int(a.known)
    return known, len(questions)


def llm_direct(questions, client):
    """The 1.7B alone picks a letter — the LLM baseline, brain bypassed."""
    sysmsg = "Answer the multiple-choice question with ONLY the letter of the correct option."
    correct = 0
    for q in questions:
        opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(q["options"]))
        prompt = f"Question: {q['question']}\nOptions:\n{opts}\nAnswer:"
        out = client.complete(prompt, sysmsg)
        m = re.search(r"[A-J]", out.upper())
        pick = m.group(0) if m else "?"
        correct += int(pick == q["answer"])
    return correct, len(questions)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"=== MMLU-Pro on the current architecture (n={n}) ===\n")
    qs = fetch_mmlu_pro(n)

    bk, bt = brain_coverage(qs)
    print(f"brain-only coverage:   {bk}/{bt}   (it declines what it doesn't know — honest)")

    ck, ct = llm_direct(qs, OllamaClient("qwen3:1.7B"))
    rand = 100.0 / 10
    print(f"qwen3:1.7B direct:     {ck}/{ct}  = {ck/ct*100:.0f}%   (random ~{rand:.0f}%)")
    print("\nHonest read: the symbolic core adds nothing here (no grad knowledge "
          "ingested);\nthe number is just a weak 1.7B on a hard 10-choice test. "
          "Lifting it = ingest knowledge,\nnot tune the brain. GPQA is gated on "
          "HuggingFace (needs auth) — not fetchable here.")


if __name__ == "__main__":
    main()
