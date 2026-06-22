#!/usr/bin/env python3
"""
learn_by_reading.py — the autonomous self-extension loop (the "learns logic" thesis).

    corpus (text)  ->  LLM extracts numeric examples  ->  guided induction
    discovers the formula  ->  held-out VERIFY  ->  STORE as a policy in the Brain
    ->  the Brain can now answer NEW instances of a law it learned BY READING.

The LLM is only the reading EYE (text -> numbers); the discovery is the brain's own
guided induction; the gate is held-out verification; the result lives in the C++
Brain (brain.policy_add) and is used by brain.policy_solve. Nothing is stored unless
it generalizes — induction proposes, verification disposes.

Run with the interpreter that has brain2 (and a model, or the stub):
    venv2/bin/python3 learn_by_reading.py            # deterministic stub reader
    venv2/bin/python3 learn_by_reading.py --real     # real local qwen3:1.7B reader
"""

import json
import random
import sys

import brain2
from policy_induction import guided_induce, _render
from llm_adapter import OllamaClient, StubClient, _first_json

EXTRACT_SYS = ("Extract every numeric quantity mentioned as flat JSON "
               '{name: number}. Use the physical name (mass, acceleration, force). '
               "Output ONLY the JSON object.")


def read_examples(corpus, client):
    """LLM reads each sentence -> a row of numeric values."""
    rows = []
    for s in corpus:
        d = _first_json(client.complete(s, EXTRACT_SYS))
        if d:
            try:
                rows.append({k: float(v) for k, v in d.items()})
            except (TypeError, ValueError):
                pass
    return rows


def learn(corpus, inputs, target, client, brain):
    """Read -> induce -> verify -> store in the brain. Returns (expr, n_rows, nodes)."""
    rows = read_examples(corpus, client)
    expr, nodes = guided_induce([dict(r) for r in rows], inputs, target)
    if expr is None:
        return None, len(rows), nodes
    brain.policy_add(target, inputs, expr)        # store the discovered law in the Brain
    return expr, len(rows), nodes


def _demo(real=False):
    # a corpus describing a hidden law (force = mass * acceleration), varied numbers
    rng = random.Random(0)
    samples = [(rng.randint(2, 9), rng.randint(2, 9)) for _ in range(10)]
    corpus = [f"An object of mass {m} under acceleration {a} feels a force of {m * a}."
              for m, a in samples]

    if real:
        client = OllamaClient("qwen3:1.7B")
    else:                                         # deterministic reader for testing
        client = StubClient({c: json.dumps({"mass": m, "accel": a, "force": m * a})
                             for c, (m, a) in zip(corpus, samples)})

    brain = brain2.Brain(som_rows=8, som_cols=8, n_dims=8)
    print(f"=== learn_by_reading — read text, discover the law, store it "
          f"({'qwen3:1.7B' if real else 'stub'}) ===\n")
    print(f"  read {len(corpus)} sentences, e.g.: {corpus[0]!r}\n")

    expr, n, nodes = learn(corpus, ["mass", "accel"], "force", client, brain)
    if expr is None:
        print("  (could not discover a law from the reading)")
        return
    print(f"  DISCOVERED from reading: force = {_render(expr)}   "
          f"(from {n} examples, {nodes} search nodes, held-out verified)\n")

    # the brain now answers a NEW instance of the law it learned by reading
    brain.teach_fact("crate", "mass", 7.0)
    brain.teach_fact("crate", "accel", 5.0)
    ans = brain.policy_solve("crate", "force")
    print(f"  brain.policy_solve crate.force = {ans}   (expected 35 — a law learned")
    print("  by READING, applied to a new object, via the C++ Brain)")


if __name__ == "__main__":
    _demo(real="--real" in sys.argv)
