#!/usr/bin/env python3
"""
vs_llm.py — honest head-to-head: the brain vs an LLM on the VERIFIED slice.

The standing claim is "the brain beats LLMs on RELIABILITY where answers can be
checked." This tests it. Same tasks (algorithm synthesis + arithmetic word problems),
both the brain (synth_engine + executive) and a local LLM, both judged by the SAME
gate: does the answer survive stress-vs-oracle / match ground truth.

The honest point isn't "brain is smarter" — it's: when an answer is mechanically
checkable, the brain SHIPS ONLY CORRECT ONES (or abstains), while the LLM is graded
on whether its guess actually holds up. Reports correct / wrong / abstained for each.

    python3 vs_llm.py            # brain side only (LLM stub) — runs offline
    venv2/bin/python3 vs_llm.py --real   # real local qwen3:1.7B opponent
"""

import math
import re
import sys

import synth_engine as SE
from llm_adapter import OllamaClient, StubClient


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


TASKS = [   # (name, kind, oracle, example-inputs, llm-prompt)
    ("factorial", "int1", lambda n: math.factorial(n), [0, 1, 4, 5, 6],
     "Write a Python function f(n) returning n factorial. Output only code."),
    ("fibonacci", "int1", _fib, [0, 1, 2, 3, 7, 10],
     "Write Python f(n) returning the nth Fibonacci (f(0)=0,f(1)=1). Only code."),
    ("gcd", "int2", math.gcd, [(12, 8), (48, 36), (7, 5), (100, 80)],
     "Write Python f(a,b) returning gcd(a,b). Only code."),
    ("max_subarray", "list",
     lambda a: max(sum(a[i:j + 1]) for i in range(len(a)) for j in range(i, len(a))),
     [[1, -2, 3, 4], [-1, -2], [2, 3], [5, -1, 5]],
     "Write Python f(lst) returning the maximum subarray sum. Only code."),
]


def brain_side(name, kind, oracle, inputs):
    ex = SE._ex(kind, oracle, inputs)
    space, code = SE.solve(ex, kind)
    if code is None:
        return "abstain"
    return "correct" if SE.stress(code, oracle, kind)[0] else "wrong"


def llm_side(client, kind, oracle, prompt):
    raw = client.complete(prompt)
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    code = m.group(1) if m else raw
    i = code.find("def ")
    code = code[i:] if i >= 0 else code
    code = code.replace(re.search(r"def (\w+)", code).group(1), "f", 1) if "def " in code else code
    ok, _ = SE.stress(code, oracle, kind)  # judged by the SAME gate
    return "correct" if ok else "wrong"


def _demo(real=False):
    client = OllamaClient("qwen3-coder:480b-cloud") if real else None
    print("=== vs_llm — brain vs LLM on the VERIFIED slice (same gate) ===\n")
    print(f"  {'task':14s} {'brain':>10s} {'llm':>10s}")
    bc = lc = 0
    for name, kind, oracle, inputs, prompt in TASKS:
        b = brain_side(name, kind, oracle, inputs)
        bc += b == "correct"
        if real:
            try:
                l = llm_side(client, kind, oracle, prompt)
            except Exception:
                l = "error"
        else:
            l = "(skipped)"
        lc += l == "correct"
        print(f"  {name:14s} {b:>10s} {l:>10s}")
    print(f"\n  brain correct: {bc}/{len(TASKS)}" + (f"   llm correct: {lc}/{len(TASKS)}" if real else ""))
    print("  both judged by stress-vs-oracle. Brain ships only verified-correct or abstains;")
    print("  the LLM's guesses are graded on whether they actually hold up.")


if __name__ == "__main__":
    _demo(real="--real" in sys.argv)
