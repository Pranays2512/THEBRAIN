#!/usr/bin/env python3
"""
code_loop.py — the middle-tier product, closed: spec -> code -> verify -> repair.

The brain owns the SPEC (description + test cases = the logic). An LLM writes the
code. The verifier RUNS it against the tests. On failure, the failure is fed back
and the LLM repairs — looping until the code passes or the budget runs out. Only
verified code ships; a wrong-but-plausible draft never does. An unreliable code
generator becomes a reliable code producer because the gate is mechanical.

  spec(desc, fn, tests) -> [generate -> verify -> (repair)]* -> verified code | honest fail

Uses qwen3-coder:480b-cloud (the code model) with --real; a deterministic stub
otherwise (demonstrates the repair loop: buggy draft -> rejected -> fixed -> admitted).

    python3 code_loop.py            # stub (offline, shows repair)
    venv2/bin/python3 code_loop.py --real   # real qwen3-coder
"""

import re
import sys

from code_verify import verify
from llm_adapter import OllamaClient, StubClient


def extract_code(text):
    """Pull the function body out of an LLM reply (strip markdown fences/prose)."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    # keep from the first 'def' onward
    i = code.find("def ")
    return code[i:] if i >= 0 else code


def _prompt(spec, prior):
    desc, fn, tests = spec
    ex = "; ".join(f"{fn}{a}=={r}" for a, r in tests[:4])
    p = (f"Write a Python function `{fn}` that {desc}. "
         f"It must satisfy: {ex}. Output ONLY the function code.")
    if prior:
        code, detail = prior
        p = (f"Your code failed: {detail}\n\n{code}\n\n"
             f"Fix it. The function `{fn}` must satisfy: {ex}. "
             "Output ONLY the corrected function code.")
    return p


def solve(spec, client, max_tries=3):
    desc, fn, tests = spec
    prior = None
    for attempt in range(1, max_tries + 1):
        code = extract_code(client.complete(_prompt(spec, prior)))
        ok, detail = verify(code, fn, tests)
        if ok:
            return code, attempt, True, detail
        prior = (code, detail)
    return code, max_tries, False, detail


def _demo(real=False):
    specs = [
        ("returns n factorial", "fact", [((0,), 1), ((1,), 1), ((5,), 120), ((6,), 720)]),
        ("returns the nth Fibonacci number (fib(0)=0, fib(1)=1)", "fib",
         [((0,), 0), ((1,), 1), ((7,), 13), ((10,), 55)]),
        ("returns True if the string is a palindrome", "is_pal",
         [(("racecar",), True), (("hello",), False), (("",), True)]),
    ]
    if real:
        client = OllamaClient("qwen3-coder:480b-cloud")
    else:
        # stub: buggy draft first; on a repair prompt ("Fix it") returns the fix
        buggy = "def fact(n):\n r=1\n for i in range(2,n):\n  r*=i\n return r\n"
        fixed = "def fact(n):\n r=1\n for i in range(2,n+1):\n  r*=i\n return r\n"
        client = StubClient({"Fix it": fixed, "factorial": buggy})
        specs = specs[:1]                         # stub only knows factorial

    print(f"=== code_loop — spec -> code -> verify -> repair "
          f"({'qwen3-coder' if real else 'stub'}) ===\n")
    for desc, fn, tests in specs:
        code, tries, ok, detail = solve((desc, fn, tests), client)
        status = "VERIFIED ✓" if ok else "FAILED ✗"
        print(f"  {fn:8s} ({desc[:38]}): {status} in {tries} attempt(s) — {detail}")
    print("\n  only code that passes every test ships; failures trigger repair.")


if __name__ == "__main__":
    _demo(real="--real" in sys.argv)
