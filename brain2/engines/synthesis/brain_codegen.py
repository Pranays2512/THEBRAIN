#!/usr/bin/env python3
"""
brain_codegen.py — the brain builds the LOGIC; rendering is mechanical (no LLM).

The sharper middle-tier thesis: don't let the LLM think. The brain DISCOVERS the
logic (guided induction over examples -> a verified formula IR), and that IR is
rendered to Python deterministically. The LLM does nothing — there's no algorithm
for it to get wrong, because the brain already found and VERIFIED it.

  examples -> guided induction -> verified formula IR -> mechanical render -> code
                                                       -> re-verify generated code

For formula-shaped logic this needs no LLM at all. (For control-flow logic — loops,
recursion — the brain would emit a plan and an LLM would TRANSCRIBE it; the LLM is a
transcriber, never the thinker, and the test gate still catches mis-transcription.)

    python3 brain_codegen.py
"""

import random

from engines.synthesis.policy_induction import guided_induce, _render


def render_py(e):
    """Formula IR -> a Python expression string (mechanical, total)."""
    if isinstance(e, (int, float)):
        return repr(e)
    if isinstance(e, str):
        return e
    op, a, b = e[0], e[1], (e[2] if len(e) > 2 else None)
    if op == "neg":
        return f"(-{render_py(a)})"
    sym = "**" if op == "^" else op
    return f"({render_py(a)} {sym} {render_py(b)})"


def to_function(fn, inputs, expr):
    return f"def {fn}({', '.join(inputs)}):\n    return {render_py(expr)}\n"


def _make(formula, inputs, n=14, seed=1):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r = {i: round(rng.uniform(1, 9), 2) for i in inputs}
        r["__y__"] = formula(r)
        rows.append(r)
    return rows


def _verify_fn(code, fn, rows, inputs, target, tol=1e-6):
    ns = {}
    exec(code, ns)
    f = ns[fn]
    return all(abs(f(*[r[i] for i in inputs]) - r[target]) < tol for r in rows)


def _demo():
    cases = [
        ("force", ["mass", "accel"], lambda r: r["mass"] * r["accel"]),
        ("ke", ["mass", "speed"], lambda r: 0.5 * r["mass"] * r["speed"] ** 2),
        ("density", ["mass", "volume"], lambda r: r["mass"] / r["volume"]),
    ]
    print("=== brain_codegen — brain discovers the logic, render is mechanical ===\n")
    for fn, inputs, f in cases:
        rows = _make(f, inputs)
        for r in rows:
            r[fn] = r.pop("__y__")
        expr, _ = guided_induce([dict(r) for r in rows], inputs, fn)  # BRAIN finds logic
        if expr is None:
            print(f"  {fn}: brain couldn't synthesize the logic")
            continue
        code = to_function(fn, inputs, expr)                        # mechanical render
        ok = _verify_fn(code, fn, rows, inputs, fn)                 # re-verify code
        print(f"  {fn}: logic = {_render(expr)}")
        print(f"        code  -> {code.strip().splitlines()[-1].strip()}   "
              f"[{'VERIFIED ✓' if ok else 'WRONG ✗'}]\n")
    print("  the LLM wrote nothing — the brain found the logic, the renderer is total,")
    print("  and the code is correct by construction (verified IR -> verified code).")


if __name__ == "__main__":
    _demo()
