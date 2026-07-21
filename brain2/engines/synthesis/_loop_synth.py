#!/usr/bin/env python3
"""
loop_synth.py — the brain synthesizes ALGORITHMS (loops), not just formulas.

brain_codegen built straight-line formulas. This adds CONTROL FLOW: the brain
searches a small imperative DSL — an accumulator loop — for a program matching
input/output examples, verifies on held-out cases, and renders it to Python. No
LLM: the brain finds the algorithm (init + loop range + update step), the renderer
is mechanical, the code is re-verified.

  DSL:  acc = INIT ; for i in RANGE: acc = UPDATE(acc, i) ; return acc
  search INIT x RANGE x UPDATE  ->  fits examples + held-out  ->  render -> verify

Covers the fold/accumulator family (factorial, sums, sum-of-squares, max, count) —
a broad class of real algorithms with loops. Not arbitrary programs (no recursion
trees / sorting), but genuine control-flow synthesis beyond formulas.

    python3 loop_synth.py
"""

# search space: inits, loop ranges, update steps (name -> (python expr, fn))
INITS = [0, 1]
RANGES = [                       # (python lo, python hi, fn(n)->range)
    ("1", "n + 1", lambda n: range(1, n + 1)),
    ("2", "n + 1", lambda n: range(2, n + 1)),
    ("0", "n",     lambda n: range(0, n)),
    ("1", "n",     lambda n: range(1, n)),
]
UPDATES = {
    "acc + i":      lambda acc, i: acc + i,
    "acc * i":      lambda acc, i: acc * i,
    "acc + i * i":  lambda acc, i: acc + i * i,
    "max(acc, i)":  lambda acc, i: max(acc, i),
    "acc + 1":      lambda acc, i: acc + 1,
}


def run(init, rng_fn, upd_fn, n):
    acc = init
    for i in rng_fn(n):
        acc = upd_fn(acc, i)
    return acc


def synthesize(examples):
    """Find (init, range, update) fitting train + held-out. Returns a render spec."""
    if len(examples) < 4:
        return None
    cut = max(3, int(len(examples) * 0.6))
    train, hold = examples[:cut], examples[cut:]
    for init in INITS:
        for lo, hi, rfn in RANGES:
            for ucode, ufn in UPDATES.items():
                try:
                    if all(run(init, rfn, ufn, n) == y for n, y in train) and \
                       all(run(init, rfn, ufn, n) == y for n, y in hold):
                        return {"init": init, "lo": lo, "hi": hi, "upd": ucode}
                except Exception:
                    continue
    return None


def render(fn, spec):
    return (f"def {fn}(n):\n"
            f"    acc = {spec['init']}\n"
            f"    for i in range({spec['lo']}, {spec['hi']}):\n"
            f"        acc = {spec['upd']}\n"
            f"    return acc\n")


def _verify(code, fn, examples):
    ns = {}
    exec(code, {"max": max, "range": range}, ns)
    f = ns[fn]
    return all(f(n) == y for n, y in examples)


def _demo():
    cases = {
        "fact": [(0, 1), (1, 1), (4, 24), (5, 120), (6, 720)],            # factorial
        "tri":  [(1, 1), (2, 3), (3, 6), (5, 15), (10, 55)],              # 1+..+n
        "sumsq": [(1, 1), (2, 5), (3, 14), (4, 30), (5, 55)],             # 1²+..+n²
        "biggest": [(3, 3), (5, 5), (7, 7), (10, 10), (2, 2)],            # max in 1..n
    }
    print("=== loop_synth — brain synthesizes ALGORITHMS (loops), no LLM ===\n")
    for fn, ex in cases.items():
        spec = synthesize(ex)
        if spec is None:
            print(f"  {fn:8s}: no loop program found in the DSL")
            continue
        code = render(fn, spec)
        ok = _verify(code, fn, ex)
        body = f"acc={spec['init']}; for i in range({spec['lo']},{spec['hi']}): acc={spec['upd']}"
        print(f"  {fn:8s}: {body}   [{'VERIFIED ✓' if ok else 'WRONG ✗'}]")
    print("\n  the brain searched init x range x update, found the loop that fits the")
    print("  examples, verified it held out, and rendered Python — control flow, no LLM.")


if __name__ == "__main__":
    _demo()
