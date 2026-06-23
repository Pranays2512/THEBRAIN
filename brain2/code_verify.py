#!/usr/bin/env python3
"""
code_verify.py — a CODE verifier (the middle-tier product's gate).

Middle tier: the brain provides the logic/spec, an LLM writes the code, and the
brain's verifier CHECKS it before anything ships. Code is the cheapest domain to
verify — you just run it against the spec's test cases. Only code that passes
every test is admitted; a buggy candidate is rejected, no matter how plausible.

  spec = (function name, [(args, expected), ...])
  candidate code -> run in a restricted namespace on each test -> all pass? admit.

This is why the middle tier is strong: correctness is mechanically checkable, so an
unreliable generator (the LLM) is made reliable by the verifier gate.

SECURITY: candidate code is exec'd to test it (the whole point is running it). Runs
in a namespace with a minimal builtins whitelist — for verifying dev/LLM-generated
functions in an authorized loop, not untrusted input.

    python3 code_verify.py
"""

_SAFE_BUILTINS = {b: __builtins__[b] if isinstance(__builtins__, dict)
                  else getattr(__builtins__, b)
                  for b in ("range", "len", "min", "max", "abs", "sum", "int",
                            "float", "bool", "enumerate", "sorted", "list", "dict",
                            "str", "zip", "map", "filter", "all", "any", "round")}


def verify(code, fn_name, tests):
    """Run candidate `code`, call fn_name on each test; True iff all match.
    Returns (passed, detail)."""
    ns = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(code, ns)                              # define the function
    except Exception as e:
        return False, f"compile/define error: {e}"
    fn = ns.get(fn_name)
    if not callable(fn):
        return False, f"no function '{fn_name}' defined"
    for args, expected in tests:
        try:
            got = fn(*args)
        except Exception as e:
            return False, f"{fn_name}{args} raised {e}"
        if got != expected:
            return False, f"{fn_name}{args} = {got}, expected {expected}"
    return True, f"all {len(tests)} tests pass"


def _demo():
    tests = [((0,), 1), ((1,), 1), ((5,), 120), ((6,), 720)]   # spec: factorial
    correct = ("def fact(n):\n"
               "    r = 1\n"
               "    for i in range(2, n + 1):\n"
               "        r = r * i\n"
               "    return r\n")
    buggy = ("def fact(n):\n"
             "    r = 1\n"
             "    for i in range(2, n):\n"       # off-by-one: range(2, n) misses n
             "        r = r * i\n"
             "    return r\n")

    print("=== code_verify — brain spec + (LLM) code -> verifier gate ===\n")
    print("  spec: factorial, tests = fact(0)=1, fact(1)=1, fact(5)=120, fact(6)=720\n")
    for label, code in (("correct candidate", correct), ("buggy candidate", buggy)):
        ok, detail = verify(code, "fact", tests)
        print(f"  {label:18s} -> {'ADMITTED ✓' if ok else 'REJECTED ✗'}  ({detail})")
    print("\n  the verifier runs the code against the spec; only code that passes")
    print("  EVERY test ships. An unreliable code generator is made reliable here.")


if __name__ == "__main__":
    _demo()
