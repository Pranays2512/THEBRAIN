#!/usr/bin/env python3
"""
brain_store.py — the brain ACCUMULATES verified knowledge across sessions.

Until now every demo re-derived everything from scratch and forgot it on exit. This
is the self-extension made permanent: a persistent store of what the brain has
LEARNED — discovered formulas (policies), facts, and synthesized functions — each
gated by verification before it's admitted, saved to disk, and reloaded next run. The
brain never re-derives what it already knows; it builds on top.

  session: load store -> learn only the NEW things (verify -> admit) -> save
  across sessions the store GROWS, and prior knowledge is reused immediately.

    python3 brain_store.py        # runs two sessions, shows the store growing
"""

import json
import os
import random

from engines.synthesis.policy_induction import guided_induce, _render
from engines.synthesis import synth_engine as SE

STORE = os.path.join(os.path.dirname(__file__), "brain_store")


def _ev(e, env):
    if isinstance(e, (int, float)):
        return e
    if isinstance(e, str):
        return env[e]
    a, b = _ev(e[1], env), _ev(e[2], env)
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b, "^": a ** b}[e[0]]


class BrainStore:
    def __init__(self, path=STORE):
        self.path = path
        self.policies = {}     # name -> {"inputs":[...], "expr":[...]}
        self.facts = {}        # "entity|rel" -> value
        self.functions = {}    # name -> code
        self.load()

    def load(self):
        for f, attr in (("policies.json", "policies"), ("facts.json", "facts"),
                        ("functions.json", "functions")):
            p = os.path.join(self.path, f)
            if os.path.exists(p):
                setattr(self, attr, json.load(open(p)))

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        for f, attr in (("policies.json", "policies"), ("facts.json", "facts"),
                        ("functions.json", "functions")):
            json.dump(getattr(self, attr), open(os.path.join(self.path, f), "w"), indent=1)

    def knows_policy(self, name): return name in self.policies
    def knows_function(self, name): return name in self.functions

    def add_policy(self, name, inputs, expr):
        self.policies[name] = {"inputs": list(inputs), "expr": _tl(expr)}

    def add_function(self, name, code):
        self.functions[name] = code

    def use_policy(self, name, env):
        p = self.policies[name]
        return _ev(_ft(p["expr"]), env)

    def use_function(self, name, *args):
        ns = {}
        exec(self.functions[name], {"range": range, "len": len, "list": list,
                                    "max": max, "min": min}, ns)
        return ns[name](*args)

    def summary(self):
        return f"{len(self.policies)} policies, {len(self.functions)} functions, {len(self.facts)} facts"


def _tl(e):  return [_tl(x) for x in e] if isinstance(e, tuple) else e
def _ft(e):  return tuple(_ft(x) for x in e) if isinstance(e, list) else e


# ── a session: learn only what's NEW, verify, store ──────────────────────────
def _data(fn, inputs, n=14, seed=1):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r = {i: round(rng.uniform(1, 9), 2) for i in inputs}
        r["__y__"] = fn(r)
        rows.append(r)
    for r in rows:
        r["__name__"] = r.pop("__y__")
    return rows


def session(store, formula_tasks, code_tasks, label):
    print(f"\n=== {label} ===")
    print(f"  loaded: {store.summary()}")
    for name, inputs, fn in formula_tasks:
        if store.knows_policy(name):
            print(f"  policy   {name:10s} already known — reuse")
            continue
        rows = [{**{i: r[i] for i in inputs}, name: r["__name__"]} for r in _data(fn, inputs)]
        expr, _ = guided_induce([dict(r) for r in rows], inputs, name)
        if expr:
            store.add_policy(name, inputs, expr)
            print(f"  policy   {name:10s} DISCOVERED = {_render(expr)} (verified) -> stored")
    for name, kind, ex, oracle in code_tasks:
        if store.knows_function(name):
            print(f"  function {name:10s} already known — reuse")
            continue
        sp, code = SE.solve(ex, kind)
        if code and SE.stress(code, oracle, kind)[0]:
            # rename the synthesized 'f' to the task name
            store.add_function(name, code.replace("def f(", f"def {name}("))
            print(f"  function {name:10s} SYNTHESIZED (survives stress) -> stored")
    store.save()
    print(f"  saved:  {store.summary()}")


def _demo():
    import shutil
    shutil.rmtree(STORE, ignore_errors=True)             # fresh brain for a clean demo

    forms = [
        ("force", ["mass", "accel"], lambda r: r["mass"] * r["accel"]),
        ("ke", ["mass", "speed"], lambda r: 0.5 * r["mass"] * r["speed"] ** 2),
        ("density", ["mass", "volume"], lambda r: r["mass"] / r["volume"]),
    ]
    codes = [
        ("factorial", "int1", SE._ex("int1", lambda n: __import__("math").factorial(n),
                                     [0, 1, 4, 5, 6]), lambda n: __import__("math").factorial(n)),
        ("fibonacci", "int1", SE._ex("int1", _fib, [0, 1, 2, 3, 7, 10]), _fib),
        ("gcd", "int2", SE._ex("int2", __import__("math").gcd,
                               [(12, 8), (48, 36), (7, 5), (100, 80), (17, 13)]),
         __import__("math").gcd),
    ]
    print("=== brain_store — accumulate verified knowledge across sessions ===")
    # Session 1: knows nothing -> learns a subset
    session(BrainStore(), forms[:2], codes[:2], "Session 1 (fresh brain)")
    # Session 2: reloads -> remembers, skips known, learns the rest, REUSES a memory
    s2 = BrainStore()
    session(s2, forms, codes, "Session 2 (reloaded — grows on top)")
    print("\n  reuse a remembered LAW + FUNCTION (loaded from disk, not re-derived):")
    print(f"    force(mass=5, accel=3) = {s2.use_policy('force', {'mass': 5, 'accel': 3})}")
    print(f"    fibonacci(10)          = {s2.use_function('fibonacci', 10)}")
    print("\n  the brain remembered across sessions, skipped what it knew, grew on top,")
    print("  and reused stored knowledge — accumulation, every item verified before storage.")


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


if __name__ == "__main__":
    _demo()
