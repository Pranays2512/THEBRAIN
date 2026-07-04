#!/usr/bin/env python3
"""math_synth.py — the brain LEARNS arithmetic instead of calling it.

The C++ core has frozen skills: MATH_MUL is host `a * b`, MATH_QUAD a baked
quadratic solver (logic_engine.hpp). The basal ganglia only learns WHICH frozen
op to route to — never what multiply IS. Math is called, not known.

This grounds arithmetic from the floor, reusing the PROVEN synthesis engine
(tree_reason.solve — the same A* program_synth uses). The substrate the string
DSL lacked:

  floor atoms:  Z (zero), S (successor +1), P (predecessor -1, truncated)
  variables:    a, b, r   (operands + the recursive result)
  combinator:   primitive recursion, on EITHER argument (the `repeat` we added):
     recurse on a:  f(0, b) = BASE(b)   ;  f(Sa, b) = STEP(a, b, f(a,b))
     recurse on b:  f(a, 0) = BASE(a)   ;  f(a, Sb) = STEP(a, b, f(a,b))

The synthesiser searches BASE+STEP whose function matches the I/O examples —
VERIFIED, then checked on held-out large inputs (generalisation, not memory).
Solved functions are CACHED into the basis (library abstraction); each new op is
built from earlier learned ones. That cache is the reattach unit.

Claim under test: from only {S, P} + recursion (no host + - * /), synthesise
add, sub, mul, pow — each grounded in the last. If yes, the brain LEARNED math.

    /opt/homebrew/bin/python3.13 math_synth.py       # proof + library
    from math_synth import LearnedArithmetic         # importable module
"""
from tree_reason import SearchProblem, solve


# ── work budget: grounded arithmetic is O(value) (unary recursion), so a bad
# candidate like mul(r,r) loops astronomically. Count iterations; explosive
# candidates blow the budget and get PRUNED. Reset per top-level evaluation.
_WORK = [0]
_BUDGET = [200_000]


class WorkExceeded(Exception):
    pass


def _tick():
    _WORK[0] += 1
    if _WORK[0] > _BUDGET[0]:
        raise WorkExceeded


def safe_call(f, a, b, budget=50_000_000):
    """Top-level call with a fresh, generous budget (real answers, not search)."""
    _WORK[0] = 0
    _BUDGET[0] = budget
    return f(a, b)


# ── the grounded floor — the ONLY host arithmetic is S (+1) and P (-1) ───────
def atom_components():
    return {
        "Z": (0, lambda env, args: 0),                    # zero
        "S": (1, lambda env, args: args[0] + 1),          # successor (host +1 — floor)
        "P": (1, lambda env, args: max(0, args[0] - 1)),  # predecessor (host -1 — floor)
    }


def base_components(survivor):
    """BASE sees the atoms + the ONE variable that survives the base case."""
    c = atom_components()
    c[survivor] = (0, (lambda v: lambda env, args: env[v])(survivor))
    return c


def step_components(library):
    """STEP sees atoms + all three vars + previously LEARNED 2-ary functions."""
    c = atom_components()
    for v in ("a", "b", "r"):
        c[v] = (0, (lambda vv: lambda env, args: env[vv])(v))
    for name, fn in library.items():
        c[name] = (2, (lambda f: lambda env, args: f(args[0], args[1]))(fn))
    return c


def eval_postfix(tokens, components, env):
    stack = []
    for t in tokens:
        arity, fn = components[t]
        if len(stack) < arity:
            return None
        args = [stack.pop() for _ in range(arity)][::-1]
        stack.append(fn(env, args))
    return stack[0] if len(stack) == 1 else None


def make_function(schema, base_tokens, step_tokens, library):
    """Build f(a,b) by primitive recursion on `schema` arg. No host +-*/ on the
    data — only S/P atoms and the recursion combinator (the loop)."""
    scomp = step_components(library)
    if schema == "a":
        bcomp = base_components("b")

        def f(a, b):
            acc = eval_postfix(base_tokens, bcomp, {"b": b})       # f(0,b)=BASE(b)
            for i in range(a):
                _tick()
                acc = eval_postfix(step_tokens, scomp, {"a": i, "b": b, "r": acc})
            return acc
    else:  # schema == "b"
        bcomp = base_components("a")

        def f(a, b):
            acc = eval_postfix(base_tokens, bcomp, {"a": a})       # f(a,0)=BASE(a)
            for j in range(b):
                _tick()
                acc = eval_postfix(step_tokens, scomp, {"a": a, "b": j, "r": acc})
            return acc
    return f


# ── synthesis as search — reuse tree_reason.solve (the proven A* engine) ─────
class StepSynth(SearchProblem):
    def __init__(self, examples, schema, base_tokens, library, max_len=6):
        self.examples = examples
        self.schema = schema
        self.base = base_tokens
        self.library = library
        self.comp = step_components(library)
        self.max_len = max_len

    def initial(self):     return ((), 0)
    def key(self, s):      return s
    def heuristic(self, s): return 0

    def is_goal(self, state):
        tokens, depth = state
        if depth != 1 or not tokens:
            return False
        f = make_function(self.schema, self.base, tokens, self.library)
        try:
            for (a, b), out in self.examples:
                _WORK[0] = 0                       # fresh budget per candidate
                _BUDGET[0] = 200_000
                if f(a, b) != out:
                    return False
            return True
        except Exception:                          # WorkExceeded / malformed → prune
            return False

    def moves(self, state):
        tokens, depth = state
        if len(tokens) >= self.max_len:
            return
        for name, (arity, _) in self.comp.items():
            if depth >= arity:
                yield (name, (tokens + (name,), depth - arity + 1), 1.0)


def _base_candidates(survivor):
    return [(survivor,), ("Z",), ("Z", "S"), (survivor, "S"), (survivor, "P")]


def synthesize(examples, library, max_len=5, node_budget=120_000):
    """Find (schema, base, step) whose primrec function fits ALL examples,
    trying recursion on either argument. Returns first (shortest-per-base)
    solution. Reuses solve()."""
    total = 0
    for schema in ("a", "b"):
        survivor = "b" if schema == "a" else "a"
        for base in _base_candidates(survivor):
            prob = StepSynth(examples, schema, base, library, max_len=max_len)
            path, cost, nodes = solve(prob, max_nodes=node_budget)
            total += nodes
            if path is not None:
                step = path[-1][1][0]
                f = make_function(schema, base, step, library)
                return f, schema, base, step, total
    return None, None, None, None, total


# ── readability ──────────────────────────────────────────────────────────────
def pretty(tokens):
    st, ar = [], {"Z": 0, "S": 1, "P": 1, "a": 0, "b": 0, "r": 0}
    for t in tokens:
        a = ar.get(t, 2)
        if a == 0:
            st.append("0" if t == "Z" else t)
        elif a == 1:
            st.append(("S" if t == "S" else "pred") + f"({st.pop()})")
        else:
            y = st.pop(); x = st.pop(); st.append(f"{t}({x}, {y})")
    return st[0] if len(st) == 1 else "?"


# ── the importable module: a self-built arithmetic library ───────────────────
class LearnedArithmetic:
    """Synthesises add, sub, mul, pow from {S,P}+recursion on construction.
    Each op is grounded in the previous. Exposes .lib (name -> callable) — the
    unit the brain routes to in place of frozen C++ ops."""

    CURRICULUM = [
        ("add", "a", [((2, 3), 5), ((0, 4), 4), ((3, 1), 4), ((1, 1), 2), ((4, 0), 4), ((5, 2), 7)]),
        ("sub", "b", [((5, 3), 2), ((4, 4), 0), ((7, 2), 5), ((3, 5), 0), ((9, 0), 9), ((6, 1), 5)]),
        ("mul", "a", [((2, 3), 6), ((0, 5), 0), ((3, 4), 12), ((1, 9), 9), ((4, 2), 8), ((5, 3), 15)]),
        ("pow", "b", [((2, 3), 8), ((3, 2), 9), ((5, 0), 1), ((2, 4), 16), ((1, 5), 1), ((4, 1), 4)]),
    ]
    ORACLE = {"add": lambda a, b: a + b, "sub": lambda a, b: max(0, a - b),
              "mul": lambda a, b: a * b, "pow": lambda a, b: a ** b}

    def __init__(self, verbose=False):
        self.lib = {}
        self.programs = {}          # name -> (schema, base, step)
        self.report = {}            # name -> {"nodes","train_ok","holdout_ok"}
        for name, _kind, examples in self.CURRICULUM:
            f, schema, base, step, nodes = synthesize(examples, self.lib)
            if f is None:
                self.report[name] = {"nodes": None, "train_ok": False, "holdout_ok": False}
                if verbose: print(f"  ✗ {name}: FAILED")
                continue
            oracle = self.ORACLE[name]
            # grounded arithmetic is O(value); keep holdout small (esp. pow, which
            # is genuinely exp-slow grounded — that is WHY a fast-path reattaches)
            holdout = {"add": [(7, 8), (12, 5), (9, 9), (11, 3), (0, 0)],
                       "sub": [(9, 4), (12, 5), (7, 7), (3, 8), (10, 0)],
                       "mul": [(7, 8), (9, 6), (12, 4), (5, 5), (0, 9)],
                       "pow": [(2, 6), (3, 4), (5, 3), (4, 3), (6, 2)]}[name]
            hok = all(safe_call(f, a, b) == oracle(a, b) for a, b in holdout)
            self.lib[name] = f
            self.programs[name] = (schema, base, step)
            self.report[name] = {"nodes": nodes, "train_ok": True, "holdout_ok": hok}
            if verbose:
                b = "b" if schema == "a" else "a"
                zero = "0" if schema == "a" else "a"
                arg = "Sa" if schema == "a" else "Sb"
                print(f"  ✓ {name:4s} searched {nodes:6d} progs   "
                      f"{name}({zero.replace('0','0,b') if schema=='a' else 'a,0'}) = {pretty(base)} ; "
                      f"{name}({arg},{'b' if schema=='a' else 'a'}) = {pretty(step)}   "
                      f"holdout {'✓' if hok else '✗'}")

    def __call__(self, name, a, b):
        return self.lib[name](a, b)


# ── proof / demo ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  math_synth — learn arithmetic from {succ, pred}, don't call it")
    print("=" * 70)
    print("\nSynthesising curriculum (each op grounded in the previous):\n")
    la = LearnedArithmetic(verbose=True)

    print("\nGeneralisation on inputs NEVER in any example:")
    for name in ("add", "sub", "mul", "pow"):
        if name not in la.lib:
            continue
        f, oracle = la.lib[name], la.ORACLE[name]
        probes = {"add": [(13, 7), (20, 6), (99, 1)], "sub": [(20, 6), (13, 7), (8, 15)],
                  "mul": [(13, 7), (11, 9), (15, 6)], "pow": [(2, 8), (3, 5), (7, 3)]}[name]
        cells = ", ".join(f"{name}{p}={safe_call(f, *p)}" for p in probes)
        allok = all(safe_call(f, *p) == oracle(*p) for p in probes)
        print(f"  {name}: {cells}   {'✓' if allok else '✗ WRONG'}")

    n_ok = sum(1 for r in la.report.values() if r["train_ok"] and r["holdout_ok"])
    print("\n" + "=" * 70)
    print(f"  LEARNED {n_ok}/{len(la.CURRICULUM)} ops from succ+pred. No host + - * / in any procedure.")
    print(f"  add←succ, sub←pred, mul←add, pow←mul.  Library = the reattach unit.")
    print("=" * 70)


if __name__ == "__main__":
    main()
