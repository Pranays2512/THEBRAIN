#!/usr/bin/env python3
"""
factorizer.py — the make/break loop's other BREAK half: decompose SOLVED artifacts
into reusable parts, growing the DSL from what the brain has already built.

Composition with a FIXED primitive set has a hand-coded ceiling. Composition + factoring
grows the set: take every formula/program the brain solved, find the repeated structural
shape, promote it to a NEW named primitive, rewrite the library to call it. Next search
reuses it -> solves harder things -> factor again. (This is DreamCoder's wake/sleep
abstraction step, in miniature.)

Expressions are nested tuples: ('*', ('+', 'a', 'b'), 'c'); leaves are var names / numbers.
A discovered primitive Pk(x,y,...) is the most frequent SHARED SHAPE (same operator
skeleton, leaves abstracted to holes). Rewrites are verified to evaluate IDENTICALLY on
random bindings — factoring may never change meaning.

Honest limit: this abstracts over leaf VALUES with a matching operator skeleton (concrete
structural reuse). Deeper anti-unification (abstracting differing sub-shapes too) is the
next rung; this is the first, verifiable step.
"""

import random
from collections import Counter

OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
       "*": lambda a, b: a * b, "/": lambda a, b: a / b if b else 0.0}


def _is_leaf(t):
    return not isinstance(t, tuple)


def shape(t):
    """Return (skeleton, leaves): skeleton keeps internal ops, every leaf -> ('hole', i)
    in left-to-right order; leaves is the list of leaf subtrees in that order."""
    leaves = []

    def walk(n):
        if _is_leaf(n):
            leaves.append(n)
            return ("hole", len(leaves) - 1)
        return (n[0],) + tuple(walk(k) for k in n[1:])
    skel = walk(t)
    return skel, leaves


def _size(t):
    return 1 if _is_leaf(t) else 1 + sum(_size(k) for k in t[1:])


def _subtrees(t):
    if _is_leaf(t):
        return
    yield t
    for k in t[1:]:
        yield from _subtrees(k)


def eval_tree(t, env, prims):
    if isinstance(t, str):
        return env[t]
    if isinstance(t, (int, float)):
        return t
    if t[0] == "call":                       # ('call', 'Pk', [arg, ...])
        _, name, args = t
        arity, skel = prims[name]
        vals = [eval_tree(a, env, prims) for a in args]
        return _eval_skel(skel, vals)
    vals = [eval_tree(k, env, prims) for k in t[1:]]
    return OPS[t[0]](*vals)


def _eval_skel(skel, vals):
    if isinstance(skel, tuple) and skel and skel[0] == "hole":
        return vals[skel[1]]
    if not isinstance(skel, tuple):
        return skel
    return OPS[skel[0]](*[_eval_skel(k, vals) for k in skel[1:]])


def _rewrite(t, target_skel, name):
    """Replace every subtree whose shape == target_skel with ('call', name, leaves)."""
    if _is_leaf(t):
        return t
    sk, lv = shape(t)
    if sk == target_skel:
        return ("call", name, lv)
    return (t[0],) + tuple(_rewrite(k, target_skel, name) for k in t[1:])


def factor(library, min_count=2, min_size=3, prims=None):
    """library: list of (name, expr-tree). Discover the most frequent shared shape,
    promote it to a primitive, rewrite the library to use it. Returns
    (new_library, prims, discovered) where discovered is (prim_name, skeleton, arity)
    or None if nothing repeats enough."""
    prims = dict(prims or {})
    counts = Counter()
    for _, tree in library:
        seen = set()
        for st in _subtrees(tree):
            if _size(st) < min_size:
                continue
            sk, _ = shape(st)
            if sk not in seen:                # count once per program (cross-program reuse)
                counts[sk] += 1
                seen.add(sk)
    best = [(c, sk) for sk, c in counts.items() if c >= min_count]
    if not best:
        return library, prims, None
    best.sort(key=lambda x: (x[0], _skel_size(x[1])), reverse=True)
    _, skel = best[0]
    name = "P%d" % len(prims)
    arity = _skel_arity(skel)
    prims[name] = (arity, skel)
    new_lib = [(nm, _rewrite(tree, skel, name)) for nm, tree in library]
    return new_lib, prims, (name, skel, arity)


def _skel_arity(skel):
    if isinstance(skel, tuple) and skel and skel[0] == "hole":
        return 1
    if not isinstance(skel, tuple):
        return 0
    return sum(_skel_arity(k) for k in skel[1:])


def _skel_size(skel):
    if not isinstance(skel, tuple) or (skel and skel[0] == "hole"):
        return 1
    return 1 + sum(_skel_size(k) for k in skel[1:])


def _verify(lib_before, lib_after, prims, trials=200, seed=0):
    """Factoring must preserve meaning: every rewritten formula evaluates identically."""
    rng = random.Random(seed)
    vars_ = sorted({v for _, t in lib_before for v in _vars(t)})
    for _ in range(trials):
        env = {v: rng.uniform(-5, 5) for v in vars_}
        for (n0, b), (n1, a) in zip(lib_before, lib_after):
            if abs(eval_tree(b, env, {}) - eval_tree(a, env, prims)) > 1e-9:
                return False, n0
    return True, None


def _vars(t):
    if isinstance(t, str):
        return {t}
    if _is_leaf(t):
        return set()
    out = set()
    for k in t[1:]:
        out |= _vars(k)
    return out


def _demo():
    print("=== factorizer — grow the DSL from solved formulas ===\n")
    library = [
        ("momentum", ("*", "m", "v")),                       # m*v
        ("work",     ("*", "F", "d")),                       # F*d
        ("area",     ("*", "l", "w")),                       # l*w
        ("kinetic",  ("*", ("*", "m", "v"), "v")),           # m*v*v (uses m*v inside)
    ]
    print("before:")
    for n, t in library:
        print("   %-9s %s" % (n, t))

    new_lib, prims, disc = factor(library, min_count=3, min_size=3)
    print("\n  discovered primitive:", disc[0], "=", disc[1], "  arity", disc[2],
          "  (a*b — the MUL2 shape, seen in 3 formulas)")
    print("\nafter (DSL grew by one reusable primitive):")
    for n, t in new_lib:
        print("   %-9s %s" % (n, t))

    ok, who = _verify(library, new_lib, prims)
    print("\n  meaning preserved (verified on 200 random bindings):", ok)
    print("  Next search can now call", disc[0], "directly — the space grew itself.")


if __name__ == "__main__":
    _demo()
