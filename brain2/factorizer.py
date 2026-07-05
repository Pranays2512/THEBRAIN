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
       "*": lambda a, b: a * b, "/": lambda a, b: a / b if b else 0.0,
       "^": lambda a, b: a ** b}


def _is_leaf(t):
    return not isinstance(t, tuple)


def to_sexpr(t):
    """Serialize an expression tree to an S-expression string for the C++ evaluator."""
    if _is_leaf(t):
        return str(t)
    return "(" + " ".join([t[0]] + [to_sexpr(k) for k in t[1:]]) + ")"


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


# ── parameterized abstraction (anti-unification) ──────────────────────────────
# shape()/factor() above abstract LEAVES but need an identical operator skeleton.
# Anti-unification generalizes further: where two trees DIFFER (different op, or
# leaf-vs-subtree), introduce a parameter that captures the whole differing subtree.
# So (a+b)*c and (x-y)*z — which share NO leaf-only skeleton (+ vs -) — generalize to
# P(u,v)=u*v. This catches idioms that share outer structure but differ inside.

def antiunify(t1, t2):
    """Most-specific generalization of two trees: keep matching structure, replace each
    point of disagreement with a hole. Returns a skeleton with ('hole', i) params."""
    ctr = [0]

    def au(a, b):
        if (not _is_leaf(a) and not _is_leaf(b)
                and a[0] == b[0] and len(a) == len(b)):
            return (a[0],) + tuple(au(x, y) for x, y in zip(a[1:], b[1:]))
        if a == b:                              # identical leaves -> keep concrete
            return a
        h = ("hole", ctr[0]); ctr[0] += 1
        return h
    return _canon(au(t1, t2))


def _canon(skel):
    """Renumber holes 0..k-1 in pre-order so a pattern has a canonical positional form."""
    ctr = [0]

    def walk(s):
        if isinstance(s, tuple) and s and s[0] == "hole":
            i = ctr[0]; ctr[0] += 1
            return ("hole", i)
        if not isinstance(s, tuple):
            return s
        return (s[0],) + tuple(walk(k) for k in s[1:])
    return walk(skel)


def instance_match(pattern, tree, binds):
    """Is `tree` an instance of `pattern` (holes match any subtree)? On success append
    the hole-fillers (left-to-right) to `binds` and return True."""
    if isinstance(pattern, tuple) and pattern and pattern[0] == "hole":
        binds.append(tree)
        return True
    if _is_leaf(pattern) or _is_leaf(tree):
        return pattern == tree
    if pattern[0] != tree[0] or len(pattern) != len(tree):
        return False
    return all(instance_match(p, t, binds) for p, t in zip(pattern[1:], tree[1:]))


def _pattern_arity(p):
    if isinstance(p, tuple) and p and p[0] == "hole":
        return 1
    if not isinstance(p, tuple):
        return 0
    return sum(_pattern_arity(k) for k in p[1:])


def _kept_nodes(p):                              # non-hole structural nodes (specificity)
    if not isinstance(p, tuple) or (p and p[0] == "hole"):
        return 0
    return 1 + sum(_kept_nodes(k) for k in p[1:])


def factor_au(library, min_count=2, min_kept=2, prims=None):
    """Like factor(), but discovers a parameterized pattern via anti-unification, so it
    can abstract idioms that share outer structure but differ inside. Returns
    (new_library, prims, (name, pattern, arity)) or (library, prims, None)."""
    prims = dict(prims or {})
    subs = [st for _, t in library for st in _subtrees(t) if _size(st) >= min_kept]
    # candidate patterns = pairwise anti-unifications
    cands = {}
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            p = antiunify(subs[i], subs[j])
            if _kept_nodes(p) >= min_kept:
                cands[p] = None
    best, best_score = None, -1
    for p in cands:
        count = sum(1 for st in subs if instance_match(p, st, []))
        if count < min_count:
            continue
        score = count * _kept_nodes(p)           # reuse x specificity
        if score > best_score:
            best, best_score = p, score
    if best is None:
        return library, prims, None
    name = "Q%d" % len(prims)
    arity = _pattern_arity(best)
    prims[name] = (arity, best)

    def rewrite(t):
        if _is_leaf(t):
            return t
        binds = []
        if instance_match(best, t, binds):
            return ("call", name, [rewrite(b) for b in binds])
        return (t[0],) + tuple(rewrite(k) for k in t[1:])
    new_lib = [(nm, rewrite(tree)) for nm, tree in library]
    return new_lib, prims, (name, best, arity)


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

    # parameterized abstraction: idioms that share OUTER structure but differ inside
    print("\n=== parameterized (anti-unification): catches what leaf-only misses ===\n")
    lib2 = [
        ("h1", ("*", ("+", "a", "b"), "c")),     # (a+b)*c
        ("h2", ("*", ("-", "x", "y"), "z")),     # (x-y)*z  — inner op differs (+/-)
    ]
    print("before (no shared leaf-only skeleton: + vs -):")
    for n, t in lib2:
        print("   %-4s %s" % (n, t))
    _, _, leaf = factor(lib2, min_count=2, min_size=3)
    print("  leaf-only factor() finds:", leaf)
    new2, prims2, au = factor_au(lib2, min_count=2, min_kept=1)
    print("  anti-unify factor_au() finds:", au[0], "=", au[1], "arity", au[2])
    for n, t in new2:
        print("   %-4s %s" % (n, t))
    ok2, _ = _verify(lib2, new2, prims2)
    print("  meaning preserved:", ok2, " — abstracted over the differing (+/-) subtree.")


if __name__ == "__main__":
    _demo()
