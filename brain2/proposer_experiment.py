#!/usr/bin/env python3
"""
proposer_experiment.py — Phase 1 go/no-go: does a learned PROPOSER beat blind
search at choosing math policies?

This is the decisive cheap experiment for the whole reasoning stack. It mirrors
program_synth_policy's blind-vs-guided comparison, but in a MATH domain where the
proposer's value is unambiguous: integration by parts.

  ∫ x^n · e^x   needs by-parts with u = the polynomial. Pick u = e^x and the
  degree climbs forever (diverges). Blind search explores the diverging branch
  before backtracking; a proposer that learned "u = polynomial" (the LIATE rule,
  learned from data, not hardcoded) goes straight to the answer.

Every answer is VERIFIED by differentiating it back (calculus_engine), so we are
never measuring fast-but-wrong. The metric is NODES EXPANDED (recursive calls):
fewer = smarter search.

DECISION RULE: guided expands >= 3x fewer nodes at equal-or-better solve rate ->
GREENLIGHT the reasoning stack. guided ~= blind -> the features below don't
capture the structure; fix features() before building anything downstream.

Run:  python3 proposer_experiment.py
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from calculus_engine import CalculusEngine, simplify, _num
from integral_engine import IntegralEngine
from physics_engine import contains, ev

VAR = "x"
_ce = CalculusEngine()
_ie = IntegralEngine()


# ── expression helpers ───────────────────────────────────────────────────────
def degree_in_x(e):
    """Polynomial degree of e in x (0 if no x; None if not a polynomial)."""
    if _num(e):
        return 0
    if isinstance(e, str):
        return 1 if e == VAR else 0
    op = e[0]
    if op == "^" and e[1] == VAR and isinstance(e[2], int):
        return e[2]
    if op in ("+", "-"):
        a, b = degree_in_x(e[1]), degree_in_x(e[2])
        return None if a is None or b is None else max(a, b)
    if op == "*":
        a, b = degree_in_x(e[1]), degree_in_x(e[2])
        return None if a is None or b is None else a + b
    return None                                   # trig/exp/log/quotient: not poly


def is_polynomial(e):
    return degree_in_x(e) is not None


def is_transcendental(e):
    """A trig/exp/log somewhere with x inside (the 'hard to integrate' factor)."""
    if _num(e) or isinstance(e, str):
        return False
    if e[0] in ("sin", "cos", "exp", "ln") and contains(e[1], VAR):
        return True
    return any(is_transcendental(a) for a in e[1:] if isinstance(a, (tuple,)))


def depth(e):
    if _num(e) or isinstance(e, str):
        return 0
    return 1 + max((depth(a) for a in e[1:] if isinstance(a, tuple)), default=0)


def _nfold(e):
    """Fold constant arithmetic and rewrite (/ u c) as (* 1/c u). Without this,
    calculus_engine's quotient rule nests ((4^2)^2)^2... through repeated
    by-parts until it overflows — answers correct, but un-evaluable."""
    if _num(e) or isinstance(e, str):
        return e
    op = e[0]
    args = [_nfold(a) for a in e[1:]]
    if op == "neg" and _num(args[0]):
        return -args[0]
    if op in ("+", "-", "*", "/", "^") and all(_num(a) for a in args):
        a, b = args[0], args[1]
        try:
            return {"+": a + b, "-": a - b, "*": a * b,
                    "/": a / b if b else e, "^": a ** b}[op]
        except (ZeroDivisionError, OverflowError):
            return (op, *args)
    if op == "/" and _num(args[1]) and args[1] != 0:   # u / c  ->  (1/c) * u
        return ("*", 1.0 / args[1], args[0])
    return (op, *args)


def base_integrate(e):
    """Closed-form antiderivative via the deterministic engine, EXTENDED to see
    through neg and constant-multiples — by-parts needs V=∫dv for these too, and
    the raw engine doesn't know them (this gap was sinking the trig chains)."""
    F = _ie.integrate(e)
    if F is not None:
        return F
    if isinstance(e, tuple):
        if e[0] == "neg":
            inner = base_integrate(e[1])
            return ("neg", inner) if inner is not None else None
        if e[0] == "*" and not contains(e[1], VAR):
            inner = base_integrate(e[2])
            return ("*", e[1], inner) if inner is not None else None
        if e[0] == "*" and not contains(e[2], VAR):
            inner = base_integrate(e[1])
            return ("*", e[2], inner) if inner is not None else None
    return None


# ── moves: the candidate policies at one integrand ───────────────────────────
@dataclass
class Move:
    rule: str                 # "base" | "const_mult" | "linearity" | "by_parts:u=left/right"
    subgoals: list            # integrands still to integrate (may be empty)
    assemble: object          # fn(list_of_antiderivs) -> antiderivative for THIS integrand
    u: object = None          # for by-parts: the chosen u (for features)
    dv: object = None         # for by-parts: the chosen dv (for features)


def _diff(e):
    return simplify(_ce.diff(e, VAR).expr)


def candidate_moves(e):
    """All rule-valid moves at integrand e. The search/proposer decides order."""
    moves = []

    # base case: the deterministic engine closes it outright
    F = base_integrate(e)
    if F is not None:
        moves.append(Move("base", [], lambda parts, F=F: F))
        return moves                              # solved here; no need to branch

    if not contains(e, VAR):                      # ∫ c dx = c*x
        moves.append(Move("base", [], lambda parts, e=e: ("*", e, VAR)))
        return moves

    op = e[0]
    if op == "neg":                               # ∫ -a = -∫a  (closes by-parts leftovers)
        moves.append(Move("neg", [e[1]], lambda parts: ("neg", parts[0])))
        return moves
    if op in ("+", "-"):                          # linearity: ∫(a±b) = ∫a ± ∫b
        a, b = e[1], e[2]
        moves.append(Move("linearity", [a, b],
                          lambda parts, op=op: (op, parts[0], parts[1])))
        return moves

    if op == "*":
        a, b = e[1], e[2]
        # constant multiple: pull the x-free factor out
        if not contains(a, VAR):
            moves.append(Move("const_mult", [b],
                              lambda parts, a=a: ("*", a, parts[0])))
            return moves
        if not contains(b, VAR):
            moves.append(Move("const_mult", [a],
                              lambda parts, b=b: ("*", b, parts[0])))
            return moves
        # by-parts: ∫ u dv = u*V - ∫ V du.   Two candidate assignments.
        for u, dv, tag in ((a, b, "left"), (b, a, "right")):
            V = base_integrate(dv)                # need V = ∫dv in closed form
            if V is None:
                continue
            du = _diff(u)
            subgoal = simplify(("*", V, du))      # ∫ V du
            moves.append(Move(
                f"by_parts:u={tag}", [subgoal],
                lambda parts, u=u, V=V: ("-", ("*", u, V), parts[0]),
                u=u, dv=dv))
    return moves


# ── features of a move — THIS IS THE DESIGN CHOICE YOU OWN ────────────────────
# The proposer scores a move from these. Whether guided beats blind depends on
# whether these signals capture the real structure (here: "u = polynomial wins").
# Start minimal, then add signals and re-run the benchmark to see the gap move.
N_FEATURES = 6


def features(e, move):
    """Vector describing a candidate move at integrand e. TODO: extend."""
    u = move.u if move.u is not None else e
    dv = move.dv if move.dv is not None else 0
    du = degree_in_x(u)
    return np.array([
        1.0,                                          # bias
        float(du if du is not None else 0),           # degree of u (0, not -1: the
                                                      # -1 sentinel inverted the score)
        1.0 if is_polynomial(u) else 0.0,             # u polynomial?  (want yes)
        1.0 if is_transcendental(dv) else 0.0,        # dv transcendental? (want yes)
        1.0 if is_transcendental(u) else 0.0,         # u transcendental? (want no)
        float(depth(e)),                              # how nested
    ], dtype=float)


# ── the recursive searcher (blind or proposer-guided) ────────────────────────
@dataclass
class Counter:
    nodes: int = 0


NODE_CAP = 1200   # per-case budget: blind blows past it on hard cases, the
                  # proposer stays well under — that gap is the whole point.


def integrate_search(e, score=None, counter=None, max_depth=22):
    """Return a verified antiderivative of e, or None. Counts nodes expanded.
    score=None -> blind (fixed move order). score(e, move)->float -> proposer
    orders moves by it (higher first). Bails at NODE_CAP so a blown-up blind
    search terminates (counts as unsolved) instead of hanging."""
    if counter is None:
        counter = Counter()
    counter.nodes += 1
    if max_depth <= 0 or counter.nodes > NODE_CAP:
        return None

    moves = candidate_moves(e)
    if score is not None:                          # proposer: order by learned score
        moves = sorted(moves, key=lambda m: -score(e, m))

    for m in moves:
        parts, ok = [], True
        for g in m.subgoals:
            sub = integrate_search(g, score, counter, max_depth - 1)
            if sub is None:
                ok = False
                break
            parts.append(sub)
        if ok:
            return _nfold(simplify(m.assemble(parts)))
    return None


def verify(integrand, antideriv, at=1.3, tol=1e-4):
    """Differentiate the answer back: does it recover the integrand?"""
    if antideriv is None:
        return False
    d = _nfold(simplify(_ce.diff(antideriv, VAR).expr))
    try:
        return abs(ev(d, {VAR: at}) - ev(integrand, {VAR: at})) < tol
    except Exception:
        return False


# ── training: harvest (features -> won?) from blindly-solved instances ────────
def harvest_dataset(corpus):
    """(X, Y): one row per move tried while solving, Y=1 on a winning path."""
    X, Y = [], []
    for e in corpus:
        _harvest(e, X, Y)
    return np.array(X), np.array(Y)


def train_linear(corpus):
    """Linear proposer (lstsq). Can't model feature interactions -> weak."""
    X, Y = harvest_dataset(corpus)
    if not len(X):
        return np.zeros(N_FEATURES)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def train_tree(corpus):
    """Decision-tree proposer (binary win/lose). Splits on the LIATE flags ->
    captures the 'u-poly AND dv-transcendental' interaction the linear can't."""
    from program_synth_tree import DecisionTree
    X, Y = harvest_dataset(corpus)
    if not len(X):
        return None
    tree = DecisionTree(n_ops=2, max_depth=6, min_samples=4).fit(X, Y.astype(int))
    return tree


def linear_score(W):
    return lambda e, m: float(features(e, m) @ W)


def tree_score(tree):
    return lambda e, m: float(tree.predict_dist(features(e, m))[1])  # P(win)


# kept for back-compat with the old entry point
def train_proposer(corpus):
    return train_linear(corpus)


def _harvest(e, X, Y, counter=None, max_depth=22):
    """Like integrate_search, but records which move at each node solved it."""
    if counter is None:
        counter = Counter()
    counter.nodes += 1
    if max_depth <= 0 or counter.nodes > NODE_CAP:
        return None
    for m in candidate_moves(e):
        parts, ok = [], True
        for g in m.subgoals:
            sub = _harvest(g, X, Y, counter, max_depth - 1)
            if sub is None:
                ok = False
                break
            parts.append(sub)
        if ok:
            # this move is on a winning path: positive example
            X.append(features(e, m))
            Y.append(1.0)
            return simplify(m.assemble(parts))
        else:
            X.append(features(e, m))             # tried, did not pan out: negative
            Y.append(0.0)
    return None


# ── the benchmark: blind vs linear vs tree ───────────────────────────────────
def benchmark(train, test):
    W = train_linear(train)
    tree = train_tree(train)
    scorers = {
        "blind":  None,
        "linear": linear_score(W),
        "tree":   tree_score(tree),
    }

    def run(score):
        nodes = solved = 0
        for e in test:
            c = Counter()
            F = integrate_search(e, score, c)
            nodes += c.nodes
            solved += 1 if verify(e, F) else 0
        return nodes / len(test), solved, len(test)

    print("=== proposer go/no-go — integration search ===\n")
    res = {name: run(s) for name, s in scorers.items()}
    n = res["blind"][2]
    print(f"  {'method':10s} {'avg nodes':>10s} {'solved':>9s}")
    for name in ("blind", "linear", "tree"):
        nd, sv, _ = res[name]
        print(f"  {name:10s} {nd:10.1f} {sv:>6}/{n}")

    bn, bs, _ = res["blind"]
    tn, ts, _ = res["tree"]
    ratio = bn / max(tn, 1e-9)
    print(f"\n  tree proposer explores ~{ratio:.1f}x fewer nodes than blind "
          f"(solve {ts}/{n} vs {bs}/{n})")
    verdict = "GREENLIGHT ✅" if ratio >= 3.0 and ts >= bs else "not yet ⚠️"
    print(f"  DECISION (tree >=3x & no solve regression): {verdict}")
    return ratio, res


# ── a corpus that exercises by-parts both ways (so fixed-order blind is wrong half the time)
def make_corpus():
    x = VAR
    poly = [x, ("^", x, 2), ("^", x, 3), ("^", x, 4)]
    trans = [("exp", x), ("sin", x), ("cos", x)]
    cases = []
    for p in poly:
        for t in trans:
            cases.append(("*", p, t))            # poly on the left
            cases.append(("*", t, p))            # poly on the right (blind's blind spot)
    return cases


def _degree(case):
    return max(degree_in_x(case[1]) or 0, degree_in_x(case[2]) or 0)


def main():
    corpus = make_corpus()
    # Train on EASY cases (low degree, cheap to harvest); TEST on HARD ones
    # (high degree, where blind's wrong by-parts branch genuinely explodes).
    # This also makes it an honest GENERALIZATION test, not memorization.
    train = [c for c in corpus if _degree(c) <= 2]
    test = [c for c in corpus if _degree(c) >= 3]
    print(f"train: {len(train)} easy cases (deg<=2)   test: {len(test)} hard cases (deg>=3)\n")
    benchmark(train, test)


if __name__ == "__main__":
    main()
