#!/usr/bin/env python3
"""
test_search_engine.py — hardening tests for the General Search engine.

Pins the engine's guarantees: optimality (non-negative costs + admissible
heuristic), determinism, cycle termination, clean no-solution / node-cap
handling, input validation — plus correctness on every domain it powers.
"""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engines.reasoning.tree_reason import SearchProblem, solve, search, LinearEquation, BridgePuzzle
from engines.reasoning.tree_domains import NQueens, WaterJugs, Rewrite

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


class Graph(SearchProblem):
    """Tiny weighted graph; optimal A->D cost is 3 (A-B-C-D), not 4 (A-C...)."""
    EDGES = {"A": [("B", 1), ("C", 4)], "B": [("C", 1)], "C": [("D", 1)], "D": []}

    def initial(self): return "A"
    def is_goal(self, s): return s == "D"
    def key(self, s): return s
    def heuristic(self, s): return 0
    def moves(self, s):
        for nxt, c in self.EDGES[s]:
            yield (f"{s}->{nxt}", nxt, c)


def run():
    print("\nSearch engine — hardening tests")

    # 1. optimality on a known graph
    path, cost, _ = solve(Graph())
    check("finds optimal path cost (3, not 4)", cost == 3)
    check("optimal path is A-B-C-D", [s for _, s in path] == ["B", "C", "D"])

    # 2. determinism
    r1, r2 = solve(Graph()), solve(Graph())
    check("deterministic (same result twice)", r1 == r2)

    # 3. domain: bridge puzzle known optimum = 17
    _, cost, _ = solve(BridgePuzzle([1, 2, 5, 10]))
    check("bridge puzzle optimum = 17", cost == 17)

    # 4. domain: algebra 3x - 5 = x + 7 -> x = 6
    path, _, _ = solve(LinearEquation("3x - 5 = x + 7"))
    left, right = path[-1][1]
    check("algebra solves x = 6", left["x"] == 1 and right["1"] == Fraction(6))

    # 5. domain: N-queens(6) solved and valid (no attacks)
    path, _, _ = solve(NQueens(6))
    s = path[-1][1]
    valid = len(s) == 6 and all(
        s[i] != s[j] and abs(s[i] - s[j]) != j - i
        for i in range(6) for j in range(i + 1, 6))
    check("N-queens(6) solved and valid", valid)

    # 6. no solution -> (None, None, nodes)
    path, cost, _ = solve(NQueens(3))           # 3-queens has no solution
    check("no solution -> None", path is None and cost is None)

    # 7. domain: water jugs reaches 4L
    path, _, _ = solve(WaterJugs([3, 5], 4))
    check("water jugs reaches 4L", any(4 in st for _, st in path))

    # 8. domain: rewrite babab -> aabbb
    path, _, _ = solve(Rewrite("babab", "aabbb", [("ba", "ab")]))
    check("rewrite derives aabbb", path[-1][1] == "aabbb")

    # 9. node cap respected -> clean no-result
    path, _, nodes = solve(NQueens(8), max_nodes=3)
    check("node cap returns cleanly", path is None and nodes <= 3)

    # 10. goal == start -> empty path
    class Trivial(Graph):
        def is_goal(self, s): return s == "A"
    path, cost, _ = solve(Trivial())
    check("goal at start -> empty optimal path", path == [] and cost == 0)

    # 11. negative cost rejected
    class Neg(Graph):
        def moves(self, s):
            yield ("bad", "D", -1)
    try:
        solve(Neg()); check("negative cost rejected", False)
    except ValueError:
        check("negative cost rejected", True)

    # 12. invalid max_nodes rejected
    try:
        solve(Graph(), max_nodes=0); check("invalid max_nodes rejected", False)
    except ValueError:
        check("invalid max_nodes rejected", True)

    # 13. malformed problem rejected
    try:
        solve(object()); check("malformed problem rejected", False)
    except TypeError:
        check("malformed problem rejected", True)

    # 14. ergonomic search() wrapper
    res = search(Graph())
    check("search() returns SearchResult", res.solved and res.cost == 3)
    check("search() reports unsolved cleanly", not search(NQueens(3)).solved)

    print(f"\nSearch engine: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
