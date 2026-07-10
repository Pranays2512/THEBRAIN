#!/usr/bin/env python3
"""
tree_domains.py — same search engine, more kinds of reasoning.

Each domain below is a different KIND of problem — constraint satisfaction,
state-space puzzle, symbolic rewriting — yet all of them plug into the one
tree_reason.solve engine by supplying (operators, goal, heuristic). Nothing
about the search changes; only the rules do. That is the whole point: the
reasoning core is shared, each domain is a plug-in.

  N-Queens        constraint satisfaction (place queens, none attacking)
  Water jugs      state-space planning with a numeric goal
  Rewrite/proof   derive a target form from rules (a tiny formal system)
"""

from core.reasoning.tree_reason import SearchProblem, solve


# ── constraint satisfaction: N-Queens ────────────────────────────────────────
class NQueens(SearchProblem):
    def __init__(self, n):
        self.n = n

    def initial(self):
        return ()                       # columns chosen so far, one per row

    def is_goal(self, s):
        return len(s) == self.n

    def key(self, s):
        return s

    def heuristic(self, s):
        return self.n - len(s)          # rows still to place (optimistic)

    def moves(self, s):
        row = len(s)
        for col in range(self.n):
            if all(col != c and abs(col - c) != row - r for r, c in enumerate(s)):
                yield (f"queen at (row {row}, col {col})", s + (col,), 1)


def board(s):
    n = len(s)
    return "\n".join("    " + " ".join("Q" if s[r] == c else "." for c in range(n))
                     for r in range(n))


# ── state-space planning: water jugs ─────────────────────────────────────────
class WaterJugs(SearchProblem):
    def __init__(self, caps, target):
        self.caps = tuple(caps)
        self.target = target

    def initial(self):
        return tuple(0 for _ in self.caps)

    def is_goal(self, s):
        return self.target in s

    def key(self, s):
        return s

    def moves(self, s):
        caps = self.caps
        for i in range(len(caps)):
            if s[i] < caps[i]:
                ns = list(s); ns[i] = caps[i]
                yield (f"fill jug{i} (->{caps[i]}L)", tuple(ns), 1)
            if s[i] > 0:
                ns = list(s); ns[i] = 0
                yield (f"empty jug{i}", tuple(ns), 1)
            for j in range(len(caps)):
                if i != j and s[i] > 0 and s[j] < caps[j]:
                    amt = min(s[i], caps[j] - s[j])
                    ns = list(s); ns[i] -= amt; ns[j] += amt
                    yield (f"pour jug{i}->jug{j}", tuple(ns), 1)


# ── symbolic rewriting / proof: sort by a single rule ────────────────────────
class Rewrite(SearchProblem):
    def __init__(self, start, goal, rules):
        self.start, self.goal, self.rules = start, goal, rules

    def initial(self):
        return self.start

    def is_goal(self, s):
        return s == self.goal

    def key(self, s):
        return s

    def heuristic(self, s):
        return 0 if s == self.goal else 1

    def moves(self, s):
        for lhs, rhs in self.rules:
            i = s.find(lhs)
            while i != -1:
                yield (f"apply  {lhs} -> {rhs}", s[:i] + rhs + s[i + len(lhs):], 1)
                i = s.find(lhs, i + 1)


def main():
    print("=== tree_domains — one search engine, three kinds of reasoning ===\n")

    print("1. N-QUEENS (constraint satisfaction): place 6 non-attacking queens")
    path, _, nodes = solve(NQueens(6))
    final = path[-1][1]
    print(board(final))
    print(f"   solved (searched {nodes} states)\n")

    print("2. WATER JUGS (state-space planning): measure exactly 4L with 3L & 5L jugs")
    path, cost, nodes = solve(WaterJugs([3, 5], 4))
    for label, st in path:
        print(f"    {label:22s} {list(st)}")
    print(f"   reached 4L in {len(path)} steps (searched {nodes} states)\n")

    print("3. REWRITE / PROOF: derive 'aabbb' from 'babab' using only  ba -> ab")
    path, _, nodes = solve(Rewrite("babab", "aabbb", [("ba", "ab")]))
    state = "babab"
    print(f"    start: {state}")
    for label, st in path:
        print(f"    {label:18s} {st}")
    print(f"   derived (searched {nodes} states)\n")

    print("Same solve() engine for all three — only the operators and goal differ.")


if __name__ == "__main__":
    main()
