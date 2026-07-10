#!/usr/bin/env python3
"""
tree_reason.py — the branch / prune / search reasoner.

The second reasoning engine for the project. The binding memory does ONE thing:
deductive closure over relations (A>B>C => A>C). This does the general thing
the scratchpad + PUCT were reaching for: explore a space of states by applying
rule-valid operators, prune the dead branches, and search to a goal — then read
back the path as the worked steps.

Transitive closure is just the simplest case of this (search whose only move is
"follow a relation"). The power comes from what the operators are. Define a
domain's operators + goal and the SAME engine solves it. Below: linear algebra
and the classic bridge-and-torch puzzle — two very different problems, one
search core, each showing its work.

This is honest about what it is: a clean general search (A* / uniform-cost),
not the untrained BG controller. The brain's learned critic could later supply
the heuristic; here the heuristic is hand-given per domain. Operators are
hand-defined per domain — that is the real, known limit (one domain at a time).
What's NOT limited: the solutions it finds inside those rules, including ones
you never handed it.
"""

import heapq
import itertools
from fractions import Fraction


# ── general search engine ────────────────────────────────────────────────────
class SearchProblem:
    def initial(self): raise NotImplementedError
    def moves(self, state): raise NotImplementedError      # yield (label, next, cost)
    def is_goal(self, state): raise NotImplementedError
    def heuristic(self, state): return 0
    def key(self, state): return state                     # hashable canonical form


from collections import namedtuple

SearchResult = namedtuple("SearchResult", "solved path cost nodes")


def solve(problem, max_nodes=500_000):
    """Best-first (A*) search. Returns (path, total_cost, nodes_expanded);
    (None, None, nodes) if no solution within max_nodes.

    Guarantees: with non-negative step costs and an admissible heuristic
    (heuristic never over-estimates true cost-to-goal; the default 0 is always
    admissible), the returned path is OPTIMAL. Tie-breaking is deterministic, so
    the same problem always yields the same result. Visited states are kept at
    their best known cost, so cycles terminate.
    """
    if not isinstance(max_nodes, int) or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    for m in ("initial", "moves", "is_goal", "heuristic", "key"):
        if not callable(getattr(problem, m, None)):
            raise TypeError(f"problem is missing required method: {m}()")

    start = problem.initial()
    tie = itertools.count()
    frontier = [(problem.heuristic(start), 0, next(tie), start, [])]
    best = {problem.key(start): 0}
    expanded = 0
    while frontier and expanded < max_nodes:
        f, g, _, state, path = heapq.heappop(frontier)
        if problem.is_goal(state):
            return path, g, expanded
        expanded += 1
        for label, nxt, cost in problem.moves(state):
            if cost < 0:
                raise ValueError(f"negative step cost {cost} breaks A* optimality")
            ng = g + cost
            k = problem.key(nxt)
            if k in best and best[k] <= ng:
                continue                                   # prune: worse than known
            best[k] = ng
            heapq.heappush(frontier,
                           (ng + problem.heuristic(nxt), ng, next(tie), nxt,
                            path + [(label, nxt)]))
    return None, None, expanded


def search(problem, max_nodes=500_000):
    """Ergonomic wrapper: returns a SearchResult(solved, path, cost, nodes)."""
    path, cost, nodes = solve(problem, max_nodes)
    return SearchResult(path is not None, path or [], cost, nodes)


# ── domain 1: linear equations ───────────────────────────────────────────────
# State = (left, right), each a dict {'x': Fraction, '1': Fraction} meaning
# left.x * x + left.1  =  right.x * x + right.1.
def _parse_side(s):
    expr = {'x': Fraction(0), '1': Fraction(0)}
    s = s.replace('-', '+-').replace(' ', '')
    for term in filter(None, s.split('+')):
        if term.endswith('x'):
            c = term[:-1]
            c = '1' if c in ('', '+') else ('-1' if c == '-' else c)
            expr['x'] += Fraction(c)
        else:
            expr['1'] += Fraction(term)
    return expr


def parse_equation(text):
    l, r = text.split('=')
    return (_parse_side(l), _parse_side(r))


def _fmt_side(e):
    parts = []
    if e['x'] != 0:
        c = e['x']
        parts.append("x" if c == 1 else ("-x" if c == -1 else f"{c}x"))
    if e['1'] != 0 or not parts:
        parts.append(str(e['1']))
    out = " + ".join(parts).replace("+ -", "- ")
    return out


def fmt_equation(state):
    l, r = state
    return f"{_fmt_side(l)} = {_fmt_side(r)}"


class LinearEquation(SearchProblem):
    def __init__(self, text):
        self.start = parse_equation(text)

    def initial(self):
        return self.start

    def is_goal(self, state):
        l, r = state
        return l['x'] == 1 and l['1'] == 0 and r['x'] == 0  # x = constant

    def heuristic(self, state):
        l, r = state
        h = 0
        if r['x'] != 0: h += 1          # x still on the right
        if l['1'] != 0: h += 1          # constant still on the left
        if l['x'] != 1: h += 1          # x not yet isolated to coeff 1
        return h

    def key(self, state):
        l, r = state
        return (l['x'], l['1'], r['x'], r['1'])

    def moves(self, state):
        l, r = state
        # candidate rule-valid moves; the search decides which actually help.
        # subtract a constant that appears (clears it from a side)
        for b, who in ((l['1'], 'left'), (r['1'], 'right')):
            if b != 0:
                nl = {'x': l['x'], '1': l['1'] - b}
                nr = {'x': r['x'], '1': r['1'] - b}
                verb = f"subtract {b}" if b > 0 else f"add {-b}"
                yield (f"{verb} on both sides", (nl, nr), 1)
        # subtract an x-term that appears (moves x across)
        for a, who in ((l['x'], 'left'), (r['x'], 'right')):
            if a != 0:
                nl = {'x': l['x'] - a, '1': l['1']}
                nr = {'x': r['x'] - a, '1': r['1']}
                verb = f"subtract {a}x" if a > 0 else f"add {-a}x"
                yield (f"{verb} on both sides", (nl, nr), 1)
        # divide both sides by the left x-coefficient to isolate x
        if l['x'] not in (0, 1) and r['x'] == 0:
            a = l['x']
            nl = {'x': l['x'] / a, '1': l['1'] / a}
            nr = {'x': r['x'] / a, '1': r['1'] / a}
            yield (f"divide both sides by {a}", (nl, nr), 1)


# ── domain 2: bridge & torch puzzle ──────────────────────────────────────────
# State = (frozenset of people still on the start side, torch_side).
# People are identified by their crossing time. Minimize total time.
class BridgePuzzle(SearchProblem):
    def __init__(self, times):
        self.times = tuple(sorted(times))

    def initial(self):
        return (frozenset(self.times), 'start')

    def is_goal(self, state):
        here, torch = state
        return len(here) == 0 and torch == 'far'

    def key(self, state):
        return (state[0], state[1])

    def moves(self, state):
        here, torch = state
        far = frozenset(self.times) - here
        if torch == 'start':
            group, dest, ts = here, 'far', 'start'
        else:
            group, dest, ts = far, 'start', 'far'
        for size in (1, 2):
            for movers in itertools.combinations(sorted(group), size):
                cost = max(movers)
                if torch == 'start':
                    nstate = (here - set(movers), 'far')
                    label = f"{list(movers)} cross  (+{cost})"
                else:
                    nstate = (here | set(movers), 'start')
                    label = f"{list(movers)} return (+{cost})"
                yield (label, nstate, cost)


# ── demo ─────────────────────────────────────────────────────────────────────
def show_equation(text):
    prob = LinearEquation(text)
    path, cost, nodes = solve(prob)
    print(f"solve:  {text}")
    if path is None:
        print("  (no solution found)")
        return
    state = prob.initial()
    print(f"  start: {fmt_equation(state)}")
    for label, nxt in path:
        print(f"    → {label:34s} {fmt_equation(nxt)}")
    l, r = path[-1][1] if path else prob.initial()
    print(f"  ANSWER: x = {r['1']}")
    print(f"  (searched {nodes} states, solved in {len(path)} steps)\n")


def show_bridge(times):
    prob = BridgePuzzle(times)
    path, cost, nodes = solve(prob)
    print(f"bridge puzzle: people with crossing times {sorted(times)}, one torch, max 2 cross")
    if path is None:
        print("  (no solution)")
        return
    for label, _ in path:
        print(f"    {label}")
    print(f"  MINIMUM TOTAL TIME: {cost}")
    print(f"  (searched {nodes} states; this optimum was found, not given)\n")


def main():
    print("=== tree_reason — branch, prune, search to goal, show the steps ===\n")
    print("DOMAIN 1: linear algebra (operators = subtract/divide both sides)\n")
    for eq in ("2x + 3 = 10", "3x - 5 = x + 7", "5x + 2 = 2x - 10"):
        show_equation(eq)
    print("DOMAIN 2: planning — the bridge puzzle (operators = legal crossings)")
    print("same search engine, completely different operators\n")
    show_bridge([1, 2, 5, 10])


if __name__ == "__main__":
    main()
