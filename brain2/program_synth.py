#!/usr/bin/env python3
"""
program_synth.py — write code from examples, by search, verifiably.

Give it input/output examples. It SEARCHES a small DSL of string operations
for a program that reproduces every example, then applies that program to new
inputs to show it learned the transformation — not memorized the answers.

This is the code version of tree_reason: state = a partial program, operators =
DSL primitives, goal = "the program reproduces all examples." The search finds
the shortest program that passes. The result is verifiable by construction — it
was searched to satisfy the spec, not guessed (which is what makes an LLM's code
plausible-but-wrong). Same engine as algebra and the bridge puzzle; only the
operators and goal differ.

Honest scope: it writes programs in the DSL given here (string ops). Richer
programs need a richer DSL (more operators) and, past a few steps, learned
search guidance — exactly the tree_learn pattern. It does not invent new
primitives. Within the DSL, every program it returns is correct on the spec.
"""

from core.reasoning.tree_reason import SearchProblem, solve


# ── the DSL: composable string -> string primitives ──────────────────────────
DSL = {
    "lower":         str.lower,
    "upper":         str.upper,
    "title":         str.title,
    "strip":         str.strip,
    "first_word":    lambda s: s.split()[0],
    "last_word":     lambda s: s.split()[-1],
    "initials":      lambda s: "".join(w[0] for w in s.split()),
    "reverse_words": lambda s: " ".join(reversed(s.split())),
}


def run(program, s):
    for name in program:
        s = DSL[name](s)
    return s


class Synthesize(SearchProblem):
    """Search the space of DSL programs for one matching all examples."""

    def __init__(self, examples, max_len=4):
        self.examples = examples
        self.max_len = max_len

    def initial(self):
        return ()                       # empty program (identity)

    def is_goal(self, prog):
        for inp, out in self.examples:
            try:
                if run(prog, inp) != out:
                    return False
            except Exception:
                return False
        return True

    def key(self, prog):
        return prog

    def heuristic(self, prog):
        return 0                        # BFS -> shortest correct program

    def moves(self, prog):
        if len(prog) >= self.max_len:
            return
        for name in DSL:
            yield (f"then {name}", prog + (name,), 1)


def synthesize(examples, tests):
    prob = Synthesize(examples)
    path, _, nodes = solve(prob)
    prog = path[-1][1] if path else ()

    print("  examples given:")
    for inp, out in examples:
        print(f'      "{inp}"  ->  "{out}"')
    if path is None:
        print(f"  no program in the DSL reproduces these (searched {nodes}).\n")
        return
    pretty = " -> ".join(prog) if prog else "(identity)"
    print(f"  SYNTHESIZED program:  {pretty}    (searched {nodes} programs)")
    print("  generalizes to new inputs (never in the examples):")
    for t in tests:
        try:
            print(f'      "{t}"  ->  "{run(prog, t)}"')
        except Exception:
            print(f'      "{t}"  ->  (program not applicable)')
    print()


def main():
    print("=== program_synth — write code from examples, by search, verifiably ===\n")

    print("1. extract initials")
    synthesize([("John Smith", "JS"), ("Mary Jane", "MJ")],
               ["Alice Cooper", "grace hopper"])

    print("2. first name, uppercased  (2-op program: it COMPOSES)")
    synthesize([("John Smith", "JOHN"), ("bob dylan", "BOB")],
               ["alice cooper", "Ada Lovelace"])

    print("3. last name, lowercased")
    synthesize([("John SMITH", "smith"), ("Mary JANE", "jane")],
               ["Grace HOPPER"])

    print("4. swap the two words")
    synthesize([("John Smith", "Smith John"), ("bob dylan", "dylan bob")],
               ["alice cooper"])

    print("5. a transformation NOT in the DSL (honest failure)")
    synthesize([("John Smith", "Smith, John")], ["Mary Jane"])

    print("Same search engine as algebra and the bridge puzzle — here the")
    print("operators are code, and every program returned is correct on the spec.")


if __name__ == "__main__":
    main()
