#!/usr/bin/env python3
"""
synthesis_engine.py — hardened Verifiable program synthesis (milestone #6).

Give it input/output examples; it searches a DSL (via the hardened search
engine) for a program that reproduces ALL of them, and returns it. The result
is correct BY CONSTRUCTION — the search goal is "matches every example" — so it
is verified, not guessed. It then generalizes to inputs it never saw, and fails
honestly (returns not-found) when no DSL program fits.

    se = SynthesisEngine()
    r = se.synthesize([("John Smith", "JOHN"), ("bob dylan", "BOB")])
    r.found, r.source          -> True, "upper -> first_word"
    r.apply("ada lovelace")    -> "ADA"   (generalizes; never shown)

Built on the hardened tree_reason search (optimal, deterministic): it returns
the SHORTEST correct program.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.reasoning.tree_reason import SearchProblem, solve


class SynthesisError(ValueError):
    pass


def _safe(fn):
    def g(s):
        try:
            return fn(s)
        except Exception:
            return None
    return g


DSL = {
    "lower":         str.lower,
    "upper":         str.upper,
    "title":         str.title,
    "capitalize":    str.capitalize,
    "strip":         str.strip,
    "first_word":    _safe(lambda s: s.split()[0]),
    "last_word":     _safe(lambda s: s.split()[-1]),
    "initials":      _safe(lambda s: "".join(w[0] for w in s.split())),
    "reverse_words": lambda s: " ".join(reversed(s.split())),
    "no_spaces":     lambda s: s.replace(" ", ""),
}
OPS = list(DSL)


def run(program, s):
    for name in program:
        s = DSL[name](s)
        if s is None:
            raise ValueError("op not applicable")
    return s


class SynthesisResult:
    def __init__(self, found, program):
        self.found = found
        self.program = program          # tuple of op names, or None

    @property
    def source(self):
        if not self.found:
            return "(no program found)"
        return " -> ".join(self.program) if self.program else "(identity)"

    def apply(self, s):
        if not self.found:
            raise SynthesisError("no program to apply")
        return run(self.program, s)

    def __repr__(self):
        return f"SynthesisResult(found={self.found}, program={self.source!r})"


class _Synthesize(SearchProblem):
    def __init__(self, examples, max_len):
        self.examples = examples
        self.max_len = max_len

    def initial(self):
        return ()

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
        return 0

    def moves(self, prog):
        if len(prog) >= self.max_len:
            return
        for name in OPS:
            yield (name, prog + (name,), 1)


class SynthesisEngine:
    def __init__(self, max_len=4):
        if not isinstance(max_len, int) or max_len < 1:
            raise SynthesisError("max_len must be a positive integer")
        self.max_len = max_len

    @staticmethod
    def _validate(examples):
        examples = list(examples)
        if not examples:
            raise SynthesisError("need at least one example")
        for pair in examples:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                    or not all(isinstance(x, str) for x in pair)):
                raise SynthesisError(f"each example must be (str, str), got {pair!r}")
        return examples

    def synthesize(self, examples, max_nodes=200_000):
        examples = self._validate(examples)
        path, _, _ = solve(_Synthesize(examples, self.max_len), max_nodes)
        if path is None:
            return SynthesisResult(False, None)
        prog = path[-1][1] if path else ()       # () => identity solves it
        return SynthesisResult(True, prog)


def _demo():
    se = SynthesisEngine()
    for ex in ([("John Smith", "JS"), ("Mary Jane", "MJ")],
               [("John Smith", "JOHN"), ("bob dylan", "BOB")],
               [("John Smith", "Smith, John")]):
        r = se.synthesize(ex)
        print(f"  {ex[0][0]!r}->{ex[0][1]!r} ...  =>  {r.source}")
        if r.found:
            print(f"      applies to a new input: 'ada lovelace' -> {r.apply('ada lovelace')!r}")


if __name__ == "__main__":
    _demo()
