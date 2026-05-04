"""
math_sequences.py — Generates structured math/logic training sequences.

Produces sequences of (concept_vector, word) pairs the brain can learn from.
Each sequence = one "thought" — brain sees it as a stream of activations + words.

Types:
  arithmetic  : [2, +, 3, =, 5]
  comparison  : [5, >, 3, =, true]
  negation    : [!, hot, =, cold]
  sequence    : [1, ->, 2, ->, 3, ->, 4]  (counting)
  algebra     : [x, +, 3, =, 7, ->, x, =, 4]  (variable solving)
  physics     : [fire, causes, heat]  (causal reasoning)
"""

import numpy as np
from typing import List, Tuple, Iterator
from concept_encoder import ConceptEncoder

# (concept_string, word_label) — word heard alongside the concept activation
Step = Tuple[str, str]
Sequence = List[Step]

class MathSequenceGenerator:
    def __init__(self, n_dims: int, max_n: int = 20):
        self.enc     = ConceptEncoder(n_dims)
        self.n_dims  = n_dims
        self.max_n   = max_n
        self.rng     = np.random.default_rng(42)

        # Pre-register number concepts
        self._numbers = [str(i) for i in range(max_n + 1)]
        for n in self._numbers:
            self.enc.encode(n)

        # Register operator concepts
        for sym in ["+", "-", "*", "=", ">", "<", "->", "!", "true", "false",
                    "x", "y", "z", "causes", "prevents", "isa", "hasa"]:
            self.enc.encode(sym)

    def encode_seq(self, sequence: Sequence) -> List[Tuple[np.ndarray, str]]:
        """Convert (concept_str, word) pairs to (vec, word) pairs."""
        return [(self.enc.encode(c), w) for c, w in sequence]

    # ── Generators ──────────────────────────────────────────────────

    def arithmetic(self) -> Sequence:
        """a + b = c"""
        a = int(self.rng.integers(0, self.max_n // 2))
        b = int(self.rng.integers(0, self.max_n // 2))
        c = a + b
        if c > self.max_n:
            return self.arithmetic()
        return [(str(a), str(a)), ("+", "plus"),
                (str(b), str(b)), ("=", "equals"), (str(c), str(c))]

    def subtraction(self) -> Sequence:
        """a - b = c  (a >= b, result >= 0)"""
        b = int(self.rng.integers(0, self.max_n // 2))
        a = int(self.rng.integers(b, self.max_n))
        c = a - b
        return [(str(a), str(a)), ("-", "minus"),
                (str(b), str(b)), ("=", "equals"), (str(c), str(c))]

    def comparison(self) -> Sequence:
        """a > b = true/false"""
        a = int(self.rng.integers(0, self.max_n))
        b = int(self.rng.integers(0, self.max_n))
        result = "true" if a > b else "false"
        op     = ">" if a > b else "<"
        return [(str(a), str(a)), (op, "greater" if op == ">" else "less"),
                (str(b), str(b)), ("=", "equals"), (result, result)]

    def counting(self) -> Sequence:
        """n -> n+1 -> n+2 -> ..."""
        start = int(self.rng.integers(0, self.max_n - 4))
        length = int(self.rng.integers(3, 6))
        seq = []
        for i in range(length):
            n = start + i
            if n > self.max_n:
                break
            seq.append((str(n), str(n)))
            if i < length - 1:
                seq.append(("->", "then"))
        return seq

    def negation(self) -> Sequence:
        """! true = false, ! false = true"""
        val = self.rng.choice(["true", "false"])
        result = "false" if val == "true" else "true"
        return [("!", "not"), (val, val), ("=", "equals"), (result, result)]

    def causal(self) -> Sequence:
        """concept causes/prevents/isa concept"""
        pairs = [
            ("fire",    "causes",   "heat"),
            ("water",   "causes",   "wet"),
            ("ice",     "causes",   "cold"),
            ("sun",     "causes",   "light"),
            ("rain",    "causes",   "wet"),
            ("drop",    "causes",   "fall"),
            ("fire",    "causes",   "burn"),
            ("eat",     "causes",   "full"),
            ("sleep",   "causes",   "rest"),
            ("water",   "prevents", "fire"),
            ("cold",    "prevents", "heat"),
            ("dog",     "isa",      "animal"),
            ("cat",     "isa",      "animal"),
            ("tree",    "isa",      "plant"),
            ("apple",   "isa",      "fruit"),
            ("bird",    "hasa",     "wings"),
            ("fish",    "hasa",     "fins"),
            ("human",   "hasa",     "hands"),
        ]
        a, rel, b = pairs[int(self.rng.integers(0, len(pairs)))]
        return [(a, a), (rel, rel), (b, b)]

    def variable_solve(self) -> Sequence:
        """x + b = c → x = a  (simple linear)"""
        a = int(self.rng.integers(1, self.max_n // 2))
        b = int(self.rng.integers(1, self.max_n // 2))
        c = a + b
        if c > self.max_n:
            return self.arithmetic()
        return [("x", "x"), ("+", "plus"), (str(b), str(b)),
                ("=", "equals"), (str(c), str(c)),
                ("->", "therefore"),
                ("x", "x"), ("=", "equals"), (str(a), str(a))]

    def all_types(self) -> Iterator[Sequence]:
        """Infinite iterator cycling all sequence types."""
        generators = [
            self.arithmetic,
            self.arithmetic,    # 2x weight — most common
            self.subtraction,
            self.comparison,
            self.counting,
            self.negation,
            self.causal,
            self.causal,        # 2x weight — important for understanding
            self.variable_solve,
        ]
        while True:
            g = generators[int(self.rng.integers(0, len(generators)))]
            yield g()

    def batch(self, n: int) -> List[Sequence]:
        gen = self.all_types()
        return [next(gen) for _ in range(n)]
