"""
math_sequences.py — Structured math/logic/physics/language training sequences.

Curriculum levels:
  1 = basic   : arithmetic, subtraction, comparison, counting, negation, causal
  2 = medium  : multiplication, chained causality, analogies, spatial transitivity
  3 = full    : physics laws, syllogisms, multi-step algebra, set logic, temporal

Usage:
  gen = MathSequenceGenerator(n_dims=32, curriculum=1)  # start easy
  gen.curriculum = 2                                     # upgrade mid-training
  for seq in gen.all_types():
      ...
"""

import numpy as np
from typing import List, Tuple, Iterator
from concept_encoder import ConceptEncoder

Step     = Tuple[str, str]   # (concept_string, word_label)
Sequence = List[Step]


class MathSequenceGenerator:
    def __init__(self, n_dims: int, max_n: int = 20, curriculum: int = 1):
        self.enc        = ConceptEncoder(n_dims)
        self.n_dims     = n_dims
        self.max_n      = max_n
        self.curriculum = curriculum          # 1 / 2 / 3
        self.rng        = np.random.default_rng(42)

        # Pre-register all concepts so vectors are stable
        for n in range(max_n + 1):
            self.enc.encode(str(n))

        for sym in [
            "+", "-", "*", "/", "=", ">", "<", "->", "!", "mod",
            "true", "false", "x", "y", "z",
            "causes", "prevents", "isa", "hasa", "needs", "produces",
            "if", "then", "all", "some", "not", "and", "or",
            "opposite", "above", "below", "inside", "outside",
            "before", "after", "implies", "therefore", "because",
            "force", "mass", "acceleration", "energy", "heat",
            "pressure", "speed", "distance", "time", "voltage",
            "current", "resistance", "greater", "less", "equals",
        ]:
            self.enc.encode(sym)

    def encode_seq(self, seq: Sequence) -> List[Tuple[np.ndarray, str]]:
        return [(self.enc.encode(c), w) for c, w in seq]

    # ── Level 1: Basic ───────────────────────────────────────────────

    def arithmetic(self) -> Sequence:
        a = int(self.rng.integers(0, self.max_n // 2))
        b = int(self.rng.integers(0, self.max_n // 2))
        c = a + b
        if c > self.max_n:
            return self.arithmetic()
        return [(str(a), str(a)), ("+", "plus"),
                (str(b), str(b)), ("=", "equals"), (str(c), str(c))]

    def subtraction(self) -> Sequence:
        b = int(self.rng.integers(0, self.max_n // 2))
        a = int(self.rng.integers(b, self.max_n))
        c = a - b
        return [(str(a), str(a)), ("-", "minus"),
                (str(b), str(b)), ("=", "equals"), (str(c), str(c))]

    def comparison(self) -> Sequence:
        a = int(self.rng.integers(0, self.max_n))
        b = int(self.rng.integers(0, self.max_n))
        op     = ">" if a > b else ("<" if a < b else "=")
        result = "true" if a > b else "false"
        op_word = "greater" if op == ">" else ("less" if op == "<" else "equals")
        return [(str(a), str(a)), (op, op_word),
                (str(b), str(b)), ("=", "equals"), (result, result)]

    def counting(self) -> Sequence:
        start  = int(self.rng.integers(0, self.max_n - 4))
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
        val    = self.rng.choice(["true", "false"])
        result = "false" if val == "true" else "true"
        return [("!", "not"), (val, val), ("=", "equals"), (result, result)]

    def causal(self) -> Sequence:
        pairs = [
            ("fire",    "causes",   "heat"),
            ("fire",    "causes",   "burn"),
            ("fire",    "causes",   "light"),
            ("water",   "causes",   "wet"),
            ("water",   "prevents", "fire"),
            ("ice",     "causes",   "cold"),
            ("ice",     "isa",      "water"),
            ("sun",     "causes",   "light"),
            ("sun",     "causes",   "heat"),
            ("rain",    "causes",   "wet"),
            ("gravity", "causes",   "fall"),
            ("eat",     "causes",   "full"),
            ("sleep",   "causes",   "rest"),
            ("cold",    "prevents", "heat"),
            ("dog",     "isa",      "animal"),
            ("cat",     "isa",      "animal"),
            ("tree",    "isa",      "plant"),
            ("apple",   "isa",      "fruit"),
            ("bird",    "hasa",     "wings"),
            ("fish",    "hasa",     "fins"),
            ("human",   "hasa",     "hands"),
            ("plant",   "needs",    "sunlight"),
            ("plant",   "needs",    "water"),
            ("animal",  "needs",    "food"),
            ("magnet",  "causes",   "attraction"),
            ("virus",   "causes",   "disease"),
            ("exercise","causes",   "strength"),
            ("cold",    "causes",   "ice"),
            ("heat",    "causes",   "steam"),
        ]
        a, rel, b = pairs[int(self.rng.integers(0, len(pairs)))]
        return [(a, a), (rel, rel), (b, b)]

    def variable_solve(self) -> Sequence:
        a = int(self.rng.integers(1, self.max_n // 2))
        b = int(self.rng.integers(1, self.max_n // 2))
        c = a + b
        if c > self.max_n:
            return self.arithmetic()
        return [("x", "x"), ("+", "plus"), (str(b), str(b)),
                ("=", "equals"), (str(c), str(c)),
                ("->", "therefore"),
                ("x", "x"), ("=", "equals"), (str(a), str(a))]

    # ── Level 2: Medium ──────────────────────────────────────────────

    def multiplication(self) -> Sequence:
        a = int(self.rng.integers(1, 6))
        b = int(self.rng.integers(1, min(6, self.max_n // max(a, 1) + 1)))
        c = a * b
        if c > self.max_n:
            a, b = 2, 3; c = 6
        return [(str(a), str(a)), ("*", "times"),
                (str(b), str(b)), ("=", "equals"), (str(c), str(c))]

    def modular(self) -> Sequence:
        b = int(self.rng.integers(2, 6))
        a = int(self.rng.integers(b, self.max_n))
        c = a % b
        return [(str(a), str(a)), ("mod", "mod"),
                (str(b), str(b)), ("=", "equals"), (str(c), str(c))]

    def chained_causality(self) -> Sequence:
        """A causes B, B causes C → A causes C  (transitivity)"""
        chains = [
            ("fire",   "heat",      "burn"),
            ("rain",   "wet",       "mud"),
            ("sun",    "light",     "vision"),
            ("cold",   "ice",       "slip"),
            ("stress", "weakness",  "disease"),
            ("study",  "knowledge", "success"),
            ("virus",  "disease",   "weakness"),
            ("water",  "rust",      "damage"),
            ("heat",   "steam",     "pressure"),
        ]
        A, B, C = chains[int(self.rng.integers(0, len(chains)))]
        return [
            (A, A), ("causes", "causes"), (B, B),
            ("and", "and"),
            (B, B), ("causes", "causes"), (C, C),
            ("->", "therefore"),
            (A, A), ("causes", "causes"), (C, C),
        ]

    def analogy(self) -> Sequence:
        """hot opposite cold, fast opposite slow (opposite pairs)"""
        pairs = [
            ("hot",   "cold"),  ("up",    "down"),  ("big",   "small"),
            ("fast",  "slow"),  ("dark",  "light"), ("hard",  "soft"),
            ("wet",   "dry"),   ("loud",  "quiet"), ("old",   "young"),
            ("open",  "closed"),("empty", "full"),  ("sharp", "dull"),
            ("true",  "false"), ("good",  "bad"),   ("strong","weak"),
        ]
        idx = int(self.rng.integers(0, len(pairs)))
        a, not_a = pairs[idx]
        idx2 = (idx + 1 + int(self.rng.integers(0, len(pairs) - 1))) % len(pairs)
        b, not_b = pairs[idx2]
        return [
            (a, a), ("opposite", "opposite"), (not_a, not_a),
            (b, b), ("opposite", "opposite"), (not_b, not_b),
        ]

    def spatial_transitivity(self) -> Sequence:
        """A above B, B above C → A above C"""
        objects = ["table", "book", "cup", "shelf", "box", "plate", "lamp", "chair"]
        idx = int(self.rng.integers(0, len(objects) - 2))
        A, B, C = objects[idx], objects[idx + 1], objects[idx + 2]
        rel = self.rng.choice(["above", "below", "inside"])
        return [
            (A, A), (rel, rel), (B, B),
            (B, B), (rel, rel), (C, C),
            ("->", "therefore"),
            (A, A), (rel, rel), (C, C),
        ]

    # ── Level 3: Advanced ────────────────────────────────────────────

    def physics_law(self) -> Sequence:
        """F = m * a  style dimensional relationships"""
        laws = [
            [("force",    "force"),    ("=", "equals"), ("mass",     "mass"),     ("*", "times"),   ("acceleration", "acceleration")],
            [("speed",    "speed"),    ("=", "equals"), ("distance", "distance"), ("/", "divided"), ("time",         "time")],
            [("pressure", "pressure"), ("=", "equals"), ("force",    "force"),    ("/", "divided"), ("area",         "area")],
            [("voltage",  "voltage"),  ("=", "equals"), ("current",  "current"),  ("*", "times"),   ("resistance",   "resistance")],
            [("energy",   "energy"),   ("=", "equals"), ("force",    "force"),    ("*", "times"),   ("distance",     "distance")],
        ]
        return laws[int(self.rng.integers(0, len(laws)))]

    def syllogism(self) -> Sequence:
        """All A are B, X is A → X is B"""
        examples = [
            ("dogs",   "animals",    "fido",     "animal"),
            ("birds",  "animals",    "robin",    "animal"),
            ("plants", "organisms",  "oak",      "organism"),
            ("metals", "conductors", "copper",   "conductor"),
            ("humans", "mortal",     "socrates", "mortal"),
            ("fruits", "food",       "apple",    "food"),
            ("fish",   "animals",    "salmon",   "animal"),
        ]
        idx = int(self.rng.integers(0, len(examples)))
        A, B, x, b = examples[idx]
        return [
            ("all", "all"), (A, A), ("isa", "isa"), (B, B),
            (x, x), ("isa", "isa"), (A, A),
            ("->", "therefore"),
            (x, x), ("isa", "isa"), (b, b),
        ]

    def temporal_order(self) -> Sequence:
        """A before B, B before C → A before C"""
        events = [
            ("plant", "grow",    "harvest"),
            ("learn", "practice","master"),
            ("birth", "life",    "death"),
            ("rain",  "flood",   "drought"),
            ("seed",  "sprout",  "tree"),
        ]
        A, B, C = events[int(self.rng.integers(0, len(events)))]
        return [
            (A, A), ("before", "before"), (B, B),
            (B, B), ("before", "before"), (C, C),
            ("->", "therefore"),
            (A, A), ("before", "before"), (C, C),
        ]

    def set_membership(self) -> Sequence:
        """X isa A, A isa B → X isa B  (transitivity via sets)"""
        chains = [
            ("fido",  "dog",    "mammal"),
            ("eagle", "bird",   "animal"),
            ("oak",   "tree",   "plant"),
            ("iron",  "metal",  "element"),
            ("mars",  "planet", "body"),
        ]
        X, A, B = chains[int(self.rng.integers(0, len(chains)))]
        return [
            (X, X), ("isa", "isa"), (A, A),
            (A, A), ("isa", "isa"), (B, B),
            ("->", "therefore"),
            (X, X), ("isa", "isa"), (B, B),
        ]

    # ── Generator ────────────────────────────────────────────────────

    def all_types(self) -> Iterator[Sequence]:
        """Infinite iterator cycling all sequence types for current curriculum."""

        # Level 1 generators with weights
        level1 = [
            (self.arithmetic,    3),
            (self.subtraction,   2),
            (self.comparison,    2),
            (self.counting,      1),
            (self.negation,      1),
            (self.causal,        3),
            (self.variable_solve,1),
        ]

        # Level 2 adds
        level2 = level1 + [
            (self.multiplication,      2),
            (self.modular,             1),
            (self.chained_causality,   2),
            (self.analogy,             2),
            (self.spatial_transitivity,1),
        ]

        # Level 3 adds
        level3 = level2 + [
            (self.physics_law,    2),
            (self.syllogism,      2),
            (self.temporal_order, 1),
            (self.set_membership, 1),
        ]

        table = {1: level1, 2: level2, 3: level3}

        while True:
            pool = table.get(self.curriculum, level3)
            generators, weights = zip(*pool)
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            idx = self.rng.choice(len(generators), p=weights)
            yield generators[idx]()

    def batch(self, n: int) -> List[Sequence]:
        gen = self.all_types()
        return [next(gen) for _ in range(n)]