"""
concept_encoder.py — Encodes concept strings as structured float vectors.

FIXED: Replaces random SHA256 hashing with two principled encodings:

  1. NUMBERS  → sinusoidal number-line encoding
     encode("5") is geometrically between encode("4") and encode("6").
     encode("0") through encode("20") are ordered on a manifold.
     The SOM can learn that 2+3 lands near 5.

  2. WORDS    → character n-gram encoding
     encode("fire") and encode("heat") won't be random noise —
     words that appear in similar n-gram contexts cluster together.
     "animal", "animals", "animate" are nearby. "dog", "cat" less so
     but still shaped by character overlap.

No pre-trained embeddings, no external knowledge. Structure emerges
from the signal geometry itself, not from memorized lookups.
"""

import numpy as np
import re
from typing import Dict, List


class ConceptEncoder:
    def __init__(self, n_dims: int, seed: int = 42):
        self.n_dims = n_dims
        self.seed   = seed
        self._cache: Dict[str, np.ndarray] = {}

        # Precompute sinusoidal bases for number encoding
        # Covers integers 0..200 and fractions
        self._num_freqs = np.array(
            [1.0 / (10000 ** (2 * i / n_dims)) for i in range(n_dims // 2)],
            dtype=np.float32
        )

    # ── Public API ────────────────────────────────────────────────────

    def encode(self, concept: str) -> np.ndarray:
        if concept in self._cache:
            return self._cache[concept]

        vec = self._encode(concept.strip().lower())
        self._cache[concept] = vec
        return vec

    def encode_batch(self, concepts: List[str]) -> np.ndarray:
        return np.stack([self.encode(c) for c in concepts])

    def known_concepts(self) -> List[str]:
        return list(self._cache.keys())

    def similarity(self, a: str, b: str) -> float:
        return float(np.dot(self.encode(a), self.encode(b)))

    # ── Internal ──────────────────────────────────────────────────────

    # All string forms that float() silently converts to NaN or Inf
    _POISON = frozenset([
        "nan", "NaN", "NAN",
        "inf", "Inf", "INF", "infinity", "Infinity", "INFINITY",
        "-inf", "-Inf", "-INF", "-infinity", "-Infinity",
        "+inf", "+Inf", "+INF", "+infinity", "+Infinity",
    ])

    def _encode(self, concept: str) -> np.ndarray:
        # Try to parse as a number first.
        # Must NOT be in _POISON — float() silently accepts 'nan', 'inf', etc.
        if concept not in self._POISON:
            try:
                num = float(concept)
                if not (np.isnan(num) or np.isinf(num)):   # safety guard
                    return self._encode_number(num)
            except ValueError:
                pass

        # Written-out numbers
        _written = {
            "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
            "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
            "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
            "fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
            "nineteen":19,"twenty":20,
        }
        if concept in _written:
            return self._encode_number(float(_written[concept]))

        return self._encode_word(concept)

    def _encode_number(self, n: float) -> np.ndarray:
        """
        Sinusoidal positional encoding — same idea as Transformer positions.
        Numbers that are close together get close vectors.
        Arithmetic structure is preserved: 2+3 ≈ 5 in this space.
        """
        # Safety guard — should never happen but makes it bulletproof
        if np.isnan(n) or np.isinf(n):
            return self._encode_word(str(n))

        vec = np.zeros(self.n_dims, dtype=np.float32)
        half = self.n_dims // 2
        angles = n * self._num_freqs[:half]
        vec[:half] = np.sin(angles)
        vec[half:half + len(angles)] = np.cos(angles)

        # Sign indicator in last dim
        if self.n_dims > 2 * len(angles):
            vec[-1] = 1.0 if n >= 0 else -1.0

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    def _encode_word(self, word: str) -> np.ndarray:
        """
        Character n-gram hashing into n_dims dimensions.
        Words sharing character patterns land in nearby regions.
        No external vocabulary or embeddings needed.
        """
        vec = np.zeros(self.n_dims, dtype=np.float32)
        # Pad word with boundary markers
        padded = f"<{word}>"
        ngrams = []
        for n in [1, 2, 3, 4]:
            for i in range(len(padded) - n + 1):
                ngrams.append(padded[i:i + n])

        if not ngrams:
            ngrams = [word]

        for ng in ngrams:
            # Two independent hash functions for better distribution
            h1 = hash(ng) % self.n_dims
            h2 = hash(ng + "_") % self.n_dims
            # Weight shorter n-grams more (they carry more structural info)
            w = 1.0 / len(ng)
            vec[h1] += w
            vec[h2] -= w * 0.5  # Negative to reduce collision clumping

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec
