"""
concept_encoder.py — Encodes concept strings as stable float vectors.

Each concept gets a deterministic unit vector seeded by its string hash.
No pre-trained embeddings — purely structural, learns meaning from co-occurrence.

After training, the SOM clusters related concepts near each other.
This is grounding: fire/heat/burn will cluster because they co-occur in chains.
"""

import numpy as np
import hashlib
from typing import Dict, List

class ConceptEncoder:
    def __init__(self, n_dims: int, seed: int = 42):
        self.n_dims  = n_dims
        self.seed    = seed
        self._cache: Dict[str, np.ndarray] = {}

    def encode(self, concept: str) -> np.ndarray:
        if concept in self._cache:
            return self._cache[concept]

        # Deterministic: same concept always same vector
        h = int(hashlib.sha256(concept.encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        vec = rng.standard_normal(self.n_dims).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        self._cache[concept] = vec
        return vec

    def encode_batch(self, concepts: List[str]) -> np.ndarray:
        return np.stack([self.encode(c) for c in concepts])

    def known_concepts(self) -> List[str]:
        return list(self._cache.keys())

    def similarity(self, a: str, b: str) -> float:
        return float(np.dot(self.encode(a), self.encode(b)))
