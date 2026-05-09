"""
conceptnet_loader.py — Loads ConceptNet relations as training sequences.

Downloads ConceptNet CSV if not present (~500MB).
Filters to English, high-weight relations only.
Converts to (concept_vec, word) sequences for brain training.

ConceptNet relations used:
  Causes, CausesDesire, CapableOf, IsA, HasA, HasProperty,
  PartOf, UsedFor, ReceivingAction, Desires, MotivatedByGoal

Usage:
    loader = ConceptNetLoader(n_dims=32)
    loader.download_if_needed()
    for seq in loader.sequences(max_seqs=100000):
        # feed to brain
"""

import os
import csv
import gzip
import urllib.request
import numpy as np
from typing import Iterator, List, Tuple, Optional
from concept_encoder import ConceptEncoder

Step     = Tuple[str, str]
Sequence = List[Step]

CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
CONCEPTNET_PATH = os.path.join(os.path.dirname(__file__), "conceptnet-assertions-5.7.0.csv.gz")

# Relations to include — causal/structural (not linguistic meta-relations)
USEFUL_RELATIONS = {
    "/r/Causes",
    "/r/CausesDesire",
    "/r/CapableOf",
    "/r/IsA",
    "/r/HasA",
    "/r/HasProperty",
    "/r/PartOf",
    "/r/UsedFor",
    "/r/ReceivingAction",
    "/r/Desires",
    "/r/MotivatedByGoal",
    "/r/NotCapableOf",
    "/r/NotDesires",
    "/r/Antonym",
}

RELATION_WORD = {
    "/r/Causes":         "causes",
    "/r/CausesDesire":   "makes_want",
    "/r/CapableOf":      "can",
    "/r/IsA":            "isa",
    "/r/HasA":           "hasa",
    "/r/HasProperty":    "is",
    "/r/PartOf":         "partof",
    "/r/UsedFor":        "usedfor",
    "/r/ReceivingAction":"receives",
    "/r/Desires":        "wants",
    "/r/MotivatedByGoal":"goal",
    "/r/NotCapableOf":   "cannot",
    "/r/NotDesires":     "not_want",
    "/r/Antonym":        "opposite",
}

def _extract_concept(uri: str) -> Optional[str]:
    """Extract English concept name from ConceptNet URI like /c/en/fire"""
    parts = uri.split("/")
    if len(parts) < 4:
        return None
    if parts[2] != "en":  # English only
        return None
    concept = parts[3].replace("_", " ")
    # Skip multi-word for now (too complex for early training)
    if " " in concept or len(concept) > 20:
        return None
    return concept.lower()


class ConceptNetLoader:
    def __init__(self, n_dims: int, min_weight: float = 1.0, vocab_cap: int = 5000):
        self.enc        = ConceptEncoder(n_dims)
        self.n_dims     = n_dims
        self.min_weight = min_weight
        self.vocab_cap  = vocab_cap
        self._allowed_words = None

    def _build_vocab(self, path: str):
        from collections import Counter
        import json

        print(f"Scanning ConceptNet to build top {self.vocab_cap} vocabulary...")
        counts = Counter()
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 5 or row[1] not in USEFUL_RELATIONS:
                    continue
                try:
                    weight = float(json.loads(row[4]).get("weight", 1.0))
                    if weight < self.min_weight:
                        continue
                except Exception:
                    pass

                concept_a = _extract_concept(row[2])
                concept_b = _extract_concept(row[3])
                if concept_a:
                    counts[concept_a] += 1
                if concept_b:
                    counts[concept_b] += 1

        top = counts.most_common(self.vocab_cap)
        self._allowed_words = {word for word, _ in top}
        if top:
            print(f"  Vocab built. Top={top[0][0]}:{top[0][1]} floor={top[-1][0]}:{top[-1][1]}")
        else:
            print("  Vocab built, but no usable words were found.")

    def download_if_needed(self, path: str = CONCEPTNET_PATH) -> str:
        if os.path.exists(path):
            print(f"ConceptNet found: {path}")
            return path
        print(f"Downloading ConceptNet (~500MB) to {path} ...")
        print("This only happens once.")
        urllib.request.urlretrieve(CONCEPTNET_URL, path,
            reporthook=lambda b, bs, t: print(
                f"  {b*bs/1e6:.0f}MB / {t/1e6:.0f}MB", end="\r"))
        print(f"\nDownload complete.")
        return path

    def sequences(self,
                  path: str = CONCEPTNET_PATH,
                  max_seqs: int = 500_000) -> Iterator[Sequence]:
        """
        Yields training sequences from ConceptNet.
        Format: [(concept_a_vec, word_a), (relation_word_vec, relation_word),
                 (concept_b_vec, word_b)]
        """
        if not os.path.exists(path):
            print(f"ConceptNet not found at {path}. Run download_if_needed() first.")
            return

        if self._allowed_words is None and self.vocab_cap > 0:
            self._build_vocab(path)

        opener = gzip.open if path.endswith(".gz") else open
        count  = 0

        with opener(path, "rt", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if count >= max_seqs:
                    break
                if len(row) < 5:
                    continue

                # row: [assertion_uri, relation, subject, object, json_info]
                relation = row[1]
                if relation not in USEFUL_RELATIONS:
                    continue

                # Parse weight from JSON info
                try:
                    import json
                    info   = json.loads(row[4])
                    weight = float(info.get("weight", 1.0))
                    if weight < self.min_weight:
                        continue
                except Exception:
                    pass

                concept_a = _extract_concept(row[2])
                concept_b = _extract_concept(row[3])
                if not concept_a or not concept_b:
                    continue

                if self._allowed_words and (
                    concept_a not in self._allowed_words or
                    concept_b not in self._allowed_words
                ):
                    continue

                rel_word = RELATION_WORD.get(relation, "relates")

                # Build sequence: A relation B
                seq: Sequence = [
                    (concept_a, concept_a),
                    (rel_word,  rel_word),
                    (concept_b, concept_b),
                ]
                yield seq
                count += 1

    def load_as_list(self, path: str = CONCEPTNET_PATH,
                     max_seqs: int = 100_000) -> List[Sequence]:
        return list(self.sequences(path, max_seqs))

    def concept_count(self, path: str = CONCEPTNET_PATH,
                      max_seqs: int = 50_000) -> int:
        seen = set()
        for seq in self.sequences(path, max_seqs):
            for concept, _ in seq:
                seen.add(concept)
        return len(seen)
