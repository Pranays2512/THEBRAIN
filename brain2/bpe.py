#!/usr/bin/env python3
"""
bpe.py — minimal dependency-free byte-pair encoding for brain2's hybrid tokenizer.

Hybrid scheme (see scorecard / training): words present in the frozen GloVe
vocabulary are emitted whole (one token, GloVe-grounded, keeps binding memory
stable). Out-of-vocabulary words are split into learned BPE subword pieces, so
morphologically related OOV words share pieces ("runner"/"running" share "run").

Pieces are marked with a leading "##" continuation marker (except the first
piece of a word) so the LM can distinguish word-initial from word-internal
units and so detokenization is unambiguous.

The merge table is plain JSON and intentionally framework-free: it can be
ported to C++ for in-brain tokenization later without changing the format.
"""

import json
from collections import Counter

END = "</w>"
CONT = "##"  # continuation marker on non-initial pieces


class BPE:
    def __init__(self, merges=None):
        # merges: ordered list of [a, b] pairs (earliest = highest priority)
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._cache = {}

    # ── training ──────────────────────────────────────────────────────────
    def train(self, word_counts, num_merges):
        """word_counts: dict[str,int]. Learns up to num_merges merge rules."""
        words = {tuple(w) + (END,): c for w, c in word_counts.items() if w}
        self.merges = []
        for _ in range(num_merges):
            pairs = Counter()
            for symbols, c in words.items():
                for p in zip(symbols[:-1], symbols[1:]):
                    pairs[p] += c
            if not pairs:
                break
            best, best_c = pairs.most_common(1)[0]
            if best_c < 2:
                break  # no recurring pair worth a merge
            self.merges.append(best)
            merged = best[0] + best[1]
            words = {self._apply_one(sym, best, merged): c for sym, c in words.items()}
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._cache.clear()

    @staticmethod
    def _apply_one(symbols, pair, merged):
        out, i, n = [], 0, len(symbols)
        while i < n:
            if i < n - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    # ── encoding ──────────────────────────────────────────────────────────
    def encode_word(self, word):
        """Split a single word into BPE pieces with continuation markers."""
        if word in self._cache:
            return self._cache[word]
        symbols = list(word) + [END]
        while len(symbols) > 1:
            best, best_rank = None, None
            for p in zip(symbols[:-1], symbols[1:]):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            symbols = list(self._apply_one(symbols, best, best[0] + best[1]))
        # strip end marker, attach continuation markers
        pieces = [s[:-len(END)] if s.endswith(END) else s for s in symbols]
        pieces = [p for p in pieces if p]
        out = [pieces[0]] + [CONT + p for p in pieces[1:]] if pieces else []
        self._cache[word] = out
        return out

    def tokenize(self, text, keep_whole):
        """Hybrid: words in keep_whole stay whole; others -> BPE pieces."""
        toks = []
        for w in text.split():
            if w in keep_whole:
                toks.append(w)
            else:
                toks.extend(self.encode_word(w))
        return toks

    # ── persistence ───────────────────────────────────────────────────────
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": [list(m) for m in self.merges]}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(merges=json.load(f)["merges"])


def train_from_corpus(pairs, num_merges, keep_whole):
    """Train BPE on the OOV portion of a corpus of {input,target} dicts.

    Training only on words NOT kept whole focuses the limited merge budget on
    the subword structure the tokenizer will actually use.
    """
    counts = Counter()
    for p in pairs:
        for field in ("input", "target"):
            for w in p.get(field, "").split():
                if w not in keep_whole:
                    counts[w] += 1
    bpe = BPE()
    bpe.train(counts, num_merges)
    return bpe


if __name__ == "__main__":
    # smoke test
    b = BPE()
    b.train({"running": 5, "runner": 4, "run": 9, "jumping": 3, "jumper": 2}, 50)
    for w in ("running", "runner", "jumped", "run"):
        print(f"{w:10s} -> {b.encode_word(w)}")
