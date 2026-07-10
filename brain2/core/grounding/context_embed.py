#!/usr/bin/env python3
"""
context_embed.py — word meaning from CONTEXT, not a hand table (toward open comprehension).

The parser only knew words it was TOLD (plus a hand synonym map). Open comprehension needs
meaning to come from context, the way grounding derives perceptual concepts from data
instead of being told. This builds count-based distributional vectors: a word's meaning is
the company it keeps (co-occurrence over a corpus). Then an unseen/paraphrase word maps to
the nearest KNOWN canonical word by contextual similarity — no synonym table, no training.

This is the FUZZY proposer half of the membrane: context gives a rough reading
(velocity ~ speed), the crisp reasoner still verifies the structured query it proposes.
A paraphrase routes only if its meaning lands on something checkable.

Honest limit: count-based vectors need a corpus and give SIMILARITY, not precise meaning;
quality scales with corpus size. Real open comprehension needs far more text. This is the
lightweight proof that meaning-from-context generalizes past the hand table.
"""

import math
import re
from collections import defaultdict

CORPUS = """
the rocket has high speed
the rocket travels at high velocity
the rocket moves with great pace
the car moves at great speed
the car runs at high velocity
a fast car has high speed
the object has large mass
the object has large weight
heavy things have large mass
heavy things have large weight
a dense object has large mass
the engine makes strong force
the engine makes strong push
a strong push is a large force
""".strip().split("\n")


def build(corpus=CORPUS, window=2):
    """Co-occurrence vectors: vec[w][c] = how often context word c appears near w."""
    vecs = defaultdict(lambda: defaultdict(float))
    for line in corpus:
        toks = re.findall(r"[a-z]+", line.lower())
        for i, w in enumerate(toks):
            for j in range(max(0, i - window), min(len(toks), i + window + 1)):
                if j != i:
                    vecs[w][toks[j]] += 1.0
    return vecs


def cosine(a, b):
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def nearest(word, canonicals, vecs):
    """Map a word to the nearest CANONICAL word by contextual similarity (no hand table)."""
    if word not in vecs:
        return None, 0.0
    sims = [(c, cosine(vecs[word], vecs[c])) for c in canonicals if c in vecs and c != word]
    if not sims:
        return None, 0.0
    return max(sims, key=lambda x: x[1])


def _demo():
    vecs = build()
    canon = ["speed", "mass", "force"]      # the canonical relation tokens the brain knows
    print("=== context_embed — meaning from context, not a hand table ===\n")
    print("  canonical tokens the brain knows:", canon, "\n")
    # paraphrases NOT in any synonym table — mapped purely by co-occurrence context
    for w in ["velocity", "pace", "weight", "push"]:
        c, s = nearest(w, canon, vecs)
        print("  '%-9s' --context--> '%s'   (similarity %.2f)" % (w, c, s))
    print("\n  None of these were hand-mapped. 'velocity'/'pace' share context with 'speed';")
    print("  'weight' with 'mass'; 'push' with 'force' — meaning inferred from the corpus.")
    print("  Fuzzy reading proposes the canonical; the crisp reasoner still verifies the query.")


if __name__ == "__main__":
    _demo()
