#!/usr/bin/env python3
"""
nl_query.py — Phase 4, rung 1: talk to the executive in WORDS (no LLM).

The reasoning core (means_ends + policy_pack) only answered formal input —
Need('sample','speed'). This is the lexical front: it maps a natural question to
that Need by EMBEDDING SIMILARITY over the known vocabulary (GloVe, already in
the repo). So "what is the velocity of the sample?" resolves to the `speed`
relation because their vectors are close — synonym/paraphrase robustness earned
from the embedding substrate, not from hand-listing every phrasing, and with no
LLM.

Honest scope: simple "what/how is the <rel> of <entity>" shapes. Entity is matched
lexically against known subjects; relation is matched by cosine against the
vocabulary of policy targets + fact relations. Open/complex phrasing (clauses,
multi-step, anaphora) is the structural-parser / LLM-distillation job — the next
rung, not this one.

    NLQueryParser(...).answer("how much force does the sample have?")
      -> Need('sample','force') -> executive -> 19.6 (verified)
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.knowledge.policy_pack import build_facts, build_memory
from engines.synthesis.policy_proposer import Solver

GLOVE = os.path.join(os.path.dirname(__file__), "glove.6B.50d.txt")

STOP = {"what", "whats", "is", "are", "the", "of", "a", "an", "how", "much",
        "many", "does", "do", "did", "have", "has", "for", "compute", "find",
        "calculate", "give", "me", "tell", "s", "to", "in", "on", "value"}

# abbreviation / multi-word relations -> the words to embed them by
REL_WORDS = {
    "ke": ["kinetic", "energy"], "pe": ["potential", "energy"],
    "molar_mass": ["molar", "mass"], "conc_mass": ["concentration", "mass"],
    "molarity": ["concentration"],
}


def load_glove(path=GLOVE, needed=None):
    """word -> 50d vector. If `needed` is given, keep only those (one fast pass)."""
    vecs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            sp = line.split()
            w = sp[0]
            if needed is not None and w not in needed:
                continue
            vecs[w] = np.asarray(sp[1:], dtype=np.float32)
    return vecs


def _rel_words(rel):
    return REL_WORDS.get(rel, rel.split("_"))


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


class NLQueryParser:
    def __init__(self, entities, relations, glove):
        self.entities = set(entities)
        self.relations = list(relations)
        self.glove = glove
        # one vector per relation = mean of its word-vectors present in GloVe
        self.rel_vec = {}
        for r in self.relations:
            vs = [glove[w] for w in _rel_words(r) if w in glove]
            if vs:
                self.rel_vec[r] = np.mean(vs, axis=0)
        # quick lexical index: a word -> relation it names
        self.lex = {}
        for r in self.relations:
            for w in _rel_words(r) + [r]:
                self.lex[w] = r

    def match_relation(self, tokens):
        """Map content tokens to the best (relation, score): exact name, then a
        shared-prefix morphological hit, then embedding-nearest. Reusable across
        attribute queries (rung 1) and relational/comparison queries (rung 2)."""
        # 1. exact lexical hit (a token that names a relation)
        rel = next((self.lex[t] for t in tokens if t in self.lex), None)
        if rel is not None:
            return rel, 1.0
        # 2. morphological hit: shared 4-char prefix (dense -> density)
        for t in tokens:
            if len(t) < 4:
                continue
            m = next((r for r in self.relations
                      if r.startswith(t[:4]) or t.startswith(r[:4])), None)
            if m:
                return m, 0.9
        # 3. else embedding-nearest relation across content tokens
        best = (0.35, None)                       # threshold floor
        for t in tokens:
            if t not in self.glove:
                continue
            for r, rv in self.rel_vec.items():
                c = _cos(self.glove[t], rv)
                if c > best[0]:
                    best = (c, r)
        return best[1], best[0]

    def parse(self, sentence):
        toks = [t for t in re.findall(r"[a-z_]+", sentence.lower()) if t not in STOP]
        entity = next((t for t in toks if t in self.entities), None)
        content = [t for t in toks if t != entity]
        rel, score = self.match_relation(content)
        return entity, rel, score


def _demo():
    kb, mem = build_facts(), build_memory()
    entities = {"sample"}
    relations = sorted({p.target for ps in mem.by_target.values() for p in ps} |
                       {"mass", "accel", "speed", "distance", "area",
                        "molar_mass", "volume"})

    # load only the vectors we need (relation words + demo query words) -> fast
    queries = [
        "how much force does the sample have?",
        "what is the velocity of the sample?",      # velocity ~ speed (synonym!)
        "what is the momentum of the sample?",
        "compute the kinetic energy of the sample",  # 'kinetic energy' -> ke
        "what is the pressure of the sample?",
        "how dense is the sample?",                  # dense ~ density
    ]
    needed = set()
    for r in relations:
        needed.update(_rel_words(r) + [r])
    for q in queries:
        needed.update(re.findall(r"[a-z_]+", q.lower()))
    glove = load_glove(needed=needed)

    parser = NLQueryParser(entities, relations, glove)

    print("=== nl_query — ask the executive in words (GloVe match, no LLM) ===\n")
    for q in queries:
        ent, rel, sc = parser.parse(q)
        if ent is None or rel is None:
            print(f"  > {q}\n      (couldn't map to a known entity/relation)\n")
            continue
        ans = Solver(kb, mem, use_proposer=True).solve(ent, rel)
        ans_s = "—" if ans is None else f"{ans:.4g}"
        print(f"  > {q}")
        print(f"      -> Need({ent}, {rel})   [match {sc:.2f}]   = {ans_s}\n")


if __name__ == "__main__":
    _demo()
