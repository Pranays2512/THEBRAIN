#!/usr/bin/env python3
"""
student_trainer.py — train the small LOCAL student on the distilled corpus.

The teacher (cloud model) produced varied phrasings -> Need labels in
distill_data.jsonl. This trains a lightweight student to map a NEW question to its
Need, generalizing beyond nl_query's lexical matcher via embeddings. Deployed it's
tiny, local, free, private — the big model never runs at inference.

Two student heads over GloVe sentence vectors (mean of content-word embeddings):
  * centroid — one mean vector per relation, predict by cosine (fast, crude).
  * kNN      — cosine-weighted vote over the k nearest TRAINING questions
               (stronger on small, diverse data; keeps every example).
No backprop, no GPU — the project's lightweight discipline. Held-out eval picks
the winner.

    st = Student.train(rows, glove)
    st.predict("how quick is the rocket?", method="knn")  -> ('attribute','rocket','speed')
"""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from nl_query import load_glove, STOP, _cos

DATA = os.path.join(os.path.dirname(__file__), "distill_data.jsonl")


def load_dataset(path=DATA):
    with open(path) as f:
        return [json.loads(line) for line in f]


def sentence_vec(question, glove, dim=50):
    """Mean of content-word GloVe vectors — a cheap sentence embedding."""
    toks = [t for t in re.findall(r"[a-z_]+", question.lower()) if t not in STOP]
    vs = [glove[t] for t in toks if t in glove]
    return np.mean(vs, axis=0) if vs else np.zeros(dim)


class Student:
    def __init__(self, glove, k=5):
        self.glove = glove
        self.k = k
        self.examples = []         # [(vec, rel)]
        self.centroids = {}
        self.entities = set()

    @classmethod
    def train(cls, rows, glove, k=5):
        s = cls(glove, k)
        buckets, wc = {}, {}       # wc: word -> {rel: count}
        for r in rows:
            rel = r["label"]["rel"]
            v = sentence_vec(r["question"], glove)
            s.examples.append((v, rel))
            buckets.setdefault(rel, []).append(v)
            for w in set(re.findall(r"[a-z_]+", r["question"].lower())):
                if w not in STOP and w in glove:
                    wc.setdefault(w, {}).setdefault(rel, 0)
                    wc[w][rel] += 1
            for key in ("entity", "x", "y"):
                if key in r["label"]:
                    s.entities.add(r["label"][key])
        s.centroids = {rel: np.mean(vs, axis=0) for rel, vs in buckets.items()}
        # DISCRIMINATIVE words: keep w for rel only if it concentrates there
        # (>=60% of its occurrences, >=2 times) — drops filler/entity words.
        s.rel_words = {}
        for w, rels in wc.items():
            tot = sum(rels.values())
            for rel, c in rels.items():
                if c >= 2 and c / tot >= 0.6:
                    s.rel_words.setdefault(rel, {})[w] = glove[w]
        return s

    def _rel_wordmax(self, question):
        """Score a relation by the BEST single (query-word, relation-word) cosine
        — the salient content word decides, instead of a diluted sentence mean."""
        qw = [self.glove[w] for w in re.findall(r"[a-z_]+", question.lower())
              if w not in STOP and w in self.glove]
        if not qw:
            return None
        best = (-1.0, None)
        for rel, words in self.rel_words.items():
            sc = max(_cos(q, wv) for q in qw for wv in words.values())
            if sc > best[0]:
                best = (sc, rel)
        return best[1]

    def _rel_centroid(self, v):
        return max(self.centroids, key=lambda r: _cos(v, self.centroids[r])) \
            if self.centroids else None

    def _rel_knn(self, v):
        sims = sorted(((_cos(v, ev), rel) for ev, rel in self.examples), reverse=True)
        votes = {}
        for sim, rel in sims[:self.k]:
            votes[rel] = votes.get(rel, 0.0) + max(sim, 0.0)   # cosine-weighted
        return max(votes, key=votes.get) if votes else None

    def predict(self, question, method="wordmax"):
        toks = re.findall(r"[a-z_]+", question.lower())
        ents = [t for t in toks if t in self.entities]
        if method == "wordmax":
            rel = self._rel_wordmax(question)
        elif method == "knn":
            rel = self._rel_knn(sentence_vec(question, self.glove))
        else:
            rel = self._rel_centroid(sentence_vec(question, self.glove))
        if len(ents) >= 2:
            return "compare", (ents[0], ents[1]), rel
        return "attribute", (ents[0] if ents else None), rel


def accuracy(student, rows, method):
    if not rows:
        return 0.0
    ok = sum(student.predict(r["question"], method)[2] == r["label"]["rel"] for r in rows)
    return ok / len(rows)


def _demo():
    if not os.path.exists(DATA):
        print("no distill_data.jsonl — run distill_harness.py first.")
        return
    import random
    rows = load_dataset()
    random.seed(1)
    random.shuffle(rows)
    cut = int(len(rows) * 0.7)
    train_rows, test_rows = rows[:cut], rows[cut:]

    needed = {r["label"]["rel"] for r in rows}
    for r in rows:
        needed.update(re.findall(r"[a-z_]+", r["question"].lower()))
    glove = load_glove(needed=needed)

    student = Student.train(train_rows, glove, k=5)
    print("=== student_trainer — distilled parser (held-out eval) ===\n")
    print(f"  corpus {len(rows)} ex, {len(student.centroids)} relations, "
          f"train {len(train_rows)} / test {len(test_rows)}\n")
    for m in ("centroid", "knn", "wordmax"):
        print(f"  {m:9s} held-out rel accuracy: {accuracy(student, test_rows, m):.0%}")
    print("\n  wordmax on held-out test questions:")
    for r in test_rows[:8]:
        p = student.predict(r["question"], "wordmax")
        mark = "OK " if p[2] == r["label"]["rel"] else "XX "
        print(f"    {mark}{r['question'][:44]:44s} -> {p[2]} (true {r['label']['rel']})")


if __name__ == "__main__":
    _demo()
