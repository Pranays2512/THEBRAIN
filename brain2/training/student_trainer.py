#!/usr/bin/env python3
"""
student_trainer.py — train the small LOCAL student on the distilled corpus.

The teacher (big cloud model) produced varied phrasings -> Need labels in
distill_data.jsonl. This trains a lightweight student to map a NEW question to
its Need, generalizing BEYOND nl_query's lexical matcher because it learns from
the teacher's phrasing diversity. Deployed, it's tiny, local, free, private — the
big model never runs at inference.

The student is a nearest-centroid classifier over GloVe sentence vectors (mean of
content-word embeddings): one centroid per relation, predict by cosine. No
backprop, no GPU, interpretable — the project's lightweight discipline. (Swap in
program_synth_tree.DecisionTree for non-linear splits if centroids plateau.)

    st = Student.train("distill_data.jsonl", glove)
    st.predict("how quick is the rocket?")   -> ('attribute', 'rocket', 'speed')

NOTE: trained on MOCK templated data it will look near-perfect — that proves the
pipeline, not generalization. Real generalization needs the real cloud corpus.
"""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from adapters.nl_query import load_glove, STOP, _cos

DATA = os.path.join(os.path.dirname(__file__), "distill_data.jsonl")


def load_dataset(path=DATA):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def sentence_vec(question, glove, dim=50):
    """Mean of content-word GloVe vectors — a cheap sentence embedding."""
    toks = [t for t in re.findall(r"[a-z_]+", question.lower()) if t not in STOP]
    vs = [glove[t] for t in toks if t in glove]
    return np.mean(vs, axis=0) if vs else np.zeros(dim)


class Student:
    def __init__(self, glove):
        self.glove = glove
        self.centroids = {}        # rel -> mean sentence vector
        self.entities = set()

    @classmethod
    def train(cls, path_or_rows, glove, k=5):
        if isinstance(path_or_rows, list):
            rows = path_or_rows
        else:
            rows = load_dataset(path_or_rows)
        s = cls(glove)
        buckets = {}
        for r in rows:
            rel = r["label"]["rel"]
            buckets.setdefault(rel, []).append(sentence_vec(r["question"], glove))
            # collect entity vocabulary from the labels
            for key in ("entity", "x", "y"):
                if key in r["label"]:
                    s.entities.add(r["label"][key])
        s.centroids = {rel: np.mean(vs, axis=0) for rel, vs in buckets.items()}
        return s

    def predict(self, question):
        """-> (intent, entity_or_pair, rel). Entity is lexical; rel is the
        nearest centroid; intent is 'compare' if two entities appear."""
        toks = re.findall(r"[a-z_]+", question.lower())
        ents = [t for t in toks if t in self.entities]
        v = sentence_vec(question, self.glove)
        rel = max(self.centroids, key=lambda r: _cos(v, self.centroids[r])) \
            if self.centroids else None
        if len(ents) >= 2:
            return "compare", (ents[0], ents[1]), rel
        return "attribute", (ents[0] if ents else None), rel

    def confident_rel(self, question):
        v = sentence_vec(question, self.glove)
        if not self.centroids:
            return None, 0.0
        scores = [(r, _cos(v, c)) for r, c in self.centroids.items()]
        best_r, best_score = max(scores, key=lambda x: x[1])
        return best_r, best_score


def _accuracy(student, rows):
    ok = 0
    for r in rows:
        _, _, rel = student.predict(r["question"])
        ok += (rel == r["label"]["rel"])
    return ok / len(rows)


def _demo():
    if not os.path.exists(DATA):
        print("no distill_data.jsonl — run distill_harness.py first.")
        return
    rows = load_dataset(DATA)
    rels = {r["label"]["rel"] for r in rows}
    # load glove for dataset words + relation words + a few NOVEL probe words
    needed = set(rels)
    for r in rows:
        needed.update(re.findall(r"[a-z_]+", r["question"].lower()))
    probes = ["how quick is the rocket?",            # quick ~ speed (unseen word)
              "what's the heft of the sample?",       # heft ~ mass/density (unseen)
              "tell me the rocket velocity"]          # velocity ~ speed (unseen)
    for q in probes:
        needed.update(re.findall(r"[a-z_]+", q.lower()))
    glove = load_glove(needed=needed)

    student = Student.train(DATA, glove)
    print("=== student_trainer — lightweight distilled parser ===\n")
    print(f"  trained on {len(rows)} examples, {len(student.centroids)} relations")
    print(f"  train accuracy (rel): {_accuracy(student, rows):.0%}  "
          f"(high here = MOCK templates; real data tests generalization)\n")
    print("  generalization probes (words NOT in training):")
    for q in probes:
        intent, ent, rel = student.predict(q)
        print(f"    {q:38s} -> {intent}/{ent}/{rel}")


if __name__ == "__main__":
    _demo()
