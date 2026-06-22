#!/usr/bin/env python3
"""
eval_front.py — honest end-to-end eval of the language front + ablation.

Measures the WHOLE front (lexical -> student -> ...), not just the student head, on
HELD-OUT questions (train the student on 70%, evaluate on the other 30% — no
leakage). Ablates the tiers so we see what each actually contributes:

    lexical-only   : the hand/embedding matcher alone
    + student      : add the distilled student (the real runtime front, no LLM)

Honest: this is relation-resolution accuracy on the real qwen3-coder corpus (610
questions, 9 entities). It tells us whether the architecture holds beyond toys
before we build more on it.

    venv2/bin/python3 eval_front.py        (100d GloVe if present, else 50d)
"""

import os
import random
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from nl_query import NLQueryParser, load_glove, _rel_words, STOP
from student_trainer import Student, load_dataset, DATA

STRONG = 0.9
GLOVE_100 = os.path.join(os.path.dirname(__file__), "glove.6B.100d.txt")


def resolve(lex, student, q, ent_set, use_student):
    """Mirror of nl_front.resolve, ablatable: lexical-strong/agree, then student
    when confident; returns the resolved relation (or None)."""
    toks = [t for t in re.findall(r"[a-z_]+", q.lower()) if t not in STOP]
    ent = next((t for t in toks if t in ent_set), None)
    content = [t for t in toks if t != ent]
    lr, sc = lex.match_relation(content)
    if not use_student:
        return (lr if lr is not None and sc >= STRONG else None)
    srel, conf = student.confident_rel(q)
    if lr is not None and sc >= STRONG and lr == srel:
        return lr
    if conf >= 0.85:
        return srel
    if lr is not None and sc >= STRONG:
        return lr
    return None


def main():
    rows = [r for r in load_dataset(DATA) if r["intent"] == "attribute"]
    random.seed(1)
    random.shuffle(rows)
    cut = int(len(rows) * 0.7)
    train, test = rows[:cut], rows[cut:]

    entities = {r["label"]["entity"] for r in rows}
    relations = sorted({r["label"]["rel"] for r in rows})
    needed = set(relations) | entities
    for r in relations:
        needed.update(_rel_words(r))
    for r in rows:
        needed.update(re.findall(r"[a-z_]+", r["question"].lower()))
    path = GLOVE_100 if os.path.exists(GLOVE_100) else None
    glove = load_glove(path=path, needed=needed) if path else load_glove(needed=needed)

    student = Student.train(train, glove, k=5)
    lex = NLQueryParser(entities, relations, glove)

    print("=== eval_front — held-out relation resolution + ablation ===")
    print(f"  corpus {len(rows)} attribute Qs, {len(entities)} entities, "
          f"{len(relations)} relations | train {len(train)} / test {len(test)}")
    print(f"  embeddings: {'100d' if path else '50d'}\n")
    for label, use_student in (("lexical-only", False), ("+ student (full)", True)):
        ok = abstain = 0
        for r in test:
            got = resolve(lex, student, r["question"], entities, use_student)
            if got is None:
                abstain += 1
            elif got == r["label"]["rel"]:
                ok += 1
        n = len(test)
        print(f"  {label:18s} accuracy {ok}/{n} = {ok/n:.0%}   (abstained {abstain})")


if __name__ == "__main__":
    main()
