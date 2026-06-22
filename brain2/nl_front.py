#!/usr/bin/env python3
"""
nl_front.py — the runtime language front: lexical matcher + DISTILLED student.

Closes the distillation loop. A question's relation is resolved by:
  1. the lexical matcher (nl_query) when it has a STRONG hit (exact name / prefix /
     close synonym) — high precision, zero training;
  2. otherwise the distilled student (student_trainer, trained on the cloud
     teacher's varied phrasings) — generalizes to wording the matcher misses.
Entity is lexical. The resolved Need goes to the verified executive (facts +
policies + proposer). The big teacher never runs here — only its distilled student.

So the system speaks both the phrasings you hand-listed AND the ones the teacher
taught the student, with a verified number out the end and no LLM at inference.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from reasoning_engine import ReasoningEngine
from means_ends import Policy
from policy_pack import PACK
from policy_proposer import MultiPolicyMemory, Solver
from nl_query import NLQueryParser, load_glove, _rel_words, STOP
from student_trainer import Student, load_dataset, DATA

STRONG = 0.9      # lexical confidence at/above which we trust the matcher over the student


class Front:
    def __init__(self, kb, mem, lexical: NLQueryParser, student: Student, solvable):
        self.kb, self.mem, self.lex, self.student = kb, mem, lexical, student
        self.entities = set(lexical.entities) | set(student.entities)
        self.solvable = solvable      # relations the executive can actually answer

    def resolve(self, q):
        toks = [t for t in re.findall(r"[a-z_]+", q.lower()) if t not in STOP]
        ent = next((t for t in toks if t in self.entities), None)
        content = [t for t in toks if t != ent]
        lr, sc = self.lex.match_relation(content)
        sr = self.student.predict(q, "wordmax")[2]
        # agree + strong lexical -> trust the matcher; else trust the trained
        # student when its answer is solvable (it handles false-friends like
        # 'work out'->force); fall back to lexical otherwise.
        if lr is not None and sc >= STRONG and lr == sr:
            return ent, lr, "lexical"
        if sr in self.solvable:
            return ent, sr, "student"
        if lr in self.solvable:
            return ent, lr, "lexical"
        return ent, (sr or lr), "student"

    def answer(self, q):
        ent, rel, src = self.resolve(q)
        if ent is None or rel is None:
            return f"(couldn't map: entity={ent}, rel={rel})", src
        val = Solver(self.kb, self.mem, use_proposer=True).solve(ent, rel)
        return (f"{ent}.{rel} = {'—' if val is None else f'{val:.4g}'}", src)


def _build():
    kb = ReasoningEngine()
    facts = {
        "rocket": {"mass": 1000, "accel": 12, "speed": 300, "volume": 2.0,
                   "molar_mass": 2, "area": 1.5, "distance": 4},
        "sample": {"mass": 2, "accel": 9.8, "speed": 30, "volume": 0.5,
                   "molar_mass": 18, "area": 0.25, "distance": 5},
    }
    for ent, fs in facts.items():
        for r, v in fs.items():
            kb.learn(ent, r, str(v))
    mem = MultiPolicyMemory()
    for t, ins, expr in PACK:
        mem.add(Policy(t, ins, expr))
    # align distillation vocab with the executive: the teacher used 'energy';
    # the executive only had ke/pe, so add energy = kinetic energy (½mv²).
    mem.add(Policy("energy", ("mass", "speed"),
                   ("*", 0.5, ("*", "mass", ("^", "speed", 2)))))

    rows = load_dataset(DATA)
    entities = set(facts) | {r["label"].get("entity", "") for r in rows} | \
               {r["label"].get("x", "") for r in rows}
    entities.discard("")
    relations = sorted({t for t, _, _ in PACK} | {r for fs in facts.values() for r in fs})

    needed = set(relations) | entities
    for r in relations:
        needed.update(_rel_words(r))
    for row in rows:
        needed.update(re.findall(r"[a-z_]+", row["question"].lower()))
    glove = load_glove(needed=needed)

    student = Student.train(rows, glove, k=5)
    lexical = NLQueryParser(entities, relations, glove)
    solvable = set(mem.by_target) | {r for fs in facts.values() for r in fs}
    return Front(kb, mem, lexical, student, solvable)


def _demo():
    front = _build()
    queries = [
        "what is the speed of the rocket?",       # lexical exact
        "how swiftly is the rocket moving?",      # novel -> student
        "how heavy is the sample?",               # heavy -> mass
        "what's the rocket's momentum?",          # lexical
        "could you work out the force on the sample?",   # phrasing -> student
        "how much energy does the rocket carry?", # energy
        "what is the density of the sample?",      # computed via policy
    ]
    print("=== nl_front — lexical + distilled student -> verified executive ===\n")
    for q in queries:
        ans, src = front.answer(q)
        print(f"  > {q}")
        print(f"      [{src:7s}] {ans}\n")


if __name__ == "__main__":
    _demo()
