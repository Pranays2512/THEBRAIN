#!/usr/bin/env python3
"""
structural_parser.py — generalize over sentence STRUCTURE, not just words (the harder layer).

context_embed gave lexical generalization (paraphrase words). This gives STRUCTURAL
generalization: the same meaning across different sentence SHAPES, and compound/comparison
shapes the old keyword template ("X of Y") could never handle. It parses by ROLE
(entity/relation, found by slot not by position) and recognizes composite structures, then
PROPOSES a structured query the crisp solver VERIFIES — propose-and-verify, the membrane.

  single   : one entity + one relation, ANY phrasing  ("force of the rocket", "the
             rocket's force", "how much mass does the sample have")
  compound : >=2 relations, one entity            ("the mass and speed of the rocket")
  compare  : 2 entities + a relation + a compare word ("which has more force, rocket or sample")

Order-independent role extraction = robustness to phrasing; compound/compare = shapes beyond
any single template. The crisp side still answers, so a misparse that doesn't compute abstains.

Honest limit: still a small grammar of structural patterns (single/compound/compare), not a
full compositional grammar; nested clauses / conditionals are the next rung.
"""

import re

from engines.grounding import context_embed as CE
from engines.reasoning.reasoning_engine import ReasoningEngine
from engines.reasoning.means_ends import PolicyMemory, FactSource, PolicySource, MeansEndsSolver, Need, Policy

COMPARE = {"greater", "more", "less", "bigger", "smaller", "heavier", "lighter", "most", "than"}
SYN = {"velocity": "speed", "weight": "mass", "push": "force", "pace": "speed"}


class StructuralParser:
    def __init__(self, entities, relations, ctx_map=None):
        self.entities, self.relations = set(entities), set(relations)
        self.ctx = ctx_map or {}

    def _norm(self, toks):
        return [self.ctx.get(SYN.get(t, t), SYN.get(t, t)) for t in toks]

    def parse(self, text):
        toks = self._norm(re.findall(r"[a-z_]+", text.lower()))
        ts = set(toks)
        ents = [t for t in toks if t in self.entities]
        rels = [t for t in dict.fromkeys(toks) if t in self.relations]   # de-dup, keep order
        if (ts & COMPARE) and len(ents) >= 2 and rels:
            return {"kind": "compare", "entities": ents[:2], "relation": rels[0]}
        if len(rels) >= 2 and ents:
            return {"kind": "compound", "entity": ents[0], "relations": rels}
        if ents and rels:
            return {"kind": "single", "entity": ents[0], "relation": rels[0]}
        return {"kind": "unknown"}


def build_brain():
    fkb = ReasoningEngine()
    for ent, fs in {"rocket": {"mass": "1000", "accel": "12", "speed": "300", "volume": "2"},
                    "sample": {"mass": "2", "accel": "9.8", "speed": "30", "volume": "0.5"}}.items():
        for r, v in fs.items():
            fkb.learn(ent, r, v)
    mem = PolicyMemory()
    for t, ins, e in [("force", ("mass", "accel"), ("*", "mass", "accel")),
                      ("density", ("mass", "volume"), ("/", "mass", "volume"))]:
        mem.add(Policy(t, ins, e))
    return fkb, mem


def answer(q, fkb, mem):
    solve = MeansEndsSolver([FactSource(fkb), PolicySource(mem)]).solve
    if q["kind"] == "single":
        v = solve(Need(q["entity"], q["relation"]))
        return None if v is None else "%s.%s = %.4g" % (q["entity"], q["relation"], v)
    if q["kind"] == "compound":
        parts = [(r, solve(Need(q["entity"], r))) for r in q["relations"]]
        if any(v is None for _, v in parts):
            return None
        return ", ".join("%s=%.4g" % (r, v) for r, v in parts)
    if q["kind"] == "compare":
        a, b = q["entities"]; r = q["relation"]
        va, vb = solve(Need(a, r)), solve(Need(b, r))
        if va is None or vb is None:
            return None
        hi = a if va >= vb else b
        return "%s (%.4g) vs %s (%.4g) -> %s has more %s" % (a, va, b, vb, hi, r)
    return None


def _demo():
    fkb, mem = build_brain()
    ctx = {}                                       # learned context map (lexical layer)
    vecs = CE.build()
    for w in vecs:
        c, s = CE.nearest(w, ["force", "mass", "speed"], vecs)
        if c and s >= 0.6 and w not in ("the", "a", "is", "of", "at", "has", "high", "large"):
            ctx[w] = c
    P = StructuralParser({"rocket", "sample"},
                         {"force", "density", "mass", "speed", "accel", "volume"}, ctx)

    print("=== structural_parser — one meaning across many sentence SHAPES ===\n")
    qs = [
        "what is the force of the rocket",            # template single
        "the rocket's force",                         # reordered, no 'of'
        "how much mass does the sample have",         # different shape entirely
        "what is the mass and speed of the rocket",   # COMPOUND (2 relations)
        "which has more force, the rocket or the sample",   # COMPARE
        "is the rocket's mass greater than the sample",      # COMPARE, reordered
        "what is the vibe of the rocket",             # unparseable -> honest abstain
    ]
    for text in qs:
        q = P.parse(text)
        ans = answer(q, fkb, mem)
        tag = ans if ans is not None else "(abstain — doesn't verify)"
        print("  %-48s [%-8s] %s" % ('"' + text + '"', q["kind"], tag))
    print("\n  Reordered, compound, and comparison shapes all parse by ROLE and are verified")
    print("  by the crisp solver — structural generalization beyond the 'X of Y' template.")


if __name__ == "__main__":
    _demo()
