#!/usr/bin/env python3
"""
deeper_grammar.py — boolean / multi-clause conditions (the next structural rung).

nested_parser handled a single if/then condition. This adds compound boolean conditions:
"if A and B then C", "if A or B then C", with each atom a verified comparison and the
entity carried across clauses (pronoun-style). Each atom is solved by the crisp engine, so
the whole boolean is verified end to end; an unparseable/unverifiable atom -> abstain.

  atom      : "<entity?> <relation> (greater|less) than <number>"
  condition : atom (and|or atom)*   evaluated left-to-right within one operator
  rule      : "if <condition> then <query>"

Honest limit: one boolean operator per condition (all-and or all-or), comparisons against
numeric literals. Mixed and/or precedence and clause-vs-clause comparisons are the next rung.
"""

import re

from engines.reasoning.structural_parser import build_brain
from engines.reasoning.means_ends import FactSource, PolicySource, MeansEndsSolver, Need

SYN = {"velocity": "speed", "weight": "mass", "push": "force"}
ENTS = {"rocket", "sample"}
RELS = {"force", "density", "mass", "speed", "accel", "volume"}
GT = ("greater", "more", "above", "bigger")


class DeeperParser:
    def __init__(self, fkb, mem):
        self.solve = MeansEndsSolver([FactSource(fkb), PolicySource(mem)]).solve

    def _toks(self, text):
        return [SYN.get(t, t) for t in re.findall(r"[a-z_]+", text.lower())]

    def _eval_atom(self, atom_toks, nums, default_ent):
        ent = next((t for t in atom_toks if t in ENTS), default_ent)
        rel = next((t for t in atom_toks if t in RELS), None)
        num = nums.pop(0) if nums else None
        if not (ent and rel and num is not None):
            return None, ent
        lhs = self.solve(Need(ent, rel))
        if lhs is None:
            return None, ent
        gt = any(w in atom_toks for w in GT)
        return (lhs > num if gt else lhs < num), ent

    def _eval_condition(self, cond_toks, text):
        # numbers appear in the condition portion, in order
        nums = [float(n) for n in re.findall(r"\d+", text)]
        op = "and" if "and" in cond_toks else ("or" if "or" in cond_toks else None)
        # split atoms on the operator
        parts, cur = [], []
        for t in cond_toks:
            if t in ("and", "or"):
                parts.append(cur); cur = []
            else:
                cur.append(t)
        parts.append(cur)
        results, ent = [], None
        for p in parts:
            r, ent = self._eval_atom(p, nums, ent)
            if r is None:
                return None, ent
            results.append(r)
        if op == "or":
            return any(results), ent
        return all(results), ent          # default / "and"

    def answer(self, text):
        toks = self._toks(text)
        if "if" not in toks:
            return "(not a conditional)"
        i = toks.index("if")
        cut = toks.index("then") if "then" in toks else len(toks)
        cond, cons = toks[i + 1:cut], toks[cut + 1:]
        holds, c_ent = self._eval_condition(cond, text)
        if holds is None:
            return "(abstain — condition doesn't parse/verify)"
        if not holds:
            return "condition FALSE -> no answer"
        cons_ent = next((t for t in cons if t in ENTS), None) or c_ent   # 'its' -> condition ent
        cons_rel = next((t for t in cons if t in RELS), None)
        v = self.solve(Need(cons_ent, cons_rel)) if cons_rel else None
        if v is None:
            return "condition TRUE; consequent doesn't verify -> abstain"
        return "condition TRUE -> %s.%s = %.4g" % (cons_ent, cons_rel, v)


def _demo():
    fkb, mem = build_brain()
    P = DeeperParser(fkb, mem)
    print("=== deeper_grammar — boolean/multi-clause conditions (verified) ===\n")
    qs = [
        "if the rocket mass is greater than 500 and its speed is greater than 100 then what is its force",  # T and T
        "if the rocket mass is greater than 5000 and its speed is greater than 100 then what is its force", # F and T
        "if the rocket mass is greater than 5000 or its speed is greater than 100 then what is its force",  # F or T
        "if the sample mass is greater than 500 or its speed is greater than 100 then what is its density", # F or F
    ]
    for text in qs:
        print('  "%s"\n     -> %s\n' % (text, P.answer(text)))
    print("  Multi-clause boolean conditions, each atom crisp-verified; and/or combine them.")


if __name__ == "__main__":
    _demo()
