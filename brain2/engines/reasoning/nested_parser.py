#!/usr/bin/env python3
"""
nested_parser.py — compositional comprehension: a query INSIDE a query, and conditionals.

structural_parser handled flat shapes (single/compound/compare). This adds the next rung:
NESTED and CONDITIONAL structure, where one query's answer feeds another.

  superlative entity : "the force of the HEAVIEST object"  -> resolve heaviest = argmax(mass)
                        over entities, THEN force of that entity. The entity slot is itself a
                        sub-query.
  conditional        : "if the rocket's mass is greater than 500 then what is its speed" ->
                        evaluate the condition; only if it holds, answer the consequent (with
                        pronoun 'its' bound to the condition's entity).

Every sub-result is computed by the crisp solver, so the whole composite is verified end to
end; a part that doesn't verify makes the whole abstain. Propose-and-verify, recursively.

Honest limit: two compositional forms (superlative, single-condition if/then) — not a full
recursive grammar. Deeper nesting and boolean conditions (and/or) are the next rung.
"""

import re

from engines.reasoning.structural_parser import build_brain
from engines.reasoning.means_ends import FactSource, PolicySource, MeansEndsSolver, Need

SUPER = {"heaviest": ("mass", max), "lightest": ("mass", min),
         "fastest": ("speed", max), "slowest": ("speed", min),
         "densest": ("density", max), "biggest": ("volume", max)}
SYN = {"velocity": "speed", "weight": "mass", "push": "force"}
ENTS = {"rocket", "sample"}
RELS = {"force", "density", "mass", "speed", "accel", "volume"}


class NestedParser:
    def __init__(self, fkb, mem):
        self.solve = MeansEndsSolver([FactSource(fkb), PolicySource(mem)]).solve

    def _toks(self, text):
        return [SYN.get(t, t) for t in re.findall(r"[a-z_]+", text.lower())]

    def _resolve_entity(self, toks):
        """A direct entity name, or a superlative sub-query (argmax/argmin over a relation)."""
        for w in toks:
            if w in SUPER:
                rel, agg = SUPER[w]
                vals = {e: self.solve(Need(e, rel)) for e in ENTS}
                vals = {e: v for e, v in vals.items() if v is not None}
                if vals:
                    return agg(vals, key=vals.get), "%s = argmax %s" % (w, rel)
        for w in toks:
            if w in ENTS:
                return w, None
        return None, None

    def answer(self, text):
        toks = self._toks(text)
        rels = [t for t in dict.fromkeys(toks) if t in RELS]

        if "if" in toks:                            # ---- conditional ----
            i = toks.index("if")
            cut = toks.index("then") if "then" in toks else len(toks)
            cond, cons = toks[i + 1:cut], toks[cut + 1:]
            c_ent, _ = self._resolve_entity(cond)
            c_rel = next((t for t in cond if t in RELS), None)
            num = next((float(t) for t in re.findall(r"\d+", text)), None)
            gt = any(w in cond for w in ("greater", "more", "above", "bigger"))
            if not (c_ent and c_rel and num is not None):
                return "(abstain — condition not parseable)"
            lhs = self.solve(Need(c_ent, c_rel))
            if lhs is None:
                return "(abstain — condition doesn't verify)"
            holds = lhs > num if gt else lhs < num
            if not holds:
                return "condition FALSE (%s.%s=%.4g) -> no answer" % (c_ent, c_rel, lhs)
            cons_ent, _ = self._resolve_entity(cons)
            cons_ent = cons_ent or c_ent             # pronoun 'its'/'it' -> condition entity
            cons_rel = next((t for t in cons if t in RELS), None)
            v = self.solve(Need(cons_ent, cons_rel)) if cons_rel else None
            if v is None:
                return "condition true; consequent doesn't verify -> abstain"
            return "condition TRUE (%s.%s=%.4g) -> %s.%s = %.4g" % (
                c_ent, c_rel, lhs, cons_ent, cons_rel, v)

        ent, how = self._resolve_entity(toks)        # ---- nested superlative entity ----
        if ent and rels:
            v = self.solve(Need(ent, rels[0]))
            if v is None:
                return "(abstain — doesn't verify)"
            sub = "  [%s -> %s]" % (how, ent) if how else ""
            return "%s.%s = %.4g%s" % (ent, rels[0], v, sub)
        return "(abstain — unparseable)"


def _demo():
    fkb, mem = build_brain()
    P = NestedParser(fkb, mem)
    print("=== nested_parser — a query inside a query, and conditionals (verified) ===\n")
    qs = [
        "what is the force of the heaviest object",        # nested: heaviest=argmax mass
        "what is the speed of the fastest one",            # nested: fastest=argmax speed
        "if the rocket's mass is greater than 500 then what is its speed",   # cond TRUE
        "if the sample's mass is greater than 500 then what is its speed",   # cond FALSE
        "if the rocket speed is greater than 100 then what is its force",    # cond TRUE -> force
        "what is the force of the rocket",                 # plain (still works)
    ]
    for text in qs:
        print("  %-62s %s" % ('"' + text + '"', P.answer(text)))
    print("\n  Superlative entities resolve via a sub-query; conditionals gate the answer on a")
    print("  verified condition. Every piece checked by the crisp solver — compositional + sound.")


if __name__ == "__main__":
    _demo()
