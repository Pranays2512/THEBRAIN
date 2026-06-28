#!/usr/bin/env python3
"""
semantic_depth.py — extend the concept SET from definitions, not just map to a fixed one.

context_embed maps a paraphrase to an ALREADY-KNOWN concept. Semantic depth is the next
step: when a genuinely NEW concept appears, learn it (from a definition that composes known
concepts), VERIFY it, and add it — so comprehension grows past the fixed vocabulary instead
of abstaining. This is the membrane again: a definition PROPOSES a new concept; the crisp
solver VERIFIES it computes before it is admitted.

  "momentum is mass times speed"  -> new relation momentum = mass * speed -> verify on a
  known entity -> admit. Now "what is the momentum of the rocket" routes and computes,
  and a paraphrase of the new word (via context) routes to it too.

So the brain's vocabulary EXTENDS itself from language, every new concept verified by
computation. Honest limit: definitions must compose KNOWN concepts with a known operator
(times/over/plus/minus) — grounding a concept from raw perceptual data (ground_blend) is the
complementary path; truly primitive new concepts still need that.
"""

import re

from structural_parser import build_brain
from means_ends import FactSource, PolicySource, MeansEndsSolver, Need, Policy

DEF_OP = {"times": "*", "multiplied": "*", "product": "*",
          "divided": "/", "over": "/", "per": "/",
          "plus": "+", "minus": "-"}


class ConceptLearner:
    def __init__(self, fkb, mem, relations):
        self.fkb, self.mem = fkb, mem
        self.relations = set(relations)
        self.solve = MeansEndsSolver([FactSource(fkb), PolicySource(mem)]).solve

    def learn_definition(self, text):
        """Parse 'X is A <op> B', add the policy, VERIFY it computes, then admit X."""
        toks = re.findall(r"[a-z_]+", text.lower())
        if "is" not in toks:
            return None, "no definition"
        i = toks.index("is")
        target = toks[i - 1] if i > 0 else None
        body = toks[i + 1:]
        op = next((DEF_OP[w] for w in body if w in DEF_OP), None)
        args = [w for w in body if w in self.relations]
        if not (target and op and len(args) >= 2):
            return None, "definition doesn't compose known concepts"
        a, b = args[0], args[1]
        self.mem.add(Policy(target, (a, b), (op, a, b)))
        # VERIFY: it must compute on a known entity before we trust it
        test = self.solve(Need("rocket", target))
        if test is None:
            return None, "proposed concept doesn't verify"
        self.relations.add(target)
        return target, "%s = %s %s %s  (verified: rocket.%s=%.4g)" % (target, a, op, b, target, test)

    def query(self, entity, relation):
        if relation not in self.relations:
            return None
        return self.solve(Need(entity, relation))


def _demo():
    fkb, mem = build_brain()
    known = {"force", "density", "mass", "speed", "accel", "volume"}
    L = ConceptLearner(fkb, mem, known)

    print("=== semantic_depth — learn a NEW concept from a definition, then comprehend it ===\n")
    print("  before: 'momentum' is unknown ->", "known" if "momentum" in L.relations else "UNKNOWN")
    print('  query "momentum of the rocket":', L.query("rocket", "momentum"), "(cannot — abstain)\n")

    for defn in ["momentum is mass times speed",
                 "specific_volume is volume over mass"]:
        tgt, msg = L.learn_definition(defn)
        print('  learn "%s"\n    -> %s' % (defn, msg))

    print("\n  after: 'momentum' is", "KNOWN" if "momentum" in L.relations else "unknown")
    print('  query "momentum of the rocket":', "%.4g" % L.query("rocket", "momentum"))
    print('  query "momentum of the sample":', "%.4g" % L.query("sample", "momentum"))
    print('  query "specific_volume of the rocket":', "%.4g" % L.query("rocket", "specific_volume"))
    print("\n  The concept vocabulary GREW from a definition — proposed by language, verified by")
    print("  computation, then comprehended like any built-in. Semantic depth past the fixed set.")


if __name__ == "__main__":
    _demo()
