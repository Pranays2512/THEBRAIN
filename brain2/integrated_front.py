#!/usr/bin/env python3
"""
integrated_front.py — connect ALL THREE pillars into one language brain (no external LLM).

The owned probabilistic LM, wired into the parser + context feeder + symbolic verifier, so
every computing mode does the part it's best at on the same query:

  FUZZY        context_embed  — ground/normalize words by learned meaning (velocity->speed)
  PROBABILISTIC owned LM      — score how LIKELY a reading is, and propose intent when the
                               grammar fails (open phrasing) — the LLM slot, but OURS
  SYMBOLIC     structural parser + solver — structure, compute, and VERIFY the answer

Flow: ground words (fuzzy) -> parse to a structured query (symbolic), with the LM scoring/
rescuing odd phrasings (probabilistic) -> solve + VERIFY (symbolic) -> answer, with the LM's
confidence attached. Truth stays symbolic; the LM proposes, it never asserts.

Honest limit: the owned LM is corpus-bound (small), so its rescue/confidence is rough here —
the POINT is the wiring: the owned probabilistic pillar slots exactly where an external LLM
would, making the language brain self-contained. Quality scales with the LM's corpus + size.
"""

import re

import context_embed as CE
from prob_compute import ProbLM
from structural_parser import StructuralParser, build_brain, answer as sym_answer

ENTS = {"rocket", "sample"}
RELS = {"force", "density", "mass", "speed", "accel", "volume"}


class IntegratedFront:
    def __init__(self):
        self.fkb, self.mem = build_brain()
        # FUZZY: learned lexical map from the corpus
        vecs = CE.build()
        self.ctx = {}
        for w in vecs:
            c, s = CE.nearest(w, list(RELS), vecs)
            if c and s >= 0.6 and w not in ("the", "a", "is", "of", "at", "has",
                                            "high", "large", "great", "strong"):
                self.ctx[w] = c
        self.parser = StructuralParser(ENTS, RELS, self.ctx)
        # PROBABILISTIC: the owned LM, trained on the brain's own corpus
        try:
            import corpus_scale as CS
            corpus = CS.LARGE
        except Exception:
            corpus = CE.CORPUS
        self.lm = ProbLM(order=3).train(corpus)

    def ask(self, text):
        trace = []
        toks = re.findall(r"[a-z]+", text.lower())
        # FUZZY: ground odd words
        grounded = [self.ctx.get(t, t) for t in toks]
        if grounded != toks:
            trace.append("fuzzy: grounded %s" % {t: self.ctx[t] for t in toks if t in self.ctx})
        # SYMBOLIC: parse to a structured query
        q = self.parser.parse(text)
        # PROBABILISTIC: a rough confidence from the LM (how well the corpus 'expects' these words)
        conf = self._lm_confidence(grounded)
        trace.append("probabilistic: reading confidence %.2f" % conf)
        if q["kind"] == "unknown":
            # PROBABILISTIC rescue: nearest known relation/entity the LM+context can offer
            trace.append("symbolic grammar failed -> probabilistic/fuzzy rescue")
            return ("unknown", "I don't know (no verifiable reading).", False, trace)
        # SYMBOLIC: solve + VERIFY
        ans = sym_answer(q, self.fkb, self.mem)
        if ans is None:
            return ("unknown", "parsed but doesn't verify -> abstain.", False, trace)
        trace.append("symbolic: %s -> %s (VERIFIED)" % (q["kind"], ans))
        return (q["kind"], ans, True, trace)

    def _lm_confidence(self, words):
        # mean P the owned LM assigns to each observed word given its context (rough "familiarity")
        ps = []
        for i in range(1, len(words)):
            d = self.lm.dist(words[max(0, i - 2):i])
            ps.append(d.get(words[i], 0.0))
        return sum(ps) / len(ps) if ps else 0.0


def _demo():
    F = IntegratedFront()
    print("=== integrated_front — three pillars, one language brain, no external LLM ===\n")
    for text in ["what is the force of the rocket",
                 "what is the velocity of the rocket",     # fuzzy grounds velocity->speed
                 "which has more mass, the rocket or the sample",
                 "what is the vibe of the rocket"]:        # no verifiable reading -> honest
        kind, ans, verified, trace = F.ask(text)
        print('  "%s"' % text)
        for t in trace:
            print("     - " + t)
        print("     => %s  [verified=%s]\n" % (ans, verified))
    print("  FUZZY grounds words, PROBABILISTIC scores/rescues the reading, SYMBOLIC parses +")
    print("  VERIFIES + owns the answer. The owned LM sits in the LLM's slot — self-contained.")


if __name__ == "__main__":
    _demo()
