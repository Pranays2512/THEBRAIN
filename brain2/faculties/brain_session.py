#!/usr/bin/env python3
"""
brain_session.py — the operational shell: a bootable, ingestable, queryable brain.

Ties the product spine into one running system. A persistent KnowledgeBase loads
into the Brain at boot; the Mind answers via eyes -> brain -> mouth; you can ingest
more knowledge live and watch coverage grow mid-session. The REPL is thin glue —
the logic lives in BrainSession (so it is testable).

    sess = BrainSession()
    sess.boot_conceptnet()
    sess.ask("what is a dog?")
    sess.ingest_text("A whale is a mammal. It lives in the ocean.")
    sess.ask("what is a whale?")

CLI:
    python3 brain_session.py [kb.json]        # boot from a saved KB (or ConceptNet)
    :ingest <file>   :stats   :coverage   :save <file>   :quit
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.knowledge.knowledge_base import KnowledgeBase
from core.reasoning.neuro_bridge import Mind, Brain, RuleEyes, GrammarMouth


class BrainSession:
    def __init__(self, kb=None):
        self.kb = kb or KnowledgeBase()
        self.brain = Brain()
        self.mind = Mind(RuleEyes(), self.brain, GrammarMouth())
        self._loaded = 0
        self._sync()

    def _sync(self):
        """Learn any KB facts not yet in the brain (idempotent). Returns the
        number of NEWLY learned facts this call."""
        delta = self.kb.into(self.brain)
        self._loaded += delta
        return delta

    # ── bootstrapping knowledge ──────────────────────────────────────────────
    def boot_conceptnet(self, min_weight=2.0):
        self.kb.ingest_conceptnet(min_weight=min_weight)
        return self._sync()

    def ingest_text(self, text, source="text"):
        self.kb.ingest_text(text, source=source)
        return self._sync()

    def ingest_file(self, path):
        with open(path, encoding="utf-8") as f:
            return self.ingest_text(f.read(), source=os.path.basename(path))

    def ingest_dir(self, path):
        total = 0
        for name in sorted(os.listdir(path)):
            if name.endswith(".txt"):
                total += self.ingest_file(os.path.join(path, name))
        return total

    # ── use ──────────────────────────────────────────────────────────────────
    def ask(self, text):
        return self.mind.respond(text)

    def coverage(self, questions):
        eyes = RuleEyes()
        answered = sum(int(self.brain.answer(eyes.parse(q)).known) for q in questions)
        return answered, len(questions)

    def stats(self):
        s = self.kb.stats()
        s["loaded_into_brain"] = self._loaded
        return s

    def save(self, path):
        self.kb.save(path)

    @classmethod
    def load(cls, path):
        return cls(kb=KnowledgeBase.load(path))


def _repl(sess):                                   # pragma: no cover (interactive)
    print("brain ready. ask a question, or :stats / :ingest <file> / :coverage / :save <f> / :quit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in (":quit", ":q"):
            break
        if line == ":stats":
            print(sess.stats())
        elif line.startswith(":ingest "):
            print(f"  +{sess.ingest_file(line.split(maxsplit=1)[1])} facts")
        elif line.startswith(":save "):
            sess.save(line.split(maxsplit=1)[1]); print("  saved")
        elif line == ":coverage":
            a, t = sess.coverage(["what is a dog?", "what is a car?", "what is a whale?"])
            print(f"  coverage {a}/{t}")
        else:
            print("  " + sess.ask(line))


def main():
    sess = BrainSession.load(sys.argv[1]) if len(sys.argv) > 1 else BrainSession()
    if not sess.kb.facts:
        print("no KB given — booting from ConceptNet...")
        sess.boot_conceptnet()
    print(f"loaded {sess._loaded} facts.")
    _repl(sess)


def _demo():
    print("=== brain_session — bootable, ingestable, queryable ===\n")
    sess = BrainSession()
    print(f"boot ConceptNet: {sess.boot_conceptnet()} facts loaded\n")
    for q in ["what is a dog?", "differentiate sin(x^2)", "what is a whale?"]:
        print(f"  > {q}\n    {sess.ask(q)}")
    print(f"\n  ingest: +{sess.ingest_text('A whale is a mammal. It lives in the ocean.')} facts")
    print(f"  > what is a whale?\n    {sess.ask('what is a whale?')}")


if __name__ == "__main__":
    # interactive terminal -> REPL; otherwise (gate / pipe) -> demo
    main() if sys.stdin.isatty() and "--demo" not in sys.argv else _demo()
