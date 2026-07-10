#!/usr/bin/env python3
"""
test_brain_session.py — the operational shell: boot, ingest, ask, persist.

Pins that a session learns ingested knowledge into the brain, answers both
knowledge and math through the eyes->brain->mouth pipeline, grows coverage when
fed more, and round-trips its knowledge base.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faculties.brain_session import BrainSession

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nBrainSession — boot, ingest, ask, persist")
    sess = BrainSession()

    # 1. ingest learns into the brain and reports the DELTA
    d1 = sess.ingest_text("A dog is an animal. A cat is a mammal.")
    check("ingest reports newly-learned delta", d1 >= 2)
    d2 = sess.ingest_text("A dog is an animal.")          # all duplicates
    check("re-ingesting duplicates learns nothing new", d2 == 0)

    # 2. answers knowledge through the pipeline
    check("answers an ingested entity", "dog" in sess.ask("what is dog?").lower())
    check("honest on un-ingested entity",
          "don't know" in sess.ask("what is dragon?").lower())

    # 3. math works through the same session (eyes route to engines)
    check("differentiates via session", "3*x^2" in sess.ask("differentiate x^3"))
    check("solves via session", "x = 2" in sess.ask("solve 2*x + 3 = 7 for x"))

    # 4. coverage grows with ingestion
    qs = ["what is dog?", "what is cat?", "what is whale?"]
    a0, t = sess.coverage(qs)
    sess.ingest_text("A whale is a mammal.")
    a1, _ = sess.coverage(qs)
    check("coverage rises after ingesting", a1 > a0)

    # 5. stats reflect what was loaded
    s = sess.stats()
    check("stats report facts + loaded count",
          s["facts"] > 0 and s["loaded_into_brain"] >= s["facts"] - 5)

    # 6. KB round-trips through disk
    with tempfile.TemporaryDirectory() as dd:
        p = os.path.join(dd, "kb.json")
        sess.save(p)
        reborn = BrainSession.load(p)
        check("reloaded session answers from saved KB",
              "dog" in reborn.ask("what is dog?").lower())

    print(f"\nBrain session: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
