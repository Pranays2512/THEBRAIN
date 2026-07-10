#!/usr/bin/env python3
"""
test_llm_adapter.py — LLM as eyes & mouth, without weakening the exact core.

Pins that math notation still parses without the model, that the LLM-eyes rescues
phrasings the rules miss, and — critically — that VERIFIED answers are rendered
deterministically (the model never touches the numbers), so attaching an LLM can
only widen coverage, never corrupt a verified result.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_adapter import LLMEyes, LLMMouth, StubClient
from core.reasoning.neuro_bridge import Mind, Brain, Answer

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_ok = True


def check(name, cond):
    global _ok
    _ok = _ok and bool(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def run():
    print("\nLLMAdapter — eyes & mouth (stub model)")

    # eyes: exact math parses WITHOUT consulting the model
    silent = StubClient({})
    eyes = LLMEyes(silent)
    q = eyes.parse("differentiate sin(x^2)")
    check("exact math parsed without the LLM", q.kind == "differentiate")

    # eyes: the model rescues an odd phrasing the rules miss
    smart = StubClient({"slope of x squared": '{"kind":"differentiate","expr":"x^2"}'})
    q2 = LLMEyes(smart).parse("what is the slope of x squared?")
    check("LLM-eyes rescues odd phrasing", q2.kind == "differentiate")
    check("rescued expr parsed to a tuple", q2.payload["expr"] == ("^", "x", 2))

    # eyes: bad/empty model output falls back to language, not a crash
    q3 = LLMEyes(StubClient({"foo": "not json"})).parse("tell me about foo")
    check("garbage model output falls back to language", q3.kind == "language")

    # mouth: a VERIFIED answer is rendered deterministically (model ignored)
    junk = StubClient({"": "HALLUCINATED NONSENSE"})
    mouth = LLMMouth(junk)
    verified = Answer("solve", known=True, verified=True, value=2, steps=["", "x = 2"])
    out = mouth.render(verified)
    check("verified answer rendered deterministically", "x = 2" in out and "NONSENSE" not in out)

    # mouth: an unknown answer stays deterministic + honest
    unknown = Answer("integrate", known=False, note="no elementary form")
    check("unknown stays deterministic", "can't integrate" in mouth.render(unknown).lower())

    # mouth: a language answer DOES get the model's phrasing
    polish = StubClient({"fruit": "An apple is a lovely red fruit."})
    lang = Answer("language", known=True, verified=False, value="apple is a fruit, red")
    check("language answer polished by the LLM",
          LLMMouth(polish).render(lang) == "An apple is a lovely red fruit.")

    # end-to-end through the Mind
    brain = Brain()
    brain.teach("apple", "isa", "fruit")
    mind = Mind(LLMEyes(smart), brain, LLMMouth(StubClient({})))
    check("end-to-end: rescued math answered",
          "2*x" in mind.respond("what is the slope of x squared?"))
    check("end-to-end: verified math stays exact",
          "x = 2" in mind.respond("solve 2*x + 3 = 7 for x"))

    print(f"\nLLM adapter: {'READY' if _ok else 'NEEDS FIX'}")
    return _ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
