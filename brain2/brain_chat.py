#!/usr/bin/env python3
"""
brain_chat.py — one front door to every faculty.

A thin router: detect the KIND of input and dispatch to the engine that handles
it. Math notation is a formal grammar (route to the exact math engines); anything
else is natural language (route to the controlled conversation engine, which
itself handles facts, how/why chains, and arithmetic word problems). The router
holds no intelligence — it only decides who answers.

    bc = BrainChat()
    bc.learn("dog", "isa", "animal"); bc.set_transitive("isa")
    bc.respond("differentiate sin(x^2)")   -> calculus
    bc.respond("is a dog an animal?")      -> reasoning
    bc.respond("I have 10 apples ...")     -> word math

Honest scope: dispatch over the existing engines' scopes — no new reasoning, just
a unified entry point. Math is exact; natural language stays controlled.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from adapters.math_chat import MathChat
from core.math.math_parser import parse, ParseError, FUNCS
from conversation_engine import ConversationEngine

MATH_INTENT = ("differentiate", "derivative", "integrate", "integral", "antiderivative")


class BrainChat:
    def __init__(self):
        self.math = MathChat()
        self.lang = ConversationEngine()

    # teaching delegates to the language/reasoning side
    def learn(self, s, rel, o):
        return self.lang.learn(s, rel, o)

    def add_rule(self, a, b, c):
        self.lang.add_rule(a, b, c)

    def set_transitive(self, rel):
        self.lang.set_transitive(rel)

    def _is_math(self, text):
        low = text.lower()
        if any(w in low for w in MATH_INTENT):
            return True
        if "solve" in low and "=" in text:
            return True
        # a bare expression: must have an operator or a function, and parse
        core = re.sub(r"\bfor\s+[a-z]\b", "", low).strip().rstrip("?.")
        if re.search(r"[+\-*/^=]", core) or any(f"{f}(" in low for f in FUNCS):
            try:
                parse(core)
                return True
            except ParseError:
                return False
        return False

    def respond(self, text):
        return self.math.respond(text) if self._is_math(text) else self.lang.respond(text)


def _demo():
    bc = BrainChat()
    # teach a little world knowledge + a causal chain
    for s, r, o in [("dog", "isa", "pet"), ("pet", "isa", "animal"),
                    ("apple", "isa", "fruit"), ("apple", "is", "red")]:
        bc.learn(s, r, o)
    for s, o in [("sunlight", "photosynthesis"), ("photosynthesis", "sugar"),
                 ("sugar", "fruit")]:
        bc.learn(s, "leads_to", o)
    bc.set_transitive("isa")
    bc.set_transitive("leads_to")

    print("=== brain_chat — one front door, every faculty ===\n")
    for q in [
        "hello",
        "what is apple?",
        "is a dog an animal?",
        "how does fruit grow?",
        "I have 10 apples and give 3 away, how many do I have left?",
        "differentiate sin(x^2)",
        "integrate cos(x)",
        "solve 2*x + 3 = 7 for x",
    ]:
        print(f"  > {q}")
        print(f"    {bc.respond(q)}\n")


if __name__ == "__main__":
    _demo()
