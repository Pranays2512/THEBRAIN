#!/usr/bin/env python3
"""
whole_brain.py — one front over the whole system. The pieces, made whole.

A single ask(text) routes a request to the right faculty and returns a verified
answer with provenance:

  COMPUTE   "force of the rocket"        -> means-ends executive over facts+policies
  FACTUAL   "is a dog a mammal" / "what  -> ReasoningEngine over real knowledge
             can a bird do"                 (transitive isa + property inheritance)
  CODE      "write a factorial function" -> synth_engine (verified) or recall from store
  UNKNOWN   anything else                -> honest "I don't know"

Discovered/synthesized knowledge PERSISTS in the BrainStore — ask for the same
function twice and the second time it's recalled, not re-synthesized. Every answer
carries how it was produced and whether it's verified.

    python3 whole_brain.py
"""

import math
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from reasoning_engine import ReasoningEngine
from core_knowledge import CORE_FACTS
from means_ends import PolicyMemory, FactSource, PolicySource, MeansEndsSolver, Need
import synth_engine as SE
from brain_store import BrainStore

CODE_WORDS = {"function", "code", "algorithm", "write", "implement", "program", "def"}
CODE_TASKS = {  # name -> (kind, examples, oracle)
    "factorial": ("int1", [0, 1, 4, 5, 6], lambda n: math.factorial(n)),
    "fibonacci": ("int1", [0, 1, 2, 3, 7, 10], None),     # oracle set below
    "gcd": ("int2", [(12, 8), (48, 36), (7, 5), (100, 80)], math.gcd),
    "triangular": ("int1", [1, 2, 3, 5, 8], lambda n: n * (n + 1) // 2),
}


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
CODE_TASKS["fibonacci"] = ("int1", CODE_TASKS["fibonacci"][1], _fib)


class WholeBrain:
    def __init__(self):
        self.store = BrainStore()
        # FACTUAL: real-world knowledge + inheritance
        self.kre = ReasoningEngine()
        for s, r, o in CORE_FACTS:
            self.kre.learn(s, r, o)
        self.kre.set_transitive("isa")
        for prop in ("has", "can", "lives_in"):
            self.kre.add_rule("isa", prop, prop)
        # COMPUTE: physics facts + policies via the means-ends executive
        self.fkb = ReasoningEngine()
        for ent, fs in {"rocket": {"mass": "1000", "accel": "12", "speed": "300", "volume": "2"},
                        "sample": {"mass": "2", "accel": "9.8", "speed": "30", "volume": "0.5"}}.items():
            for r, v in fs.items():
                self.fkb.learn(ent, r, v)
        self.mem = PolicyMemory()
        for t, ins, e in [("force", ("mass", "accel"), ("*", "mass", "accel")),
                          ("density", ("mass", "volume"), ("/", "mass", "volume")),
                          ("momentum", ("mass", "speed"), ("*", "mass", "speed")),
                          ("energy", ("mass", "speed"), ("*", 0.5, ("*", "mass", ("^", "speed", 2))))]:
            self.mem.add(__import__("means_ends").Policy(t, ins, e))
        self.entities = {"rocket", "sample"}
        self.relations = {"force", "density", "momentum", "energy", "mass", "speed", "accel", "volume"}
        self.concepts = {s for s, _, _ in CORE_FACTS} | {o for _, _, o in CORE_FACTS}

    def ask(self, text):
        toks = re.findall(r"[a-z_]+", text.lower())
        ts = set(toks)
        if CODE_WORDS & ts:
            return self._code(toks)
        # COMPUTE first (before the loose 'is'/'can' factual checks)
        rel = next((t for t in toks if t in self.relations), None)
        ent = next((t for t in toks if t in self.entities), None)
        if rel and ent:
            v = MeansEndsSolver([FactSource(self.fkb), PolicySource(self.mem)]).solve(Need(ent, rel))
            if v is not None:
                return ("compute", f"{ent}.{rel} = {v:.4g}", True)
        # FACTUAL: abilities
        if "can" in ts:
            subj = next((t for t in toks if t in self.concepts and self.kre.ask_all(t, "can")), None)
            if subj:
                return ("factual", f"{subj} can: {sorted(self.kre.ask_all(subj, 'can'))}", True)
        # FACTUAL: is-a, only among KNOWN concepts (so 'meaning of life' -> unknown)
        known = [t for t in toks if t in self.concepts]
        if "is" in ts and len(known) >= 2:
            for x in known:
                for y in known:
                    if x != y and self.kre.reaches(x, "isa", y)[0]:
                        return ("factual", f"Yes — {' -> '.join(self.kre.reaches(x,'isa',y)[1])}", True)
            return ("factual", f"No (no isa path among {known})", True)
        return ("none", "I don't know.", False)

    def _code(self, toks):
        name = next((t for t in toks if t in CODE_TASKS), None)
        if name is None:
            return ("code", "can't synthesize that yet (outside the synth DSLs; needs the LLM tier).", False)
        if self.store.knows_function(name):
            return ("code", f"recalled from memory:\n{self.store.functions[name].strip()}", True)
        kind, raw, oracle = CODE_TASKS[name]
        ex = SE._ex(kind, oracle, raw)
        sp, code = SE.solve(ex, kind)
        if code and SE.stress(code, oracle, kind)[0]:
            code = code.replace("def f(", f"def {name}(")
            self.store.add_function(name, code)
            self.store.save()
            return ("code", f"synthesized + verified, stored:\n{code.strip()}", True)
        return ("code", "couldn't synthesize a verified program.", False)


def _demo():
    import shutil
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "brain_store"), ignore_errors=True)
    b = WholeBrain()
    qs = [
        "what is the force of the rocket?",
        "is a dog a mammal?",
        "what can a bird do?",
        "write a function for factorial",
        "write a factorial function",            # second time -> recalled from memory
        "what is the density of the sample?",
        "write the quicksort algorithm",         # outside synth DSLs -> honest
        "what is the meaning of life?",          # unknown -> honest
    ]
    print("=== whole_brain — one front: compute / factual / code, verified ===\n")
    for q in qs:
        route, ans, ok = b.ask(q)
        mark = "✓" if ok else "·"
        print(f"  > {q}\n    [{route:7s} {mark}] {ans}\n")


if __name__ == "__main__":
    _demo()
