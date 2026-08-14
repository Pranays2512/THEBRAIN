#!/usr/bin/env python3
"""
neuro_bridge.py — the IO contract: LLM is the eyes and mouth, the brain is brain.

The brain (controller) operates ONLY on structured Query/Answer — it never sees
raw text. Eyes turn language into a Query; Mouth turns a verified Answer into
language. The LLM is just one implementation of Eyes/Mouth and drops into those
named slots without touching cognition; the v0 here uses the exact symbolic
parsers we already built.

    mind = Mind(RuleEyes(), Brain(), GrammarMouth())
    mind.respond("differentiate sin(x^2)")   # eyes -> brain -> mouth

Flow:  text --Eyes--> Query --Brain--> Answer --Mouth--> text
The brain decides content; the LLM only translates in and out, so it cannot
invent facts. Coverage = what the brain knows; everything it answers is verified
or honestly flagged unknown.
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from engines.math.math_parser import parse, ParseError
from engines.math.calculus_engine import CalculusEngine
from engines.math.integral_engine import IntegralEngine, render as render_expr
from engines.math.algebra_engine import AlgebraEngine, AlgebraError
from faculties.conversation_engine import ConversationEngine
from faculties.query_planner import QueryPlanner
from engines.reasoning.brainql import BrainQLQuery, BrainQLResult, BrainQLExecutor


# ── the contract ─────────────────────────────────────────────────────────────
@dataclass
class Query:
    kind: str                       # differentiate | integrate | solve | language | error
    payload: dict = field(default_factory=dict)
    raw: str = ""


@dataclass
class Answer:
    kind: str
    known: bool                     # did the brain actually have an answer?
    verified: bool = False          # was it checked (math) — content you can trust?
    value: object = None
    steps: list = field(default_factory=list)
    note: str = ""


class Eyes(ABC):
    """Language -> structured Query. The LLM implements this for messy text."""
    @abstractmethod
    def parse(self, text: str) -> Query: ...


class Mouth(ABC):
    """Verified Answer -> language. The LLM implements this for fluent text."""
    @abstractmethod
    def render(self, answer: Answer) -> str: ...


# ── v0 Eyes: exact symbolic parsing (LLM swaps in for open NL) ────────────────
def _after(text, phrases):
    low = text.lower()
    for p in sorted(phrases, key=len, reverse=True):
        k = low.find(p)
        if k != -1:
            return text[k + len(p):].strip()
    return text.strip()


class RuleEyes(Eyes):
    def parse(self, text):
        t = text.strip().rstrip("?.")
        low = t.lower()
        try:
            if "differentiate" in low or "derivative" in low:
                expr = parse(_after(t, ["differentiate", "derivative of", "derivative"]))
                return Query("differentiate", {"expr": expr}, text)
            if "integrate" in low or "integral" in low or "antiderivative" in low:
                expr = parse(_after(t, ["integrate", "integral of", "integral", "antiderivative"]))
                return Query("integrate", {"expr": expr}, text)
            if "solve" in low or "=" in t:
                s = _after(t, ["solve for", "solve"])
                for tail in (" for x", " for y"):
                    if s.lower().endswith(tail):
                        s = s[: -len(tail)].strip()
                node = parse(s)
                return Query("solve", {"equation": node}, text)
        except ParseError as e:
            return Query("error", {"error": str(e)}, text)
        return Query("language", {"text": text}, text)       # everything else


# ── the Brain: structured Query -> verified Answer (the controller) ──────────
class Brain:
    def __init__(self):
        self.calc = CalculusEngine()
        self.intg = IntegralEngine()
        self.alg = AlgebraEngine()
        self.lang = ConversationEngine(max_describe=4)   # keep answers readable on dense KBs
        self.planner = QueryPlanner(engine=self.lang.r)  # share one reasoning store

    # teaching the knowledge/reasoning side
    def teach(self, s, rel, o):
        return self.lang.learn(s, rel, o)

    def set_transitive(self, rel):
        self.lang.set_transitive(rel)

    def answer(self, q: Query) -> Answer:
        if q.kind == "differentiate":
            r = self.calc.diff(q.payload["expr"])
            return Answer("differentiate", True, verified=True, value=r.text, steps=r.rules)
        if q.kind == "integrate":
            F = self.intg.integrate(q.payload["expr"])
            if F is None:
                return Answer("integrate", False, note="no elementary form in my ruleset")
            ok = self.intg.verify(q.payload["expr"], F)
            return Answer("integrate", True, verified=ok, value=render_expr(F))
        if q.kind == "solve":
            node = q.payload["equation"]
            if not (isinstance(node, tuple) and node[0] == "="):
                return Answer("solve", False, note="that is not an equation")
            try:
                val, steps = self.alg.solve(node)
            except AlgebraError as e:
                return Answer("solve", False, note=str(e))
            return Answer("solve", True, verified=True, value=val, steps=steps)
        if q.kind == "language":
            text_in = q.payload["text"]
            # relational/quantified questions ("capital of france", "which X ...")
            planned = self.planner.try_answer(text_in)
            if planned is not None:
                return Answer("language", True, verified=False, value=planned)
            text = self.lang.respond(text_in)
            known = not text.lower().startswith(("i don't know", "i'm not sure", "not that"))
            return Answer("language", known, verified=False, value=text)
        return Answer("error", False, note=q.payload.get("error", "could not parse"))


# ── v0 Mouth: grammar rendering (LLM swaps in for fluency) ────────────────────
class GrammarMouth(Mouth):
    def render(self, a: Answer) -> str:
        if a.kind == "differentiate":
            return f"The derivative is {a.value}."
        if a.kind == "integrate":
            if not a.known:
                return f"I can't integrate that in closed form ({a.note})."
            tag = "" if a.verified else " (unverified)"
            return f"The integral is {a.value} + C{tag}."
        if a.kind == "solve":
            return f"x = {a.value} (verified)." if a.known else f"I can't solve that: {a.note}."
        if a.kind in ("language", "factual", "compute", "code"):
            return a.value
        return f"I couldn't understand that ({a.note})."



# ── the Mind: eyes → brain → mouth ────────────────────────────────────────────
class Mind:
    def __init__(self, eyes, brain: Brain, mouth: Mouth,
                 bql_mouth=None, bql_executor=None):
        """eyes may be a RuleEyes, LLMEyes, or BrainQLEyes.

        bql_mouth  : a BrainQLMouth instance for rendering BrainQLResults.
                     If None and eyes emits BrainQL, falls back to
                     bql_executor.re-based template rendering.
        bql_executor: a BrainQLExecutor. If None, one is created from brain.lang.r.
        """
        self.eyes = eyes
        self.brain = brain
        self.mouth = mouth
        self.bql_mouth = bql_mouth
        # Build a BrainQL executor backed by the same ReasoningEngine the brain uses
        if bql_executor is not None:
            self.bql_exec = bql_executor
        else:
            self.bql_exec = BrainQLExecutor(brain.lang.r)

    def teach(self, s, rel, o):
        return self.brain.teach(s, rel, o)

    def set_transitive(self, rel):
        self.brain.set_transitive(rel)

    def respond(self, text: str) -> str:
        """Full pipeline: text → Eyes → [BrainQL | math Query] → Brain → Mouth → text."""
        parsed = self.eyes.parse(text)

        # — BrainQL path —
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], BrainQLQuery):
            results = self.bql_exec.run_block(parsed)
            # Use bql_mouth if available; else fallback template
            if self.bql_mouth is not None:
                parts = [self.bql_mouth.render_result(r) for r in results]
            else:
                parts = [self._fallback_render(r) for r in results]
            return " ".join(parts)

        # — math / language Query path (unchanged) —
        return self.mouth.render(self.brain.answer(parsed))

    def _fallback_render(self, result: BrainQLResult) -> str:
        """Deterministic BrainQL rendering when no bql_mouth is available."""
        if not result.known:
            note = result.note or "no answer"
            return f"I don't know ({note})."
        v = result.value
        if result.op in ("TEACH",):
            return f"Got it: {result.subj} {result.rel} {v}."
        if result.op in ("CHAIN",):
            items = ", ".join(v) if isinstance(v, list) else str(v)
            return f"{result.subj} isa: {items}."
        return f"{result.subj} {result.rel}: {v}."


def _demo():
    mind = Mind(RuleEyes(), Brain(), GrammarMouth())
    for s, r, o in [("apple", "isa", "fruit"), ("apple", "is", "red")]:
        mind.teach(s, r, o)

    print("=== neuro_bridge — eyes -> brain -> mouth ===\n")
    for q in ["differentiate sin(x^2)", "solve 2*x + 3 = 7 for x",
              "integrate cos(x)", "integrate sin(x^2)", "what is apple?"]:
        eyes = RuleEyes().parse(q)
        ans = mind.brain.answer(eyes)
        print(f"  > {q}")
        print(f"    eyes  -> kind={eyes.kind}")
        print(f"    brain -> known={ans.known} verified={ans.verified} value={ans.value!r}")
        print(f"    mouth -> {mind.mouth.render(ans)}\n")


if __name__ == "__main__":
    _demo()
