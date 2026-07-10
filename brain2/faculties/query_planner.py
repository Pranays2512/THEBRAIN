#!/usr/bin/env python3
"""
query_planner.py — let the brain DECOMPOSE a question, then reason.

Instead of scattered "if 'how' in text" intents, this turns a (possibly
multi-part) question into a STACK of small structured queries, answers each over
the reasoning engine, and composes one reply. The pipeline mirrors how you'd
break the question down by hand:

  intent      : question / statement / ... (from the AppraisalEngine)
  quantifier  : how many -> count | which -> which | ways -> list | else single
  subject     : the entity in focus (vitamin_c)
  relation    : the predicate asked about (provides / helps), matched or inferred
  -> push each sub-question on a stack, solve, then COMBINE the answers.

Honest scope: the slot extraction is still controlled grammar (the comprehension
wall is unmoved) — but the structure, decomposition, and composition are real and
replace the ad-hoc intent routing with one mechanism.

    qp = QueryPlanner(engine)
    qp.answer("in how many ways can we get vitamin C? "
              "which fruit provides vitamin C?")
"""

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.reasoning.reasoning_engine import ReasoningEngine
from faculties.appraisal_engine import AppraisalEngine
from faculties.conversation_engine import oxford

# natural verb -> canonical relation (controlled alias map)
VERB_RELATION = {
    "get": "provides", "gets": "provides", "provide": "provides",
    "provides": "provides", "give": "provides", "gives": "provides",
    "source": "provides", "help": "helps", "helps": "helps", "aid": "helps",
}
STOP = {"in", "a", "an", "the", "of", "to", "we", "i", "you", "can", "could",
        "do", "does", "is", "are", "how", "many", "much", "which", "what",
        "who", "way", "ways", "from", "and", "for", "our", "us", "by", "that"}


@dataclass
class Query:
    intent: str          # question / statement / greeting ...
    quantifier: str      # count | which | list | single
    subject: str         # entity in focus
    relation: str        # predicate (canonical relation)
    category: str        # for "which X ...", the class X (e.g. fruit)
    text: str            # the sub-question this came from


class QueryPlanner:
    def __init__(self, engine=None):
        self.r = engine or ReasoningEngine()
        self.appraiser = AppraisalEngine()

    def learn(self, s, rel, o):
        return self.r.learn(s, rel, o)

    def set_transitive(self, rel):
        self.r.set_transitive(rel)

    # ── known vocabulary (from the facts) ────────────────────────────────────
    def _entities(self):
        return {s for s, _, _ in self.r.kb.facts} | {o for _, _, o in self.r.kb.facts}

    def _categories(self):
        return {o for _, r, o in self.r.kb.facts if r == "isa"}

    def _relations(self):
        return {r for _, r, _ in self.r.kb.facts}

    # ── parse: text -> stack of structured sub-queries ───────────────────────
    def parse(self, text):
        parts = [p.strip() for p in re.split(r"[?.]", text) if p.strip()]
        return [self._plan(p) for p in parts if p]

    def _tokens(self, text):
        # known entities can be multi-word ("vitamin c" -> vitamin_c); fold bigrams
        words = re.findall(r"[a-z']+", text.lower())
        ents = self._entities()
        out, i = [], 0
        while i < len(words):
            if i + 1 < len(words) and f"{words[i]}_{words[i+1]}" in ents:
                out.append(f"{words[i]}_{words[i+1]}")
                i += 2
            else:
                out.append(words[i])
                i += 1
        return out

    def _plan(self, part):
        toks = self._tokens(part)
        tset = set(toks)
        intent = self.appraiser.appraise(part).type
        ents, cats, rels = self._entities(), self._categories(), self._relations()

        # inverse pattern: "<rel> of <obj>"  (the capital OF france -> who?)
        if "of" in toks:
            i = toks.index("of")
            relw = toks[i - 1] if i > 0 else None
            obj = next((t for t in toks[i + 1:] if t in ents), None)
            cand = relw if relw in rels else (f"{relw}_of" if relw and f"{relw}_of" in rels else None)
            if cand and obj:
                return Query(intent, "inverse", obj, cand, None, part)

        if "how" in tset and "many" in tset:
            quant = "count"
        elif {"way", "ways"} & tset:
            quant = "list"
        elif {"which", "what"} & tset:
            quant = "which"
        else:
            quant = "single"

        category = next((t for t in toks if t in cats and
                         (("which" in tset) or ("what" in tset))), None)
        # subject: a known entity that isn't the category (prefer a real entity)
        subject = next((t for t in toks if t in ents and t != category), None)
        # relation: a verb in the text mapped to a relation, else inferred
        relation = next((VERB_RELATION[t] for t in toks
                         if t in VERB_RELATION and VERB_RELATION[t] in rels), None)
        if relation is None and subject is not None:
            into = self.r.relations_into(subject)
            relation = into[0] if len(into) == 1 else (into[0] if into else None)
        return Query(intent, quant, subject, relation, category, part)

    # ── execute one sub-query against the engine ─────────────────────────────
    def execute(self, q):
        if q.subject is None or q.relation is None:
            return ("none", [])
        if q.quantifier == "inverse":
            return ("inverse", self.r.subjects_with(q.relation, q.subject))
        if q.quantifier in ("count", "which"):
            srcs = self.r.subjects_with(q.relation, q.subject)
            if q.category:
                srcs = [s for s in srcs if self.r.reaches(s, "isa", q.category)[0]
                        or (s, "isa", q.category) in self.r.kb.facts]
            return (q.quantifier, srcs)
        if q.quantifier == "list":
            return ("list", self.r.process_branches(q.subject, q.relation))
        ans, _ = self.r.ask(q.subject, q.relation)
        return ("single", [ans] if ans else [])

    # ── verbalize + compose the stack ────────────────────────────────────────
    def _say(self, q, kind, items):
        sub = q.subject.replace("_", " ") if q.subject else "it"
        if kind in ("count", "which") and items:
            names = oxford([s.replace("_", " ") for s in items])
            if kind == "count":
                return f"We can get {sub} in {len(items)} ways: from {names}."
            cat = (q.category or "thing").replace("_", " ")
            return f"The {cat}s that provide {sub} are {names}."
        if kind == "list" and items:
            steps = [" -> ".join(p.replace("_", " ") for p in path) for path in items]
            return f"{sub.capitalize()} helps in {len(items)} ways: " + "; ".join(steps) + "."
        if kind == "single" and items and items[0]:
            return f"{sub.capitalize()}: {items[0].replace('_', ' ')}."
        return f"I can't answer that about {sub}."

    # general verbalizer + success-or-None resolver (used by the Brain) ───────
    def _verbalize(self, q, kind, items):
        sub = q.subject.replace("_", " ")
        names = oxford([s.replace("_", " ") for s in items])
        rel = q.relation.replace("_", " ")
        if kind == "inverse":
            noun = q.relation[:-3] if q.relation.endswith("_of") else q.relation
            return f"The {noun.replace('_', ' ')} of {sub} is {names}."
        if kind == "count":
            return f"There are {len(items)}: {names}."
        if kind == "which":
            return f"{names[0].upper() + names[1:]} {rel} {sub}."
        if kind == "list":
            steps = [" -> ".join(p.replace('_', ' ') for p in path) for path in items]
            return f"{sub.capitalize()}: " + "; ".join(steps) + "."
        return f"{sub.capitalize()} {rel} {names}."        # single

    def try_answer(self, text):
        """Answer relational/quantified questions, or None if it can't — so the
        caller can fall back to another faculty."""
        out, prev = [], None
        for q in self.parse(text):
            kind, items = self.execute(q)
            if kind == "none" or not items:
                continue
            s = self._verbalize(q, kind, items)
            out.append(("Specifically, " + s[0].lower() + s[1:])
                       if prev and q.subject == prev else s)
            prev = q.subject
        return " ".join(out) if out else None

    def answer(self, text):
        queries = self.parse(text)
        if not queries:
            return "I'm not sure what you're asking."
        sentences, prev_subject = [], None
        for q in queries:
            kind, items = self.execute(q)
            s = self._say(q, kind, items)
            if prev_subject and q.subject == prev_subject:
                s = "Specifically, " + s[0].lower() + s[1:]
            sentences.append(s)
            prev_subject = q.subject
        return " ".join(sentences)


def _demo():
    qp = QueryPlanner()
    for s, o in [("orange", "vitamin_c"), ("lemon", "vitamin_c"),
                 ("strawberry", "vitamin_c")]:
        qp.learn(s, "provides", o)
    for s in ("orange", "lemon", "strawberry"):
        qp.learn(s, "isa", "fruit")
    for s, o in [("vitamin_c", "immune_system"), ("immune_system", "fighting_infection"),
                 ("vitamin_c", "energy")]:
        qp.learn(s, "helps", o)
    qp.set_transitive("isa")
    qp.set_transitive("helps")

    print("=== query_planner — decompose, reason, compose ===\n")
    for q in [
        "in how many ways can we get vitamin C?",
        "which fruit provides vitamin C?",
        "in how many ways can we get vitamin C? which fruit provides vitamin C?",
        "in which ways does vitamin C help?",
    ]:
        print(f"  > {q}")
        print(f"    {qp.answer(q)}\n")


if __name__ == "__main__":
    _demo()
