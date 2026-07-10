#!/usr/bin/env python3
"""
nl_front.py — the runtime language front: lexical matcher + DISTILLED student.

Closes the distillation loop. A question's relation is resolved by:
  1. the lexical matcher (nl_query) when it has a STRONG hit (exact name / prefix /
     close synonym) — high precision, zero training;
  2. otherwise the distilled student (student_trainer, trained on the cloud
     teacher's varied phrasings) — generalizes to wording the matcher misses.
Entity is lexical. The resolved Need goes to the verified executive (facts +
policies + proposer). The big teacher never runs here — only its distilled student.

So the system speaks both the phrasings you hand-listed AND the ones the teacher
taught the student, with a verified number out the end and no LLM at inference.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.reasoning.reasoning_engine import ReasoningEngine
from core.reasoning.means_ends import Policy
from policy_pack import PACK
from core.synthesis.policy_proposer import MultiPolicyMemory, Solver
from nl_query import NLQueryParser, load_glove, _rel_words, STOP
from training.student_trainer import Student, load_dataset, DATA

STRONG = 0.9      # lexical confidence at/above which we trust the matcher over the student
CONF_FLOOR = 0.85  # student confidence floor: real keyword hits score ~1.0 (the
                   # teacher's phrasings cover the word); spurious matches sit ~0.75,
                   # so this cleanly separates "answer" from "escalate to the LLM".


class Front:
    """Escalation ladder: lexical -> distilled student -> local LLM. Each rung is
    accepted only if it yields a Need the executive can actually SOLVE; the LLM is
    invoked only when the cheaper rungs fail (so it costs nothing on easy queries)."""
    def __init__(self, kb, mem, lexical: NLQueryParser, student: Student, solvable,
                 llm_parser=None, cpp_engine=None, brain=None):
        self.kb, self.mem, self.lex, self.student = kb, mem, lexical, student
        self.entities = set(lexical.entities) | set(student.entities)
        self.solvable = solvable
        self.llm = llm_parser
        self.cpp = cpp_engine          # if set, solve via the C++ brain2.PolicyEngine
        self.brain = brain             # if set, solve via the C++ Brain (the mind itself)

    def _entity(self, q):
        toks = [t for t in re.findall(r"[a-z_]+", q.lower()) if t not in STOP]
        return next((t for t in toks if t in self.entities), None)

    def _solve(self, ent, rel):
        if not (ent and rel and rel in self.solvable):
            return None
        if self.brain is not None:                # the C++ Brain reasons natively
            return self.brain.policy_solve(ent, rel)
        if self.cpp is not None:                  # C++ brain2.PolicyEngine
            return self.cpp.solve(ent, rel)
        return Solver(self.kb, self.mem, use_proposer=True).solve(ent, rel)

    def resolve(self, q):
        """Confidence ladder: a tier is accepted only when it's CONFIDENT, not
        merely solvable (everything is solvable for a known entity). When the
        cheap tiers abstain, escalate to the LLM; if nothing confident resolves,
        say so honestly."""
        ent = self._entity(q)
        content = [t for t in re.findall(r"[a-z_]+", q.lower())
                   if t not in STOP and t != ent]
        lr, sc = self.lex.match_relation(content)
        srel, conf = self.student.confident_rel(q)

        # 1. strong lexical that AGREES with the student
        if lr is not None and sc >= STRONG and lr == srel:
            return ent, lr, "lexical"
        # 2. the student, when its wordmax confidence clears the floor (it
        #    overrides lexical false-friends like 'work out' -> force)
        if conf >= CONF_FLOOR:
            return ent, srel, "student"
        # 3. escalate: the LLM (invoked only here — the expensive tail)
        if self.llm is not None:
            le, lrel = self.llm.parse(q)
            if (le or ent) and lrel in self.solvable:
                return (le or ent), lrel, "llm"
        return ent, None, "none"

    def answer(self, q):
        ent, rel, src = self.resolve(q)
        val = self._solve(ent, rel)
        if val is None:
            return "I don't know.", "none"
        return f"{ent}.{rel} = {val:.4g}", src


def _build(llm=None, use_cpp=False, use_brain=False):
    kb = ReasoningEngine()
    facts = {
        "rocket": {"mass": 1000, "accel": 12, "speed": 300, "volume": 2.0,
                   "molar_mass": 2, "area": 1.5, "distance": 4},
        "sample": {"mass": 2, "accel": 9.8, "speed": 30, "volume": 0.5,
                   "molar_mass": 18, "area": 0.25, "distance": 5},
    }
    for ent, fs in facts.items():
        for r, v in fs.items():
            kb.learn(ent, r, str(v))
    policies = list(PACK) + [
        # align distillation vocab with the executive: teacher used 'energy';
        # executive only had ke/pe, so add energy = kinetic energy (½mv²).
        ("energy", ("mass", "speed"), ("*", 0.5, ("*", "mass", ("^", "speed", 2))))]
    mem = MultiPolicyMemory()
    for t, ins, expr in policies:
        mem.add(Policy(t, ins, expr))

    rows = load_dataset(DATA)
    entities = set(facts) | {r["label"].get("entity", "") for r in rows} | \
               {r["label"].get("x", "") for r in rows}
    entities.discard("")
    relations = sorted({t for t, _, _ in PACK} | {r for fs in facts.values() for r in fs})

    needed = set(relations) | entities
    for r in relations:
        needed.update(_rel_words(r))
    for row in rows:
        needed.update(re.findall(r"[a-z_]+", row["question"].lower()))
    glove = load_glove(needed=needed)

    student = Student.train(rows, glove, k=5)
    lexical = NLQueryParser(entities, relations, glove)
    solvable = set(mem.by_target) | {r for fs in facts.values() for r in fs}

    cpp_engine = brain = None
    if use_brain:                                 # the C++ Brain IS the reasoner
        import brain2
        brain = brain2.Brain(som_rows=8, som_cols=8, n_dims=8)
        for ent, fs in facts.items():
            for r, v in fs.items():
                brain.teach_fact(ent, r, float(v))
        for t, ins, expr in policies:
            brain.policy_add(t, list(ins), expr)
    elif use_cpp:                                 # solve via the native C++ reasoner
        import brain2
        cmem = brain2.PolicyMemory()
        for t, ins, expr in policies:
            cmem.add(t, list(ins), expr)
        def fact(e, r):
            ans, _ = kb.ask(e, r)
            return float(ans) if ans is not None else None
        cpp_engine = brain2.PolicyEngine(cmem, fact, True)

    llm_parser = None
    if llm:                                       # attach the rung-3 LLM tier
        from rung3_parser import LLMParser
        from adapters.llm_adapter import OllamaClient, StubClient
        client = OllamaClient("qwen3:1.7B") if llm == "real" else StubClient({
            "oomph": '{"entity":"rocket","rel":"force"}'})   # idiom only the LLM cracks
        llm_parser = LLMParser(client, entities | set(facts), solvable)
    return Front(kb, mem, lexical, student, solvable, llm_parser, cpp_engine, brain)


def _demo(llm="stub", use_cpp=False, use_brain=False):
    front = _build(llm=llm, use_cpp=use_cpp, use_brain=use_brain)
    queries = [
        "what is the speed of the rocket?",       # lexical exact
        "how swiftly is the rocket moving?",      # novel -> student
        "what's the rocket's momentum?",          # lexical
        "could you work out the force on the sample?",   # phrasing -> student
        "what is the density of the sample?",      # computed via policy
        "what's the oomph of the rocket?",        # idiom -> cheap tiers abstain -> LLM
        "what is the wisdom of the rocket?",      # unanswerable -> honest "I don't know"
    ]
    tag = "lexical + student + LLM" if llm else "lexical + student"
    engine = "C++ Brain.policy_solve" if use_brain else \
             "C++ PolicyEngine" if use_cpp else "Python executive"
    print(f"=== nl_front — full ladder ({tag}) -> {engine} ===\n")
    for q in queries:
        ans, src = front.answer(q)
        print(f"  > {q}")
        print(f"      [{src:8s}] {ans}\n")


if __name__ == "__main__":
    _demo(llm="real" if "--real" in sys.argv else "stub",
          use_cpp="--cpp" in sys.argv, use_brain="--brain" in sys.argv)
