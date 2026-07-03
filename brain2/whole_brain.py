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
from appraisal_engine import AppraisalEngine

CODE_WORDS = {"function", "code", "algorithm", "write", "implement", "program", "def"}
# Paraphrase -> canonical token, so routing isn't brittle to exact wording. A real
# semantic router needs sentence embeddings; this is the cheap robustness layer that
# stops common paraphrases from falling through to "I don't know".
SYNONYMS = {
    "velocity": "speed", "fast": "speed",
    "weight": "mass", "heavy": "mass", "massive": "mass",
    "acceleration": "accel", "accelerating": "accel",
    "make": "write", "create": "write", "build": "write", "generate": "write",
    "method": "function", "subroutine": "function", "routine": "function", "procedure": "function",
}
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
        # LEARNED context map: meaning from a corpus, not a hand table. Any corpus word
        # whose context strongly matches a known relation becomes an automatic synonym —
        # this is open-comprehension's fuzzy proposer; the crisp solver still verifies.
        # OPEN-LANGUAGE: read declarative prose into VERIFIED events (the open-lang track).
        # Fuzzy/positional parse proposes an Event; the crisp membrane disposes (admit verified,
        # reject contradiction/type-violation, abstain on the unknown). Same membrane as compute.
        from reading_loop import EventReader
        from type_oracle import TypeOracle
        from verb_learn import VerbLearner
        self.verbs = {"eat", "chase", "like", "see", "run", "catch", "drink"}
        self.verb_constraints = {"eat": {"agent": {"animal"}, "patient": {"animal", "plant", "food"}},
                                 "chase": {"agent": {"animal"}, "patient": {"animal"}}}
        _oracle = TypeOracle()
        self.reader = EventReader(self.concepts | self.entities, self.verbs, type_of=_oracle,
                                  constraints=self.verb_constraints, learner=VerbLearner(_oracle))
        # WHOLE: the NEURAL perception substrate (C++ Brain) + emotional appraisal. Every input
        # is perceived (SOM/episodic/emotion evolve) BEFORE the symbolic front answers — neural
        # senses (novelty, feeling), symbolic owns truth. Guarded: degrades if brain2 absent.
        self.appraiser = AppraisalEngine()
        self._seen = set()                         # perceived-token memory -> novelty sense
        try:
            import brain2
            self.brain = brain2.Brain(som_rows=16, som_cols=16, n_dims=32)
            self._perceive_mode = brain2.ErrorMode.FULL
        except Exception:
            self.brain = None
        # SELF-EXTENSION + VERIFICATION faculties (were built + tested but orphaned; wired now):
        # a persistent library of code checks learned from breaks.
        try:
            from check_library import CheckLibrary
            self.checks = CheckLibrary(path=os.path.join(os.path.dirname(__file__), "brain_store"))
        except Exception:
            self.checks = None
        import context_embed as CE
        STOP = {"the", "a", "an", "is", "are", "of", "at", "with", "has", "have", "had",
                "makes", "made", "to", "in", "on", "and", "or", "it", "its", "that", "this",
                "things", "thing", "great", "high", "large", "strong", "fast", "dense", "heavy"}
        vecs = CE.build()
        self.ctx_map = {}
        for w in vecs:
            if w in self.relations or w in self.entities or w in STOP:
                continue
            sims = sorted(((CE.cosine(vecs[w], vecs[c]), c)
                           for c in self.relations if c in vecs and c != w), reverse=True)
            # admit only a confident, unambiguous mapping (high sim AND clear margin)
            if sims and sims[0][0] >= 0.6 and (len(sims) < 2 or sims[0][0] - sims[1][0] >= 0.1):
                self.ctx_map[w] = sims[0][1]

    def ask(self, text):
        toks = [self.ctx_map.get(SYNONYMS.get(t, t), SYNONYMS.get(t, t))
                for t in re.findall(r"[a-z_]+", text.lower())]
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
        # OPEN-LANGUAGE: a DECLARATIVE (not a question) -> read it as a verified event.
        # Question-hood keys on the first token / '?', not word-presence, so "the dog did not
        # eat the fish" (declarative, has 'did') still reads as an event.
        first = toks[0] if toks else ""
        is_question = text.strip().endswith("?") or first in {
            "what", "how", "why", "who", "can", "is", "are", "does", "did", "will",
            "could", "would", "should", "which", "when", "where"}
        if not is_question:
            ev = self._read_event(text)
            if ev is not None:
                return ev
        return ("none", "I don't know.", False)

    def _read_event(self, text):
        """Read a declarative into an Event and report the membrane's disposition. Returns a
        response tuple, or None if nothing verb-like was found (fall through to 'I don't know')."""
        b = dict(self.reader.stats)
        evs, rel = self.reader.read(text)
        if not evs:
            return None
        self.reader.acquire()                   # learn verbs seen enough times -> future crisp
        s = self.reader.stats
        e = evs[-1]
        desc = "%s%s %s %s" % ("not " if e.polarity < 0 else "", e.agent, e.verb, e.patient or "")
        cause = " (CAUSE)" if rel is not None else ""
        if s["admit"] > b.get("admit", 0):
            return ("event", f"learned + verified: {desc.strip()}{cause}", True)
        if s["reject"] > b.get("reject", 0):
            return ("event", f"rejected (contradicts known / type-violation): {desc.strip()}", False)
        return ("event", f"held — can't verify yet (unknown verb/type): {desc.strip()}", False)

    # ── the WHOLE brain: perceive (neural) -> feel (appraisal) -> answer (verified) ──
    def _perceive(self, text):
        """Neural + affective sense of the input, BEFORE the symbolic answer. Runs the real
        C++ perception (SOM/episodic/emotion evolve) as a side effect; returns the readable
        projections: novelty (fraction of unseen tokens) and the emotional appraisal. Membrane:
        this SENSES (soft, never authoritative); the symbolic front still owns the answer."""
        toks = re.findall(r"[a-z_]+", text.lower())
        novel = [t for t in toks if t not in self._seen]
        novelty = len(novel) / len(toks) if toks else 0.0
        self._seen.update(toks)
        if self.brain is not None:
            try:
                self.brain.perceive_text(text, self._perceive_mode)  # real neural perception
            except Exception:
                pass
        ap = self.appraiser.appraise(text)
        dom = max(ap.frame, key=ap.frame.get) if ap.frame else None
        felt = dom if (dom and ap.frame.get(dom, 0) > 0) else "neutral"
        return {"novelty": round(novelty, 2), "utterance": ap.type, "felt": felt,
                "perceived": self.brain is not None}

    def sense(self, text):
        """The whole brain in one call: perceive+feel (neural/affective) THEN answer (verified).
        Returns the perception plus the crisp answer — all faculties in one runtime."""
        perc = self._perceive(text)
        kind, msg, ok = self.ask(text)
        return {"perception": perc, "answer": {"kind": kind, "msg": msg, "verified": ok}}

    # ── self-extension + verification faculties (formerly orphaned, now wired in) ──
    def self_check(self):
        """Verification-health introspection: synthesize a task's invariants and audit them —
        are they catching wrong answers, or spuriously rejecting correct ones? Wires
        synth_invariant + verifier_monitor + invariant_miner into the front."""
        import synth_invariant as SI, verifier_monitor as VM
        mine, hold = [0, 1, 2, 3, 4], [5, 6, 7]
        inv = SI.task_invariants(math.factorial, mine, hold)
        correct = [(x, math.factorial(x)) for x in mine + hold]
        wrong = [lambda n: n * n, lambda n: n + 1]                 # known-wrong candidates
        report = VM.audit(inv, correct, wrong, list(range(8)))
        return {"task": "factorial", "invariants": sorted(inv),
                "health": {k: v[0] for k, v in report.items()}}

    def self_extend(self):
        """Autonomous self-improvement: conjecture -> sandbox-test against a trusted principle
        -> bank verified laws, learning which shapes work. Wires autonomous_loop into the
        front. The membrane holds: only sandbox-verified conjectures are banked."""
        import autonomous_loop as AL
        prop, banked, total = AL.Proposer(), {}, 0
        for gap, true_law in AL.GAPS:
            for name, fn, shape in prop.order():
                total += 1
                if AL.sandbox_test(fn, true_law):
                    banked[gap] = name
                    prop.reward_shape(shape)
                    break
        return {"banked": banked, "conjectures_tested": total}

    def introspect(self):
        """The whole brain's self-report: known symbols, learned verbs, cached code checks,
        and live verification health — the faculties that were orphaned, now reachable."""
        base = {"eat", "chase", "like", "see", "run", "catch", "drink"}
        return {"entities": sorted(self.entities), "relations": sorted(self.relations),
                "verbs_learned": sorted(self.reader.verbs - base),
                "code_checks": (dict(self.checks.inv) if self.checks else {}),
                "verification": self.self_check()}

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

    # THE WHOLE BRAIN: perceive (neural) -> feel (appraisal) -> answer (verified), one runtime
    print("=== whole brain — sense(): perceive + feel + answer in one call ===\n")
    print("  neural perception attached (C++ Brain):", b.brain is not None, "\n")
    for t in ["the dog ate the fish", "the dog ate the fish",           # repeat -> novelty drops
              "the dog did not eat the fish",                            # contradiction + felt
              "quantum entanglement defies locality",                    # novel, held
              "what is the momentum of the rocket?"]:                    # verified compute
        r = b.sense(t); p, a = r["perception"], r["answer"]
        print(f"  > {t}")
        print(f"    perceive: novelty={p['novelty']:.2f}  felt={p['felt']}  ({p['utterance']})")
        print(f"    answer  : [{a['kind']} {'✓' if a['verified'] else '·'}] {a['msg'][:58]}\n")


if __name__ == "__main__":
    _demo()
