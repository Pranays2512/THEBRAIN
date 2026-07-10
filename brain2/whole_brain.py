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
from core.reasoning.reasoning_engine import ReasoningEngine
from core.knowledge.core_knowledge import CORE_FACTS
from core.reasoning.means_ends import PolicyMemory, FactSource, PolicySource, MeansEndsSolver, Need
from core.synthesis import synth_engine as SE
from core.store.brain_store import BrainStore
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
        self._proposer = None                       # lazy online_proposer2 (guided code synth)
        try:
            import json
            from core.knowledge.concept_memory import ConceptMemory
            from core.knowledge.semantic_memory import SemanticMemory
            cp = os.path.join(self.store.path, "concepts.json")
            sp = os.path.join(self.store.path, "semantic.json")
            # LOAD prior sessions' discoveries so the brain accumulates across restarts
            self.concept_mem = ConceptMemory.load(cp) if os.path.exists(cp) else ConceptMemory()
            self.semantic = SemanticMemory()
            if os.path.exists(sp):
                self.semantic.replay(json.load(open(sp)))
        except Exception:
            self.concept_mem = self.semantic = None
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
        from core.store.type_oracle import TypeOracle
        from core.events.verb_learn import VerbLearner
        from event_predict import EventPredictor
        self.verbs = {"eat", "chase", "like", "see", "run", "catch", "drink"}
        self.verb_constraints = {"eat": {"agent": {"animal"}, "patient": {"animal", "plant", "food"}},
                                 "chase": {"agent": {"animal"}, "patient": {"animal"}}}
        _oracle = TypeOracle()
        # predictive processing: the reader PREDICTS the next event and learns from the error,
        # so surprise is a real semantic signal (unexpected event), not lexical novelty.
        self.reader = EventReader(self.concepts | self.entities, self.verbs, type_of=_oracle,
                                  constraints=self.verb_constraints, learner=VerbLearner(_oracle),
                                  predictor=EventPredictor())
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
            from core.store.check_library import CheckLibrary
            self.checks = CheckLibrary(path=os.path.join(os.path.dirname(__file__), "brain_store"))
        except Exception:
            self.checks = None
        from core.grounding import context_embed as CE
        STOP = {"the", "a", "an", "is", "are", "of", "at", "with", "has", "have", "had",
                "makes", "made", "to", "in", "on", "and", "or", "it", "its", "that", "this",
                "things", "thing", "great", "high", "large", "strong", "fast", "dense", "heavy"}
        vecs = CE.build()
        self.vecs = vecs                            # learned concept vectors (for concept_blend)
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
        # RICHER queries first (compare / compound / boolean / nested) — they'd otherwise be
        # mis-caught by the single-fact compute path or the event reader.
        _RICH = {"heavier", "lighter", "faster", "slower", "denser", "bigger", "greater",
                 "more", "less", "than", "heaviest", "lightest", "fastest", "slowest",
                 "densest", "biggest"}
        if (_RICH & ts) or " and " in f" {text.lower()} " or " or " in f" {text.lower()} " \
                or sum(t in self.relations for t in toks) >= 2:
            r = self.ask_rich(text)
            if r is not None:
                return ("compute", r, True)
        # COMPUTE (before the loose 'is'/'can' factual checks)
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
        # richer query comprehension (compare / compound / boolean / nested) before giving up
        rich = self.ask_rich(text)
        if rich is not None:
            return ("compute", rich, True)
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
        from mouth import say_event
        desc = say_event(e).rstrip(".")            # the brain says it in its OWN learned grammar
        cause = " (CAUSE)" if rel is not None else ""
        if s["admit"] > b.get("admit", 0):
            return ("event", f"learned + verified: {desc}{cause}", True)
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
        neural = None
        if self.brain is not None:
            try:                                        # real C++ perception (recompiled to
                pr = self.brain.perceive_text(text, self._perceive_mode)  # RETURN PerceiveResult)
                neural = {"bmu": pr.bmu, "surprise": round(pr.prediction_error, 4),
                          "valence": round(pr.valence, 4)}
            except Exception:
                pass
        ap = self.appraiser.appraise(text)
        dom = max(ap.frame, key=ap.frame.get) if ap.frame else None
        felt = dom if (dom and ap.frame.get(dom, 0) > 0) else "neutral"
        # novelty (token-level) is the differentiating readable signal; `neural` carries the
        # real C++ SOM/predictor state (bmu/surprise/valence) — meaningful once the Brain is
        # semantically grounded + trained (a fresh SOM collapses these to ~constant).
        return {"novelty": round(novelty, 2), "utterance": ap.type, "felt": felt,
                "perceived": self.brain is not None, "neural": neural}

    def sense(self, text):
        """The whole brain in one call: perceive+feel (neural/affective) THEN answer (verified).
        Returns the perception plus the crisp answer — all faculties in one runtime."""
        perc = self._perceive(text)
        kind, msg, ok = self.ask(text)
        # SEMANTIC surprise from predictive processing (set if the answer read an event) — the
        # real 'how expected was this?' signal, distinct from lexical novelty above.
        if kind == "event" and self.reader.last_surprise is not None:
            perc["surprise"] = round(self.reader.last_surprise, 2)
        return {"perception": perc, "answer": {"kind": kind, "msg": msg, "verified": ok}}

    # ── self-extension + verification faculties (formerly orphaned, now wired in) ──
    def self_check(self):
        """Verification-health introspection: synthesize a task's invariants and audit them —
        are they catching wrong answers, or spuriously rejecting correct ones? Wires
        synth_invariant + verifier_monitor + invariant_miner into the front."""
        import verifier_monitor as VM; from core.synthesis import synth_invariant as SI
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

    def generate(self, corpus=None, seed=("<s>",), n=3):
        """The PROBABILISTIC pillar (lightweight): an n-gram model over text the brain has
        (its admitted events rendered to sentences, or a given corpus) — distributions,
        entropy, GENERATION. Complements the heavy owned Transformer (neural_lm_torch) used
        in training; this is the in-process, torch-free generator. Wires prob_compute."""
        from core.math.prob_compute import ProbLM
        if corpus is None:                              # build from what the brain has read
            from mouth import say_event
            corpus = [say_event(e) for e in getattr(self.reader, "events", [])]
            corpus += ["a dog is an animal", "an animal is a living thing",
                       "the rocket has large mass", "energy depends on mass and speed"]
        lm = ProbLM(order=3).train(corpus)
        return {"trained_on": len(corpus), "vocab": len(lm.vocab),
                "samples": [" ".join(lm.generate(seed_rng=i)) for i in range(n)],
                "entropy_at_start": round(lm.entropy(list(seed)), 3)}

    def check_dimensions(self, expr, target):
        """A units VERIFIER: is `expr` dimensionally sound for the `target` quantity (e.g.
        mass*accel is a force, mass*speed is not)? A second membrane beyond numeric checking —
        catches type-of-quantity errors a value check can't. Wires dimensional_verify."""
        from core.math.dimensional_verify import dimensionally_sound
        try:
            return bool(dimensionally_sound(expr, target))
        except Exception:
            return None

    def test_conjecture(self, conjecture):
        """The brain designs experiments to test its OWN guess against a principle it already
        trusts (energy conservation), admitting only what survives — active experimentation,
        no answer key. Wires conjecture_sandbox. `conjecture` is f(mass, velocity) -> KE
        (the true law is ½·m·v²; a guess that matches on random drops is admitted)."""
        from core.synthesis.conjecture_sandbox import design_and_test
        ok, worst, counter = design_and_test(conjecture)
        return {"admitted": bool(ok), "worst_error": round(worst, 4), "counterexample": counter}

    def write_code_robust(self, kind, oracle, inputs):
        """Self-correcting synthesis: synthesize, STRESS against the oracle, and if it breaks on
        a counterexample fold that in and re-synthesize — the refuter closing the loop so an
        overfit fixes itself with no hand-holding. Wires refute_synth."""
        from core.synthesis.refute_synth import synth_self_correct
        code, log = synth_self_correct(kind, oracle, inputs)
        return {"code": code, "iterations": len(log), "verified": code is not None}

    def save_state(self):
        """Persist everything the brain DISCOVERED this session so it accumulates across
        restarts: banked policies + facts + code (brain_store), named concepts
        (concept_memory), and associative memory (semantic_memory). Verified-only — nothing
        fuzzy is written, and the crisp store is the source of truth on reload."""
        import json
        self.store.save()
        try:
            if self.concept_mem is not None:
                self.concept_mem.save(os.path.join(self.store.path, "concepts.json"))
            if self.semantic is not None:
                self.semantic.save(os.path.join(self.store.path, "semantic.json"))
        except Exception:
            pass
        return {"policies": len(self.store.policies), "functions": len(self.store.functions),
                "concepts": len(getattr(self.concept_mem, "concepts", {}) or {}),
                "semantic_facts": len(getattr(self.semantic, "facts", []) or [])}

    def remember(self, subj, rel, obj):
        """Associative memory write (semantic_memory): stores a relation and supports fuzzy
        recall + similarity a plain dict cannot. Complements the crisp store."""
        if self.semantic is None:
            return False
        self.semantic.learn(subj, rel, obj)
        return True

    def recall_similar(self, token, k=5):
        """What is this token associatively like? (semantic_memory.similar) — the fuzzy
        neighbour lookup, distinct from crisp isa closure."""
        if self.semantic is None:
            return []
        try:
            return self.semantic.similar(token, k)
        except Exception:
            return []

    def ground(self):
        """Perception -> symbol -> reasoning: the brain sees raw vectors, recognizes their
        category on the SOM, ASSERTS the grounded category as a fact, and INFERS properties it
        was never told (grounded meaning, not LLM-given). Wires ground_reason. Guarded — needs
        the C++ brain2; returns how many properties it inferred from perception alone."""
        if self.brain is None:
            return {"grounded": False, "reason": "C++ brain2 unavailable"}
        from core.grounding import ground_reason as GR
        r = GR.ground_and_reason(reasoner=ReasoningEngine())
        return {"grounded": True, "inferred_correct": f"{r['correct']}/{r['total']}",
                "sample": r["results"][:3]}

    def ground_numeric(self):
        """Ground CONTINUOUS quantities: perceive raw vectors, DECODE numeric values, assert
        them as facts, and let the PolicyEngine compute from what it perceived (not values it
        was told). Wires ground_numeric. Guarded on C++ brain2."""
        if self.brain is None:
            return {"grounded": False, "reason": "C++ brain2 unavailable"}
        from core.grounding import ground_numeric as GN
        r = GN.ground_and_compute()
        return {"grounded": True, "within_10pct": f"{r['hits']}/{r['total']}",
                "sample": r["results"][:3]}

    def learn_heuristic(self, n_tasks=60, probes=12, seed=7):
        """Search that IMPROVES with experience (learned_guidance): fit a cost-to-goal estimate
        from solved instances, then guide the A* engine so it expands far fewer states while
        staying correct (it still solves). Wires learned_guidance into the front; returns the
        measured blind-vs-learned node reduction. Domain-agnostic — demonstrated on the puzzle
        it is proven on, the same engine (tree_reason) the synthesis paths use."""
        import random
        from core.reasoning import learned_guidance as LG
        from core.reasoning.tree_learn import EightPuzzle, features, manhattan, scramble
        h = LG.LearnedHeuristic(features)
        h.train(LG.collect_examples(EightPuzzle, scramble, manhattan))
        rng = random.Random(seed)
        starts = [scramble(80, rng) for _ in range(probes)]
        blind, ok_b = LG._avg_nodes(None, starts)
        learned, ok_l = LG._avg_nodes(h, starts)
        return {"blind_states": round(blind), "learned_states": round(learned),
                "speedup": round(blind / max(learned, 1), 1),
                "solved": f"{ok_l}/{len(starts)}", "correct": ok_l >= ok_b}

    # ── CREATIVITY faculties: originate new knowledge, membrane-gated ────────────
    # These were built + verified in isolation but orphaned (no importer). Wired here
    # into the front so the running brain can conjecture across domains, blend concepts,
    # map analogies, and induce rules — every product VERIFIED before it is banked.

    def cross_domain(self):
        """Find a law two DISTINCT domains secretly share (curiosity into the adjacent
        possible). Factors the union of the brain's banked policies via anti-unification;
        a skeleton recurring across differently-named laws is a cross-domain insight —
        verified to reconstruct every input formula. Banks the shared shape into the store."""
        import curiosity_cross as CC
        libs = [(t, p.expr) for t, p in self.mem.by_target.items()]
        if len(libs) < 2:
            return {"discovered": None, "reason": "need >=2 banked laws"}
        r = CC.cross_domain_laws(libs)
        disc = r["discovery"]
        if not r["new"] or not r["verified"] or not disc:
            return {"discovered": None, "reason": "no verified shared structure"}
        name, skel = disc[0], disc[1]                  # (name, pattern, arity)
        self.store.add_policy(name, (), skel)          # bank the shared shape (verified)
        self.store.save()
        # give the discovered structure a NAMED, reusable identity (concept_memory)
        cname = name
        if self.concept_mem is not None:
            try:
                cname = self.concept_mem.register(skel, [n for n, _ in r["new"]])
            except Exception:
                pass
        return {"discovered": name, "concept": cname, "shape": skel, "verified": True,
                "unified": [n for n, _ in r["new"]]}

    def blend(self, a=None, b=None):
        """Invent a concept in EMPTY feature space (novelty primitive). Fuses two distant
        known concepts into a point outside every category; admitted only if verifiably
        novel (nearest known concept farther than the cluster radius). Proposes — grounding
        still decides usefulness."""
        from core.knowledge import concept_blend as CB
        # CE vectors are SPARSE co-occurrence dicts; densify to aligned lists over the shared
        # context vocabulary so concept_blend's per-dimension fuse/distance is well-defined.
        grounded = [c for c in (self.concepts | self.relations) if c in self.vecs]
        dims = sorted({k for c in grounded for k in self.vecs[c]})
        pool = {c: [self.vecs[c].get(k, 0.0) for k in dims] for c in grounded}
        if len(pool) < 3 or not dims:
            return {"novel": False, "reason": "too few grounded concepts"}
        # data-driven radius = median nearest-neighbour distance (what "same concept" means here)
        names = sorted(pool)
        nn = []
        for n in names:
            d = min(CB.dist(pool[n], pool[m]) for m in names if m != n)
            nn.append(d)
        radius = sorted(nn)[len(nn) // 2]
        if a is None or b is None:                     # pick the two most distant concepts
            best, pair = -1.0, None
            for i, x in enumerate(names):
                for y in names[i + 1:]:
                    d = CB.dist(pool[x], pool[y])
                    if d > best:
                        best, pair = d, (x, y)
            a, b = pair
        if a not in pool or b not in pool:
            return {"novel": False, "reason": "concepts not grounded"}
        r = CB.propose(a, b, pool, radius)
        r["parents"] = [a, b]
        r.pop("vector", None)                          # keep the report light
        return r

    def analogize(self, source, target):
        """Structure-map two domains given as (subj, rel, obj) triples over a shared relation
        vocabulary; return the object correspondence + analogical predictions (HYPOTHESES to
        verify, not truths). Ambiguous/structure-poor domains yield no mapping, honestly."""
        from core.events.analogy_engine import AnalogyEngine
        mapping, transfers = AnalogyEngine().map_domains(list(source), list(target))
        return {"mapping": mapping,
                "predictions": [(s, r, o) for s, r, o, _ in transfers]}

    def induce(self, episodes, promote=True):
        """Mine rules (A tends to precede B) from observed episodes, VERIFY on a held-out
        split (reject train-only coincidences), and — if promote — install the survivors into
        the factual reasoner so they become chainable knowledge. Originating rules from data."""
        import random
        from core.synthesis.inductive_engine import InductiveLearner
        eps = [list(e) for e in episodes if len(e) >= 2]
        if len(eps) < 4:
            return {"promoted": [], "rejected": [], "reason": "too few episodes"}
        random.Random(0).shuffle(eps)              # deterministic: a pattern must appear in
        cut = max(2, int(0.7 * len(eps)))          # BOTH splits to verify (else "untested")
        learner = InductiveLearner()
        promoted, rejected = learner.mine(eps[:cut], eps[cut:])
        if promote and promoted:
            learner.promote_into(self.kre, promoted)   # discovered rules -> usable knowledge
        return {"promoted": [(r.a, r.b, r.conf_test) for r in promoted],
                "rejected": [(a, b) for a, b, _ in rejected]}

    def read_to_law(self, corpus, inputs, target, client):
        """The extractor rung: an LLM reads prose into numeric rows, the brain INDUCES a law
        over them and VERIFIES it before storing (teacher proposes data, brain disposes the
        law). Needs an LLM `client` with .complete(); the induced law is membrane-checked, so
        a corpus that supports no law yields None, not a guess."""
        import learn_by_reading as LBR

        class _Adapter:                                # bridge to brain_store.add_policy
            def __init__(s, store): s.store = store
            def policy_add(s, t, ins, expr):
                s.store.add_policy(t, ins, expr); s.store.save()
        expr, n_rows, nodes = LBR.learn(corpus, tuple(inputs), target, client, _Adapter(self.store))
        return {"law": expr, "rows": n_rows, "nodes": nodes, "stored": expr is not None}

    def create(self):
        """The unifying idle cycle: the brain extends ITSELF between queries. Induce rules from
        what it has read, find a cross-domain shared law, invent a novel concept — each product
        passes the membrane (held-out verify / structural verify / novelty check) before it is
        banked. Nothing fuzzy reaches the truth store. Returns what it originated this pass."""
        # 1. induce rules from the reader's admitted event history (agent->verb->patient chains)
        episodes = [[e.agent, e.verb, e.patient] for e in getattr(self.reader, "events", [])
                    if e.agent and e.verb and e.patient]
        induced = self.induce(episodes) if episodes else {"promoted": [], "reason": "no reading yet"}
        # 2. cross-domain law from banked policies
        crossed = self.cross_domain()
        # 3. invent a verified-novel concept
        blended = self.blend()
        # 4. curiosity: where is the brain's predictor weakest? (prediction-error gaps)
        curious = None
        if episodes:
            try:
                from curiosity_loop import CuriosityLoop
                cl = CuriosityLoop()
                cl.observe(episodes)
                cl.tick()
                curious = {"error": cl.error(), "gaps": cl.curiosity_gaps()}
            except Exception:
                pass
        persisted = self.save_state()                  # discoveries survive the session
        return {"induced": induced, "cross_domain": crossed, "blended": blended,
                "curiosity": curious, "persisted": persisted}

    def run_loop(self, ticks=5, verbose=False):
        """The STANDING unifying loop (roadmap's always-on cycle): tick the create() faculties
        repeatedly so they COMPOUND — each pass runs self_extend (autonomous conjecture→sandbox
        →bank), induce, cross_domain, blend, curiosity, and persists. Tracks growth per tick;
        stops early once a tick adds nothing new (converged). This is the orchestrator that was
        missing — the modules existed, the loop that keeps them running unattended did not."""
        history = []
        prev = (len(self.store.policies), len(getattr(self.concept_mem, "concepts", {}) or {}))
        for t in range(ticks):
            self.self_extend()                          # autonomous conjecture -> sandbox -> bank
            out = self.create()                         # induce/cross-domain/blend + persist
            now = (len(self.store.policies), len(getattr(self.concept_mem, "concepts", {}) or {}))
            grew = now != prev
            rec = {"tick": t, "policies": now[0], "concepts": now[1],
                   "induced": len(out["induced"].get("promoted", [])),
                   "curiosity_gaps": (out.get("curiosity") or {}).get("gaps", []),
                   "grew": grew}
            history.append(rec)
            if verbose:
                print(f"  tick {t}: policies={now[0]} concepts={now[1]} "
                      f"induced={rec['induced']} grew={grew}")
            if not grew and t > 0:                      # converged: nothing new to discover
                break
            prev = now
        return {"ticks_run": len(history), "history": history,
                "final": self.save_state()}

    def wonder(self):
        """Curiosity -> a question. The brain asks about the most surprising event it has seen
        (surprise-gated salience), voicing it through its own mouth. Returns None if nothing
        has surprised it yet. This is the predictive loop closing into active learning: high
        prediction error -> attention -> a question worth answering."""
        if not self.reader.salient:
            return None
        from mouth import ask
        return {"question": ask(self.reader.salient[-1]),
                "curious_about": self.reader.curiosity()}

    def introspect(self):
        """The whole brain's self-report: known symbols, learned verbs, cached code checks,
        and live verification health — the faculties that were orphaned, now reachable."""
        base = {"eat", "chase", "like", "see", "run", "catch", "drink"}
        return {"entities": sorted(self.entities), "relations": sorted(self.relations),
                "verbs_learned": sorted(self.reader.verbs - base),
                "code_checks": (dict(self.checks.inv) if self.checks else {}),
                "creativity": ["cross_domain", "blend", "analogize", "induce",
                               "read_to_law", "create"],   # originate faculties, membrane-gated
                "learned_search": ["_guided_solve (online_proposer2)",
                                   "learn_heuristic (learned_guidance)"],  # search that improves
                "code_gen": ["_code (int algorithms: composable/early/fold/two)",
                             "write_transform (string transforms: program_synth)"],
                "grounding": ["ground (perception->symbol->fact->infer, ground_reason)",
                              "ground_numeric (perceive quantities->compute, ground_numeric)"],
                "probabilistic": ["generate (n-gram over read text, prob_compute)"],
                "verifiers": ["check_dimensions (dimensional_verify)",
                              "test_conjecture (conjecture_sandbox)",
                              "write_code_robust (refute_synth)"],
                "memory": ["remember / recall_similar (semantic_memory)",
                           "concept_mem (concept_memory, names discoveries)",
                           "create.curiosity (curiosity_loop)"],
                "verification": self.self_check()}

    def _guided_solve(self, ex, kind, oracle):
        """Code synthesis ordered by a learned proposer (online_proposer2): a task-signature
        keys which space to try first, and stress-survival rewards/penalizes that space, so
        repeated synthesis learns the right space per task FAMILY and NEW tasks transfer by
        signature. Falls back to the static engine if the proposer is unavailable — the
        verifier gates every result, so guidance changes SPEED, never correctness."""
        if self._proposer is None:
            try:
                from core.synthesis.online_proposer2 import FeatureProposer
                self._proposer = FeatureProposer()
            except Exception:
                self._proposer = False
        if self._proposer:
            name, code, _ = self._proposer.solve(ex, kind, oracle)
            if code:
                return name, code
        return SE.solve(ex, kind)                        # static-order fallback

    def ask_rich(self, q):
        """Richer QUERY comprehension the flat router can't parse — generalize over sentence
        STRUCTURE, not surface words: compare ('is the rocket heavier than the sample'),
        compound ('mass and speed of the rocket'), boolean conditions ('mass > 100 and speed <
        50'), and nested (superlative 'the fastest object', if/then). Answered by the SAME
        verified compute core (means-ends over facts+policies). Returns a string, or None to
        fall back to ask(). (Wires structural_parser / deeper_grammar / nested_parser.)"""
        from core.reasoning import structural_parser as SP; from core.reasoning import deeper_grammar as DG; from core.reasoning import nested_parser as NP
        for M in (DG, NP):                              # inject THIS brain's live vocabulary
            M.ENTS = set(self.entities)
            M.RELS = set(self.relations)
        # comparative adjectives -> the relation they compare on, so 'heavier' finds 'mass'
        cmp_map = {"heavier": "mass", "lighter": "mass", "faster": "speed",
                   "slower": "speed", "denser": "density", "bigger": "volume"}
        ctx = {**self.ctx_map, **cmp_map}
        low = q.lower()
        toks = set(re.findall(r"[a-z_]+", low))

        def _ok(a):
            return a and "abstain" not in a and "not a conditional" not in a \
                and "not parseable" not in a and "no answer" not in a

        try:
            if "if" in toks:                            # conditional: boolean (and/or) or single
                a = DG.DeeperParser(self.fkb, self.mem).answer(q)   # if-then, boolean clauses
                if _ok(a):
                    return a
                a = NP.NestedParser(self.fkb, self.mem).answer(q)   # if-then, single clause
                if _ok(a):
                    return a
            if toks & set(NP.SUPER):                    # superlative sub-query (needs a relation)
                a = NP.NestedParser(self.fkb, self.mem).answer(q)
                if _ok(a):
                    return a
            parsed = SP.StructuralParser(self.entities, self.relations, ctx).parse(q)
            if parsed["kind"] in ("compare", "compound", "single"):
                a = SP.answer(parsed, self.fkb, self.mem)
                if a:
                    return a
        except Exception:
            pass
        return None

    def write_transform(self, examples):
        """Synthesize a STRING-transform program from (input, output) examples, by verified
        search over a text DSL (program_synth) — e.g. [("John Smith","JOHN")] -> upper∘first.
        The returned program is correct on the examples BY CONSTRUCTION and generalizes; None
        if no program in the DSL fits (honest miss)."""
        from core.synthesis import program_synth as PS
        from core.reasoning.tree_reason import solve
        path, _, nodes = solve(PS.Synthesize(list(examples)))
        if path is None:
            return {"program": None, "nodes": nodes}
        prog = path[-1][1]
        return {"program": list(prog) or ["identity"],
                "apply": (lambda s, _p=prog: PS.run(_p, s)),
                "nodes": nodes,
                "verified": all(PS.run(prog, i) == o for i, o in examples)}

    def _code(self, toks):
        name = next((t for t in toks if t in CODE_TASKS), None)
        if name is None:
            return ("code", "can't synthesize that yet (outside the synth DSLs; needs the LLM tier).", False)
        if self.store.knows_function(name):
            return ("code", f"recalled from memory:\n{self.store.functions[name].strip()}", True)
        kind, raw, oracle = CODE_TASKS[name]
        ex = SE._ex(kind, oracle, raw)
        sp, code = self._guided_solve(ex, kind, oracle)
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
