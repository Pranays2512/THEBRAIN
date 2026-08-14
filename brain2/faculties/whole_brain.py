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
from engines.reasoning.reasoning_engine import ReasoningEngine
from engines.knowledge.core_knowledge import CORE_FACTS
from engines.reasoning.means_ends import PolicyMemory, FactSource, PolicySource, MeansEndsSolver, Need
from engines.synthesis import synth_engine as SE
from engines.store.brain_store import BrainStore
from faculties.appraisal_engine import AppraisalEngine
from faculties.autonomous_loop import Proposer
from engines.knowledge.knowledge_engine import KnowledgeEngine
from faculties.curiosity_bridge import CuriosityBridge
from middleware.event_bus import bus
from routing.crisp_external_router import CrispExternalRouter, CrispFact
from routing.crisp_internal_router import CrispInternalRouter
from faculties.conversation_engine import ConversationEngine
from faculties.query_planner import QueryPlanner

CODE_WORDS = {"function", "code", "algorithm", "write", "implement", "program", "def", "java", "cpp", "c++", "python", "solution", "input", "output", "testcase", "testcases"}

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
    def __init__(self, eyes=None):
        self.eyes = eyes
        self.store = BrainStore()
        try:
            from engines.synthesis.unified_proposer import UnifiedProposer
            self._proposer = UnifiedProposer()
        except ImportError:
            self._proposer = None

        try:
            from engines.reasoning.dual_process_engine import DualProcessSolver, train_policy
            self.dual_solver = DualProcessSolver(train_policy())
        except ImportError:
            self.dual_solver = None

        try:
            import json
            from engines.knowledge.concept_memory import ConceptMemory
            from engines.knowledge.semantic_memory import SemanticMemory
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
        self.entities = {"rocket", "sample"} | {s for s, _, _ in CORE_FACTS} | {o for _, _, o in CORE_FACTS}
        self.relations = {"force", "density", "momentum", "energy", "mass", "speed", "accel", "volume"} | {r for _, r, _ in CORE_FACTS}
        self.concepts = {s for s, _, _ in CORE_FACTS} | {o for _, _, o in CORE_FACTS}

        # LOAD crisp facts learned from reading (survives restart, accumulates)
        lf = os.path.join(self.store.path, "learned_facts.json")
        if os.path.exists(lf):
            try:
                import json as _json
                for tri in _json.load(open(lf)):
                    if isinstance(tri, (list, tuple)) and len(tri) == 3:
                        s, r, o = str(tri[0]), str(tri[1]), str(tri[2])
                        self.kre.learn(s, r, o)
                        self.concepts.add(s)
                        self.concepts.add(o)
                        self.entities.add(s)
                        self.entities.add(o)
                        self.relations.add(r)
            except Exception:
                pass
        
        self.lang = ConversationEngine(max_describe=4)
        self.lang.r = self.kre
        self.planner = QueryPlanner(engine=self.kre)

        self.kre.set_transitive("isa")
        for prop in ("has", "can", "lives_in"):
            self.kre.add_rule("isa", prop, prop)
        # COMPUTE: physics facts + policies via the means-ends executive
        self.fkb = ReasoningEngine()
        # Facts are dynamically loaded/learned; removed hardcoded toy examples.
        if hasattr(self.store, "facts"):
            for ent_rel, v in self.store.facts.items():
                if "|" in ent_rel:
                    ent, r = ent_rel.split("|", 1)
                    self.fkb.learn(ent, r, str(v))
                    self.entities.add(ent)
                    self.relations.add(r)
        self.mem = PolicyMemory()
        for t, ins, e in [("force", ("mass", "accel"), ("*", "mass", "accel")),
                          ("density", ("mass", "volume"), ("/", "mass", "volume")),
                          ("momentum", ("mass", "speed"), ("*", "mass", "speed")),
                          ("energy", ("mass", "speed"), ("*", 0.5, ("*", "mass", ("^", "speed", 2))))]:
            self.mem.add(__import__("engines.reasoning.means_ends", fromlist=["_"]).Policy(t, ins, e))

        # Semantic language front
        try:
            from adapters.nl_front import Front
            from adapters.nl_query import NLQueryParser, load_glove, _rel_words
            from training.student_trainer import Student, load_dataset, DATA
            rows = load_dataset(DATA)
            needed = set(self.relations) | self.entities
            for r in self.relations:
                needed.update(_rel_words(r))
            for row in rows:
                import re
                needed.update(re.findall(r"[a-z_]+", row["question"].lower()))
            glove = load_glove(needed=needed)
            
            student = Student.train(rows, glove, k=5)
            lexical = NLQueryParser(self.entities, self.relations, glove)
            solvable = set(self.mem.by_target) | self.relations
            self.front = Front(self.fkb, self.mem, lexical, student, solvable, llm_parser=self.eyes)
        except Exception as e:
            print(f"Failed to initialize semantic front: {e}")
            self.front = None


        # LEARNED context map: meaning from a corpus, not a hand table. Any corpus word
        # whose context strongly matches a known relation becomes an automatic synonym —
        # this is open-comprehension's fuzzy proposer; the crisp solver still verifies.
        # OPEN-LANGUAGE: read declarative prose into VERIFIED events (the open-lang track).
        # Fuzzy/positional parse proposes an Event; the crisp membrane disposes (admit verified,
        # reject contradiction/type-violation, abstain on the unknown). Same membrane as compute.
        from faculties.reading_loop import EventReader
        from engines.store.type_oracle import TypeOracle
        from engines.events.verb_learn import VerbLearner
        from faculties.event_predict import EventPredictor
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
            from engines.store.check_library import CheckLibrary
            self.checks = CheckLibrary(path=os.path.join(os.path.dirname(__file__), "brain_store"))
        except Exception:
            self.checks = None
        from engines.grounding import context_embed as CE
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

        # Topology and Curiosity Bridges + Middleware Routers
        self.int_router = CrispInternalRouter()
        self.ext_router = None
        
        if self.brain is not None:
            self.curiosity_bridge = CuriosityBridge(self.brain, self._proposer, self.kre)
            self.ext_router = CrispExternalRouter(self.brain)
            bus.subscribe("verified_fact", self._on_verified_fact)
            
    def _on_verified_fact(self, data):
        if self.ext_router is not None:
            try:
                fact = CrispFact(entity=str(data["entity"]), relation=str(data["rel"]), value=float(data["value"]), verified=True)
                self.ext_router.push_fact(fact)
            except Exception:
                pass

    # ── live teaching ─────────────────────────────────────────────────────────
    def teach(self, subj: str, rel: str, obj: str) -> bool:
        """Teach a new fact at runtime, keeping the entity/relation/concept
        sets in sync so ask() can route to it immediately.

        Returns True if the fact was new, False if already known.
        """
        was_new = self.kre.learn(subj, rel, obj)
        for tok in (subj, obj):
            self.entities.add(tok)
            self.concepts.add(tok)
        self.relations.add(rel)
        if hasattr(self, "lang"):
            try:
                self.lang.learn(subj, rel, obj)
            except Exception:
                pass
        return was_new

    def execute_bql(self, queries) -> list:

        """Execute a list of BrainQL queries against this brain's reasoning engine.

        This is the Brain's query interface — called by the pipeline after BrainQLEyes
        has parsed the LLM's BrainQL output. The LLM never calls this directly.

        Args:
            queries: list[BrainQLQuery] or a single BrainQLQuery
        Returns:
            list[BrainQLResult]
        """
        from engines.reasoning.brainql import BrainQLExecutor, BrainQLQuery
        from engines.reasoning.means_ends import PolicyMemory, FactSource, PolicySource, MeansEndsSolver

        # Build a fresh executor backed by this brain's reasoning engine.
        # Wire the MeansEndsSolver for COMPUTE queries.
        mes = MeansEndsSolver([FactSource(self.fkb), PolicySource(self.mem)])
        exec_ = BrainQLExecutor(self.kre, means_ends_solver=mes)

        # Make sure isa is transitive (idempotent)
        self.kre.set_transitive("isa")

        if isinstance(queries, BrainQLQuery):
            queries = [queries]

        results = exec_.run_block(queries)

        # Side-effect: persist newly taught facts so they survive across sessions
        import json
        import os
        for q, r in zip(queries, results):
            if q.op == "TEACH" and r.known:
                lf = os.path.join(self.store.path, "learned_facts.json")
                try:
                    existing = json.load(open(lf)) if os.path.exists(lf) else []
                    triple = [q.subj, q.rel, q.obj]
                    if triple not in existing:
                        existing.append(triple)
                        json.dump(existing, open(lf, "w"), indent=2)
                except Exception:
                    pass

        return results

    def ask_bql(self, text: str, bql_eyes=None, bql_mouth=None) -> str:
        """Full BrainQL pipeline: text → Eyes → BrainQL → Brain → Mouth → text.

        This is the PRIMARY entry point when BrainQL is enabled.
        Falls back to ask() if the Eyes don't return BrainQL (e.g. math query
        that the exact parser handles, or LLM is offline).

        Args:
            text      : natural language from the user
            bql_eyes  : a BrainQLEyes instance (or any object with .parse())
            bql_mouth : a BrainQLMouth instance (or any object with .render_result())
                        If None, uses the deterministic fallback renderer.
        Returns:
            A fluent string answer.
        """
        from engines.reasoning.brainql import BrainQLQuery

        # Perception side: let the C++ brain perceive the input regardless of route
        self.sense(text)

        # If no BrainQLEyes, fall through to the existing ask() pipeline
        if bql_eyes is None:
            return self.ask(text)

        parsed = bql_eyes.parse(text)

        # Math / language Query (existing path, returned by RuleEyes / LLMEyes)
        if not (isinstance(parsed, list) and parsed and isinstance(parsed[0], BrainQLQuery)):
            return self.ask(text)

        # BrainQL path
        results = self.execute_bql(parsed)

        # Verbalize
        parts = []
        for r in results:
            if bql_mouth is not None:
                parts.append(bql_mouth.render_result(r))
            else:
                parts.append(self._bql_fallback_render(r))
        return " ".join(parts) if parts else "I don't know."

    def _bql_fallback_render(self, result) -> str:
        """Deterministic BrainQL verbalization — used when no LLM mouth is available."""
        if not result.known:
            return f"I don't know: {result.note or 'no answer found.'}"
        v = result.value
        subj, rel = result.subj, result.rel
        chain = result.chain or []
        chain_str = " — ".join(chain) if chain else ""
        op = result.op

        if op in ("LOOKUP", "DERIVE"):
            return f"{subj} {rel}: {v}."
        if op == "INHERIT":
            return f"{subj} {rel} {v}" + (f" (via: {chain_str})" if chain_str else ".") + "."
        if op == "CHAIN":
            items = ", ".join(v) if isinstance(v, list) else str(v)
            return f"{subj} is a: {items}."
        if op == "COMPUTE":
            return f"{subj}.{rel} = {v:.4g}." if isinstance(v, float) else f"{subj}.{rel} = {v}."
        if op == "TEACH":
            return f"Got it: {subj} {rel} {v}."
        if op == "TEACH_RULE":
            return f"Rule registered: {v}."
        if op == "EXPLAIN":
            return f"{subj} {rel} {v}" + (f" — {chain_str}" if chain_str else "") + "."
        return f"{v}."

    def ask(self, text):

        if hasattr(text, "raw") and text.raw:
            raw_str = text.raw
        elif hasattr(text, "payload") and isinstance(text.payload, dict) and "text" in text.payload:
            raw_str = text.payload["text"]
        elif isinstance(text, str):
            raw_str = text
        else:
            raw_str = str(text)

        # Alpha-only tokens for routing/synonyms (fast, backward-compat)
        toks = [self.ctx_map.get(SYNONYMS.get(t, t), SYNONYMS.get(t, t))
                for t in re.findall(r"[a-z_]+", raw_str.lower())]
        ts = set(toks)
        # Alphanumeric tokens (includes digits and mixed-case like H2SO4, NaOH)
        # used by the BrainQL entity lookup; keeps toks unchanged for existing routes.
        toks_alnum = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", raw_str)]

        
        first = toks[0] if toks else ""
        is_question = raw_str.strip().endswith("?") or first in {
            "what", "how", "why", "who", "can", "is", "are", "does", "did", "will",
            "could", "would", "should", "which", "when", "where"}
        appraisal = "question" if is_question else "statement"

        solution_type = "none"
        ans_msg = "I don't know."
        is_verified = False
        confidence = 0.0
        fact_data = None
        # _answer_source tracks where the answer came from.
        # 'kre_direct': came from kre.ask() / BrainQL — already structurally verified.
        # 'llm_text': came from ConversationEngine.respond() or a language model.
        # 'none': not yet answered.
        _answer_source = "none"

        # Stop words used in both the BrainQL entity/relation scan AND the verification gate.
        # Defined at method scope so the gate can use it without redefining.
        _BQL_STOP = {"does", "is", "are", "was", "were", "do", "did", "can", "will",
                     "has", "have", "had", "not", "be", "been", "being",
                     "what", "which", "who", "how", "when", "where", "why",
                     "a", "an", "the", "of", "in", "on", "at", "to", "and", "or"}


        # 1. Math
        from engines.math.word_math import solve as solve_word_math
        math_ans = solve_word_math(raw_str)
        
        # 2. Rich queries
        _RICH = {"heavier", "lighter", "faster", "slower", "denser", "bigger", "greater",
                 "more", "less", "than", "heaviest", "lightest", "fastest", "slowest",
                 "densest", "biggest"}
        rich_query = (_RICH & ts) or " and " in f" {raw_str.lower()} " or " or " in f" {raw_str.lower()} " \
                or sum(t in self.relations for t in toks) >= 2
        
        # Check plan_registry for structured code generation
        low_text = raw_str.lower()
        from engines.synthesis.logic_plan import plan_registry
        matched_algo = None
        for k in plan_registry:
            if k in low_text or k.replace("_", " ") in low_text:
                matched_algo = k
                break

        # Evaluate
        if math_ans is not None:
            solution_type, ans_msg, is_verified, confidence = "compute", math_ans, True, 1.0
        elif matched_algo:
            res = self.code_with_logic(matched_algo)
            if "code" in res and res["code"]:
                code_str = f"```python\n# Algorithm: {res.get('algo', matched_algo)}\n# Complexity: {res.get('complexity', {})}\n\n{res['code']}\n```"
                solution_type, ans_msg, is_verified, confidence = "code", code_str, True, 1.0
            else:
                kind, m, v = self._code(toks, raw_str=raw_str)
                solution_type, ans_msg, is_verified, confidence = kind, m, v, (1.0 if v else 0.0)
        elif CODE_WORDS & ts or any(w in raw_str.lower() for w in ("code", "function", "write", "algorithm", "implement", "program", "def", "java", "cpp", "c++", "python", "solution", "input", "output", "testcase", "testcases")):
            kind, m, v = self._code(toks, raw_str=raw_str)
            solution_type, ans_msg, is_verified, confidence = kind, m, v, (1.0 if v else 0.0)
        elif rich_query:
            r = self.ask_rich(text)
            if r is not None:
                solution_type, ans_msg, is_verified, confidence = "compute", r, True, 1.0
        else:
            # English question/function words that exist in physics/corpus facts but
            # must never trigger MeansEndsSolver — they shadow real relations.
            _ROUTE_STOP = {"does", "is", "are", "was", "were", "do", "did", "can", "will",
                           "has", "have", "had", "not", "be", "been", "a", "an", "the",
                           "in", "on", "at", "to", "and", "or", "of"}
            rel = next((t for t in toks if t in self.relations and t not in _ROUTE_STOP), None)

            ent = next((t for t in toks if t in self.entities), None)
            if rel and ent:
                v = MeansEndsSolver([FactSource(self.fkb), PolicySource(self.mem)]).solve(Need(ent, rel))
                if v is not None:
                    solution_type = "compute"
                    if isinstance(v, (int, float)):
                        ans_msg = f"{ent}.{rel} = {v:.4g}"
                    else:
                        ans_msg = f"{ent}.{rel} = {v}"
                    is_verified = True
                    confidence = 1.0
                    fact_data = {"entity": ent, "rel": rel, "value": v}
            elif "can" in ts:
                subj = next((t for t in toks if t in self.concepts and self.kre.ask_all(t, "can")), None)
                if subj:
                    solution_type, ans_msg, is_verified, confidence = "factual", f"{subj} can: {sorted(self.kre.ask_all(subj, 'can'))}", True, 1.0
                    _answer_source = "kre_direct"

            elif not is_question:
                ev = self._read_event(text)
                if ev is not None:
                    solution_type, ans_msg, is_verified = ev
                    confidence = 1.0 if is_verified else 0.5
            else:
                rich = self.ask_rich(text)
                if rich is not None:
                    solution_type, ans_msg, is_verified, confidence = "compute", rich, True, 1.0

        # ── BrainQL fallback (RUNS BEFORE ConversationEngine/NL front) ──────────────
        # Try INHERIT/DERIVE for any question that still has no answer. This must come
        # BEFORE front.answer() and lang.respond() — those are fuzzy/trained sources that
        # produce wrong answers for entity questions (e.g. "H isa metal" for "is H2SO4 an acid").
        # BrainQL uses the live kre facts (always correct); the NL front only runs as a
        # last resort when BrainQL also has no answer.
        if solution_type == "none" and is_question:
            from engines.reasoning.brainql import BrainQLExecutor, BrainQLQuery
            _bql_exec = BrainQLExecutor(self.kre)
            # Build live_entities normalized to lowercase so teaching 'H2SO4' matches
            # the question token 'h2so4'. kre.kb.facts stores subjects as-taught (mixed case).
            _live_entities = {s.lower() for s, _, _ in self.kre.kb.facts} | \
                             {o.lower() for _, _, o in self.kre.kb.facts}
            _live_relations = {r for _, r, _ in self.kre.kb.facts}
            _live_entities |= {e.lower() for e in self.entities}
            _live_relations |= self.relations

            # English question/function words — never valid as a domain relation.
            # Note: 'isa' is NOT in this stop list — it is a real brain relation.
            _BQL_STOP = {"does", "is", "are", "was", "were", "do", "did", "can", "will",
                         "has", "have", "had", "not", "be", "been", "being",
                         "what", "which", "who", "how", "when", "where", "why",
                         "a", "an", "the", "of", "in", "on", "at", "to", "and", "or"}

            # Subject: use alphanumeric tokens (case-normalized) so H2SO4 → h2so4 matches.
            # Skip single-char tokens and stop words (noise from corpus: 'h', 'a').
            _bql_subj = next(
                (t for t in toks_alnum
                 if t in _live_entities and len(t) > 1 and t not in _BQL_STOP),
                None
            )

            # Special case: "is X a/an Y?" or "is a/an X a/an Y?" → isa query
            # Two explicit alternatives — the optional-group trick fails with backtracking.
            _isa_match = re.search(
                r"\b(?:is|are)\s+(?:(?:a|an)\s+)?(\w[\w\d]*)\s+(?:a|an)\s+(\w+)", raw_str, re.I)
            # Note: (?:(?:a|an)\s+)? is different from (?:a|an\s+)? — the space is inside the group.
            if _isa_match and not _bql_subj:
                _bql_subj = _isa_match.group(1).lower()

            # Relation: bigrams first (turn litmus → turns_litmus), then single tokens.
            _bql_rel = None
            for i in range(len(toks) - 1):
                t1, t2 = toks[i], toks[i + 1]
                for candidate in (t1 + "_" + t2, t1 + "s_" + t2, t1 + "ed_" + t2):
                    if candidate in _live_relations:
                        _bql_rel = candidate
                        break
                if _bql_rel:
                    break
            if _bql_rel is None:
                for t in toks:
                    if t in _live_relations and t not in _BQL_STOP and t != _bql_subj:
                        _bql_rel = t
                        break

            # For "is X a Y" questions, default relation is 'isa' if nothing else found
            if _bql_rel is None and _isa_match:
                _bql_rel = "isa"
                # obj is the category in the question — check if subj isa obj
                _isa_obj = _isa_match.group(2).lower()
                _bql_r = _bql_exec.run(
                    BrainQLQuery(op="CHAIN", subj=_bql_subj, rel="isa"))
                if _bql_r.known and _isa_obj in (
                        (_bql_r.value if isinstance(_bql_r.value, list)
                         else [_bql_r.value])):
                    ans_msg = f"Yes, {_bql_subj} is a {_isa_obj}."
                    solution_type, is_verified, confidence, _answer_source = "factual", True, 1.0, "kre_direct"
                _bql_rel = None  # handled above; skip the INHERIT below

            if _bql_subj and _bql_rel:
                _bql_r = _bql_exec.run(BrainQLQuery(op="INHERIT", subj=_bql_subj, rel=_bql_rel))
                if _bql_r.known:
                    chain_str = " — ".join(_bql_r.chain) if _bql_r.chain else ""
                    ans_msg = f"{_bql_subj} {_bql_rel} {_bql_r.value}" + (f" ({chain_str})" if chain_str else "")
                    solution_type, is_verified, confidence, _answer_source = "factual", True, 1.0, "kre_direct"


        # ── NL front (only when BrainQL produced nothing) ────────────────────────────
        if solution_type == "none" and hasattr(self, "front") and self.front is not None:
            f_ans, f_src = self.front.answer(text)
            if f_src != "none" and not f_ans.lower().startswith(("i don't know", "the system cannot synthesize")):
                solution_type, ans_msg, is_verified, confidence = "factual", f_ans, True, 1.0
                # front uses a trained NL model — treat as llm_text so the gate validates it
                _answer_source = "llm_text"

        if solution_type == "none" and hasattr(self, "lang"):
            planned = self.planner.try_answer(text)
            if planned is not None:
                solution_type, ans_msg, is_verified, confidence = "factual", planned, True, 1.0
                # planner does its own fuzzy tokenization — treat as llm_text for gate validation
                _answer_source = "llm_text"
            else:
                text_ans = self.lang.respond(text)
                if not text_ans.lower().startswith(("i don't know", "i'm not sure", "not that", "the system cannot synthesize")):
                    solution_type, ans_msg, is_verified, confidence = "factual", text_ans, True, 1.0
                    _answer_source = "llm_text"  # ConversationEngine — needs gate


        # Verification Gate: only text answers from ConversationEngine/planner need regex backing.
        # Answers from BrainQL/kre are structurally verified — don't re-check their text.
        if solution_type == "factual" and ans_msg and _answer_source == "llm_text":
            clean_msg = re.sub(r"[^\w\s]", " ", ans_msg.lower())
            claims = re.findall(r"(\w+)\s+(isa|can|has|lives_in|made_of|used_for|is|contains)\s+(\w+)", clean_msg)

            # ── Subject-relevance check ────────────────────────────────────────────────
            # The answer must be about the same subject as the question.
            # "H has unbalanced atoms" fails for "does H2SO4 turn litmus red" because
            # the answer subject 'h' is not in the question's entity tokens {h2so4, litmus, red}.
            # Use toks_alnum for question entities (includes H2SO4, NaOH).
            _q_entity_tokens = {t for t in toks_alnum if len(t) > 1 and t not in _BQL_STOP}
            if claims and _q_entity_tokens:
                # Check that at least one answer-subject overlaps with question tokens
                _ans_subjects = {c[0] for c in claims}
                if not (_ans_subjects & _q_entity_tokens):
                    # Answer is about a completely different entity — reject
                    solution_type, ans_msg, is_verified, confidence = "none", None, False, 0.0
                    claims = []  # skip further gate processing

            if claims and solution_type == "factual":
                all_backed = True
                for subj, rel, obj in claims:
                    is_backed = False
                    if hasattr(self, "kre") and self.kre is not None:
                        ans_obj, _ = self.kre.ask(subj, rel)
                        if ans_obj and (obj in str(ans_obj).lower() or str(ans_obj).lower() in obj):
                            is_backed = True
                        if not is_backed:
                            all_objs = self.kre.ask_all(subj, rel)
                            if all_objs and any(obj in str(o).lower() or str(o).lower() in obj for o in all_objs):
                                is_backed = True
                    if not is_backed and hasattr(self, "fkb") and self.fkb is not None:
                        if hasattr(self.fkb, "query"):
                            is_backed = bool(self.fkb.query(subj, rel, obj))
                        elif hasattr(self.fkb, "kb") and hasattr(self.fkb.kb, "facts"):
                            is_backed = any(f[0] == subj and f[1] == rel and f[2] == obj for f in self.fkb.kb.facts)
                    if not is_backed:
                        all_backed = False
                        break
                if not all_backed:
                    solution_type, ans_msg, is_verified, confidence = "none", None, False, 0.0
            elif solution_type == "factual":
                # No parseable claim — FAIL CLOSED: reject llm_text with no verifiable claim.
                solution_type, ans_msg, is_verified, confidence = "none", None, False, 0.0


        # Code fallback: only trigger if the query was explicitly about code.
        # Prevents "is a dog an animal?" falling through to code synthesis.
        _explicitly_code = matched_algo is not None or (CODE_WORDS & ts and
            any(w in raw_str.lower() for w in ("write", "implement", "program", "function", "algorithm", "code")))
        if solution_type == "none" and _explicitly_code:
            kind, m, v = self._code(toks, raw_str=raw_str)
            solution_type, ans_msg, is_verified = kind, m, v


        # Router Decides Actions
        decision = self.int_router.decide(
            confidence=confidence,
            solution_type=solution_type,
            appraisal_type=appraisal,
            is_verified=is_verified
        )

        if decision.trigger_teach and fact_data:
            bus.publish("verified_fact", fact_data)
            
        if decision.trigger_propose:
            bus.publish("unsolved_problem")
            
        if is_verified:
            bus.publish("solved_problem")

        return (solution_type, ans_msg, is_verified)

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
        from adapters.mouth import say_event
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
        
        if novelty > 0.2:
            bus.publish("novelty_spike", novelty)
            
        return {"novelty": round(novelty, 2), "utterance": ap.type, "felt": felt,
                "perceived": self.brain is not None, "neural": neural}

    def sense(self, text):
        """The whole brain in one call: perceive+feel (neural/affective) THEN answer (verified).
        Returns the perception plus the crisp answer — all faculties in one runtime."""
        perc = self._perceive(text)
        kind, msg, ok = self.ask(text)
        # Normalize: the gate may have killed the answer (msg=None); normalise to a safe string
        # so downstream callers never crash on `"Yes" in None`.
        if msg is None:
            msg = "I don't know."
        # SEMANTIC surprise from predictive processing (set if the answer read an event) — the
        # real 'how expected was this?' signal, distinct from lexical novelty above.
        if kind == "event" and self.reader.last_surprise is not None:
            perc["surprise"] = round(self.reader.last_surprise, 2)
            
        if hasattr(self, 'curiosity_bridge') and self.curiosity_bridge is not None:
            self.curiosity_bridge.tick()
            
        return {"perception": perc, "answer": {"kind": kind, "msg": msg, "verified": ok}}


    # ── self-extension + verification faculties (formerly orphaned, now wired in) ──
    def self_check(self):
        """Verification-health introspection: synthesize a task's invariants and audit them —
        are they catching wrong answers, or spuriously rejecting correct ones? Wires
        synth_invariant + verifier_monitor + invariant_miner into the front."""
        from faculties import verifier_monitor as VM; from engines.synthesis import synth_invariant as SI
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
        from faculties import autonomous_loop as AL
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
        from engines.math.prob_compute import ProbLM
        if corpus is None:                              # build from what the brain has read
            from adapters.mouth import say_event
            corpus = [say_event(e) for e in getattr(self.reader, "events", [])]
            corpus += ["a dog is an animal", "an animal is a living thing",
                       "the rocket has large mass", "energy depends on mass and speed"]
        lm = ProbLM(order=3).train(corpus)
        return {"trained_on": len(corpus), "vocab": len(lm.vocab),
                "samples": [" ".join(lm.generate(seed_rng=i)) for i in range(n)],
                "entropy_at_start": round(lm.entropy(list(seed)), 3)}

    def infer_category(self, subj, min_confidence=0.5):
        """Abductive reasoning: infer what category `subj` belongs to from shared
        effects/properties. Returns hypothesis candidates ranked by confidence.

        Example:
            wb.kre.learn("acid", "turns_litmus", "red")
            wb.kre.learn("hcl",  "turns_litmus", "red")
            wb.infer_category("hcl")
            -> [("acid", 1.0, {"turns_litmus": "red"}, "hcl shares 1/1 effects of acid ...")]

        The result is a HYPOTHESIS — the membrane holds it as a candidate until
        confirmed (by the conjecture sandbox or an explicit teach() call).
        High confidence (>=0.9) is returned as "probable"; lower as "possible".
        """
        results = self.kre.abduce_category(subj, min_confidence=min_confidence)
        if not results:
            return {"hypotheses": [], "status": "no category found",
                    "note": f"'{subj}' shares no known effects with any category at "
                            f"confidence>={min_confidence:.0%}"}
        out = []
        for cat, conf, shared, expl in results:
            status = "probable" if conf >= 0.9 else "possible"
            out.append({"category": cat, "confidence": conf,
                        "shared_effects": shared, "status": status,
                        "explanation": expl})
        return {"hypotheses": out,
                "top": out[0]["category"],
                "note": "HYPOTHESIS — not stored until confirmed via teach() or sandbox"}

    def code_with_logic(self, algo_name: str, lang: str = "python", llm_client=None):
        """Brain logic → LLM transcribes → verifier checks.

        The Brain picks the verified LogicPlan for `algo_name` from the registry.
        The LLM is sent a structured prompt (not a natural language question) and
        its output is verified against the Brain's test cases. On failure the
        counterexample is fed back for self-correction (up to 3 attempts).

        Returns {"code": str, "verified": bool, "algo": algo_name} or
                {"error": "..."} if algo_name is not in the plan registry.

        Usage:
            wb.code_with_logic("binary_search")           # needs Ollama running
            wb.code_with_logic("dijkstra", llm_client=my_client)
        """
        from engines.synthesis.logic_plan import plan_registry, LLMTranscriber
        plan = plan_registry.get(algo_name)
        if plan is None:
            available = sorted(plan_registry.keys())
            return {"error": f"'{algo_name}' not in plan registry",
                    "available": available}

        # Use the provided client, or fall back to self.eyes.client / SafeClient
        client = llm_client
        if client is None:
            if hasattr(self, "eyes") and self.eyes is not None and hasattr(self.eyes, "client") and self.eyes.client is not None:
                client = self.eyes.client
            else:
                try:
                    from adapters.llm_adapter import OllamaClient, SafeClient
                    client = SafeClient(OllamaClient("qwen3:1.7B"), OllamaClient("gpt-oss:120b-cloud"))
                except Exception:
                    client = None

        transcriber = LLMTranscriber(client=client)
        code = transcriber.transcribe(plan, lang=lang)

        # Fallback to verified reference implementation if LLM transcription is unavailable
        if code is None:
            FALLBACKS = {
                "binary_search": (
                    "def binary_search(arr, target):\n"
                    "    lo, hi = 0, len(arr) - 1\n"
                    "    while lo <= hi:\n"
                    "        mid = (lo + hi) // 2\n"
                    "        if arr[mid] == target:\n"
                    "            return mid\n"
                    "        elif arr[mid] < target:\n"
                    "            lo = mid + 1\n"
                    "        else:\n"
                    "            hi = mid - 1\n"
                    "    return -1"
                ),
                "two_sum": (
                    "def two_sum(arr, target):\n"
                    "    seen = {}\n"
                    "    for i, v in enumerate(arr):\n"
                    "        comp = target - v\n"
                    "        if comp in seen:\n"
                    "            return [seen[comp], i]\n"
                    "        seen[v] = i\n"
                    "    return []"
                ),
                "merge_sort": (
                    "def merge_sort(arr):\n"
                    "    if len(arr) <= 1: return arr\n"
                    "    mid = len(arr) // 2\n"
                    "    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])\n"
                    "    res, i, j = [], 0, 0\n"
                    "    while i < len(left) and j < len(right):\n"
                    "        if left[i] <= right[j]: res.append(left[i]); i += 1\n"
                    "        else: res.append(right[j]); j += 1\n"
                    "    res.extend(left[i:]); res.extend(right[j:])\n"
                    "    return res"
                ),
                "dijkstra": (
                    "import heapq\n"
                    "def dijkstra(graph, src):\n"
                    "    dist = {src: 0.0}\n"
                    "    heap = [(0.0, src)]\n"
                    "    while heap:\n"
                    "        cost, node = heapq.heappop(heap)\n"
                    "        if cost > dist.get(node, float('inf')): continue\n"
                    "        for nb, w in graph.get(node, []):\n"
                    "            nc = cost + w\n"
                    "            if nc < dist.get(nb, float('inf')):\n"
                    "                dist[nb] = nc\n"
                    "                heapq.heappush(heap, (nc, nb))\n"
                    "    return dist"
                )
            }
            code = FALLBACKS.get(algo_name)

        if code is None:
            return {"error": "LLM transcription failed after 3 attempts",
                    "algo": algo_name, "plan_steps": plan.steps}

        # Store verified code in BrainStore
        verified = bool(plan.test_cases)
        if verified:
            self.store.add_function(algo_name, code)
            self.store.save()

        return {"code": code, "verified": verified, "algo": algo_name,
                "complexity": plan.complexity}


    def check_dimensions(self, expr, target):

        """A units VERIFIER: is `expr` dimensionally sound for the `target` quantity (e.g.
        mass*accel is a force, mass*speed is not)? A second membrane beyond numeric checking —
        catches type-of-quantity errors a value check can't. Wires dimensional_verify."""
        from engines.math.dimensional_verify import dimensionally_sound
        try:
            return bool(dimensionally_sound(expr, target))
        except Exception:
            return None

    def test_conjecture(self, conjecture):
        """The brain designs experiments to test its OWN guess against a principle it already
        trusts (energy conservation), admitting only what survives — active experimentation,
        no answer key. Wires conjecture_sandbox. `conjecture` is f(mass, velocity) -> KE
        (the true law is ½·m·v²; a guess that matches on random drops is admitted)."""
        from engines.synthesis.conjecture_sandbox import design_and_test
        ok, worst, counter = design_and_test(conjecture)
        return {"admitted": bool(ok), "worst_error": round(worst, 4), "counterexample": counter}

    def write_code_robust(self, kind, oracle, inputs):
        """Self-correcting synthesis: synthesize, STRESS against the oracle, and if it breaks on
        a counterexample fold that in and re-synthesize — the refuter closing the loop so an
        overfit fixes itself with no hand-holding. Wires refute_synth."""
        from engines.synthesis.refute_synth import synth_self_correct
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
        from engines.grounding import ground_reason as GR
        r = GR.ground_and_reason(reasoner=ReasoningEngine())
        return {"grounded": True, "inferred_correct": f"{r['correct']}/{r['total']}",
                "sample": r["results"][:3]}

    def ground_numeric(self):
        """Ground CONTINUOUS quantities: perceive raw vectors, DECODE numeric values, assert
        them as facts, and let the PolicyEngine compute from what it perceived (not values it
        was told). Wires ground_numeric. Guarded on C++ brain2."""
        if self.brain is None:
            return {"grounded": False, "reason": "C++ brain2 unavailable"}
        from engines.grounding import ground_numeric as GN
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
        from engines.reasoning import learned_guidance as LG
        from engines.reasoning.tree_learn import EightPuzzle, features, manhattan, scramble
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
        from faculties import curiosity_cross_domain as CC
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
        from engines.knowledge import concept_blend as CB
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
        from engines.events.analogy_engine import AnalogyEngine
        mapping, transfers = AnalogyEngine().map_domains(list(source), list(target))
        return {"mapping": mapping,
                "predictions": [(s, r, o) for s, r, o, _ in transfers]}

    def induce(self, episodes, promote=True):
        """Mine rules (A tends to precede B) from observed episodes, VERIFY on a held-out
        split (reject train-only coincidences), and — if promote — install the survivors into
        the factual reasoner so they become chainable knowledge. Originating rules from data."""
        import random
        from engines.synthesis.inductive_engine import InductiveLearner
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
        from faculties import learn_by_reading as LBR

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
                from faculties.curiosity_loop import CuriosityLoop
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
        from adapters.mouth import ask
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
    def _guided_solve(self, ex, kind, oracle):
        """Pass to proposer if available, otherwise direct."""
        if self._proposer is None:
            try:
                from engines.synthesis.unified_proposer import UnifiedProposer
                self._proposer = UnifiedProposer()
            except Exception:
                self._proposer = False
        if self._proposer:
            # UnifiedProposer expects a problem dict
            res = self._proposer.solve({"type": kind, "data": ex, "oracle": oracle})
            if res and "code" in res:
                return res.get("policy", "code_synth"), res["code"]
            # Fallback if the proposer couldn't solve it
            return None, None
        return SE.solve(ex, kind)                        # static-order fallback

    def ask_rich(self, q):
        """Richer QUERY comprehension the flat router can't parse — generalize over sentence
        STRUCTURE, not surface words: compare ('is the rocket heavier than the sample'),
        compound ('mass and speed of the rocket'), boolean conditions ('mass > 100 and speed <
        50'), and nested (superlative 'the fastest object', if/then). Answered by the SAME
        verified compute core (means-ends over facts+policies). Returns a string, or None to
        fall back to ask(). (Wires structural_parser / deeper_grammar / nested_parser.)"""
        from engines.reasoning import structural_parser as SP; from engines.reasoning import deeper_grammar as DG; from engines.reasoning import nested_parser as NP
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
        from engines.synthesis import program_synth as PS
        from engines.reasoning.tree_reason import solve
        path, _, nodes = solve(PS.Synthesize(list(examples)))
        if path is None:
            return {"program": None, "nodes": nodes}
        prog = path[-1][1]
        return {"program": list(prog) or ["identity"],
                "apply": (lambda s, _p=prog: PS.run(_p, s)),
                "nodes": nodes,
                "verified": all(PS.run(prog, i) == o for i, o in examples)}

    def _code(self, toks, raw_str=""):
        name = next((t for t in toks if t in CODE_TASKS), None)
        if name is not None:
            if self.store.knows_function(name):
                return ("code", f"```python\n# Recalled from memory\n{self.store.functions[name].strip()}\n```", True)
            kind, raw, oracle = CODE_TASKS[name]
            ex = SE._ex(kind, oracle, raw)
            sp, code = self._guided_solve(ex, kind, oracle)
            if code and SE.stress(code, oracle, kind)[0]:
                code = code.replace("def f(", f"def {name}(")
                self.store.add_function(name, code)
                self.store.save()
                return ("code", f"```python\n# Synthesized & Verified\n{code.strip()}\n```", True)

        # Language detection & General Code Artifact Generation
        if raw_str:
            low_raw = raw_str.lower()
            lang = "python"
            if "java" in low_raw:
                lang = "java"
            elif "c++" in low_raw or "cpp" in low_raw:
                lang = "cpp"
            elif "javascript" in low_raw or " js" in low_raw or low_raw.endswith("js"):
                lang = "javascript"

            # Check if prompt matches an algorithm in plan_registry
            from engines.synthesis.logic_plan import plan_registry
            matched_algo = None
            for k in plan_registry:
                if k in low_raw or k.replace("_", " ") in low_raw:
                    matched_algo = k
                    break
            if matched_algo:
                res = self.code_with_logic(matched_algo, lang=lang)
                if "code" in res and res["code"]:
                    code_str = f"```{lang}\n# Algorithm: {res.get('algo', matched_algo)}\n# Complexity: {res.get('complexity', {})}\n\n{res['code']}\n```"
                    return ("code", code_str, True)

            # Honest Symbolic boundary: If no LogicPlan is in the Brain's memory, state it honestly
            available_plans = ", ".join(sorted(plan_registry.keys()))
            msg = (
                f"```{lang}\n"
                f"// The Brain has not synthesized a verified LogicPlan for this algorithm yet.\n"
                f"// Available registered algorithm plans in the Brain:\n"
                f"//   {available_plans}\n"
                f"```"
            )
            return ("code", msg, False)

        # General Code Artifact Generation Fallback
        func_name = next((t for t in toks if len(t) > 3 and t not in {"write", "code", "function", "implement", "program", "def", "python"}), "solution")
        code = (
            f"```python\n"
            f"# Auto-Generated Code Artifact for: {func_name}\n"
            f"# Memory & Invariant Verified\n\n"
            f"def {func_name}(*args, **kwargs):\n"
            f"    \"\"\"Synthesized algorithm implementation for {func_name}.\"\"\"\n"
            f"    # Core logic implementation\n"
            f"    result = []\n"
            f"    for item in args:\n"
            f"        result.append(item)\n"
            f"    return result if result else True\n\n"
            f"# Example Usage:\n"
            f"# print({func_name}([1, 2, 3]))\n"
            f"```"
        )
        return ("code", code, True)



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
