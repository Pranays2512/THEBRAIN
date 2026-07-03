#!/usr/bin/env python3
"""brain_data.py — ingest the tagged training corpus (see docs/kimi_data_prompt.txt) and feed
every subsystem from ONE file. The bidirectional-template insight as a data pipeline:

  FACT / LAW  -> the symbolic brain (knowledge_distill: facts taught, laws verified)
              -> the student LM corpus (sentence => structure parse pairs)
  EVENT       -> event pairs (parser target + the mouth's say_event target)
              -> the predictor (ordered events -> learned verb transitions)
  ISA         -> the type oracle (grounds selectional types)
  MORPH       -> the mouth's morphology table (child-grade -> fluent, by LEARNING not an LLM)
  SEQ         -> the predictive loop (coherent event streams) + raw text for the C++ Brain

Line grammars (one per line):
  <sentence> => FACT: obj | prop | num
  <sentence> => LAW: quantity = expression
  <sentence> => EVENT: verb | agent | patient | tense | polarity(+/-)
  ISA: child | parent
  MORPH: lemma | past | present3sg
  SEQ: <sentence>
"""

from event_form import Event, POS, NEG


class BrainData:
    def __init__(self):
        self.facts = []          # (obj, prop, num_str)   [FACT]
        self.laws = []           # structure string "LAW: q = expr"   (fed to knowledge_distill)
        self.structs = []        # all right-hand structures (for knowledge_distill.parse_teacher)
        self.parse_pairs = []    # "sentence => structure"  (student LM corpus)
        self.events = []         # (sentence, Event)   [EVENT]
        self.isa = []            # (child, parent)   [ISA]
        self.morph = {}          # lemma -> {"past","present3sg"}   [MORPH]
        self.sequences = []      # [sentence]   [SEQ]
        self.units = {}          # prop -> (M,L,T)   [UNIT]  (dimensional verifier)
        self.questions = []      # (sentence, {"entity","rel"})   [ASK]
        self.chains = []         # documented multi-step derivations   [CHAIN]

    # ── parse ────────────────────────────────────────────────────────────────
    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            return cls.parse(f.read())

    @classmethod
    def parse(cls, text):
        d = cls()
        for raw in text.splitlines():
            ln = raw.strip()
            if not ln:
                continue
            if ln.startswith("ISA:"):
                p = [x.strip() for x in ln[4:].split("|")]
                if len(p) == 2 and all(p):
                    d.isa.append((p[0], p[1]))
            elif ln.startswith("UNIT:"):
                p = [x.strip() for x in ln[5:].split("|")]
                if len(p) == 2 and p[0]:
                    try:
                        d.units[p[0]] = tuple(int(x) for x in p[1].split(","))
                    except ValueError:
                        pass
            elif ln.startswith("CHAIN:"):
                d.chains.append(ln[6:].strip())
            elif ln.startswith("MORPH:"):
                p = [x.strip() for x in ln[6:].split("|")]
                if len(p) == 3 and p[0]:
                    d.morph[p[0]] = {"past": p[1], "present3sg": p[2]}
            elif ln.startswith("SEQ:"):
                s = ln[4:].strip()
                if s:
                    d.sequences.append(s)
            elif "=>" in ln:
                left, right = (x.strip() for x in ln.split("=>", 1))
                d._parse_pair(left, right)
        return d

    def _parse_pair(self, sentence, struct):
        if struct.startswith("FACT:"):
            p = [x.strip() for x in struct[5:].split("|")]
            if len(p) == 3 and all(p):
                self.facts.append((p[0], p[1], p[2]))
                self.structs.append(struct)
                self.parse_pairs.append("%s => %s" % (sentence.lower(), struct))
        elif struct.startswith("LAW:"):
            self.laws.append(struct)
            self.structs.append(struct)
            self.parse_pairs.append("%s => %s" % (sentence.lower(), struct))
        elif struct.startswith("ASK:"):
            p = [x.strip() for x in struct[4:].split("|")]
            if len(p) == 2 and all(p):
                self.questions.append((sentence, {"entity": p[0], "rel": p[1]}))
        elif struct.startswith("EVENT:"):
            p = [x.strip() for x in struct[6:].split("|")]
            if len(p) == 5 and p[0]:
                verb, agent, patient, tense, pol = p
                ev = Event(verb, agent or None, patient or None, tense or "present",
                           NEG if pol == "-" else POS)
                self.events.append((sentence, ev))
                self.parse_pairs.append("%s => %s" % (sentence.lower(), struct))

    # ── feed each subsystem ────────────────────────────────────────────────────
    def type_oracle(self):
        """A TypeOracle grounded in the ISA chains (parents-first handled by closure build)."""
        from type_oracle import TypeOracle
        return TypeOracle(triples=[(c, "isa", p) for c, p in self.isa])

    def load_morph(self):
        """Populate the mouth's learned morphology table (upgrades say_event fluency)."""
        import mouth
        mouth.MORPH.update(self.morph)
        return len(self.morph)

    def _law_dim(self, law_struct):
        """dim_consistent for a 'LAW: t = expr' string, using the UNITs. True/False/None."""
        from domain_features import dim_consistent
        from means_ends import Policy
        import knowledge_distill as KD
        try:
            target, expr = (x.strip() for x in law_struct[4:].split("=", 1))
            tree = KD.infix_to_tree(expr)
            return dim_consistent(Policy(target, tuple(sorted(_symbols(tree))), tree), self.units)
        except Exception:
            return None

    def teach_knowledge(self, fkb, mem):
        """Teach the symbolic brain: DIMENSIONAL filter first (drop provably-nonsense laws using
        the UNITs — the hard filter runs before the compute gate), then the tested distiller
        (facts learned, remaining laws verified numerically). Two verifiers, not one."""
        import knowledge_distill as KD
        structs, dim_dropped = [], 0
        for s in self.structs:
            if s.startswith("LAW:") and self.units and self._law_dim(s) is False:
                dim_dropped += 1                   # dimensionally impossible -> never even compute
                continue
            structs.append(s)
        f, l, _ = KD.parse_teacher("\n".join(structs))
        adm, rej = KD.teach(fkb, mem, f, l)
        return {"facts": len(f), "laws_admitted": len(adm), "laws_rejected": len(rej),
                "dim_dropped": dim_dropped}

    def train_predictor(self, predictor):
        """Feed the ordered EVENT stream so the predictor learns verb transitions."""
        prev = None
        for _, ev in self.events:
            predictor.learn(prev, ev)
            prev = ev
        return {"events": len(self.events), "verbs": len(predictor.base)}

    def entities(self):
        return {c for c, _ in self.isa} | {ev.agent for _, ev in self.events if ev.agent} \
            | {ev.patient for _, ev in self.events if ev.patient}

    def verbs(self):
        return {ev.verb for _, ev in self.events}

    def learn_verb_constraints(self, oracle, promote_at=2):
        """Train the event membrane FROM DATA: watch every event, induce each verb's selectional
        constraint (agent/patient types) via the conjecture->verify->admit VerbLearner. Replaces
        hand-set constraints with learned ones."""
        from verb_learn import VerbLearner
        vl = VerbLearner(oracle, promote_at=promote_at)
        for _, ev in self.events:
            vl.observe(ev)
        vl.acquire()
        return vl.constraints

    def learn_questions(self, entities):
        """Train question understanding: learn question templates from ASK pairs (the same
        conjecture->verify->admit as statement templates, held out to prove generalization)."""
        from template_memory import TemplateMemory
        if len(self.questions) < 2:
            return None, 0
        tm = TemplateMemory(entities=set(entities))
        ex = [(s, {"entity": q["entity"], "rel": q["rel"], "value": 0}) for s, q in self.questions]
        n = tm.learn_question(ex[:-1], holdout=ex[-1:])
        return tm, n

    def dim_report(self):
        """Run the DIMENSIONAL verifier (domain_features) over every law using the UNITs —
        consistent / violation (nonsense) / unknown (missing units -> abstain, never prune)."""
        if not self.units:
            return None
        res = {"consistent": 0, "violation": 0, "unknown": 0}
        for law in self.laws:
            r = self._law_dim(law)
            res["consistent" if r is True else "violation" if r is False else "unknown"] += 1
        return res

    def report(self):
        return {"facts": len(self.facts), "laws": len(self.laws), "events": len(self.events),
                "isa": len(self.isa), "morph": len(self.morph), "sequences": len(self.sequences),
                "units": len(self.units), "questions": len(self.questions),
                "chains": len(self.chains), "parse_pairs": len(self.parse_pairs)}


def _symbols(tree):
    """Variable names in an expression tree (for dimensional analysis)."""
    if isinstance(tree, str):
        return {tree}
    if isinstance(tree, (int, float)):
        return set()
    out = set()
    for x in tree[1:]:
        out |= _symbols(x)
    return out
