#!/usr/bin/env python3
"""read_book.py — hand the brain a BOOK; it reads AND trains itself on it.

Not just a parser: the orchestrator that composes the reading + reasoning + membrane +
self-training faculties into one "ingest a book" loop, and LOGS what each module
contributed. The membrane holds throughout — the LLM only PROPOSES (grounded triples);
the crisp faculties dispose and own the truth.

  raw text ─► sentence stream
      │
      ├─ StringTemplateCache.parse  ── hit ─►  triple            (NO LLM — learned pattern)
      │        │ miss
      │        └─► teacher(sentence) [LLMExtractor] ─► triple ─► learn the template
      │                                                            (next time: NO LLM)
      ▼
   route each triple:
      numeric value  ─►  numeric core (SimpleKB)                 [module: numeric_core]
      string  value  ─►  ReasoningEngine.learn (isa/has/can)     [module: reasoning_engine]
   plus:  EventReader.read(sentence)  ─► membrane admit/reject/abstain  [module: event_membrane]

  self-training pass (consolidation, no new text):
      type closure   ─►  multi-hop facts DERIVED, never stated   [module: type_closure]
      inductive_engine ► rules mined + verified from co-occurrence [module: inductive_engine]
      verb_learn     ─►  selectional constraints from events      [module: verb_learner]

The headline number is the DECAY curve: LLM-calls per batch, which falls as the template
cache learns the book's recurring sentence shapes. It never hits zero for novel structure
(the autoformalization wall) — but it falls, and the fall is the claim.

    venv2/bin/python3 read_book.py                              # offline proof (stub)
    venv2/bin/python3 read_book.py data/raw_ssc9.txt qwen3:1.7B # real: Ollama teacher
"""
import os
import re
import sys
from collections import Counter, defaultdict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from core.reasoning.reasoning_engine import ReasoningEngine
from llm_extractor import LLMExtractor


def sentences(text, max_sents=None):
    text = re.sub(r"\s+", " ", text)
    out = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)]
    out = [s for s in out if len(s.split()) >= 3 and any(c.isalpha() for c in s)]
    return out[:max_sents] if max_sents else out


def _toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _num(v):
    try:
        float(v); return True
    except (TypeError, ValueError):
        return False


class StringTemplateCache:
    """The 'learn to read like a child' lane: after the teacher labels a sentence with a
    triple, remember its SHAPE (the sentence tokens with the subject/object blanked to
    slots). A later sentence matching a known shape is parsed LOCALLY into a triple — no
    LLM. This is what makes the brain need the teacher LESS as it reads."""

    def __init__(self):
        self.templates = {}                 # pattern(tuple) -> (rel, subj_pos, obj_pos)

    def _pattern(self, toks, s_tok, o_tok):
        pat, sp, op = [], None, None
        for i, t in enumerate(toks):
            if t == s_tok and sp is None:
                pat.append("\0SUBJ"); sp = i
            elif t == o_tok and op is None:
                pat.append("\0OBJ"); op = i
            else:
                pat.append(t)
        return (tuple(pat), sp, op) if sp is not None and op is not None else (None, None, None)

    def learn(self, sentence, s, r, o):
        toks = _toks(sentence)
        st = _toks(s)[-1] if _toks(s) else None      # last word of a multiword name
        ot = _toks(o)[-1] if _toks(o) else None
        if not st or not ot or st not in toks or ot not in toks:
            return
        pat, sp, op = self._pattern(toks, st, ot)
        if pat is not None:
            self.templates.setdefault(pat, (r, sp, op))

    def parse(self, sentence):
        toks = _toks(sentence)
        for pat, (r, sp, op) in self.templates.items():
            if len(pat) != len(toks) or sp >= len(toks) or op >= len(toks):
                continue
            if all(p in ("\0SUBJ", "\0OBJ") or p == toks[i] for i, p in enumerate(pat)):
                return toks[sp], r, toks[op]
        return None


class BookTrainer:
    """Read a book and train every faculty on it, logging each one's contribution."""

    def __init__(self, client=None, with_events=True):
        self.extractor = LLMExtractor(client) if client else None
        self.cache = StringTemplateCache()
        self.kre = ReasoningEngine()                 # string relations (isa/has/can)
        self.numeric = {}                            # (entity, rel) -> float
        self.isa_pairs = set()
        self.contrib = defaultdict(Counter)          # module -> counters
        self.samples = defaultdict(list)             # module -> example outputs
        self.decay = []                              # LLM-calls per batch
        self._stated_isa = set()
        self.with_events = with_events
        self.reader = None
        if with_events:
            try:
                from reading_loop import EventReader
                from core.store.type_oracle import TypeOracle
                from core.events.verb_learn import VerbLearner
                self._oracle = TypeOracle()
                self.reader = EventReader(set(), set(), type_of=self._oracle,
                                          learner=VerbLearner(self._oracle))
            except Exception:
                self.reader = None

    # ── extraction: local template first, LLM only on a miss ────────────────────
    def _triples(self, sentence):
        local = self.cache.parse(sentence)
        if local is not None:
            self.contrib["template_cache"]["grammar_hit"] += 1
            return [local], False                    # parsed WITHOUT the LLM
        if self.extractor is None:
            return [], True
        try:
            triples = self.extractor.extract(sentence)
        except Exception:
            triples = []
        self.contrib["llm_teacher"]["calls"] += 1
        for s, r, o in triples:                      # learn the shape for next time
            self.cache.learn(sentence, s, r, o)
        return triples, True

    def _route(self, s, r, o):
        if _num(o):
            self.numeric[(s, r)] = float(o)
            self.contrib["numeric_core"]["facts"] += 1
            if len(self.samples["numeric_core"]) < 5:
                self.samples["numeric_core"].append(f"{s} {r} {o}")
        else:
            self.kre.learn(s, r, o)
            self.contrib["reasoning_engine"]["facts"] += 1
            if len(self.samples["reasoning_engine"]) < 5:
                self.samples["reasoning_engine"].append(f"{s} {r} {o}")
            if r == "isa":
                self.isa_pairs.add((s, o)); self._stated_isa.add((s, o))

    # ── read the whole book, batch by batch (decay = LLM-calls per batch) ───────
    def read(self, sents, batch=10):
        for i in range(0, len(sents), batch):
            before = self.contrib["llm_teacher"]["calls"]
            for sent in sents[i:i + batch]:
                self.contrib["book"]["sentences"] += 1
                triples, _used_llm = self._triples(sent)
                for t in triples:
                    if len(t) == 3:
                        self._route(*t)
                if self.reader is not None:
                    events, _rel = self.reader.read(sent)   # updates reader.stats + membrane
                    for ev in events:
                        if len(self.samples["event_membrane"]) < 5:
                            self.samples["event_membrane"].append(
                                f"{ev.verb}({ev.agent},{ev.patient})")
            chunk = min(batch, len(sents) - i)
            self.decay.append((self.contrib["llm_teacher"]["calls"] - before) / max(chunk, 1))

    # ── self-training / consolidation: derive what wasn't stated ────────────────
    def consolidate(self):
        # 1. type closure: multi-hop isa facts the book never stated directly
        self.kre.set_transitive("isa")
        derived = 0
        nodes = {s for s, _ in self.isa_pairs} | {o for _, o in self.isa_pairs}
        for n in nodes:
            for anc in self.kre.ask_all(n, "isa"):
                if (n, anc) not in self._stated_isa and n != anc:
                    derived += 1
                    if len(self.samples["type_closure"]) < 6:
                        self.samples["type_closure"].append(f"{n} isa {anc} (derived)")
        self.contrib["type_closure"]["derived_facts"] = derived

        # 2. inductive rules from co-occurrence over the isa chains (verify on holdout)
        try:
            from core.synthesis.inductive_engine import InductiveLearner
            eps = [[s, o] for s, o in self.isa_pairs]
            if len(eps) >= 4:
                cut = max(2, int(0.7 * len(eps)))
                prom, _ = InductiveLearner().mine(eps[:cut], eps[cut:])
                self.contrib["inductive_engine"]["rules_promoted"] = len(prom)
                for r in prom[:5]:
                    self.samples["inductive_engine"].append(f"{r.a} -> {r.b}")
        except Exception:
            pass

        # event membrane tallies (from the reader's own stats)
        if self.reader is not None:
            for k in ("admit", "reject", "abstain", "nomatch"):
                if self.reader.stats.get(k):
                    self.contrib["event_membrane"][k] = self.reader.stats[k]

        # 3. verb constraints from the events the reader admitted
        if self.reader is not None and self.reader.learner is not None:
            try:
                self.reader.learner.acquire()
                c = self.reader.learner.constraints
                self.contrib["verb_learner"]["constraints"] = len(c)
                for v, spec in list(c.items())[:5]:
                    self.samples["verb_learner"].append(f"{v}: {sorted(spec)}")
            except Exception:
                pass

    # ── the report: what each module contributed ────────────────────────────────
    def report(self):
        return {"contrib": {m: dict(c) for m, c in self.contrib.items()},
                "samples": {m: s for m, s in self.samples.items()},
                "decay": self.decay,
                "kb_facts": len(self.numeric) + self.contrib["reasoning_engine"]["facts"]}


def train_on_book(path_or_text, client=None, batch=10, max_sents=80, is_text=False):
    text = path_or_text if is_text else open(path_or_text, errors="ignore").read()
    sents = sentences(text, max_sents)
    bt = BookTrainer(client=client)
    bt.read(sents, batch=batch)
    bt.consolidate()
    rep = bt.report()
    rep["n_sentences"] = len(sents)
    return rep, bt


def _print_report(rep):
    d = rep["decay"]
    print(f"\n  sentences: {rep['n_sentences']}   verified KB facts: {rep['kb_facts']}")
    if d:
        bars = "  ".join(f"{r*100:3.0f}%" for r in d)
        print(f"\n  LLM-need per batch (should FALL as templates are learned):\n    {bars}")
        print(f"    first {d[0]*100:.0f}%  ->  last {d[-1]*100:.0f}%  "
              f"({'FELL' if d[-1] < d[0] else 'flat/rose'})")
    print("\n  per-module contribution:")
    order = ["book", "llm_teacher", "template_cache", "reasoning_engine", "numeric_core",
             "event_membrane", "type_closure", "inductive_engine", "verb_learner"]
    for m in order:
        if m in rep["contrib"]:
            c = rep["contrib"][m]
            line = ", ".join(f"{k}={v}" for k, v in c.items())
            print(f"    {m:18s} {line}")
            for ex in rep["samples"].get(m, [])[:3]:
                print(f"        e.g. {ex}")


def _offline_proof():
    from llm_adapter import StubClient
    book = (
        "The lion is an animal. The tiger is an animal. The sparrow is a bird. "
        "The eagle is a bird. The trout is a fish. The animal is a living_thing. "
        "The bird is an animal. The fish is an animal. The rose is a plant. "
        "The plant is a living_thing. The oak is a plant. The lion has weight 190."
    )
    table = {}
    for s in sentences(book):
        m = re.match(r"the (\w+) is an? (\w+)", s.lower())
        if m:
            table[s] = f'[["{m.group(1)}", "isa", "{m.group(2)}"]]'
        m2 = re.match(r"the (\w+) has weight (\d+)", s.lower())
        if m2:
            table[s] = f'[["{m2.group(1)}", "weight", "{m2.group(2)}"]]'
    print("=" * 68)
    print("  read_book — OFFLINE PROOF (synthetic book, stub teacher, no Ollama)")
    print("=" * 68)
    rep, _ = train_on_book(book, client=StubClient(table), batch=3, is_text=True)
    _print_report(rep)
    print("\n  -> template cache learns 'the X is a Y' from the teacher, then parses later")
    print("     such sentences ITSELF (LLM-need falls); closure DERIVES lion isa living_thing")
    print("     though the book never stated it. Same loop runs on a real book (path + model).")


def _real(path, model):
    from llm_adapter import OllamaClient
    print("=" * 68)
    print(f"  read_book — {path}  (teacher: Ollama {model})")
    print("=" * 68)
    rep, _ = train_on_book(path, client=OllamaClient(model), batch=10, max_sents=50)
    _print_report(rep)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        _real(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else "qwen3:1.7B")
    else:
        _offline_proof()
