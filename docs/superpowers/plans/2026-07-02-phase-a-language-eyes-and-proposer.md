# Phase A: Brain's Own Eyes + Superpowered Proposer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the brain symbolic language comprehension (parse policies learned via conjecture→verify→admit, no LM in the loop) and upgrade the proposer from single-feature groundability to a learned, domain-aware ranker that also mints novel concepts.

**Architecture:** Language parsing becomes what physics already is — a library of verified, learned rules (templates). Templates are induced from labeled examples by slotting + anti-unification, gated by verification against held-out examples, persisted like `PolicyMemory`. The proposer gains: (1) execution traces as training data, (2) a numpy logistic-regression ranker over structural features, (3) per-domain hard filters (dimensional analysis for physics/chem), (4) concept formation — cross-domain shared structures get named, stored, and reused. The membrane rule holds everywhere: fuzzy things (embeddings, ranker) only PROPOSE ordering; crisp verification admits.

**Tech Stack:** Pure Python + numpy (already a dependency). GloVe embeddings via existing `nl_query.load_glove`. No torch, no LLM at inference. Tests in `brain2/tests/test_*.py`, run with `/opt/homebrew/bin/python3.13 -m pytest` (venv is broken — never use `THEBRAIN/venv`).

**Existing interfaces this plan builds on (verified against code 2026-07-02 — the .md docs are stale, code is truth):**
- `means_ends.Policy(target, inputs, expr)` — frozen dataclass, tuple-formula expr
- `means_ends.ev(expr, env)` — tuple-formula evaluator
- `policy_proposer.MultiPolicyMemory` — many policies per target; `.candidates(target)`
- `policy_proposer.Solver(kb, mem, use_proposer)` — counts `.work`, `._solve` orders candidates by `proposer_score`
- `policy_proposer.groundable(rel, mem, kb, seen)` — [0,1] groundability
- `reasoning_engine.ReasoningEngine` — `.learn(s, r, o_str)`, `.ask(s, r) -> (ans, why)`
- `nl_front.Front.resolve(q)` — confidence ladder: lexical → student → LLM
- `nl_query.load_glove(needed=...)`, `nl_query.STOP`
- `factorizer.factor_au(lib, min_count, min_kept) -> (new, prims, disc)`, `factorizer._verify(lib, new, prims)`

---

## Part 1 — Language Eyes (parse without any LM)

### Task 1: Parse templates — representation + matcher

**Files:**
- Create: `brain2/parse_template.py`
- Test: `brain2/tests/test_parse_template.py`

A template is a tuple of items: `("w", word)` literal, `("slot", name, type)` typed hole, or `("any",)` single-token wildcard. Types: `"entity"` (must be in a known-entity set) and `"number"` (numeric literal). A successful match returns bindings `{"entity": ..., "value": ...}` plus the template's fixed `rel`.

- [ ] **Step 1: Write the failing test**

```python
# brain2/tests/test_parse_template.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parse_template import Template, tokenize, match

def test_tokenize_lowercases_and_splits_numbers():
    assert tokenize("The rocket weighs 1000 kg.") == ["the", "rocket", "weighs", "1000", "kg"]

def test_literal_template_matches_and_binds():
    t = Template(rel="mass", items=(
        ("w", "the"), ("slot", "entity", "entity"), ("w", "weighs"),
        ("slot", "value", "number"), ("w", "kg")))
    b = match(t, tokenize("the rocket weighs 1000 kg"), entities={"rocket"})
    assert b == {"entity": "rocket", "value": 1000.0}

def test_no_match_on_unknown_entity_or_wrong_shape():
    t = Template(rel="mass", items=(
        ("w", "the"), ("slot", "entity", "entity"), ("w", "weighs"),
        ("slot", "value", "number"), ("w", "kg")))
    assert match(t, tokenize("the blorp weighs 1000 kg"), entities={"rocket"}) is None
    assert match(t, tokenize("the rocket weighs heavy kg"), entities={"rocket"}) is None

def test_any_wildcard_skips_one_token():
    t = Template(rel="mass", items=(
        ("any",), ("slot", "entity", "entity"), ("w", "weighs"),
        ("slot", "value", "number"), ("any",)))
    assert match(t, tokenize("a rocket weighs 1000 tons"), entities={"rocket"}) \
        == {"entity": "rocket", "value": 1000.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/pranay./Documents/THEBRAIN && /opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_parse_template.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parse_template'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""parse_template.py — symbolic sentence templates: the crisp grammar unit.

A Template is to language what a Policy is to physics: a stored, serializable,
verifiable rule. ("w", word) items must match exactly; ("slot", name, type)
items bind a typed value; ("any",) skips one token. Matching is exact-length —
no fuzzy scoring here (the membrane: fuzz lives in the grounder, Task 5)."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    rel: str
    items: tuple


def tokenize(text):
    return re.findall(r"[a-z_]+|\d+(?:\.\d+)?", text.lower())


def _is_number(tok):
    return re.fullmatch(r"\d+(?:\.\d+)?", tok) is not None


def match(template, tokens, entities):
    if len(tokens) != len(template.items):
        return None
    bind = {}
    for item, tok in zip(template.items, tokens):
        kind = item[0]
        if kind == "w":
            if tok != item[1]:
                return None
        elif kind == "any":
            continue
        else:                                   # ("slot", name, type)
            _, name, typ = item
            if typ == "number":
                if not _is_number(tok):
                    return None
                bind[name] = float(tok)
            elif typ == "entity":
                if tok not in entities:
                    return None
                bind[name] = tok
    return bind
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_parse_template.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add brain2/parse_template.py brain2/tests/test_parse_template.py
git commit -m "feat: parse templates - crisp grammar unit (literal/slot/any matcher)"
```

---

### Task 2: Template induction — slotting + anti-unification

**Files:**
- Modify: `brain2/parse_template.py` (append)
- Test: `brain2/tests/test_parse_template.py` (append)

Two operations: `slot_example(sentence, label)` turns one labeled example into a maximally-literal template (replace entity token and value token with slots). `anti_unify(t1, t2)` generalizes two same-rel, same-length templates by wildcarding mismatched literals — exactly the move `factorizer` makes for formulas, applied to word sequences.

- [ ] **Step 1: Write the failing test (append to test file)**

```python
from parse_template import slot_example, anti_unify

def test_slot_example_replaces_entity_and_value():
    t = slot_example("the rocket weighs 1000 kg",
                     {"entity": "rocket", "rel": "mass", "value": 1000})
    assert t.rel == "mass"
    assert t.items == (("w", "the"), ("slot", "entity", "entity"), ("w", "weighs"),
                       ("slot", "value", "number"), ("w", "kg"))

def test_anti_unify_wildcards_mismatched_literals():
    a = slot_example("the rocket weighs 1000 kg",
                     {"entity": "rocket", "rel": "mass", "value": 1000})
    b = slot_example("a sample weighs 2 kg",
                     {"entity": "sample", "rel": "mass", "value": 2})
    g = anti_unify(a, b)
    assert g.items == (("any",), ("slot", "entity", "entity"), ("w", "weighs"),
                       ("slot", "value", "number"), ("w", "kg"))

def test_anti_unify_refuses_different_rel_or_length():
    a = slot_example("the rocket weighs 1000 kg",
                     {"entity": "rocket", "rel": "mass", "value": 1000})
    c = slot_example("the rocket moves at 300",
                     {"entity": "rocket", "rel": "speed", "value": 300})
    assert anti_unify(a, c) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_parse_template.py -v`
Expected: FAIL with `ImportError: cannot import name 'slot_example'`

- [ ] **Step 3: Write minimal implementation (append to parse_template.py)**

```python
def slot_example(sentence, label):
    """One labeled example -> maximally literal template. The entity token and
    the value token become slots; every other token stays a literal."""
    toks = tokenize(sentence)
    val_str = re.sub(r"\.0$", "", str(float(label["value"])))
    items = []
    for tok in toks:
        if tok == label["entity"]:
            items.append(("slot", "entity", "entity"))
        elif _is_number(tok) and float(tok) == float(label["value"]):
            items.append(("slot", "value", "number"))
        else:
            items.append(("w", tok))
    return Template(rel=label["rel"], items=tuple(items))


def anti_unify(t1, t2):
    """Generalize two templates: agreeing positions stay, disagreeing literals
    become ('any',). Different rel or length -> not the same rule, refuse."""
    if t1.rel != t2.rel or len(t1.items) != len(t2.items):
        return None
    items = []
    for a, b in zip(t1.items, t2.items):
        if a == b:
            items.append(a)
        elif a[0] == "w" and b[0] == "w":
            items.append(("any",))
        else:
            return None            # slot vs literal disagreement: structurally different
    return Template(rel=t1.rel, items=tuple(items))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_parse_template.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add brain2/parse_template.py brain2/tests/test_parse_template.py
git commit -m "feat: template induction - slotting + anti-unification over word sequences"
```

---

### Task 3: TemplateMemory — conjecture → verify → admit + persistence

**Files:**
- Create: `brain2/template_memory.py`
- Test: `brain2/tests/test_template_memory.py`

The gate, same shape as `PolicyLearner`: a conjectured template (from slotting or anti-unification) is admitted only when it correctly re-parses ALL its source examples AND at least one example it was not induced from (statistical admit, N=1 unseen — raise later). Persists to JSON like `PolicyMemory.save/load`.

- [ ] **Step 1: Write the failing test**

```python
# brain2/tests/test_template_memory.py
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from template_memory import TemplateMemory

EXAMPLES = [
    ("the rocket weighs 1000 kg", {"entity": "rocket", "rel": "mass", "value": 1000}),
    ("a sample weighs 2 kg",      {"entity": "sample", "rel": "mass", "value": 2}),
    ("the probe weighs 55 kg",    {"entity": "probe",  "rel": "mass", "value": 55}),
]
ENTITIES = {"rocket", "sample", "probe"}

def test_learn_admits_generalizing_template_and_parses_unseen():
    tm = TemplateMemory(entities=ENTITIES)
    admitted = tm.learn(EXAMPLES[:2], holdout=[EXAMPLES[2]])
    assert admitted >= 1
    parsed = tm.parse("a drone weighs 7 kg")     # unseen sentence, unseen entity word
    assert parsed is None                         # 'drone' not a known entity -> honest miss
    tm.entities.add("drone")
    assert tm.parse("a drone weighs 7 kg") == ("drone", "mass", 7.0)

def test_rejects_template_that_fails_holdout():
    tm = TemplateMemory(entities=ENTITIES)
    bad_holdout = [("the probe is 55 kg", {"entity": "probe", "rel": "mass", "value": 55})]
    admitted = tm.learn(EXAMPLES[:2], holdout=bad_holdout)   # 'is' != 'weighs'
    assert admitted == 0

def test_save_load_roundtrip():
    tm = TemplateMemory(entities=ENTITIES)
    tm.learn(EXAMPLES[:2], holdout=[EXAMPLES[2]])
    path = os.path.join(tempfile.gettempdir(), "templates.json")
    tm.save(path)
    tm2 = TemplateMemory.load(path, entities=ENTITIES)
    assert tm2.parse("the probe weighs 55 kg") == ("probe", "mass", 55.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_template_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'template_memory'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""template_memory.py — durable grammar store with the conjecture->verify->admit
gate. Mirrors PolicyMemory/PolicyLearner: induction proposes a template,
verification against held-out labeled examples disposes. Only verified
templates are stored; parsing is exact and explainable."""

import json
from parse_template import Template, tokenize, match, slot_example, anti_unify


class TemplateMemory:
    def __init__(self, entities=None):
        self.templates = []                     # list[Template], most-literal first
        self.entities = set(entities or ())

    # ── use ──────────────────────────────────────────────────────────────────
    def parse(self, sentence):
        toks = tokenize(sentence)
        for t in self.templates:
            b = match(t, toks, self.entities)
            if b is not None and "entity" in b and "value" in b:
                return b["entity"], t.rel, b["value"]
        return None

    # ── learn: conjecture -> verify -> admit ─────────────────────────────────
    def _verified(self, t, examples):
        for sent, lab in examples:
            b = match(t, tokenize(sent), self.entities)
            if b is None or b.get("entity") != lab["entity"] \
                    or float(b.get("value", float("nan"))) != float(lab["value"]) \
                    or t.rel != lab["rel"]:
                return False
        return True

    def learn(self, examples, holdout):
        """Conjecture templates from labeled examples; admit those that verify
        on all sources AND on at least one holdout example. Returns admit count."""
        conjectures = [slot_example(s, l) for s, l in examples]
        for i in range(len(conjectures)):
            for j in range(i + 1, len(conjectures)):
                g = anti_unify(conjectures[i], conjectures[j])
                if g is not None:
                    conjectures.append(g)
        admitted = 0
        for t in conjectures:
            if t in self.templates:
                continue
            if self._verified(t, examples) and \
               any(self._verified(t, [h]) for h in holdout):
                self.templates.append(t)
                admitted += 1
        # most-literal templates first: fewer wildcards = higher precision
        self.templates.sort(key=lambda t: sum(1 for it in t.items if it[0] == "any"))
        return admitted

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path):
        data = [{"rel": t.rel, "items": [list(it) for it in t.items]}
                for t in self.templates]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path, entities=None):
        m = cls(entities=entities)
        with open(path) as f:
            data = json.load(f)
        for d in data:
            m.templates.append(Template(d["rel"], tuple(tuple(it) for it in d["items"])))
        return m
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_template_memory.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add brain2/template_memory.py brain2/tests/test_template_memory.py
git commit -m "feat: TemplateMemory - grammar store with conjecture->verify->admit gate"
```

---

### Task 4: Morphology normalizer — cheap coverage multiplier

**Files:**
- Modify: `brain2/parse_template.py` (append)
- Test: `brain2/tests/test_parse_template.py` (append)

Suffix-stripping rules applied during tokenization so `weighs/weighing/weighed → weigh`. Rules are a data table, not code — later they can be learned the same conjecture→verify way. Keep it conservative: only strip when the stem is ≥3 chars (avoid `is → i`).

- [ ] **Step 1: Write the failing test (append)**

```python
from parse_template import normalize

def test_normalize_strips_verb_suffixes():
    assert normalize("weighs") == "weigh"
    assert normalize("weighing") == "weigh"
    assert normalize("weighed") == "weigh"
    assert normalize("moves") == "move"       # -es after consonant+e: 'moves'->'move'

def test_normalize_conservative_on_short_words():
    assert normalize("is") == "is"
    assert normalize("kg") == "kg"

def test_tokenize_normalized():
    from parse_template import tokenize
    assert tokenize("the rocket weighed 5 kg", normalized=True) \
        == ["the", "rocket", "weigh", "5", "kg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_parse_template.py -v`
Expected: FAIL with `ImportError: cannot import name 'normalize'`

- [ ] **Step 3: Write minimal implementation (append to parse_template.py; also change `tokenize` signature)**

```python
# suffix rules: (suffix, replacement) tried in order; first hit wins.
_SUFFIX_RULES = (("ing", ""), ("ies", "y"), ("ied", "y"), ("es", "e"),
                 ("ed", ""), ("s", ""))


def normalize(tok):
    if _is_number(tok) or len(tok) <= 3:
        return tok
    for suf, rep in _SUFFIX_RULES:
        if tok.endswith(suf) and len(tok) - len(suf) + len(rep) >= 3:
            return tok[: len(tok) - len(suf)] + rep
    return tok
```

And change `tokenize` to:

```python
def tokenize(text, normalized=False):
    toks = re.findall(r"[a-z_]+|\d+(?:\.\d+)?", text.lower())
    return [normalize(t) for t in toks] if normalized else toks
```

Then in `template_memory.py`, switch both `parse()` and `_verified()` and `learn()` call sites to `tokenize(sentence, normalized=True)`, and in `slot_example` tokenize with `normalized=True` as well (templates are stored in normal form).

- [ ] **Step 4: Run ALL tests to verify pass (normalization touches Task 1–3 code)**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_parse_template.py brain2/tests/test_template_memory.py -v`
Expected: all passed. If `test_tokenize_lowercases_and_splits_numbers` breaks, it tests the non-normalized default — it must still pass unchanged.

- [ ] **Step 5: Commit**

```bash
git add brain2/parse_template.py brain2/template_memory.py brain2/tests/test_parse_template.py
git commit -m "feat: morphology normalization - suffix rules as data, templates stored in normal form"
```

---

### Task 5: Word grounding — fuzzy proposes, crisp disposes

**Files:**
- Create: `brain2/word_grounder.py`
- Test: `brain2/tests/test_word_grounder.py`

Unknown word in an entity slot → GloVe-nearest known entity ABOVE a similarity floor becomes a *proposal*; the caller (Front / executive) must verify by solving. This is the ONLY fuzzy component in the eyes, and it never writes to the template store. Uses existing `nl_query.load_glove`.

- [ ] **Step 1: Write the failing test**

```python
# brain2/tests/test_word_grounder.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from word_grounder import WordGrounder

def _fake_glove():
    # 3-dim toy embeddings: 'automobile' near 'car', 'banana' far from both
    return {"car": np.array([1.0, 0.0, 0.0]),
            "automobile": np.array([0.95, 0.05, 0.0]),
            "banana": np.array([0.0, 0.0, 1.0])}

def test_grounds_synonym_above_floor():
    g = WordGrounder(_fake_glove(), known={"car"}, floor=0.8)
    assert g.propose("automobile") == "car"

def test_refuses_below_floor_and_oov():
    g = WordGrounder(_fake_glove(), known={"car"}, floor=0.8)
    assert g.propose("banana") is None
    assert g.propose("zzgrf") is None            # not in embeddings -> honest None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_word_grounder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'word_grounder'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""word_grounder.py — fuzzy word->entity proposal via embedding similarity.

Membrane: this NEVER answers. It proposes a known entity for an unknown word;
the executive must still solve+verify with that entity, and nothing is written
to any crisp store here. Floor keeps precision high; below it, honest None."""

import numpy as np


class WordGrounder:
    def __init__(self, glove, known, floor=0.8):
        self.glove = glove
        self.known = set(known)
        self.floor = floor

    def propose(self, word):
        v = self.glove.get(word)
        if v is None:
            return None
        best, best_sim = None, self.floor
        for k in self.known:
            u = self.glove.get(k)
            if u is None:
                continue
            sim = float(np.dot(v, u) / ((np.linalg.norm(v) * np.linalg.norm(u)) or 1.0))
            if sim >= best_sim:
                best, best_sim = k, sim
        return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_word_grounder.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add brain2/word_grounder.py brain2/tests/test_word_grounder.py
git commit -m "feat: WordGrounder - embedding similarity proposes entities, never answers"
```

---

### Task 6: Wire eyes into the Front — new rung between lexical and student

**Files:**
- Modify: `brain2/nl_front.py` (Front.resolve, Front.__init__, _build)
- Test: `brain2/tests/test_front_templates.py`

Templates become rung 1.5: after lexical, before student. Template hit = crisp parse = accept immediately (it was verified at admission). Unknown entity word in a template-shaped sentence → `WordGrounder.propose` → accept only if the executive then solves. Student and LLM rungs stay untouched — they are now the measured fallback, to be deleted per-domain when Task 7's coverage metric clears threshold.

- [ ] **Step 1: Write the failing test**

```python
# brain2/tests/test_front_templates.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from template_memory import TemplateMemory

def test_template_rung_resolves_before_student():
    """Build a minimal Front-like resolve: template memory parses a taught
    pattern for a question form, no student/LLM involved."""
    tm = TemplateMemory(entities={"rocket", "sample"})
    ex = [
        ("what is the mass of the rocket", {"entity": "rocket", "rel": "mass", "value": 0}),
        ("what is the mass of the sample", {"entity": "sample", "rel": "mass", "value": 0}),
    ]
    # question templates carry no value slot; teach with learn_question
    admitted = tm.learn_question(ex[:1], holdout=ex[1:])
    assert admitted >= 1
    assert tm.parse_question("what is the mass of the rocket") == ("rocket", "mass")
    assert tm.parse_question("what is the mass of the sample") == ("sample", "mass")
    assert tm.parse_question("what is the wisdom of the rocket") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_front_templates.py -v`
Expected: FAIL with `AttributeError: 'TemplateMemory' object has no attribute 'learn_question'`

- [ ] **Step 3: Implement question templates (append to template_memory.py)**

Question templates have an entity slot but no value slot; the rel is the answer target. Reuse the same induction/gate. Add to `TemplateMemory`:

```python
    def parse_question(self, sentence):
        toks = tokenize(sentence, normalized=True)
        for t in self.templates:
            b = match(t, toks, self.entities)
            if b is not None and "entity" in b and "value" not in b:
                return b["entity"], t.rel
        return None

    def _q_slot(self, sentence, label):
        toks = tokenize(sentence, normalized=True)
        # the rel's own word(s) stay literal; only the entity becomes a slot
        items = tuple(("slot", "entity", "entity") if tok == label["entity"]
                      else ("w", tok) for tok in toks)
        return Template(rel=label["rel"], items=items)

    def _q_verified(self, t, examples):
        for sent, lab in examples:
            b = match(t, tokenize(sent, normalized=True), self.entities)
            if b is None or b.get("entity") != lab["entity"] or t.rel != lab["rel"]:
                return False
        return True

    def learn_question(self, examples, holdout):
        conjectures = [self._q_slot(s, l) for s, l in examples]
        for i in range(len(conjectures)):
            for j in range(i + 1, len(conjectures)):
                g = anti_unify(conjectures[i], conjectures[j])
                if g is not None:
                    conjectures.append(g)
        admitted = 0
        for t in conjectures:
            if t not in self.templates and self._q_verified(t, examples) and \
               any(self._q_verified(t, [h]) for h in holdout):
                self.templates.append(t)
                admitted += 1
        self.templates.sort(key=lambda t: sum(1 for it in t.items if it[0] == "any"))
        return admitted
```

Then wire into `nl_front.Front`:
- `__init__` gains `templates: TemplateMemory = None, grounder: WordGrounder = None` params, stored as `self.templates`, `self.grounder`.
- In `resolve()`, insert after the entity extraction and BEFORE the lexical/student tiers:

```python
        # 1.5 verified grammar: a template hit is crisp — it was gated at admission
        if self.templates is not None:
            hit = self.templates.parse_question(q)
            if hit is not None:
                return hit[0], hit[1], "template"
            if self.grounder is not None and ent is None:
                toks = [t for t in re.findall(r"[a-z_]+", q.lower()) if t not in STOP]
                for tok in toks:
                    ge = self.grounder.propose(tok)
                    if ge is not None:
                        self.templates.entities.add(tok)      # try tok as alias
                        hit = self.templates.parse_question(q)
                        self.templates.entities.discard(tok)
                        if hit is not None:
                            return ge, hit[1], "template+ground"
```

In `_build()`, construct `TemplateMemory(entities=entities)`, teach it question examples generated from the same `rows` dataset the student trains on (each row has `question` + `label` with entity/rel), split 80/20 into learn/holdout, and pass it plus a `WordGrounder(glove, known=entities)` into `Front(...)`.

- [ ] **Step 4: Run new test + existing front demo to verify nothing broke**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_front_templates.py -v && /opt/homebrew/bin/python3.13 brain2/nl_front.py`
Expected: test passed; demo still answers all 7 queries with same answers (sources may shift toward `template` — that is the point).

- [ ] **Step 5: Commit**

```bash
git add brain2/template_memory.py brain2/nl_front.py brain2/tests/test_front_templates.py
git commit -m "feat: template rung 1.5 in Front - verified grammar answers before student/LLM"
```

---

### Task 7: Coverage harness — know when a rung is deletable

**Files:**
- Create: `brain2/coverage_harness.py`
- Test: `brain2/tests/test_coverage_harness.py`

The deletion metric. Splits labeled sentences into train/held-out; reports per-rung resolution: `% template`, `% student`, `% llm`, `% none`. You delete the student for a domain when template coverage on held-out ≥ your threshold (suggest 90%). Without this, removing LMs is guesswork.

- [ ] **Step 1: Write the failing test**

```python
# brain2/tests/test_coverage_harness.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from coverage_harness import coverage

def test_coverage_counts_rungs():
    def fake_resolve(q):
        return ("rocket", "mass", "template") if "mass" in q else ("rocket", None, "none")
    held_out = [("what is the mass of the rocket", "mass"),
                ("mass of rocket", "mass"),
                ("what is the wisdom of the rocket", None)]
    rep = coverage(fake_resolve, held_out)
    assert rep["template"] == 2
    assert rep["none"] == 1
    assert rep["correct"] == 3          # 'none' on unanswerable IS correct
    assert abs(rep["template_pct"] - 2 / 3) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_coverage_harness.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""coverage_harness.py — the deletion metric. Measures which rung resolves each
held-out question and whether the resolved rel is CORRECT. A student/LLM rung is
deletable for a domain when template_pct clears threshold on held-out data."""

from collections import Counter


def coverage(resolve, held_out):
    """resolve: q -> (entity, rel, source). held_out: [(question, expected_rel)]
    where expected_rel None means the honest answer is 'I don't know'."""
    by_src = Counter()
    correct = 0
    for q, expected in held_out:
        _, rel, src = resolve(q)
        by_src[src] += 1
        if rel == expected:
            correct += 1
    n = max(len(held_out), 1)
    rep = dict(by_src)
    rep["correct"] = correct
    rep["n"] = len(held_out)
    rep["template_pct"] = by_src.get("template", 0) / n
    rep["correct_pct"] = correct / n
    return rep
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_coverage_harness.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add brain2/coverage_harness.py brain2/tests/test_coverage_harness.py
git commit -m "feat: coverage harness - per-rung resolution metric, the LM deletion criterion"
```

---

## Part 2 — Superpowered Proposer

### Task 8: Trace collection — the proposer's training data

**Files:**
- Create: `brain2/proposer_trace.py`
- Modify: `brain2/policy_proposer.py` (Solver._solve logs attempts)
- Test: `brain2/tests/test_proposer_trace.py`

Every candidate-policy attempt at every choice point becomes one training row: features + succeeded-or-not. This runs ALWAYS (cheap append), so every solve — math, physics, synth — feeds the ranker. Features are structural and domain-agnostic here; domain plugins come in Task 10.

- [ ] **Step 1: Write the failing test**

```python
# brain2/tests/test_proposer_trace.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reasoning_engine import ReasoningEngine
from means_ends import Policy
from policy_proposer import MultiPolicyMemory, Solver
from proposer_trace import TraceLog, features

def _setup():
    kb = ReasoningEngine()
    for s, r, o in [("rocket", "mass", "1000"), ("rocket", "accel", "20"),
                    ("rocket", "speed", "300")]:
        kb.learn(s, r, o)
    mem = MultiPolicyMemory()
    mem.add(Policy("force", ("mass", "accel"), ("*", "mass", "accel")))
    mem.add(Policy("power", ("energy", "time"), ("/", "energy", "time")))   # dead end
    mem.add(Policy("power", ("force", "speed"), ("*", "force", "speed")))
    return kb, mem

def test_features_vector_shape_and_range():
    kb, mem = _setup()
    p = mem.candidates("power")[1]
    f = features(p, mem, lambda r: kb.ask("rocket", r)[0] is not None)
    assert len(f) == 4
    assert all(0.0 <= x <= 1.0 for x in f[:2])   # groundability, fanin norm

def test_solver_records_success_and_failure_rows():
    kb, mem = _setup()
    log = TraceLog()
    s = Solver(kb, mem, use_proposer=False)
    s.trace_log = log
    s.solve("rocket", "power")
    labels = [row["ok"] for row in log.rows if row["target"] == "power"]
    assert True in labels and False in labels     # dead policy failed, live one won
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_proposer_trace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proposer_trace'`

- [ ] **Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""proposer_trace.py — every policy attempt becomes a training row for the
learned proposer. Features are structural (computable BEFORE trying the
policy); the label is whether the attempt produced a value. JSONL on disk so
traces accumulate across runs — the proposer's experience."""

import json
from policy_proposer import groundable


def _expr_size(e):
    return 1 + sum(_expr_size(x) for x in e[1:]) if isinstance(e, tuple) else 1


def features(policy, mem, kb_has):
    """[mean input groundability, normalized fan-in, normalized expr size,
    candidate-count for target normalized] — all in [0,1]-ish, all cheap."""
    g = sum(groundable(i, mem, kb_has) for i in policy.inputs) / len(policy.inputs)
    fanin = min(len(policy.inputs) / 5.0, 1.0)
    size = min(_expr_size(policy.expr) / 20.0, 1.0)
    ncand = min(len(mem.candidates(policy.target)) / 5.0, 1.0)
    return [g, fanin, size, ncand]


class TraceLog:
    def __init__(self, path=None):
        self.rows = []
        self.path = path

    def record(self, target, feats, ok):
        row = {"target": target, "x": feats, "ok": bool(ok)}
        self.rows.append(row)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")

    @staticmethod
    def load(path):
        log = TraceLog(path=None)
        with open(path) as f:
            log.rows = [json.loads(line) for line in f if line.strip()]
        return log
```

Modify `policy_proposer.Solver`: in `__init__` add `self.trace_log = None`; in `_solve`, inside the `for p in cands:` loop, compute `feats = features(p, self.mem, self._fact)` before trying, and after the attempt call `self.trace_log.record(need.rel, feats, ok)` when `self.trace_log is not None` (import `features` from `proposer_trace` lazily inside `_solve` to avoid an import cycle: `from proposer_trace import features`).

- [ ] **Step 4: Run test to verify it passes (plus the existing demo still works)**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_proposer_trace.py -v && /opt/homebrew/bin/python3.13 brain2/policy_proposer.py`
Expected: 2 passed; demo output unchanged (proposer still ~2x less work than blind).

- [ ] **Step 5: Commit**

```bash
git add brain2/proposer_trace.py brain2/policy_proposer.py brain2/tests/test_proposer_trace.py
git commit -m "feat: proposer trace collection - every policy attempt becomes ranker training data"
```

---

### Task 9: Learned ranker — logistic regression over traces

**Files:**
- Create: `brain2/learned_proposer.py`
- Test: `brain2/tests/test_learned_proposer.py`

Tiny numpy logistic regression: `score = sigmoid(w·x + b)`. Trained on `TraceLog` rows. Falls back to raw groundability when trained on fewer than `MIN_ROWS` examples (never let an untrained ranker order the search worse than the current single feature). The membrane: the ranker only reorders candidates — the executive still verifies every answer.

- [ ] **Step 1: Write the failing test**

```python
# brain2/tests/test_learned_proposer.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from learned_proposer import LearnedRanker

def _synthetic_rows(n=200):
    # groundability (x[0]) is the true signal; other features are noise
    import random
    random.seed(7)
    rows = []
    for _ in range(n):
        g = random.random()
        rows.append({"target": "t", "x": [g, random.random(), random.random(),
                                          random.random()], "ok": g > 0.5})
    return rows

def test_untrained_falls_back_to_groundability():
    r = LearnedRanker()
    assert r.score([0.9, 0.5, 0.5, 0.5]) == 0.9

def test_trained_ranker_learns_groundability_signal():
    r = LearnedRanker()
    r.train(_synthetic_rows())
    assert r.score([0.9, 0.5, 0.5, 0.5]) > r.score([0.1, 0.5, 0.5, 0.5])

def test_save_load_roundtrip(tmp_path):
    r = LearnedRanker()
    r.train(_synthetic_rows())
    p = str(tmp_path / "ranker.json")
    r.save(p)
    r2 = LearnedRanker.load(p)
    assert abs(r2.score([0.9, 0.5, 0.5, 0.5]) - r.score([0.9, 0.5, 0.5, 0.5])) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_learned_proposer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""learned_proposer.py — the proposer grows up: logistic regression over trace
features replaces the single hand-picked groundability score. Trained from
TraceLog rows (its own experience). Below MIN_ROWS it falls back to raw
groundability so an untrained ranker can never be WORSE than today's proposer.
Membrane intact: scores only ORDER candidates; verification still decides."""

import json
import numpy as np

MIN_ROWS = 50


class LearnedRanker:
    def __init__(self):
        self.w = None
        self.b = 0.0

    def score(self, x):
        if self.w is None:
            return x[0]                          # fallback: groundability feature
        z = float(np.dot(self.w, x) + self.b)
        return 1.0 / (1.0 + np.exp(-z))

    def train(self, rows, lr=0.5, epochs=300):
        if len(rows) < MIN_ROWS:
            return False
        X = np.array([r["x"] for r in rows], dtype=float)
        y = np.array([1.0 if r["ok"] else 0.0 for r in rows])
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))
            g = p - y
            self.w -= lr * (X.T @ g) / len(y)
            self.b -= lr * float(g.mean())
        return True

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"w": None if self.w is None else self.w.tolist(),
                       "b": self.b}, f)

    @classmethod
    def load(cls, path):
        r = cls()
        with open(path) as f:
            d = json.load(f)
        r.w = None if d["w"] is None else np.array(d["w"])
        r.b = d["b"]
        return r
```

Wire into `policy_proposer.Solver`: `__init__` gains optional `ranker=None`; in `_solve`, when `self.use_proposer and self.ranker is not None`, order by `-self.ranker.score(features(p, self.mem, self._fact))` instead of `proposer_score`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_learned_proposer.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add brain2/learned_proposer.py brain2/policy_proposer.py brain2/tests/test_learned_proposer.py
git commit -m "feat: learned proposer - logistic ranker over traces, groundability fallback"
```

---

### Task 10: Domain feature plugins + dimensional hard filter

**Files:**
- Create: `brain2/domain_features.py`
- Test: `brain2/tests/test_domain_features.py`

Two mechanisms, different strengths:
1. **Soft features** appended to the ranker's vector per domain — e.g. historical success rate of this exact policy, symbol overlap between policy inputs and recently-solved needs (context relevance).
2. **Hard filter: dimensional analysis** (physics/chem). A policy whose expr is dimensionally inconsistent with its target CANNOT be right — score it 0 before search ever tries it. Dims are exponent vectors over base units (M, L, T, N); `+`/`-` require equal dims, `*` adds, `/` subtracts, `^` scales by a numeric exponent. This is the single highest-precision pruning signal available in physics/chem and it is pure structure — no learning needed.

- [ ] **Step 1: Write the failing test**

```python
# brain2/tests/test_domain_features.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from means_ends import Policy
from domain_features import dims_of, dim_consistent, success_rate_feature

# dims: exponents over (M, L, T)
UNITS = {
    "mass": (1, 0, 0), "accel": (0, 1, -2), "speed": (0, 1, -1),
    "force": (1, 1, -2), "energy": (1, 2, -2), "time": (0, 0, 1),
    "power": (1, 2, -3),
}

def test_dims_of_evaluates_expressions():
    assert dims_of(("*", "mass", "accel"), UNITS) == (1, 1, -2)          # force
    assert dims_of(("/", "energy", "time"), UNITS) == (1, 2, -3)         # power
    assert dims_of(("*", 0.5, ("*", "mass", ("^", "speed", 2))), UNITS) == (1, 2, -2)

def test_dim_consistent_accepts_right_rejects_wrong():
    good = Policy("power", ("force", "speed"), ("*", "force", "speed"))
    bad = Policy("power", ("mass", "speed"), ("+", "mass", "speed"))     # M + L/T: illegal
    assert dim_consistent(good, UNITS) is True
    assert dim_consistent(bad, UNITS) is False

def test_unknown_units_are_not_filtered():
    mystery = Policy("power", ("foo", "bar"), ("*", "foo", "bar"))
    assert dim_consistent(mystery, UNITS) is None    # unknown -> abstain, don't prune

def test_success_rate_feature():
    hist = {("power", ("force", "speed")): (8, 2)}   # 8 wins, 2 losses
    p = Policy("power", ("force", "speed"), ("*", "force", "speed"))
    assert abs(success_rate_feature(p, hist) - 0.75) < 1e-9   # laplace (8+1)/(10+2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_domain_features.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""domain_features.py — domain knowledge the proposer can use.

Hard filter: dimensional analysis. A dimensionally-inconsistent policy cannot
be correct, so it is pruned BEFORE search (score 0) — pure structure, zero
learning, the strongest pruning signal physics/chem offer. Unknown units ->
None (abstain): the filter must never prune what it does not understand.

Soft feature: per-policy historical success rate (Laplace-smoothed) — the
policy's own track record, fed to the LearnedRanker as an extra feature."""


class DimError(Exception):
    pass


def _add(a, b): return tuple(x + y for x, y in zip(a, b))
def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _mul(a, k): return tuple(x * k for x in a)


def dims_of(expr, units):
    """Exponent vector of a tuple-formula, or raise DimError on inconsistency,
    or raise KeyError on an unknown symbol."""
    if isinstance(expr, (int, float)):
        return None                              # dimensionless scalar
    if isinstance(expr, str):
        return units[expr]
    op = expr[0]
    if op == "neg":
        return dims_of(expr[1], units)
    a = dims_of(expr[1], units)
    b = dims_of(expr[2], units)
    if op in ("+", "-"):
        if a != b:
            raise DimError(f"{a} {op} {b}")
        return a
    if op == "*":
        if a is None: return b
        if b is None: return a
        return _add(a, b)
    if op == "/":
        zero = tuple(0 for _ in (a or b))
        if a is None: a = zero
        if b is None: return a
        return _sub(a, b)
    if op == "^":
        if not isinstance(expr[2], (int, float)):
            raise DimError("non-numeric exponent")
        return None if a is None else _mul(a, expr[2])
    raise DimError(f"unknown op {op!r}")


def dim_consistent(policy, units):
    """True (consistent), False (provably wrong), None (unknown units: abstain)."""
    try:
        d = dims_of(policy.expr, units)
    except DimError:
        return False
    except KeyError:
        return None
    target = units.get(policy.target)
    if target is None or d is None:
        return None
    return d == target


def success_rate_feature(policy, history):
    """Laplace-smoothed win rate of this exact policy: (wins+1)/(wins+losses+2)."""
    wins, losses = history.get((policy.target, policy.inputs), (0, 0))
    return (wins + 1) / (wins + losses + 2)
```

Wire into `policy_proposer.Solver._solve`: when a `units` dict is provided to the Solver (optional `units=None` in `__init__`), drop candidates where `dim_consistent(p, self.units) is False` before ordering. Append `success_rate_feature(p, self.history)` to the feature vector passed to the ranker (Solver gains optional `history={}` updated after each attempt: increment wins/losses keyed by `(p.target, p.inputs)`).

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_domain_features.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add brain2/domain_features.py brain2/policy_proposer.py brain2/tests/test_domain_features.py
git commit -m "feat: dimensional hard filter + success-rate feature for the proposer"
```

**Domain mapping for this machinery (design note, no extra code needed now):**
- **Math:** soft features only — expr-size delta toward goal, shared-symbol count with the target expression. Hard filter: none general (types later).
- **Coding:** success = tests pass; history feature dominates. The trace rows from `composable_synth`/`synth_engine` runs feed the SAME ranker — pass `trace_log` into those search loops exactly as in Task 8.
- **Physics:** dims hard filter + all soft features. Units table taught as facts (`kb.learn(rel, "dims", "1,1,-2")`) or a static dict per domain pack.
- **Chemistry:** same dims machinery with base units (mol, g, L); mass/charge balance later as a second hard filter with identical True/False/None contract.
- **General tasks:** history + groundability only — never invent a fake hard filter where no invariant exists.

---

### Task 11: Concept formation — name what repeats, reuse what's named

**Files:**
- Create: `brain2/concept_memory.py`
- Test: `brain2/tests/test_concept_memory.py`

The "novel concepts" mechanism. `factorizer.factor_au` already finds shared structure across domains (`curiosity_cross.py` proves it on ½mv² vs ½kx²). What's missing: the discovered structure gets no NAME, no store, no reuse count. This task adds ConceptMemory: a discovered shared primitive is registered as a candidate concept; each time it appears in a later verified solution its usage count rises; at `PROMOTE_AT` uses it is promoted — becoming a first-class policy the proposer can propose. Concepts are compression + reuse, which is what a concept IS.

- [ ] **Step 1: Write the failing test**

```python
# brain2/tests/test_concept_memory.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from concept_memory import ConceptMemory

QUAD_A = ("*", 0.5, ("*", "m", ("*", "v", "v")))     # kinetic energy
QUAD_B = ("*", 0.5, ("*", "k", ("*", "x", "x")))     # spring energy
QUAD_C = ("*", 0.5, ("*", "c", ("*", "q", "q")))     # capacitor energy (new domain!)

def test_register_names_shared_structure():
    cm = ConceptMemory(promote_at=2)
    name = cm.register(shape=("*", 0.5, ("*", "COEFF", ("*", "Q", "Q"))),
                       sources=["kinetic_energy", "spring_energy"])
    assert name == "concept_0"
    assert cm.status(name) == "candidate"

def test_matching_a_new_instance_bumps_usage_and_promotes():
    cm = ConceptMemory(promote_at=2)
    name = cm.register(shape=("*", 0.5, ("*", "COEFF", ("*", "Q", "Q"))),
                       sources=["kinetic_energy", "spring_energy"])
    hit = cm.recognize(QUAD_C)
    assert hit == (name, {"COEFF": "c", "Q": "q"})
    cm.record_use(name)                                  # capacitor solve verified
    cm.record_use(name)
    assert cm.status(name) == "promoted"

def test_recognize_rejects_different_shape():
    cm = ConceptMemory(promote_at=2)
    cm.register(shape=("*", 0.5, ("*", "COEFF", ("*", "Q", "Q"))), sources=[])
    assert cm.recognize(("+", "a", "b")) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_concept_memory.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
#!/usr/bin/env python3
"""concept_memory.py — from shared structure to NAMED, REUSABLE concept.

factorizer/curiosity_cross discover that two domains share a shape; this store
gives the shape a name and a life-cycle: candidate -> (used PROMOTE_AT times in
verified solutions) -> promoted. Promotion is the statistical admit gate —
a concept earns first-class status by proving reusable, not by being found once.
Shape variables are UPPERCASE strings; recognize() pattern-matches a concrete
expr against each stored shape and returns the variable binding."""

import json

PROMOTE_AT = 3


def _match_shape(shape, expr, bind):
    if isinstance(shape, str) and shape.isupper():       # shape variable
        if shape in bind and bind[shape] != expr:
            return None
        bind[shape] = expr
        return bind
    if isinstance(shape, tuple) and isinstance(expr, tuple) and len(shape) == len(expr):
        for s, e in zip(shape, expr):
            if _match_shape(s, e, bind) is None:
                return None
        return bind
    return bind if shape == expr else None


class ConceptMemory:
    def __init__(self, promote_at=PROMOTE_AT):
        self.concepts = {}          # name -> {shape, sources, uses, status}
        self.promote_at = promote_at
        self._n = 0

    def register(self, shape, sources):
        for name, c in self.concepts.items():            # dedupe by exact shape
            if c["shape"] == shape:
                return name
        name = f"concept_{self._n}"
        self._n += 1
        self.concepts[name] = {"shape": shape, "sources": list(sources),
                               "uses": 0, "status": "candidate"}
        return name

    def recognize(self, expr):
        for name, c in self.concepts.items():
            bind = _match_shape(c["shape"], expr, {})
            if bind is not None and bind:
                return name, bind
        return None

    def record_use(self, name):
        c = self.concepts[name]
        c["uses"] += 1
        if c["uses"] >= self.promote_at and c["status"] == "candidate":
            c["status"] = "promoted"

    def status(self, name):
        return self.concepts[name]["status"]

    def save(self, path):
        def enc(e):
            return [enc(x) for x in e] if isinstance(e, tuple) else e
        data = {n: {**c, "shape": enc(c["shape"])} for n, c in self.concepts.items()}
        with open(path, "w") as f:
            json.dump({"n": self._n, "concepts": data}, f, indent=2)

    @classmethod
    def load(cls, path):
        def dec(e):
            return tuple(dec(x) for x in e) if isinstance(e, list) else e
        m = cls()
        with open(path) as f:
            d = json.load(f)
        m._n = d["n"]
        for n, c in d["concepts"].items():
            m.concepts[n] = {**c, "shape": dec(c["shape"])}
        return m
```

**Integration (same task, after tests pass):** in `curiosity_cross._demo`-style flows, after `FZ.factor_au` returns a discovered primitive `disc`, call `concepts.register(shape=disc_as_shape, sources=[names])` — converting the factorizer's primitive variables to UPPERCASE shape variables. In the Solver, when a solve completes, walk the used policies' exprs with `concepts.recognize`; on hit, `record_use`. When a concept promotes, synthesize a `Policy(target=concept_name, inputs=sorted(vars), expr=shape_with_lowercase_vars)` and add it to the policy memory — the proposer can now PROPOSE the concept in domains that never saw it.

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/test_concept_memory.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add brain2/concept_memory.py brain2/tests/test_concept_memory.py
git commit -m "feat: ConceptMemory - shared structures get names, earn promotion by reuse"
```

---

### Task 12: End-to-end check + retire stale docs

**Files:**
- Modify: `brain2/architecture_roadmap.md` (mark stale sections, add Phase A status)
- Test: full suite

- [ ] **Step 1: Run everything**

Run: `/opt/homebrew/bin/python3.13 -m pytest brain2/tests/ -x -q 2>&1 | tail -20`
Expected: no new failures vs the pre-plan baseline (record baseline BEFORE Task 1: run the same command and save the output).

- [ ] **Step 2: Run all touched demos**

Run: `/opt/homebrew/bin/python3.13 brain2/nl_front.py && /opt/homemade/bin/python3.13 brain2/policy_proposer.py && /opt/homebrew/bin/python3.13 brain2/means_ends.py`
(NOTE: second command has a typo guard — use `/opt/homebrew/bin/python3.13` for all three.)
Expected: all three demos complete; nl_front shows `template` as source for taught-pattern queries.

- [ ] **Step 3: Update stale docs**

The user confirmed all .md files are stale. In `brain2/architecture_roadmap.md`: add a dated "Phase A implemented" entry listing the new modules (`parse_template.py`, `template_memory.py`, `word_grounder.py`, `coverage_harness.py`, `proposer_trace.py`, `learned_proposer.py`, `domain_features.py`, `concept_memory.py`) and DELETE or mark-stale any section contradicting current code. Do not rewrite history — date the corrections.

- [ ] **Step 4: Commit**

```bash
git add brain2/architecture_roadmap.md
git commit -m "docs: Phase A status + retire stale roadmap sections"
```

---

## Part 3 — Keep In Mind While Implementing (the guardrails)

**Membrane discipline (the one rule that keeps this project honest):**
1. Fuzzy components (WordGrounder, LearnedRanker, embeddings) may only PROPOSE or ORDER. If you ever find one writing to TemplateMemory/PolicyMemory/facts directly, that's the bug — even if tests pass.
2. Every admit path needs a reject test. `test_rejects_template_that_fails_holdout` and the bad-conjecture rejection in `means_ends._demo` are the pattern: for each new gate, write the test where the WRONG thing is refused. A gate you've never seen reject is decoration.

**Anti-collapse (self-training loops):**
3. Only verified parses may generate new training examples or templates. If student/LLM output feeds template induction, it must first survive the holdout gate. Unverified self-training = error amplification = model collapse.
4. Keep provenance: when a template/policy/concept is admitted, record which examples admitted it (the `sources` field in ConceptMemory is the pattern). When something later proves wrong, you can trace and evict its descendants.

**Measurement before deletion:**
5. Run `coverage_harness` on a FROZEN held-out set (never train on it, never regenerate it casually). Deleting the student/LLM rung for a domain is justified by `template_pct ≥ 0.9 AND correct_pct ≥ 0.95` on that set — not by vibes.
6. Record the pre-plan test baseline before Task 1. "No new failures" is only checkable against a recorded baseline.

**Proposer/ranker hygiene:**
7. The ranker must never be able to make search WORSE than today: the groundability fallback under MIN_ROWS is load-bearing. Same principle for future features — untrained/unknown always degrades to current behavior.
8. Hard filters must have the three-valued contract (True/False/None). `None` = abstain = do not prune. A hard filter that prunes what it doesn't understand will silently delete correct solutions — worst possible failure mode because search "works", just worse.
9. Traces are append-only JSONL. Never edit them; retrain from scratch each time (cheap at this scale). Deduplicate identical rows at train time, not at log time.
10. Watch for feedback loops: the ranker orders search → search produces traces → traces train the ranker. Policies ranked low get fewer attempts, hence fewer trace rows, hence frozen scores. Mitigation (only when you observe it): epsilon-greedy — with small probability (0.05) try a random candidate order to keep exploration alive.

**Language-specific:**
11. Templates are exact-length matchers by design. Resist adding fuzzy scoring INSIDE `match()` — the fuzz belongs in the grounder and in future multi-template composition, where it can be gated. Fuzzy matching inside the crisp matcher dissolves the membrane.
12. Precision order matters: most-literal template first (fewest `any` wildcards). A wildcard-heavy template admitted early can shadow better ones — the sort in `learn()` handles this; keep it when refactoring.
13. Morphology rules are data. When a rule misfires (e.g. `physics → physic`), fix the table, add a test, don't special-case in code.
14. Honest failure beats wrong parse everywhere: `parse() -> None` and "I don't know" are correct outputs. Never lower a floor/threshold to make a demo look better.

**Concept formation:**
15. Naming is cheap; promotion must be expensive. `promote_at` guards against coincidental structure. If concepts promote that turn out spurious, RAISE promote_at rather than adding heuristics.
16. A promoted concept is a hypothesis with a good track record, not a truth. It still goes through verification every time it's used in a solve — promotion changes what the proposer PROPOSES, never what the verifier ACCEPTS.

**Scope discipline (YAGNI applied to this plan):**
17. No SymPy in this plan — that's the next plan (search substrate). Adding it here would couple two independent risk surfaces.
18. No multi-slot templates (two entities, ranges, units conversion) until single-slot coverage is measured. Coverage data tells you WHICH extension pays.
19. The C++ port of any of this comes AFTER Python versions are verified and measured — same as every prior brain2 phase.

---

## Self-review notes

- Spec coverage: language eyes (Tasks 1–7), proposer power (8–10), domains mapped (Task 10 design note), novel concepts (11), implementation cautions (Part 3), stale docs addressed (Task 12). Nothing from the request unmapped.
- Type consistency: `Template(rel, items)`, `match(t, tokens, entities) -> dict|None`, `features(...) -> list[4]` extended by success-rate in Task 10 (ranker retrains on new width — traces from before the width change should be discarded; note this when implementing Task 10).
- Known simplification: `learn_question` duplicates `learn`'s conjecture loop — acceptable duplication now (three similar blocks beat a premature abstraction); merge only if a third variant appears.
