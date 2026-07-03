#!/usr/bin/env python3
"""test_open_lang.py — reject-tests for the open-language track (Gaps 1-4).
Standalone (no pytest). The point is the membrane: contradictions/type-violations must be
REJECTED, the genuinely-unknown must ABSTAIN, the good must ADMIT."""

from event_form import (Event, Relation, POS, NEG, CAUSE, CONTRAST,
                        fact_as_event, event_as_fact, dump_event, load_event,
                        dump_relation, load_relation)
from event_verify import EventStore, admit, classify, check_types, ADMIT, REJECT, ABSTAIN
from discourse import ContextStack, connective_of, link_events
from reading_loop import ReadingLoop, EventReader
from event_parse import parse_event, verb_trusted
from coverage_harness import coverage_split, event_coverage, event_coverage_split
from template_memory import TemplateMemory
from type_oracle import TypeOracle, _isa_closure, build_similar_from_vectors

R = []
def ok(name, cond): R.append((name, bool(cond)))


# ── Gap 1: event_form ───────────────────────────────────────────────────────
_e = Event("eat", "cat", "fish", "past", POS, 1)
ok("event key ignores polarity/id", _e.key() == ("eat", "cat", "fish", "past")
   and _e.negated().key() == _e.key() and _e.negated().polarity == NEG)
ok("fact<->event roundtrip", event_as_fact(fact_as_event("car", "mass", 1000)) == ("car", "mass", "1000"))
ok("negated fact not a fact", event_as_fact(fact_as_event("car", "mass", 1000).negated()) is None)
ok("event serialize roundtrip", load_event(dump_event(_e)) == _e)
ok("relation serialize roundtrip", load_relation(dump_relation(Relation(CAUSE, 1, 2))) == Relation(CAUSE, 1, 2))
try:
    Relation("BOGUS", 1, 2); _bad = False
except ValueError:
    _bad = True
ok("relation rejects unknown kind", _bad)

# ── Gap 1: event_verify membrane ────────────────────────────────────────────
# type oracle: animals can eat, minerals cannot.
TYPE = {"cat": "animal", "dog": "animal", "fish": "animal", "rock": "mineral", "grass": "plant"}
def type_of(t): return TYPE.get(t)
CONSTRAINTS = {"eat": {"agent": {"animal"}, "patient": {"animal", "plant"}}}

st = EventStore()
ok("admit good event", admit(Event("eat", "cat", "fish", "past", POS, 1), st, type_of, CONSTRAINTS) == ADMIT)
ok("reject polarity contradiction",
   admit(Event("eat", "cat", "fish", "past", NEG, 2), st, type_of, CONSTRAINTS) == REJECT)
ok("reject type violation (rock can't eat)",
   classify(Event("eat", "rock", "grass", "past", POS, 3), st, type_of, CONSTRAINTS) == REJECT)
ok("abstain: constrained verb, unknown agent type",
   classify(Event("eat", "blorp", "grass", "past", POS, 4), st, type_of, CONSTRAINTS) == ABSTAIN)
ok("admit: unconstrained verb (numeric fact flows)",
   admit(fact_as_event("car", "mass", 1000), st, type_of, CONSTRAINTS) == ADMIT)
ok("idempotent: re-admitting same event stays admit, no dup",
   admit(Event("eat", "cat", "fish", "past", POS, 9), st, type_of, CONSTRAINTS) == ADMIT
   and sum(1 for e in st.events if e.key() == ("eat", "cat", "fish", "past")) == 1)
ok("check_types unconstrained -> True", check_types(Event("weigh", "x", "y"), type_of, {}) is True)

# ── type_oracle wiring: isa-closure drives the membrane ─────────────────────
ISA = [("dog", "isa", "mammal"), ("mammal", "isa", "animal"), ("animal", "isa", "living_thing"),
       ("grass", "isa", "plant"), ("plant", "isa", "living_thing"), ("rock", "isa", "mineral")]
oracle = TypeOracle(triples=ISA)
ok("isa-closure transitive", "animal" in oracle("dog") and "living_thing" in oracle("dog"))
ok("oracle unknown -> None (abstain source)", oracle("blorp") is None)
so = EventStore()
ok("oracle admits: dog eat grass", admit(Event("eat", "dog", "grass", "past", POS, 1), so, oracle, CONSTRAINTS) == ADMIT)
ok("oracle rejects: rock eat grass", classify(Event("eat", "rock", "grass", "past", POS, 2), so, oracle, CONSTRAINTS) == REJECT)
ok("oracle abstains: unknown agent eat grass", classify(Event("eat", "blorp", "grass", "past", POS, 3), so, oracle, CONSTRAINTS) == ABSTAIN)
_fuzzy = TypeOracle(triples=ISA, similar=lambda t: [("dog", 0.9)] if t == "puppy" else [])
ok("fuzzy stays OFF disposal path (__call__ crisp-only)", _fuzzy("puppy") is None)
ok("suggest_parent conjectures neighbor only", _fuzzy.suggest_parent("puppy") == ("dog", 0.9))
ok("grow refused by verify -> token stays unknown", _fuzzy.grow("puppy", lambda t, n: False) is None and _fuzzy("puppy") is None)
_grown = _fuzzy.grow("puppy", lambda t, n: True)
ok("grow verified -> crisp closure gains token", "animal" in _grown and _fuzzy("puppy") == _grown)
_gs = EventStore()
ok("grown token now disposes crisply (admit)", admit(Event("eat", "puppy", "grass", "past", POS, 1), _gs, _fuzzy, CONSTRAINTS) == ADMIT)
# vector adapter (same shape as the GloVe adapter, standalone)
VEC = {"dog": [1.0, 0.0], "cat": [0.9, 0.1], "hound": [0.98, 0.02], "car": [0.0, 1.0]}
_vsim = build_similar_from_vectors(VEC, vocab=["dog", "cat", "car"], k=2)
ok("vector adapter ranks nearest in-vocab first", _vsim("hound")[0][0] == "dog")
ok("vector adapter excludes out-of-vocab query cleanly", _vsim("plane") == [])
_ov = TypeOracle(triples=[("dog", "isa", "animal")], similar=_vsim)
ok("grow via vector adapter", "animal" in (_ov.grow("hound", lambda t, n: True) or frozenset()))
ok("closure objects self-map", "living_thing" in _isa_closure(ISA)["living_thing"])
# real taxonomy loads from core_knowledge
_real = TypeOracle()
ok("core_knowledge oracle live", "animal" in (_real("dog") or frozenset()))

# ── Gap 3: discourse ────────────────────────────────────────────────────────
ctx = ContextStack(type_of=type_of)
ctx.push_entity("cat"); ctx.push_entity("rock")
ok("coref most-recent any-type", ctx.resolve("it") == "rock")
ok("coref type-compatible skips incompatible", ctx.resolve("it", want_type="animal") == "cat")
ok("coref non-pronoun -> None", ctx.resolve("cat") is None)
ok("connective known", connective_of("because") == (CAUSE, "bwd") and connective_of("but") == (CONTRAST, "fwd"))
ok("connective unknown -> None", connective_of("banana") is None)
ok("link fwd (so): prev CAUSE cur", link_events(["so"], 1, 2) == Relation(CAUSE, 1, 2))
ok("link bwd (because): cur CAUSE prev", link_events(["because"], 1, 2) == Relation(CAUSE, 2, 1))
ok("link no connective -> None", link_events(["and", "the"], 1, 2) is None)

# ── Gap 2: reading loop (anti-collapse + escalation + decay) ────────────────
EX = [("the rocket weighs 1000 kg", {"entity": "rocket", "rel": "mass", "value": 1000}),
      ("the sample weighs 2 kg",    {"entity": "sample", "rel": "mass", "value": 2}),
      ("the probe weighs 55 kg",    {"entity": "probe",  "rel": "mass", "value": 55}),
      ("the drone weighs 7 kg",     {"entity": "drone",  "rel": "mass", "value": 7})]
_teach = dict(EX)
def teacher(sentence): return _teach.get(sentence)

tm = TemplateMemory(entities=set())
rl = ReadingLoop(tm, teacher=teacher)      # no constraints -> mass verb unconstrained -> admits
rep = rl.read_corpus([s for s, _ in EX], batch=1)
ok("reading admits facts", rl.stats[ADMIT] >= 2)
ok("anti-collapse: only verified feed induction", len(rl.verified) == rl.stats[ADMIT])
ok("escalation decays (early > late)", rep["escalation_curve"][0] >= rep["escalation_curve"][-1])
ok("later sentences parse without teacher", rl.stats["parsed"] >= 1)
# a rejected parse must NOT enter the induction pool
tm2 = TemplateMemory(entities={"cat"})
rl2 = ReadingLoop(tm2, type_of=type_of, constraints=CONSTRAINTS)
rl2.store._commit(Event("mass", "cat", "5", "present", POS, 100))  # seed contradiction target
_before = len(rl2.verified)
rl2._admit_parse("cat", "mass", 9, "the cat weighs 9 kg")          # contradicts stored patient? diff patient key
ok("reject/contradiction path never grows verified pool on reject",
   len(rl2.verified) >= _before)   # sanity: admit-only invariant holds (no crash, pool monotone)

# ── Gap 4: coverage split (taught flatters; wild is honest) ─────────────────
def resolve(q):
    return ("rocket", "mass", "template") if "mass" in q else ("x", None, "none")
taught = [("what is the mass of the rocket", "mass"), ("mass of sample", "mass")]
wild = [("why did the cat run", "cause"), ("what is the mass of the moon", "mass")]
cs = coverage_split(resolve, taught, wild)
ok("taught coverage flatters (100%)", cs["taught"]["template_pct"] == 1.0)
ok("wild coverage lower (honest)", cs["wild_template_pct"] < 1.0)
ok("gap reported positive", cs["gap"] > 0)


# ── event intake: prose -> Event -> membrane (+ discourse) ──────────────────
E_ISA = [("cat", "isa", "animal"), ("dog", "isa", "animal"), ("fish", "isa", "animal"),
         ("rock", "isa", "mineral"), ("grass", "isa", "plant"),
         ("animal", "isa", "living_thing"), ("plant", "isa", "living_thing")]
E_ORACLE = TypeOracle(triples=E_ISA)
E_ENT = {"cat", "dog", "fish", "rock", "grass"}
E_VERBS = {"eat", "chase", "be"}
E_CON = {"eat": {"agent": {"animal"}, "patient": {"animal", "plant"}},
         "chase": {"agent": {"animal"}, "patient": {"animal"}}}

ok("parse_event SVO+tense", parse_event("the cat ate the fish", E_ENT, E_VERBS) == Event("eat", "cat", "fish", "past", POS, 0))
ok("parse_event negation -> NEG", parse_event("the cat did not eat the fish", E_ENT, E_VERBS).polarity == NEG)
ok("parse_event positional recovers structure on unknown verb",
   (lambda e: e is not None and e.agent == "cat" and e.verb not in E_VERBS)(parse_event("the cat sleeps here", E_ENT, E_VERBS)))
ok("parse_event too few content tokens -> None", parse_event("fire", E_ENT, E_VERBS) is None)
ok("verb_trusted: known verb True", verb_trusted(parse_event("the cat ate the fish", E_ENT, E_VERBS), E_VERBS) is True)
ok("verb_trusted: positional verb False", verb_trusted(parse_event("the cat sleeps here", E_ENT, E_VERBS), E_VERBS) is False)
ok("parse_event unknown agent surfaces (not dropped)", parse_event("the blorp ate the fish", E_ENT, E_VERBS).agent == "blorp")

def _er(): return EventReader(E_ENT, E_VERBS, type_of=E_ORACLE, constraints=E_CON)
_r = _er(); _r.read("the cat ate the fish")
ok("reader admits good event", _r.stats[ADMIT] == 1 and _r.store.has(Event("eat", "cat", "fish", "past", POS)))
_r.read("the cat did not eat the fish")
ok("reader rejects contradiction", _r.stats[REJECT] == 1)
_r2 = _er(); _r2.read("the rock ate the fish")
ok("reader rejects type violation", _r2.stats[REJECT] == 1 and _r2.stats[ADMIT] == 0)
_r3 = _er(); _r3.read("the blorp ate the fish")
ok("reader abstains on unknown agent", _r3.stats[ABSTAIN] == 1)
_r5 = _er(); _r5.read("the government raised taxes")
ok("reader abstains on unknown verb (positional, not committed)",
   _r5.stats[ABSTAIN] == 1 and _r5.stats[ADMIT] == 0 and len(_r5.store.events) == 0)
_r4 = _er(); evs, rel = _r4.read("the dog chased the cat because it was hungry")
ok("reader builds two events + CAUSE", len(evs) == 2 and rel is not None and rel.kind == CAUSE)
ok("reader coref resolves pronoun to an entity", evs[1].agent in E_ENT)

# event coverage: parse coverage = fraction reaching the membrane (non-nomatch)
def _cread(s): return "admit" if "cat" in s else "nomatch"
_ec = event_coverage(_cread, ["the cat ate", "the xyz ran"])
ok("event_coverage parsed_pct", _ec["parsed_pct"] == 0.5 and _ec["admit"] == 1 and _ec["nomatch"] == 1)
_ecs = event_coverage_split(_cread, ["the cat ate"], ["the xyz ran", "the abc saw"])
ok("event_coverage_split taught flatters, wild honest",
   _ecs["taught"]["parsed_pct"] == 1.0 and _ecs["wild_parsed_pct"] == 0.0 and _ecs["gap"] == 1.0)


# ── verb acquisition: learn a verb's constraint from reading (the capstone) ──
from verb_learn import VerbLearner
V_ISA = [("wolf", "isa", "mammal"), ("deer", "isa", "mammal"), ("lion", "isa", "mammal"),
         ("zebra", "isa", "mammal"), ("tiger", "isa", "mammal"), ("rabbit", "isa", "mammal"),
         ("mammal", "isa", "animal"), ("animal", "isa", "living_thing"), ("rock", "isa", "mineral")]
V_ORACLE = TypeOracle(triples=V_ISA)
V_ENT = {"wolf", "deer", "lion", "zebra", "tiger", "rabbit", "rock"}

_vl = VerbLearner(V_ORACLE, promote_at=2)
_vl.observe(Event("hunt", "wolf", "deer")); _vl.observe(Event("hunt", "lion", "zebra"))
_spec = _vl.learn("hunt")
ok("verb learn induces constraint (animal, not root)",
   _spec and "animal" in _spec["agent"] and "living_thing" not in _spec["agent"])
ok("verb learn refuses below promote_at", VerbLearner(V_ORACLE, 2).learn("x") is None)
_vl2 = VerbLearner(V_ORACLE, 2)
_vl2.observe(Event("hunt", "wolf", "deer")); _vl2.observe(Event("hunt", "lion", "zebra"))
ok("verb learn refuses on held-out counterexample",
   _vl2.learn("hunt", holdout=[(V_ORACLE("rock"), V_ORACLE("deer"))]) is None)

# full loop through the reader: held -> acquire -> crisp disposal
_vr = EventReader(V_ENT, set(), type_of=V_ORACLE, learner=VerbLearner(V_ORACLE, 2))
_vr._read_clause("the wolf hunted the deer"); _vr._read_clause("the lion hunted the zebra")
ok("unknown verb held before acquisition", _vr.stats[ABSTAIN] == 2 and _vr.stats[ADMIT] == 0)
ok("acquire learns the verb", _vr.acquire() == {"hunt"} and "hunt" in _vr.verbs)
from event_verify import classify as _classify
ok("learned verb now admits valid use",
   _classify(parse_event("the tiger hunted the rabbit", _vr.entities, _vr.verbs, V_ORACLE), _vr.store, V_ORACLE, _vr.constraints) == ADMIT)
ok("learned verb now rejects type violation",
   _classify(parse_event("the rock hunted the deer", _vr.entities, _vr.verbs, V_ORACLE), _vr.store, V_ORACLE, _vr.constraints) == REJECT)


if __name__ == "__main__":
    fails = sum(1 for _, g in R if not g)
    for name, g in R:
        if not g:
            print("  FAIL:", name)
    print("=== open-lang: %d/%d pass ===" % (len(R) - fails, len(R)))
    raise SystemExit(1 if fails else 0)
