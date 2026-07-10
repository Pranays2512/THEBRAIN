#!/usr/bin/env python3
"""test_phase_a.py — standalone tests for the Phase-A slice (no pytest dep).
Covers parse_template, template_memory, coverage_harness, domain_features, concept_memory.
Assertions follow the plan; slot_example stores NORMAL form (weighs->weigh) per Task 4."""

import os
import tempfile

from core.store.parse_template import Template, tokenize, match, slot_example, anti_unify, normalize
from core.store.template_memory import TemplateMemory
from core.store.coverage_harness import coverage
from domain_features import dims_of, dim_consistent, success_rate_feature
from concept_memory import ConceptMemory
from means_ends import Policy

R = []
def ok(name, cond): R.append((name, bool(cond)))


# ── parse_template ─────────────────────────────────────────────────────────
ok("tokenize", tokenize("The rocket weighs 1000 kg.") == ["the", "rocket", "weighs", "1000", "kg"])
_t = Template("mass", (("w", "the"), ("slot", "entity", "entity"), ("w", "weighs"),
                       ("slot", "value", "number"), ("w", "kg")))
ok("literal match", match(_t, tokenize("the rocket weighs 1000 kg"), {"rocket"}) == {"entity": "rocket", "value": 1000.0})
ok("no-match unknown entity", match(_t, tokenize("the blorp weighs 1000 kg"), {"rocket"}) is None)
ok("no-match wrong shape", match(_t, tokenize("the rocket weighs heavy kg"), {"rocket"}) is None)
_w = Template("mass", (("any",), ("slot", "entity", "entity"), ("w", "weighs"),
                       ("slot", "value", "number"), ("any",)))
ok("any wildcard", match(_w, tokenize("a rocket weighs 1000 tons"), {"rocket"}) == {"entity": "rocket", "value": 1000.0})

_a = slot_example("the rocket weighs 1000 kg", {"entity": "rocket", "rel": "mass", "value": 1000})
ok("slot_example (normal form)", _a.items == (("w", "the"), ("slot", "entity", "entity"), ("w", "weigh"),
                                              ("slot", "value", "number"), ("w", "kg")))
_b = slot_example("a sample weighs 2 kg", {"entity": "sample", "rel": "mass", "value": 2})
ok("anti_unify wildcards mismatch", anti_unify(_a, _b).items ==
   (("any",), ("slot", "entity", "entity"), ("w", "weigh"), ("slot", "value", "number"), ("w", "kg")))
_c = slot_example("the rocket moves at 300", {"entity": "rocket", "rel": "speed", "value": 300})
ok("anti_unify refuses diff rel", anti_unify(_a, _c) is None)
ok("normalize verbs", (normalize("weighs"), normalize("weighing"), normalize("weighed")) == ("weigh", "weigh", "weigh"))
ok("normalize conservative", (normalize("is"), normalize("kg")) == ("is", "kg"))
ok("tokenize normalized", tokenize("the rocket weighed 5 kg", normalized=True) == ["the", "rocket", "weigh", "5", "kg"])


# ── template_memory ────────────────────────────────────────────────────────
EX = [("the rocket weighs 1000 kg", {"entity": "rocket", "rel": "mass", "value": 1000}),
      ("a sample weighs 2 kg",      {"entity": "sample", "rel": "mass", "value": 2}),
      ("the probe weighs 55 kg",    {"entity": "probe",  "rel": "mass", "value": 55})]
ENT = {"rocket", "sample", "probe"}
tm = TemplateMemory(entities=ENT)
ok("learn admits >=1", tm.learn(EX[:2], holdout=[EX[2]]) >= 1)
ok("parse unseen entity -> None", tm.parse("a drone weighs 7 kg") is None)
tm.entities.add("drone")
ok("parse after adding entity", tm.parse("a drone weighs 7 kg") == ("drone", "mass", 7.0))
tm2 = TemplateMemory(entities=ENT)
ok("rejects bad holdout", tm2.learn(EX[:2], holdout=[("the probe is 55 kg", {"entity": "probe", "rel": "mass", "value": 55})]) == 0)
tm3 = TemplateMemory(entities=ENT); tm3.learn(EX[:2], holdout=[EX[2]])
_p = os.path.join(tempfile.gettempdir(), "tmpl_test.json"); tm3.save(_p)
ok("save/load roundtrip", TemplateMemory.load(_p, entities=ENT).parse("the probe weighs 55 kg") == ("probe", "mass", 55.0))
qtm = TemplateMemory(entities={"rocket", "sample"})
qex = [("what is the mass of the rocket", {"entity": "rocket", "rel": "mass", "value": 0}),
       ("what is the mass of the sample", {"entity": "sample", "rel": "mass", "value": 0})]
ok("learn_question admits", qtm.learn_question(qex[:1], holdout=qex[1:]) >= 1)
ok("parse_question rocket", qtm.parse_question("what is the mass of the rocket") == ("rocket", "mass"))
ok("parse_question none", qtm.parse_question("what is the wisdom of the rocket") is None)


# ── coverage_harness ───────────────────────────────────────────────────────
def _fake(q): return ("rocket", "mass", "template") if "mass" in q else ("rocket", None, "none")
_rep = coverage(_fake, [("what is the mass of the rocket", "mass"), ("mass of rocket", "mass"),
                        ("what is the wisdom of the rocket", None)])
ok("coverage template count", _rep["template"] == 2 and _rep["none"] == 1)
ok("coverage correct (none on unanswerable)", _rep["correct"] == 3 and abs(_rep["template_pct"] - 2/3) < 1e-9)


# ── domain_features (dimensional hard filter) ──────────────────────────────
U = {"mass": (1, 0, 0), "accel": (0, 1, -2), "speed": (0, 1, -1), "force": (1, 1, -2),
     "energy": (1, 2, -2), "time": (0, 0, 1), "power": (1, 2, -3)}
ok("dims force", dims_of(("*", "mass", "accel"), U) == (1, 1, -2))
ok("dims power", dims_of(("/", "energy", "time"), U) == (1, 2, -3))
ok("dims KE", dims_of(("*", 0.5, ("*", "mass", ("^", "speed", 2))), U) == (1, 2, -2))
ok("dim_consistent good", dim_consistent(Policy("power", ("force", "speed"), ("*", "force", "speed")), U) is True)
ok("dim_consistent bad", dim_consistent(Policy("power", ("mass", "speed"), ("+", "mass", "speed")), U) is False)
ok("dim_consistent unknown -> None", dim_consistent(Policy("power", ("foo", "bar"), ("*", "foo", "bar")), U) is None)
ok("success_rate laplace", abs(success_rate_feature(Policy("power", ("force", "speed"), ("*", "force", "speed")),
                                                     {("power", ("force", "speed")): (8, 2)}) - 0.75) < 1e-9)


# ── concept_memory ─────────────────────────────────────────────────────────
QC = ("*", 0.5, ("*", "c", ("*", "q", "q")))
cm = ConceptMemory(promote_at=2)
_nm = cm.register(shape=("*", 0.5, ("*", "COEFF", ("*", "Q", "Q"))), sources=["kinetic_energy", "spring_energy"])
ok("register names", _nm == "concept_0" and cm.status(_nm) == "candidate")
ok("recognize new instance", cm.recognize(QC) == (_nm, {"COEFF": "c", "Q": "q"}))
cm.record_use(_nm); cm.record_use(_nm)
ok("promotes at reuse", cm.status(_nm) == "promoted")
cm2 = ConceptMemory(promote_at=2); cm2.register(shape=("*", 0.5, ("*", "COEFF", ("*", "Q", "Q"))), sources=[])
ok("recognize rejects diff shape", cm2.recognize(("+", "a", "b")) is None)


if __name__ == "__main__":
    fails = sum(1 for _, g in R if not g)
    for name, g in R:
        if not g:
            print("  FAIL:", name)
    print("=== Phase-A: %d/%d pass ===" % (len(R) - fails, len(R)))
    raise SystemExit(1 if fails else 0)
