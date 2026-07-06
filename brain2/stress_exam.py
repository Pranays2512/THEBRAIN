#!/usr/bin/env python3
"""stress_exam.py — hammer the brain on its OWN trained data (uncurated).

exam.py is a curated 300-mark test. This is the opposite: pull RANDOM samples straight
from the grade 1-9 corpus and quiz the brain on them, plus adversarial cases and the
newly-wired faculties. The number that comes out is the HONEST coverage — no cherry-picked
questions. Every category reports pass-rate + a few real failures so the gaps are visible.

    /opt/homebrew/bin/python3.13 stress_exam.py
"""
import os
import random
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from brain_data import BrainData
from type_oracle import TypeOracle
import knowledge_distill as KD
from means_ends import PolicyMemory
from event_form import Event, POS
from event_verify import EventStore, classify
from mouth import say_event

FILES = (["data/taxonomy_core.txt"]
         + [f"data/math{i}.txt" for i in range(1, 10)]
         + [f"data/english{i}.txt" for i in range(1, 10)]
         + [f"data/science{i}.txt" for i in range(3, 10)]
         + [f"data/ssc{i}.txt" for i in range(6, 10)])

RNG = random.Random(1)


def load():
    fkb, mem = KD.SimpleKB(), PolicyMemory()
    datas, all_isa, all_ents, all_verbs = [], [], set(), set()
    for f in FILES:
        if not os.path.exists(f):
            continue
        d = BrainData.from_file(f)
        d.load_morph()
        d.teach_knowledge(fkb, mem)
        datas.append(d)
        all_isa += [(c, "isa", p) for c, p in d.isa]
        all_ents |= d.entities()
        all_verbs |= d.verbs()
    oracle = TypeOracle(triples=all_isa)
    constraints = BrainData.learn_verb_constraints_pooled(datas, oracle, frac=0.5)
    return fkb, mem, oracle, constraints, all_verbs, datas


def section(title):
    print(f"\n{'─'*70}\n  {title}\n{'─'*70}")


def report(name, passed, total, fails, results):
    pct = 100 * passed // max(total, 1)
    results.append((name, passed, total))
    print(f"  {name:32s} {passed:5d}/{total:<5d} ({pct:3d}%)")
    for f in fails[:3]:
        print(f"       ✗ {f}")


def main():
    print("=" * 70)
    print("  STRESS EXAM — random uncurated samples from the grade 1-9 corpus")
    print("=" * 70)
    print("  loading full corpus...", flush=True)
    fkb, mem, oracle, constraints, all_verbs, datas = load()

    facts = [(e, r, v) for d in datas for (e, r, v) in d.facts]
    props = [(e, r, v) for d in datas for (e, r, v) in d.props]
    isa = [(c, p) for d in datas for (c, p) in d.isa]
    events = [ev for d in datas for (_s, ev) in d.events]
    morph = {k: v for d in datas for k, v in d.morph.items()}
    print(f"  corpus: {len(facts)} facts, {len(props)} props, {len(isa)} isa, "
          f"{len(events)} events, {len(morph)} morph\n")

    results = []

    # 1. FACT RECALL — split CLEAN numeric facts from MALFORMED (non-numeric values in a
    # numeric slot: ranges, lists, units — a corpus data-quality issue, not a brain miss)
    section("1. FACT RECALL (clean numeric facts; malformed reported separately)")
    _isnum = re.compile(r"^-?\d+\.?\d*$")
    clean = [x for x in facts if _isnum.match(str(x[2]).strip())]
    malformed = len(facts) - len(clean)
    sample = RNG.sample(clean, min(250, len(clean)))
    ok, fails = 0, []
    for e, r, v in sample:
        got, _ = fkb.ask(e, r)
        try:
            hit = got is not None and abs(float(got) - float(v)) < 1e-6
        except (TypeError, ValueError):
            hit = False                            # key overwritten by a string value
        ok += hit
        if not hit and len(fails) < 3:
            fails.append(f"{e} | {r} -> got {got!r}, expected {v} (key collision)")
    report("clean numeric fact recall", ok, len(sample), fails, results)
    print(f"       note: {malformed}/{len(facts)} FACT lines have NON-numeric values "
          f"(ranges/lists/units) — corpus data-quality, excluded above")

    # 2. PROP RECALL — random qualitative props (string values)
    section("2. PROP RECALL (random qualitative props)")
    sample = RNG.sample(props, min(250, len(props)))
    ok, fails = 0, []
    for e, r, v in sample:
        got, _ = fkb.ask(e, r)
        hit = got is not None and str(got).lower() == str(v).lower()
        ok += hit
        if not hit and len(fails) < 3:
            fails.append(f"{e} | {r} -> got {got!r}, expected {v!r}")
    report("qualitative prop recall", ok, len(sample), fails, results)

    # 3. ISA CLOSURE — every stated child must reach its parent (transitive)
    section("3. ISA CLOSURE (transitive reachability)")
    sample = RNG.sample(isa, min(250, len(isa)))
    ok, fails = 0, []
    for c, p in sample:
        anc = oracle(c) or set()
        hit = p in anc
        ok += hit
        if not hit and len(fails) < 3:
            fails.append(f"{c} -> {p} not in closure {sorted(anc)[:5]}")
    report("isa closure", ok, len(sample), fails, results)

    # 4. MEMBRANE — real admitted events should NOT be rejected
    section("4. MEMBRANE on real events (should admit/abstain, not reject)")
    sample = RNG.sample(events, min(200, len(events)))
    store = EventStore()
    ok, fails = 0, []
    for ev in sample:
        d = classify(ev, store, oracle, constraints, all_verbs)
        hit = d != "reject"
        ok += hit
        if not hit and len(fails) < 3:
            fails.append(f"{ev.verb}({ev.agent},{ev.patient}) -> {d}")
    report("real events not falsely rejected", ok, len(sample), fails, results)

    # 5. MEMBRANE adversarial — nonsense verbs MUST NOT admit
    section("5. MEMBRANE adversarial (nonsense verbs must not admit)")
    nonsense = ["glorbify", "snorble", "flurp", "zark", "quomble", "vprex", "blimt", "draxil"]
    ents = list({ev.agent for ev in events if ev.agent})[:50]
    ok, fails = 0, []
    for v in nonsense:
        a = RNG.choice(ents) if ents else "thing"
        d = classify(Event(v, a, "water", "present", POS), store, oracle, constraints, all_verbs)
        hit = d != "admit"
        ok += hit
        if not hit:
            fails.append(f"{v}({a},water) -> {d} (should abstain)")
    report("nonsense verbs held (abstain)", ok, len(nonsense), fails, results)

    # 6. ARITHMETIC — curriculum identities via LEARNED procedures
    section("6. ARITHMETIC (curriculum identities, learned procedures)")
    _NUM = re.compile(r"^-?\d+(\.\d+)?$")
    idents = []
    for d in datas:
        for law in d.laws:
            body = law[4:].strip() if law.startswith("LAW:") else law
            if "=" not in body or "√" in body or "∠" in body:
                continue
            lhs, rhs = (x.strip() for x in body.rsplit("=", 1))
            if not _NUM.match(rhs):
                continue
            tree = KD.infix_to_tree(lhs)
            if tree is not None and _numeric_only(tree):
                idents.append((lhs, tree, float(rhs)))
    KD.reset_arith_stats()
    ok, fails = 0, []
    for expr, tree, rhs in idents:
        try:
            got = KD._eval(tree, {})
            hit = got is not None and abs(got - rhs) < 1e-9
        except Exception:
            hit = False
        ok += hit
        if not hit and len(fails) < 3:
            fails.append(f"{expr} = {rhs} (got other)")
    report("arithmetic identities", ok, len(idents), fails, results)

    # 7. MORPHOLOGY — past tense from learned morph
    section("7. MORPHOLOGY (past tense generation)")
    sample = RNG.sample(list(morph.items()), min(120, len(morph)))
    ok, fails = 0, []
    for lemma, forms in sample:
        want = forms.get("past")
        if not want:
            continue
        ev = Event(lemma, "he", "it", "past", POS)
        said = say_event(ev).lower()
        hit = want.lower() in said
        ok += hit
        if not hit and len(fails) < 3:
            fails.append(f"{lemma}/past: want '{want}', said '{said}'")
    report("past-tense morphology", ok, ok + (len(sample) - ok), fails, results)

    # 8. WIRED FACULTIES — dimensional verifier on real physics laws + rich queries
    section("8. WIRED FACULTIES (dimensional verify + rich queries)")
    from whole_brain import WholeBrain
    wb = WholeBrain()
    checks = [
        ("dim force=mass*accel", wb.check_dimensions(("*", "mass", "accel"), "force") is True),
        ("dim reject mass*speed=force", wb.check_dimensions(("*", "mass", "speed"), "force") is False),
        ("conjecture admits KE", wb.test_conjecture(lambda m, v: 0.5 * m * v * v)["admitted"]),
        ("conjecture rejects m*v", not wb.test_conjecture(lambda m, v: m * v)["admitted"]),
        ("compare query", "more" in str(wb.ask("is the rocket heavier than the sample")[1])),
        ("compound query", "speed" in str(wb.ask("mass and speed of the rocket")[1])),
        ("superlative query", "rocket" in str(wb.ask("what is the mass of the heaviest object")[1])),
        ("ground perception", wb.ground().get("inferred_correct", "0/6").startswith("6")),
    ]
    ok, fails = 0, []
    for name, hit in checks:
        ok += bool(hit)
        if not hit:
            fails.append(name)
    report("wired faculties", ok, len(checks), fails, results)

    # ── overall ──
    tot_p = sum(p for _, p, _ in results)
    tot_t = sum(t for _, _, t in results)
    print("\n" + "=" * 70)
    print(f"  OVERALL: {tot_p}/{tot_t}  ({100*tot_p//max(tot_t,1)}%)  "
          f"across {len(results)} stress categories")
    print("  (uncurated random samples from the trained corpus + adversarial + wired)")
    print("=" * 70)


def _numeric_only(tree):
    if isinstance(tree, str):
        return False
    if not isinstance(tree, tuple):
        return True
    return all(_numeric_only(x) for x in tree[1:])


if __name__ == "__main__":
    main()
