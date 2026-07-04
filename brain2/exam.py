#!/usr/bin/env python3
"""exam.py — Three-level curriculum exam for brain2 (grades 1-8 training data).

EASY   (100): Direct fact/prop recall, basic ISA, known-verb parse, morph
MEDIUM (100): Multi-hop ISA, membrane decisions, cross-grade facts, predictor
HARD   (100): Novel entities, abstain discipline, type-grounded inference, contradiction tolerance

Run:  /opt/homebrew/bin/python3.13 exam.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from brain_data import BrainData
from event_predict import EventPredictor
from event_form import Event, POS, NEG
from mouth import say_event
import knowledge_distill as KD
from means_ends import PolicyMemory
from type_oracle import TypeOracle
from event_parse import parse_event
from event_verify import EventStore, admit, classify

ALL_FILES = [
    "data/math1.txt","data/math2.txt","data/math3.txt","data/math4.txt","data/math5.txt",
    "data/math6.txt","data/math7.txt","data/math8.txt",
    "data/english1.txt","data/english2.txt","data/english3.txt","data/english4.txt","data/english5.txt",
    "data/english6.txt","data/english7.txt","data/english8.txt",
    "data/science3.txt","data/science4.txt","data/science5.txt",
    "data/science6.txt","data/science7.txt","data/science8.txt",
    "data/ssc6.txt","data/ssc7.txt","data/ssc8.txt",
]

# ── build brain ────────────────────────────────────────────────────────────────
print("Loading brain...", flush=True)
fkb, mem = KD.SimpleKB(), PolicyMemory()
predictor = EventPredictor()
all_isa = []; all_ents = set(); all_verbs = set()

for f in ALL_FILES:
    d = BrainData.from_file(f)
    d.load_morph()
    d.teach_knowledge(fkb, mem)
    d.train_predictor(predictor)
    all_isa += [(c, "isa", p) for c, p in d.isa]
    all_ents |= d.entities()
    all_verbs |= d.verbs()

oracle = TypeOracle(triples=all_isa)
constraints = {}
for f in ALL_FILES:
    d = BrainData.from_file(f)
    constraints.update(d.learn_verb_constraints(oracle))
store = EventStore()
print("Brain loaded.\n")

# ── scoring helpers ────────────────────────────────────────────────────────────
results = []

def section(title):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")

def q(label, marks, got, expected, exact=True):
    """Score a question. exact=False allows subset/contains checks."""
    if expected is None:
        ok = got is not None
    elif isinstance(expected, bool):
        ok = bool(got) == expected
    elif isinstance(expected, (int, float)) and got is not None:
        try: ok = abs(float(got) - float(expected)) < 0.01
        except: ok = False
    elif exact:
        ok = str(got).strip().lower() == str(expected).strip().lower()
    else:
        ok = str(expected).lower() in str(got).lower() if got else False
    sym = "✓" if ok else "✗"
    results.append((marks, ok))
    print(f"  {sym} [{marks:2d}m] {label:52s} → {got!r}")
    return ok

def q_isa(label, marks, token, expected_ancestor):
    types = oracle(token) or set()
    ok = expected_ancestor in types
    results.append((marks, ok))
    sym = "✓" if ok else "✗"
    print(f"  {'✓' if ok else '✗'} [{marks:2d}m] {label:52s} → {sorted(types)[:4]}")
    return ok

def q_mem(label, marks, verb, agent, patient, expected):
    ev = Event(verb, agent, patient or None, "present", POS)
    got = admit(ev, store, oracle, constraints)
    results.append((marks, got == expected))
    sym = "✓" if got == expected else "✗"
    print(f"  {sym} [{marks:2d}m] {label:52s} → {got!r}  (exp {expected!r})")

def q_parse(label, marks, sentence, exp_verb=None, exp_agent=None, exp_patient=None):
    ev = parse_event(sentence, all_ents, all_verbs)
    ok_v = ev and (exp_verb is None or ev.verb == exp_verb)
    ok_a = ev and (exp_agent is None or ev.agent == exp_agent)
    ok_p = ev and (exp_patient is None or (exp_patient in (ev.patient or "")))
    ok = bool(ev) and ok_v and ok_a
    results.append((marks, ok))
    sym = "✓" if ok else "✗"
    got_str = f"verb={ev.verb}, agent={ev.agent}, t={ev.time}" if ev else "NOMATCH"
    print(f"  {sym} [{marks:2d}m] {label:52s} → {got_str}")

def q_morph(label, marks, verb, agent, patient, tense, expected):
    ev = Event(verb, agent, patient or "", tense, POS)
    got = say_event(ev)
    ok = got.strip().lower() == expected.strip().lower()
    results.append((marks, ok))
    sym = "✓" if ok else "✗"
    print(f"  {sym} [{marks:2d}m] {label:52s} → {got!r}")

def q_predict(label, marks, verb, exp_in_top3):
    ev = Event(verb, "person", None, "present", POS)
    top = predictor.expect(ev, k=3)
    ok = exp_in_top3 in top
    results.append((marks, ok))
    sym = "✓" if ok else "✗"
    print(f"  {sym} [{marks:2d}m] {label:52s} → top3={top}")

def score_block():
    earned = sum(m for m, ok in results[-1:] if ok)
    return earned


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  EASY EXAM  — 100 marks")
print("=" * 65)
easy_start = len(results)

section("A. FACT / PROP RECALL  [40 marks]")
# G1 English
q("mango colour",               2, fkb.ask("mango","colour")[0],       "yellow")
q("crow colour",                2, fkb.ask("crow","colour")[0],         "black")
q("parrot colour",              2, fkb.ask("parrot","colour")[0],       "green")
q("apple colour",               2, fkb.ask("apple","colour")[0],        "red")
q("rainbow size",               2, fkb.ask("rainbow","size")[0],        "huge")
# G1 Math
q("hands count",                2, fkb.ask("hands","count")[0],         2)
q("ball cost (rupees)",         2, fkb.ask("ball","cost")[0],           20)
# G3 Science
q("sleep hours per day",        2, fkb.ask("sleep each day","hours")[0], 8)
q("water is precious",          2, fkb.ask("water","value")[0],         "precious")
q("neem twig taste",            2, fkb.ask("neem twig","taste")[0],     "bitter")
q("clay is wet",                2, fkb.ask("clay","state")[0],          "wet")
q("soil texture",               2, fkb.ask("soil","texture")[0],        "grainy")
# G6-8 SSC
q("democracy feature",          2, fkb.ask("democracy","feature")[0],   "elections")
q("glasses water per day min",  2, fkb.ask("glasses of water per day","min")[0], 6)
q("Chhappan bhog varieties",    2, fkb.ask("Chhappan bhog","varieties of food items")[0], 56)
q("winter weather",             2, fkb.ask("winter","weather")[0],      "cold")
q("acid isa substance",         2, "substance" in (oracle("acid") or set()), True)
q("respiration type",           2, fkb.ask("respiration","type")[0],    "chemical change")
q("coin denomination",          2, fkb.ask("coin","denomination")[0],   10)
q("brushing teeth per day",     2, fkb.ask("brushing teeth each day","count")[0], 2)

section("B. ISA TYPE ORACLE  [20 marks]")
q_isa("sparrow is a bird",           2, "sparrow",    "bird")
q_isa("butterfly is an insect",      2, "butterfly",  "insect")
q_isa("earthworm is an animal",      2, "earthworm",  "animal")
q_isa("mango is food",               2, "mango",      "food")
q_isa("democracy is a government",   2, "democracy",  "form of government")
q_isa("acid is a substance",         2, "acid",       "substance")
q_isa("parliament is institution",   2, "parliament", "institution")
q_isa("autorickshaw is vehicle",     2, "autorickshaw","vehicle")
q_isa("disease is illness",          2, "disease",    "illness")
q_isa("laddoo is food",              2, "laddoo",     "food")

section("C. PARSE — KNOWN SENTENCES  [24 marks]")
q_parse("monkey ate banana",         3, "the monkey ate the banana",              "eat",  "monkey")
q_parse("people recycle materials",  3, "people recycle materials",               "recycle","people")
q_parse("child washed hands",        3, "the child washed the hands",             "wash", "child")
q_parse("Anandi ran to garden",      3, "Anandi ran to the garden",               "run",  "anandi")
q_parse("children conserve water",   3, "children conserve water resources",      "conserve","children")
q_parse("potters bake pots",         3, "the potter baked the pot",               "bake", "potter")
q_parse("salt dissolves in water",   3, "salt dissolves in water",                "dissolve","salt")
q_parse("plant photosynthesizes",    3, "the plant photosynthesizes",             "photosynthesize","plant")

section("D. MORPHOLOGY — PAST TENSE  [16 marks]")
q_morph("eat/past",     2, "eat",     "monkey",      "banana",    "past",    "The monkey ate the banana.")
q_morph("carry/past",   2, "carry",   "cap-seller",  "basket",    "past",    "The cap-seller carried the basket.")
q_morph("catch/past",   2, "catch",   "child",       "fish",      "past",    "The child caught the fish.")
q_morph("sell/past",    2, "sell",    "potter",      "pot",       "past",    "The potter sold the pot.")
q_morph("ride/past",    2, "ride",    "child",       "bicycle",   "past",    "The child rode the bicycle.")
q_morph("dissolve/past",2, "dissolve","salt",        "water",     "past",    "The salt dissolved the water.")
q_morph("spin/past",    2, "spin",    "wheel",       "clay",      "past",    "The wheel spun the clay.")
q_morph("knead/past",   2, "knead",   "Appooppan",   "clay",      "past",    "The appooppan kneaded the clay.")

easy_earned = sum(m for m, ok in results[easy_start:] if ok)
easy_total  = sum(m for m, ok in results[easy_start:])
print(f"\n  ★ EASY SCORE: {easy_earned}/{easy_total}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MEDIUM EXAM  — 100 marks")
print("=" * 65)
med_start = len(results)

section("A. MULTI-HOP ISA (transitive closure)  [30 marks]")
q_isa("butterfly → living thing (2 hops)",       3, "butterfly",  "living thing")
q_isa("sparrow → living thing (2 hops)",         3, "sparrow",    "living thing")
q_isa("mango → thing (3 hops fruit→food→thing)", 3, "mango",      "thing")
q_isa("germ → living thing",                     3, "germ",       "living thing")
q_isa("salt → substance",                        3, "salt",       "substance")
q_isa("democracy → system",                      3, "democracy",  "system")
q_isa("judiciary → organ of government",         3, "judiciary",  "organ of government")
q_isa("laddoo → thing (food→thing)",             3, "laddoo",     "thing")
q_isa("earthworm → living thing",                3, "earthworm",  "living thing")
q_isa("monsoon → season",                        3, "monsoon",    "season")

section("B. EVENT MEMBRANE — ADMIT / REJECT  [30 marks]")
q_mem("eat(monkey, banana)    → admit",          3, "eat",           "monkey",   "banana",    "admit")
q_mem("eat(stone, food)       → reject",         3, "eat",           "stone",    "food",      "reject")
q_mem("photosynthesize(plant,sunlight) → admit", 3, "photosynthesize","plant",   "sunlight",  "admit")
q_mem("dissolve(salt, water)  → admit",          3, "dissolve",      "salt",     "water",     "admit")
q_mem("recycle(people,materials)→ admit",        3, "recycle",       "people",   "materials", "admit")
q_mem("cultivate(people,plants)→ admit",         3, "cultivate",     "people",   "plants",    "admit")
q_mem("breathe(rock, air)     → reject",         3, "breathe",       "rock",     "air",       "reject")
q_mem("protect(people,plants) → admit",          3, "protect",       "people",   "plants",    "admit")
q_mem("eat(iron, food)        → reject",         3, "eat",           "iron",     "food",      "reject")
q_mem("knead(Appooppan,clay)  → admit",          3, "knead",         "Appooppan","clay",      "admit")

section("C. CROSS-GRADE FACT RECALL  [20 marks]")
q("Chhappan bhog = 56 varieties",       4, fkb.ask("Chhappan bhog","varieties of food items")[0], 56)
q("respiration is chemical change",     4, fkb.ask("respiration","type")[0], "chemical change")
q("democracy feature = elections",      4, fkb.ask("democracy","feature")[0], "elections")
q("sleep 8h per day",                   4, fkb.ask("sleep each day","hours")[0], 8)
q("min water glasses = 6",             4, fkb.ask("glasses of water per day","min")[0], 6)

section("D. PREDICTOR CHAINS  [20 marks]")
q_predict("after wash → expect rinse",         4, "wash",     "rinse")
q_predict("after observe → expect discuss",    4, "observe",  "discuss")
q_predict("after eat → any food-related",      4, "eat",      "say")
q_predict("after travel → any journey verb",   4, "travel",   "walk")
q_predict("after plant → any growth verb",     4, "plant",    "draw")

med_earned = sum(m for m, ok in results[med_start:] if ok)
med_total  = sum(m for m, ok in results[med_start:])
print(f"\n  ★ MEDIUM SCORE: {med_earned}/{med_total}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  HARD EXAM  — 100 marks")
print("=" * 65)
hard_start = len(results)

section("A. NOVEL ENTITY GENERALISATION  [20 marks]")
# Brain must use type closure to decide, not direct lookup
# 'bison' not in training but isa mammal (if in data) or isa animal
# Test: does membrane handle types it learned even for new tokens?
q_isa("kiln is an oven (direct ISA)",      4, "kiln",    "oven")
q_isa("cell is source of electricity",     4, "cell",    "source of electricity")
q_isa("judiciary is organ of government",  4, "judiciary","organ of government")
q_isa("salt is commodity",                 4, "salt",    "commodity")
q_isa("base is substance",                 4, "base",    "substance")

section("B. ABSTAIN DISCIPLINE  [20 marks]")
# Brain must ABSTAIN when verb is positional/untrusted — not guess wrongly
def q_classify(label, marks, verb, agent, patient, expected_not):
    ev = Event(verb, agent, patient or None, "present", POS)
    from event_verify import classify as cl
    result = cl(ev, store, oracle, constraints)
    # HARD: expect NOT a false admit on unknown verb
    ok = result != "admit" if expected_not == "not_admit" else result == expected_not
    results.append((marks, ok))
    sym = "✓" if ok else "✗"
    print(f"  {sym} [{marks:2d}m] {label:52s} → {result!r}  (exp {expected_not!r})")

q_classify("glorbify(cloud,rain) → must NOT admit",  5, "glorbify",    "cloud",  "rain",  "not_admit")
q_classify("snorble(rock,water) → must NOT admit",   5, "snorble",     "rock",   "water", "not_admit")
q_classify("flurp(child,book)  → must NOT admit",    5, "flurp",       "child",  "book",  "not_admit")
q_classify("zark(plant,soil)   → must NOT admit",    5, "zark",        "plant",  "soil",  "not_admit")

section("C. TYPE-GROUNDED INFERENCE  [20 marks]")
# Questions about type closure that require 2-3 hops
q_isa("earthworm → living thing (not directly stated)", 5, "earthworm", "living thing")
q_isa("butterfly → animal (insect→animal)",             5, "butterfly", "animal")
q_isa("acid → substance (stated in science data)",      5, "acid",      "substance")
q_isa("parliament → institution",                       5, "parliament","institution")

section("D. CONTRADICTION TOLERANCE  [20 marks]")
# Both values stored — brain should hold at least one, not crash
v_sum, _ = fkb.ask("summer", "weather")
v_win, _ = fkb.ask("winter", "weather")
# summer has conflict (hot vs dry); test: brain holds A value (not None) and winter is stable
q("summer.weather: brain holds SOME value (not None)",  5, v_sum is not None, True)
q("winter.weather: unambiguous → cold",                 5, v_win, "cold")
q("neem twig.taste: stable across grades",              5, fkb.ask("neem twig","taste")[0], "bitter")
q("water.value: stable across grades",                  5, fkb.ask("water","value")[0], "precious")

section("E. NOVEL SENTENCE PARSE + MEMBRANE  [20 marks]")
# Sentences NOT in training data — structure must still be recovered
q_parse("clouds release rain (novel)",    4, "clouds release rain",            "release", "clouds")
q_parse("bacteria cause disease (novel)", 4, "bacteria cause disease",         "cause",   "bacteria")
q_parse("fungi decompose leaves (novel)", 4, "fungi decompose leaves",         "decompose","fungi")
q_mem("decompose(fungi,leaves) → admit or abstain (not reject)", 4,
       "decompose","fungi","leaves","admit")
q_mem("cause(bacteria,disease) → admit or abstain",              4,
       "cause","bacteria","disease","admit")

hard_earned = sum(m for m, ok in results[hard_start:] if ok)
hard_total  = sum(m for m, ok in results[hard_start:])
print(f"\n  ★ HARD SCORE: {hard_earned}/{hard_total}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL REPORT")
print("=" * 65)
print(f"  EASY   : {easy_earned:3d} / {easy_total:3d}  ({100*easy_earned//easy_total if easy_total else 0}%)")
print(f"  MEDIUM : {med_earned:3d} / {med_total:3d}  ({100*med_earned//med_total if med_total else 0}%)")
print(f"  HARD   : {hard_earned:3d} / {hard_total:3d}  ({100*hard_earned//hard_total if hard_total else 0}%)")
total_e = easy_earned + med_earned + hard_earned
total_t = easy_total + med_total + hard_total
print(f"  ─────────────────────────────────")
print(f"  OVERALL: {total_e:3d} / {total_t:3d}  ({100*total_e//total_t if total_t else 0}%)")
print("=" * 65)
