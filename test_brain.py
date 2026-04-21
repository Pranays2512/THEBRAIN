"""
test_brain.py — Automated test suite for THE BRAIN
Run: python test_brain.py
"""

import sys
import os

sys.dont_write_bytecode = True

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status}  {name}" + (f"  [{detail}]" if detail else ""))
    results.append((name, condition))
    return condition

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ══════════════════════════════════════════════════════════
# 1. VOCAB CORE — words have grounded MFCC vectors
# ══════════════════════════════════════════════════════════
section("1. Vocabulary Core")

from vocab_core import VOCABULARY

check("vocab loads",       len(VOCABULARY) > 80,          f"{len(VOCABULARY)} words")
check("food in vocab",     'food'   in VOCABULARY)
check("danger in vocab",   'danger' in VOCABULARY)
check("hungry in vocab",   'hungry' in VOCABULARY)
check("what in vocab",     'what'   in VOCABULARY)
check("i in vocab",        'i'      in VOCABULARY)
check("very in vocab",     'very'   in VOCABULARY)
check("little in vocab",   'little' in VOCABULARY)
check("water in vocab",    'water'  in VOCABULARY)

for word in ['food', 'hungry', 'danger']:
    vec, n_frames = VOCABULARY[word]
    check(f"{word} vector shape", vec.ndim == 1 and len(vec) > 0, f"shape={vec.shape}")
    check(f"{word} has frames",   n_frames >= 1,                  f"frames={n_frames}")


# ══════════════════════════════════════════════════════════
# 2. SEMANTIC MEMORY — store, recall, contradiction
# ══════════════════════════════════════════════════════════
section("2. Semantic Memory")

from m75_semantic import SemanticMemory
sem = SemanticMemory()

# Basic store/recall
sem.learn_from_sentence(['food', 'is', 'good'])
rels = sem.recall_relations('food', 'is')
check("store X is Y",       len(rels) > 0,                   f"food is: {rels}")
check("correct object",     rels and rels[0]['object'] == 'good')

# Name learning
sem.learn_from_sentence(['my', 'name', 'is', 'pranay'])
obj = sem._latest_relation_object('you', 'name')
check("learn user name",    obj == 'pranay',                 f"got: {obj}")

sem.learn_from_sentence(['your', 'name', 'is', 'fastbrain'])
obj = sem._latest_relation_object('i', 'name')
check("learn brain name",   obj == 'fastbrain',              f"got: {obj}")

# Causal learning
sem.learn_from_sentence(['danger', 'causes', 'fear'])
rels = sem.recall_relations('danger', 'causes')
check("learn X causes Y",   rels and 'fear' in rels[0]['object'])

sem.learn_from_sentence(['afraid', 'because', 'danger'])
rels = sem.recall_relations('afraid', 'because')
check("learn X because Y",  len(rels) > 0,  f"got: {rels}")

# Question words must NOT become subjects
sem.learn_from_sentence(['what', 'is', 'food'])
rels = sem.recall_relations('what')
check("no question-subject", len(rels) == 0,                 f"got: {rels}")

# State queries no longer return templates
ans = sem.find_relation_answer(['what', 'do', 'you', 'feel'])
check("feel query returns None", ans is None,                f"got: {ans}")
ans = sem.find_relation_answer(['what', 'do', 'you', 'want'])
check("want query returns None", ans is None,                f"got: {ans}")
ans = sem.find_relation_answer(['what', 'do', 'you', 'need'])
check("need query returns None", ans is None,                f"got: {ans}")

# Factual queries still return templates
sem.learn_from_sentence(['xylophone', 'is', 'music'])
ans = sem.find_relation_answer(['what', 'is', 'xylophone'])
check("what is X returns string", ans is not None and 'xylophone' in ans, f"got: {ans}")

# Contradiction detection
sem2 = SemanticMemory()
sem2.learn_from_sentence(['food', 'is', 'good'])
sem2.learn_from_sentence(['food', 'is', 'bad'])
check("contradiction detected",       sem2.contradiction_count() == 1,
      f"count={sem2.contradiction_count()}")
conflict = sem2.recent_contradictions(1)[0]
check("conflict has stored+new",
      conflict['stored'] == 'good' and conflict['new'] == 'bad',
      f"{conflict}")

# State relation change is NOT a contradiction
sem2.learn_from_sentence(['i', 'feel', 'calm'])
sem2.learn_from_sentence(['i', 'feel', 'afraid'])
check("state change not contradiction", sem2.contradiction_count() == 1,
      f"count={sem2.contradiction_count()}")


# ══════════════════════════════════════════════════════════
# 3. BRAIN MODULES — SelfModel, WorkingMemory, LogicModule
# ══════════════════════════════════════════════════════════
section("3. Brain Modules")

from brain_modules import (WorkingMemory, EpisodicMemory, SelfModel,
                            CuriosityModule, LogicModule)

# SelfModel
sm = SelfModel()
check("initial state calm",     sm.dominant_state() == 'calm')
sm._state['hunger'] = 0.8
check("hunger dominant hungry", sm.dominant_state() == 'hungry')
sm._state['fear']   = 0.6
check("fear overrides hungry",  sm.dominant_state() == 'afraid')
sm._state['uncertainty'] = 0.7
check("uncertainty overrides fear", sm.dominant_state() == 'uncertain')

sm2 = SelfModel()
sm2.spike_uncertainty(0.5)
check("spike uncertainty", sm2._state['uncertainty'] == 0.5)
sm2.resolve_uncertainty(1.0)
check("resolve uncertainty", sm2._state['uncertainty'] < 0.5)

words = sm2.state_words()
check("state_words returns list", isinstance(words, list))

vec = sm2.as_vector()
check("as_vector 6-dim",   vec.shape == (6,))

# WorkingMemory
wm = WorkingMemory(500)
wm.update(bmu=100, heard_words=['food'], word_to_bmu={'food': 100})
check("WM activation after update", wm.activation_strength() > 0)

wm.enter_seeking()
check("WM seeking on",     wm._seeking == True)
wm.tick_seeking()
wm.tick_seeking()
wm.tick_seeking()
check("WM seeking auto-release after 3 ticks", wm._seeking == False)

wm2 = WorkingMemory(500)
wm2.enter_seeking()
wm2.resolve_seeking()
check("WM resolve_seeking", wm2._seeking == False)

# EpisodicMemory
ep = EpisodicMemory(maxlen=10)
ep.record(['food','eat'], ['eat','food'], 'food', {'hunger':0.8}, 1.0)
ep.record(['danger'],     ['run'],        'danger', {'fear':0.6}, 0.0)
check("episodic count",     ep.count() == 2)
check("last topic danger",  ep.last_topic() == 'danger')
similar = ep.recall_similar('food')
check("recall similar food", len(similar) == 1)

# CuriosityModule with priors
cur = CuriosityModule(500)
cur._zone_counts = {'food': 500, 'danger': 300, 'rest': 400,
                    'action': 350, 'pain': 200, 'water': 20, 'social': 5}
least = cur.least_visited_zone(list(cur._zone_counts.keys()))
check("least visited is social", least == 'social',          f"got: {least}")

# LogicModule
sem3 = SemanticMemory()
sem3.store_relation('hungry', 'causes', 'want food')
sem3.store_relation('danger', 'means',  'run careful')
sem3.store_relation('xylophone', 'is',  'music')
sem3.store_relation('music',     'is',  'good')

sm3 = SelfModel()
sm3._state['hunger'] = 0.8
logic = LogicModule()
vocab = set(VOCABULARY)

# Type 1: drive inference
conc = logic.infer(sm3, sem3, 'social', [], vocab)
check("logic Type1 hunger→food words",
      any(w in conc for w in ['want','food','eat']),        f"got: {conc}")

# Type 2: zone-percept inference
sm3._state['hunger'] = 0.3  # reset hunger
conc2 = logic.infer(sm3, sem3, 'danger', [], vocab)
check("logic Type2 danger zone→run/careful",
      any(w in conc2 for w in ['run','careful']),           f"got: {conc2}")

# Type 3: 2-hop chain from heard words
conc3 = logic.infer(sm3, sem3, 'social', ['xylophone'], vocab | {'music','good'})
check("logic Type3 2-hop xylophone→music",
      'music' in conc3,                                     f"got: {conc3}")

check("logic confidence set", logic._confidence > 0)


# ══════════════════════════════════════════════════════════
# 4. ZONE ROUTING — correct zone from words
# ══════════════════════════════════════════════════════════
section("4. Zone Routing")

# Import live_fast carefully (loads brain — skip if no pkl)
import os
if not os.path.exists('brain_fast.pkl'):
    print(f"{WARN}  brain_fast.pkl missing — skipping zone routing tests")
    print(f"       Run: python train_world6_fast.py")
else:
    # Patch sys to avoid main() running
    from live_fast import (_resolve_zone, ZONE_ANCHORS, generate_response,
                           brain, curiosity, semantic, _relation_response)

    def zone_for(words):
        bmus = [brain.word_to_bmu[w] for w in words if w in brain.word_to_bmu]
        return _resolve_zone(words, bmus)

    check("food words → food zone",     zone_for(['food','eat','hungry']) == 'food')
    check("danger words → danger zone", zone_for(['danger','afraid'])     == 'danger')
    check("water words → water zone",   zone_for(['water','river'])       == 'water')
    check("rest words → rest zone",     zone_for(['tired','sleep'])       == 'rest')
    check("social words → social",      zone_for(['hi','hello'])          == 'social')
    check("plant words → rest zone",    zone_for(['plant','tree','grass'])== 'rest')
    check("river → water not food",     zone_for(['river','rain'])        == 'water')

    # Curiosity priors — social/water should be least visited at session start
    least = curiosity.least_visited_zone(list(ZONE_ANCHORS.keys()))
    check("curiosity least zone is social/water",
          least in ('social', 'water'),  f"got: {least}")

    # Generate responses — basic sanity
    def gen(words):
        bmus = [brain.word_to_bmu[w] for w in words if w in brain.word_to_bmu]
        return generate_response(bmus, words)

    r_hi = gen(['hi'])
    check("hi → non-empty response", len(r_hi) > 0,  f"got: '{r_hi}'")
    check("hi → social word",
          any(w in r_hi for w in ['hello','hi','yes','happy']),  f"got: '{r_hi}'")

    r_danger = gen(['danger'])
    check("danger → fear word",
          any(w in r_danger for w in ['run','careful','afraid','danger']),
          f"got: '{r_danger}'")

    r_food = gen(['food','eat'])
    check("food → food word",
          any(w in r_food for w in ['eat','food','hungry','want','full']),
          f"got: '{r_food}'")

    # No-template: state queries return None → routed through WordTP
    check("feel query → None (no template)",
          _relation_response(['what','do','you','feel']) is None)
    check("want query → None (no template)",
          _relation_response(['what','do','you','want']) is None)
    check("need query → None (no template)",
          _relation_response(['what','do','you','need']) is None)

    # Factual queries still return strings
    semantic.store_relation('testword', 'is', 'testmeaning')
    ans = semantic.find_relation_answer(['what', 'is', 'testword'])
    check("what is X still returns string",
          ans is not None and 'testword' in ans,  f"got: {ans}")


# ══════════════════════════════════════════════════════════
# 5. SELF MODEL STATE REPORTING
# ══════════════════════════════════════════════════════════
section("5. Self-Model State")

sm4 = SelfModel()
sm4._state['hunger']  = 0.0
sm4._state['fatigue'] = 0.0
sm4._state['fear']    = 0.0
sm4._state['uncertainty'] = 0.0
check("calm when all drives low",     sm4.dominant_state() == 'calm')

sm4._state['hunger'] = 0.65
check("hungry when hunger=0.65",      sm4.dominant_state() == 'hungry')
check("hungry state_words has food",
      any(w in ['hungry','want','food','need'] for w in sm4.state_words()))

sm4._state['hunger'] = 0.0   # reset so fatigue can dominate
sm4._state['fatigue'] = 0.65
check("tired when fatigue=0.65",      sm4.dominant_state() == 'tired')

sm4._state['fear'] = 0.55
check("afraid when fear=0.55",        sm4.dominant_state() == 'afraid')

# Drive update
sm5 = SelfModel()
init_hunger = sm5._state['hunger']
sm5.update('food', reward=1.0, wm_strength=0.3, episode_count=5)
check("hunger decreases in food zone", sm5._state['hunger'] <= init_hunger)

init_fear = sm5._state['fear']
sm5.update('danger', reward=0.0, wm_strength=0.3, episode_count=5)
check("fear increases in danger zone", sm5._state['fear'] >= init_fear)


# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print(f"\n{'═'*55}")
passed = sum(1 for _, ok in results if ok)
total  = len(results)
failed = total - passed
color  = "\033[92m" if failed == 0 else "\033[91m"
print(f"  {color}{passed}/{total} passed  ({failed} failed)\033[0m")
if failed:
    print("\n  Failed tests:")
    for name, ok in results:
        if not ok:
            print(f"    ✗  {name}")
print(f"{'═'*55}\n")
sys.exit(0 if failed == 0 else 1)
