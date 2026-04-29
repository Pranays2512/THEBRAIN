"""
test_brain.py — Systematic tests for THE BRAIN after vocab expansion.
Imports live_fast functions directly — no UI loop.
"""
import sys
sys.dont_write_bytecode = True

# Suppress setup noise for cleaner output
import io
_buf = io.StringIO()
_real_stdout = sys.stdout
sys.stdout = _buf

import live_fast as lf

sys.stdout = _real_stdout
setup_out = _buf.getvalue()
for line in setup_out.splitlines():
    if any(k in line for k in ['FastBrain', 'Word TP', 'Semantic', 'grounded words', 'remembered', 'BMU']):
        print(line)

print()

PASS = 0
FAIL = 0


def say(words_str: str) -> str:
    words = words_str.lower().split()
    bmus  = lf.feed_input(words)
    resp  = lf.generate_response(bmus, words, raw_tokens=words)
    lf.hear_own_response(resp.split() if resp else [])
    return resp


def check(label: str, words_str: str, must_contain=None, must_not_contain=None):
    global PASS, FAIL
    resp = say(words_str)
    ok = True
    reasons = []
    if must_contain:
        for w in must_contain:
            if w not in resp.split():
                ok = False
                reasons.append(f"missing '{w}'")
    if must_not_contain:
        for w in must_not_contain:
            if w in resp.split():
                ok = False
                reasons.append(f"unexpected '{w}'")
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    pad = " " * max(0, 34 - len(label))
    print(f"  [{status}] {label}{pad} | you: {words_str!r:32s} | brain: {resp!r}")
    if reasons:
        print(f"         Reasons: {', '.join(reasons)}")
    return resp


print("=" * 80)
print("  IDENTITY")
print("=" * 80)
check("self name query",       "what is your name",   must_contain=['fastbrain'])
check("who are you",           "who are you",          must_contain=['fastbrain'])
check("user name recall",      "who am i",             must_contain=['pranay'])

print()
print("=" * 80)
print("  GROUNDED DRIVES")
print("=" * 80)
check("hunger response",       "i feel hungry",        must_contain=['food'])
check("danger response",       "danger here",          must_contain=['run'])
check("tired response",        "feel tired",           must_not_contain=['danger'])
check("pain/wall response",    "wall hurt",            must_contain=['stop'])
check("safe response",         "feel safe",            must_contain=['calm'])
check("full after eat",        "eat food full",        must_not_contain=['danger'])

print()
print("=" * 80)
print("  NEW SPATIAL WORDS")
print("=" * 80)
check("found food",            "found food",           must_contain=['eat'])
check("near food",             "near food",            must_contain=['eat'])
check("far from food",         "far food",             must_contain=['food'])
check("lost response",         "i am lost",            must_not_contain=['eat'])
check("free response",         "i feel free",          must_not_contain=['danger'])
check("out danger",            "out of danger",        must_not_contain=['wall'])

print()
print("=" * 80)
print("  NEW TEMPORAL WORDS")
print("=" * 80)
check("after eat",             "after eat feel full",  must_not_contain=['danger'])
check("before sleep",          "before sleep tired",   must_contain=['sleep'])
check("again hungry",          "hungry again",         must_contain=['eat'])
check("still tired",           "still tired",          must_contain=['sleep'])

print()
print("=" * 80)
print("  NEW SOCIAL WORDS")
print("=" * 80)
check("talk with you",         "talk with you")
check("listen response",       "listen")
check("alone response",        "feel alone",           must_not_contain=['danger'])
check("with you",              "with you")

print()
print("=" * 80)
print("  NEW SENSORY/ACTION WORDS")
print("=" * 80)
check("look for food",         "look for food",        must_contain=['eat'])
check("see food",              "see food",             must_contain=['eat'])
check("try open door",         "try open door",        must_contain=['open'])
check("try push button",       "try push button",      must_contain=['push'])

print()
print("=" * 80)
print("  NEW MOOD WORDS")
print("=" * 80)
check("dark danger",           "dark danger",          must_contain=['run'])
check("light safe",            "light",                must_contain=['calm'])
check("worry response",        "i worry",              must_contain=['stop'])

print()
print("=" * 80)
print("  QUESTION WORDS")
print("=" * 80)
check("where is food",         "where is food",        must_contain=['eat'])
check("how open door",         "how do i open door",   must_contain=['open'])
check("who are you",           "who are you",          must_contain=['fastbrain'])

print()
print("=" * 80)
print("  ANTI-REPETITION  (same input 3x)")
print("=" * 80)
r1 = say("hi")
r2 = say("hi")
r3 = say("hi")
all_same = (r1 == r2 == r3)
if not all_same:
    PASS += 1
    print(f"  [PASS] hi x3 vary   r1={r1!r}  r2={r2!r}  r3={r3!r}")
else:
    FAIL += 1
    print(f"  [FAIL] hi x3 all same: {r1!r}")

print()
print("=" * 80)
print("  CORRECTION LOOP")
print("=" * 80)
bl_before = len(lf._BLACKLISTED_RESPONSES)
r1 = check("initial action",   "go push button open door")
lf.apply_feedback(positive=False)
r2 = say("push button open door go")
lf.hear_own_response(r2.split() if r2 else [])
print(f"  [INFO] after neg feedback brain: {r2!r}")
bl_grew = len(lf._BLACKLISTED_RESPONSES) > bl_before
status = "PASS" if bl_grew else "FAIL"
if bl_grew: PASS += 1
else: FAIL += 1
print(f"  [{status}] blacklist grew: {lf._BLACKLISTED_RESPONSES}")

print()
print("=" * 80)
print("  VOCAB COVERAGE")
print("=" * 80)
new_words = ['near','far','found','lost','out','in','free',
             'after','before','again','still',
             'talk','listen','alone','with',
             'look','see','try',
             'dark','light','worry',
             'where','who','how','do',
             'all','never','always']
in_vocab  = [w for w in new_words if w in lf.VOCABULARY]
in_zone   = [w for w in new_words if w in lf._WORD_TO_ZONE]
in_bmu    = [w for w in new_words if w in lf.brain.word_to_bmu]
missing   = [w for w in new_words if w not in lf.brain.word_to_bmu]
print(f"  New words in VOCABULARY    : {len(in_vocab)}/{len(new_words)}")
print(f"  New words in _WORD_TO_ZONE : {len(in_zone)}/{len(new_words)}")
print(f"  New words mapped on SOM    : {len(in_bmu)}/{len(new_words)}")
if missing:
    print(f"  Not yet on SOM             : {missing}")
cov_ok = len(in_vocab) == len(new_words) and len(in_zone) == len(new_words)
if cov_ok: PASS += 1
else: FAIL += 1
print(f"  [{'PASS' if cov_ok else 'FAIL'}] vocab + zone coverage complete")

print()
print("=" * 80)
print(f"  RESULTS:  {PASS} passed  |  {FAIL} failed  |  {PASS+FAIL} total")
print("=" * 80)
