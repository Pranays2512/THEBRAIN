#!/usr/bin/env python3
"""
test_conversation.py — Automated conversation test suite for FastBrain.

Runs a set of probes, scores each response, prints a report.
Does NOT require live interaction.

Usage:
    python test_conversation.py
    python test_conversation.py --verbose
"""

import sys, argparse

sys.dont_write_bytecode = True

# ── Import the live system (module-level state initialises on import) ─────────
# Suppress live_fast's startup prints
import io, contextlib
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    from live_fast import (
        generate_response, feed_input, semantic,
        VOCABULARY, brain, SILENCE, _relation_response
    )

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES
# Each entry:
#   input      : str   — what the user says
#   must_contain: list  — at least one of these words must appear in response
#   must_not   : list  — none of these words may appear
#   label      : str   — test name
#   category   : str   — group for summary
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [

    # ── GREETINGS ─────────────────────────────────────────────────────────────
    {   'label': 'greeting_hi',
        'category': 'greeting',
        'input': 'hi',
        'must_contain': ['hello', 'hi', 'happy'],
        'must_not': ['hurt', 'pain', 'run', 'afraid', 'eat', 'food'],
    },
    {   'label': 'greeting_hello',
        'category': 'greeting',
        'input': 'hello',
        'must_contain': ['hello', 'hi', 'happy'],
        'must_not': ['hurt', 'pain', 'run', 'afraid'],
    },

    # ── SELF IDENTITY ─────────────────────────────────────────────────────────
    {   'label': 'identity_who_are_you',
        'category': 'identity',
        'input': 'who are you',
        'must_contain': ['fastbrain', 'brain', 'think', 'learn', 'feel', 'am'],
        'must_not': ['eat', 'food', 'wall', 'hurt', 'run'],
    },
    {   'label': 'identity_what_is_your_name',
        'category': 'identity',
        'input': 'what is your name',
        'must_contain': ['fastbrain', 'name', 'brain'],
        'must_not': ['eat', 'food', 'hurt'],
    },
    {   'label': 'identity_are_you_alive',
        'category': 'identity',
        'input': 'are you alive',
        'must_contain': ['alive', 'yes', 'feel', 'hungry', 'tired', 'awake', 'sleep', 'calm'],
        'must_not': ['run', 'wall', 'hurt'],
    },
    {   'label': 'identity_do_you_think',
        'category': 'identity',
        'input': 'do you think',
        'must_contain': ['think', 'yes', 'learn', 'know', 'feel', 'brain', 'fastbrain'],
        'must_not': ['eat', 'hurt', 'run'],
    },

    # ── SELF STATE ────────────────────────────────────────────────────────────
    {   'label': 'state_how_do_you_feel',
        'category': 'self_state',
        'input': 'how do you feel',
        'must_contain': ['feel', 'calm', 'hungry', 'tired', 'good', 'afraid'],
        'must_not': ['run', 'wall', 'door', 'button'],
    },
    {   'label': 'state_are_you_hungry',
        'category': 'self_state',
        'input': 'are you hungry',
        'must_contain': ['hungry', 'eat', 'food', 'want', 'yes', 'no'],
        'must_not': ['run', 'door', 'button', 'afraid'],
    },
    {   'label': 'state_are_you_tired',
        'category': 'self_state',
        'input': 'are you tired',
        'must_contain': ['tired', 'sleep', 'calm', 'yes', 'no', 'safe', 'warm', 'awake'],
        'must_not': ['run', 'wall', 'food', 'eat'],
    },

    # ── FOOD ZONE ─────────────────────────────────────────────────────────────
    {   'label': 'food_hungry',
        'category': 'food',
        'input': 'hungry',
        'must_contain': ['eat', 'food', 'hungry', 'want', 'drink', 'water'],
        'must_not': ['run', 'afraid', 'hurt', 'sleep', 'hello'],
    },
    {   'label': 'food_eat',
        'category': 'food',
        'input': 'eat',
        'must_contain': ['eat', 'food', 'hungry', 'want'],
        'must_not': ['run', 'afraid', 'hurt', 'sleep'],
    },
    {   'label': 'food_water',
        'category': 'food',
        'input': 'water',
        'must_contain': ['water', 'drink', 'eat', 'food', 'full', 'hungry'],
        'must_not': ['run', 'afraid', 'hurt'],
    },

    # ── PAIN ZONE ─────────────────────────────────────────────────────────────
    {   'label': 'pain_wall',
        'category': 'pain',
        'input': 'wall',
        'must_contain': ['hurt', 'pain', 'stop', 'bad', 'no'],
        'must_not': ['eat', 'food', 'run', 'afraid', 'sleep'],
    },
    {   'label': 'pain_hurt',
        'category': 'pain',
        'input': 'hurt',
        'must_contain': ['hurt', 'pain', 'stop', 'bad', 'no'],
        'must_not': ['eat', 'food', 'run', 'sleep'],
    },
    {   'label': 'pain_wall_hurts',
        'category': 'pain',
        'input': 'wall hurts',
        'must_contain': ['hurt', 'pain', 'stop', 'bad', 'no', 'hurts'],
        'must_not': ['eat', 'food', 'run', 'sleep'],
    },

    # ── REST ZONE ─────────────────────────────────────────────────────────────
    {   'label': 'rest_sleep',
        'category': 'rest',
        'input': 'sleep',
        'must_contain': ['sleep', 'calm', 'tired', 'warm', 'safe'],
        'must_not': ['eat', 'food', 'run', 'afraid', 'hurt'],
    },
    {   'label': 'rest_tired',
        'category': 'rest',
        'input': 'tired',
        'must_contain': ['sleep', 'calm', 'tired', 'warm'],
        'must_not': ['eat', 'food', 'run', 'afraid'],
    },

    # ── DANGER ZONE ───────────────────────────────────────────────────────────
    {   'label': 'danger_danger',
        'category': 'danger',
        'input': 'danger',
        'must_contain': ['run', 'afraid', 'careful', 'danger'],
        'must_not': ['eat', 'food', 'sleep', 'hello', 'hurt'],
    },
    {   'label': 'danger_afraid',
        'category': 'danger',
        'input': 'afraid',
        'must_contain': ['run', 'afraid', 'careful', 'danger'],
        'must_not': ['eat', 'food', 'sleep', 'hello'],
    },
    {   'label': 'danger_here',
        'category': 'danger',
        'input': 'danger here',
        'must_contain': ['run', 'careful', 'afraid', 'danger'],
        'must_not': ['eat', 'food', 'sleep', 'hello'],
    },

    # ── ACTION ZONE ───────────────────────────────────────────────────────────
    {   'label': 'action_open_door',
        'category': 'action',
        'input': 'open door',
        'must_contain': ['open', 'door', 'go', 'move', 'push'],
        'must_not': ['eat', 'food', 'hurt', 'sleep', 'afraid'],
    },
    {   'label': 'action_push_button',
        'category': 'action',
        'input': 'push button',
        'must_contain': ['push', 'button', 'open', 'door', 'go'],
        'must_not': ['eat', 'food', 'hurt', 'sleep', 'afraid'],
    },
    {   'label': 'action_door',
        'category': 'action',
        'input': 'door',
        'must_contain': ['door', 'open', 'go', 'move'],
        'must_not': ['eat', 'food', 'hurt', 'sleep', 'afraid'],
    },

    # ── RELATION WORDS ────────────────────────────────────────────────────────
    {   'label': 'relation_do_you_like_food',
        'category': 'relation',
        'input': 'do you like food',
        'must_contain': ['like', 'food', 'yes', 'eat', 'want'],
        'must_not': ['run', 'afraid', 'hurt', 'wall'],
    },
    {   'label': 'relation_do_you_hate',
        'category': 'relation',
        'input': 'do you hate pain',
        'must_contain': ['hate', 'pain', 'yes', 'hurt', 'stop', 'bad'],
        'must_not': ['eat', 'food', 'run', 'sleep'],
    },
    {   'label': 'relation_what_do_you_like',
        'category': 'relation',
        'input': 'what do you like',
        'must_contain': ['like', 'food', 'sleep', 'calm'],
        'must_not': ['run', 'afraid', 'wall', 'hurt'],
    },

    # ── SEMANTIC TEACHING (teach then query) ─────────────────────────────────
    {   'label': 'teach_user_name',
        'category': 'semantic',
        'input': 'my name is pranay',
        'must_contain': [],        # teaching — any non-empty response OK
        'must_not': ['hurt', 'run', 'pain'],
        'teaches': ('you', 'name', 'pranay'),   # verify relation stored
    },
    {   'label': 'recall_user_name',
        'category': 'semantic',
        'input': 'what is my name',
        'must_contain': ['pranay'],
        'must_not': ['hurt', 'run', 'pain'],
        'requires_teach': ('you', 'name', 'pranay'),
    },

    # ── CAUSAL REASONING ─────────────────────────────────────────────────────
    {   'label': 'causal_teach_wall_hurts',
        'category': 'causal',
        'input': 'wall hurts because hard',
        'must_contain': [],       # teaching — any response OK
        'must_not': ['run', 'afraid', 'sleep'],
        'teaches_relation': ('wall hurts', 'because', 'hard'),
    },
    {   'label': 'causal_recall_why_hurt',
        'category': 'causal',
        'input': 'why hurt',
        'must_contain': ['hurt', 'because', 'wall', 'pain', 'stop'],
        'must_not': ['eat', 'food', 'run', 'sleep'],
    },
    {   'label': 'causal_teach_danger_afraid',
        'category': 'causal',
        'input': 'danger causes afraid',
        'must_contain': [],       # teaching
        'must_not': ['eat', 'food', 'sleep'],
    },

    # ── NO REPETITION ────────────────────────────────────────────────────────
    {   'label': 'no_word_doubling',
        'category': 'quality',
        'input': 'sleep',
        'must_contain': ['sleep', 'calm', 'tired'],
        'must_not': [],
        'check_no_repeat': True,
    },
    {   'label': 'no_zone_bleed_food_in_danger',
        'category': 'quality',
        'input': 'danger',
        'must_contain': ['run', 'afraid', 'careful'],
        'must_not': ['eat', 'food', 'hungry', 'sleep', 'calm'],
    },
    {   'label': 'no_zone_bleed_danger_in_food',
        'category': 'quality',
        'input': 'hungry',
        'must_contain': ['eat', 'food', 'want', 'full', 'hungry'],
        'must_not': ['run', 'afraid', 'careful', 'sleep', 'tired'],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_probe(test: dict, verbose: bool) -> dict:
    """Run one test, return result dict."""
    inp = test['input']
    tokens = inp.lower().split()
    heard_words = [w for w in tokens if w in VOCABULARY]

    # Pre-teach if this test requires a prior relation
    if 'requires_teach' in test:
        s, r, o = test['requires_teach']
        semantic.store_relation(s, r, o)

    # Get response — relation memory first, then generation
    heard_bmus = feed_input(heard_words) if heard_words else []
    for _ in range(4):
        brain.hear(SILENCE)
        brain.step()

    resp = _relation_response(tokens)
    if resp is None:
        resp = generate_response(heard_bmus, heard_words, raw_tokens=tokens)

    resp = resp or ''
    resp_words = set(resp.lower().split())

    # Check must_contain (at least one match if list non-empty)
    mc = test.get('must_contain', [])
    hit = any(w in resp_words for w in mc) if mc else True

    # Check must_not (none allowed)
    mn = test.get('must_not', [])
    bleed = [w for w in mn if w in resp_words]
    bleed_ok = len(bleed) == 0

    # Check no word repeat
    rep_ok = True
    if test.get('check_no_repeat'):
        words_list = resp.split()
        rep_ok = len(words_list) == len(set(words_list))

    # Check teaching side effect (triple: subject, relation, object)
    teach_ok = True
    if 'teaches' in test:
        s, r, o = test['teaches']
        semantic.learn_from_sentence(tokens)
        rels = semantic.recall_relations(s, r)
        teach_ok = any(rel['object'] == o for rel in rels)

    # Check causal relation was stored
    if 'teaches_relation' in test:
        s, r, o = test['teaches_relation']
        semantic.learn_from_sentence(tokens)
        rels = semantic.recall_relations(s, r)
        teach_ok = any(rel['object'] == o for rel in rels)

    passed = hit and bleed_ok and rep_ok and teach_ok

    if verbose:
        status = '✓' if passed else '✗'
        print(f"  [{status}] {test['label']}")
        print(f"       you: {inp!r}")
        print(f"     brain: {resp!r}")
        if not hit and mc:
            print(f"     MISS: needed one of {mc}")
        if not bleed_ok:
            print(f"     BLEED: found {bleed} (must_not)")
        if not rep_ok:
            print(f"     REPEAT: words repeat in response")
        if not teach_ok:
            print(f"     TEACH FAIL: relation not stored")

    return {
        'label':    test['label'],
        'category': test['category'],
        'passed':   passed,
        'response': resp,
        'bleed':    bleed,
        'hit':      hit,
        'rep_ok':   rep_ok,
        'teach_ok': teach_ok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  BRAIN CONVERSATION TEST SUITE")
    print(f"  {len(TESTS)} probes | vocab={len(VOCABULARY)} words")
    print("=" * 60)

    results = []
    categories: dict[str, list] = {}

    for test in TESTS:
        r = run_probe(test, verbose=args.verbose)
        results.append(r)
        cat = r['category']
        categories.setdefault(cat, []).append(r)
        if not args.verbose:
            sym = '✓' if r['passed'] else '✗'
            print(f"  {sym} {r['label']:40s}  {r['response']!r}")

    # Summary by category
    print("\n" + "─" * 60)
    print("  CATEGORY SUMMARY")
    print("─" * 60)
    total_pass = 0
    for cat, rs in sorted(categories.items()):
        n = len(rs)
        p = sum(1 for r in rs if r['passed'])
        total_pass += p
        bar = '█' * p + '░' * (n - p)
        print(f"  {cat:15s} {bar}  {p}/{n}")

    total = len(results)
    pct = total_pass / total * 100
    print("─" * 60)
    print(f"  TOTAL: {total_pass}/{total}  ({pct:.0f}%)")

    # Failures detail
    fails = [r for r in results if not r['passed']]
    if fails:
        print(f"\n  FAILURES ({len(fails)}):")
        for r in fails:
            issues = []
            if not r['hit']:      issues.append('missing word')
            if not r['rep_ok']:   issues.append('word repeat')
            if r['bleed']:        issues.append(f"zone bleed: {r['bleed']}")
            if not r['teach_ok']: issues.append('teach failed')
            print(f"    {r['label']:40s} → {', '.join(issues)}")
            print(f"         response: {r['response']!r}")

    print()
    return 0 if total_pass == total else 1


if __name__ == '__main__':
    sys.exit(main())
