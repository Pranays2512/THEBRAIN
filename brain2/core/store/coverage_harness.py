#!/usr/bin/env python3
"""coverage_harness.py — the deletion metric. Measures which rung resolves each held-out
question and whether the resolved rel is CORRECT. A student/LLM rung is deletable for a
domain when template_pct clears threshold on FROZEN held-out data — not by vibes.
(Plan Phase A, Task 7.)"""

from collections import Counter


def coverage(resolve, held_out):
    """resolve: q -> (entity, rel, source). held_out: [(question, expected_rel)] where
    expected_rel None means the honest answer is 'I don't know'."""
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


def event_coverage(read, sentences):
    """Event intake coverage. read: sentence -> disposition in {admit, reject, abstain,
    nomatch}. PARSE coverage = fraction that becomes a JUDGED event (anything but nomatch) —
    the honest 'how much open prose reaches the membrane at all' number. (nomatch = no known
    verb -> the intake couldn't even form an event.)"""
    c = Counter(read(s) for s in sentences)
    n = max(len(sentences), 1)
    parsed = sum(v for k, v in c.items() if k != "nomatch")
    rep = dict(c)
    rep["n"] = len(sentences)
    rep["parsed_pct"] = parsed / n
    return rep


def event_coverage_split(read, taught, wild):
    """Taught prose (known verbs/entities) flatters; `wild` (real prose outside the taught
    lexicon) is the honest open-language intake number. gap = how much the taught figure
    over-states current reach."""
    t = event_coverage(read, taught)
    w = event_coverage(read, wild)
    return {"taught": t, "wild": w, "gap": t["parsed_pct"] - w["parsed_pct"],
            "wild_parsed_pct": w["parsed_pct"]}


def coverage_split(resolve, taught, wild):
    """The honest open-language number. Taught-domain coverage FLATTERS: the grammar was
    fitted there. `wild` is held-out text from OUTSIDE taught domains — the real 'is language
    owned yet' signal. Reports both plus the gap so the flattering figure can't be quoted
    alone. A student/LLM rung is only deletable when WILD template_pct clears threshold."""
    t = coverage(resolve, taught)
    w = coverage(resolve, wild)
    return {"taught": t, "wild": w,
            "gap": t["template_pct"] - w["template_pct"],
            "wild_template_pct": w["template_pct"],
            "wild_correct_pct": w["correct_pct"]}
