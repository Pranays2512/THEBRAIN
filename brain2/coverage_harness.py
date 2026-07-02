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
