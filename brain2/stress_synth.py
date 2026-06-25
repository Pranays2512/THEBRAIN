#!/usr/bin/env python3
"""
stress_synth.py — stress-IN-THE-LOOP synthesis: self-corrects overfits, no hand help.

synth_engine's stress gate caught the max_list overfit AFTER synthesis, and the fix
was a human adding negative examples. This removes the human: the synthesizer itself
gates every candidate by stress-vs-oracle DURING the search, so it skips a program
that fits the examples but fails random cases and keeps looking for one that survives.

  for each candidate (in DSL order):
      fits the examples?  ->  AND survives 1000 stress cases vs oracle?  ->  return it
  the overfit (max_list init=0, fits all-positive examples) is auto-skipped; the
  search continues to init=first, which survives — with the SAME positive-only examples.

    python3 stress_synth.py
"""

from loop_synth4 import INITS, FOLD_UPD, _run_fold, render as render4
from synth_engine import stress


def _fits(spec, data):
    try:
        return all(_run_fold(spec["init"], FOLD_UPD[spec["upd"]], lst) == y
                   for lst, y in data)
    except Exception:
        return False


def synth_fold(examples, oracle, use_stress):
    """examples: [((lst,), y)]. use_stress -> gate each candidate by stress-vs-oracle."""
    data = [(a[0], y) for a, y in examples]
    for ik in INITS:
        for uc in FOLD_UPD:
            spec = {"init": ik, "upd": uc}
            if not _fits(spec, data):
                continue
            code = render4("f", "fold", spec)
            if use_stress:
                surv, _ = stress(code, oracle, "list")
                if not surv:
                    continue                     # overfit: fits examples, fails stress
            return code, spec
    return None, None


def _demo():
    # POSITIVE-ONLY examples (the trap): example-fit picks init=0, which is wrong
    # for negatives. No negative examples are added by hand.
    ex = [(([3, 1, 4],), 4), (([2, 2],), 2), (([7],), 7), (([9, 1],), 9)]

    print("=== stress_synth — synthesis self-corrects overfits (no hand-added examples) ===\n")
    print("  task: max_list, POSITIVE-ONLY examples (the overfit trap)\n")

    code, spec = synth_fold(ex, max, use_stress=False)
    surv, ce = stress(code, max, "list")
    print(f"  example-fit only : init={spec['init']}, upd={spec['upd']}  "
          f"-> stress {'survives' if surv else f'FAILS at {ce}'}  (overfit shipped)")

    code, spec = synth_fold(ex, max, use_stress=True)
    surv, ce = stress(code, max, "list")
    print(f"  stress-in-loop   : init={spec['init']}, upd={spec['upd']}  "
          f"-> stress {'SURVIVES ✓' if surv else 'fails'}  (self-corrected)")
    print("\n  same positive-only examples; the in-loop stress gate skipped init=0 and")
    print("  found init=first on its own — verification drives the search, not the human.")


if __name__ == "__main__":
    _demo()
