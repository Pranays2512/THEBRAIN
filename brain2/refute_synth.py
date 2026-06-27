#!/usr/bin/env python3
"""
refute_synth.py — wire the refuter into synthesis: self-correct an overfit with a DIAGNOSIS.

synth_engine.solve picks a program that fits the given examples. If the examples are
skewed (e.g. all-positive), it can return an overfit that passes them but is wrong
elsewhere. stress() flags that with a single counterexample. The refuter does better: it
returns WHERE the rule breaks (scope), and we feed that counterexample back as a new
constraint and re-synthesize — the loop corrects itself, with a reason, not just a retry.

  loop: solve(examples) -> refute(vs oracle) -> if it breaks, add the counterexample to
        examples and re-solve; stop when robust (or out of tries).

This is stress-in-the-loop upgraded from "reject" to "diagnose + repair", and it connects
the new break-engine to the synthesizer the brain already runs.

Honest limit: needs an oracle to refute against (same bound as all verification here); the
repair only works if the correct program is reachable in the DSL once the example is added.
"""

import refuter as RF
import synth_engine as SE


def synth_self_correct(kind, oracle, inputs, max_iters=5):
    examples = SE._ex(kind, oracle, inputs)
    log = []
    code = None
    for it in range(max_iters):
        _space, code = SE.solve(examples, kind)
        if code is None:
            log.append("  iter %d: no candidate" % it)
            return None, log
        rep = RF.refute(code, oracle, kind)
        if rep["robust"]:
            log.append("  iter %d: ROBUST  (scope: %s)" % (it, rep["scope"]))
            return code, log
        bad = rep["breaks_at"]
        log.append("  iter %d: breaks at %s  [%s]  -> add counterexample, re-synth"
                   % (it, bad, rep["scope"]))
        try:
            examples.append((bad, oracle(*bad)))      # the break becomes a new constraint
        except Exception:
            return code, log
    log.append("  gave up after %d iters" % max_iters)
    return code, log


def _demo():
    print("=== refute_synth — diagnose an overfit and repair it ===\n")

    # max_list, but ALL-POSITIVE example inputs -> the easy fit is init=0 (overfits;
    # wrong on all-negative lists). The refuter finds the negative break and repairs it.
    print("[max_list] seeded with all-positive examples (the overfit trap):")
    code, log = synth_self_correct("list", max, [[3, 1, 4], [2, 7], [5], [6, 2, 9]])
    for line in log:
        print(line)
    ok, ce = SE.stress(code, max, "list")
    print("  final program stress vs real max: %s\n" % ("PASS" if ok else "FAIL %s" % str(ce)))

    # a control: a task that's already fine -> robust on iter 0, no wasted repair
    print("[sum_list] well-posed examples:")
    code, log = synth_self_correct("list", sum, [[1, 2, 3], [], [5], [-2, 4]])
    for line in log:
        print(line)
    print("\n  The refuter turned 'this fails somewhere' into 'it fails HERE' and fed the"
          " fix back — synthesis self-corrected instead of silently shipping the overfit.")


if __name__ == "__main__":
    _demo()
