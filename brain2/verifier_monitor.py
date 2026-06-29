#!/usr/bin/env python3
"""
verifier_monitor.py — verify the verifiers: detect irregular / untrustworthy checks.

Self-minted verifiers can go wrong: a SPURIOUS one rejects correct answers (worse than no
check — it blocks truth), a USELESS one never catches anything wrong (dead weight, false
confidence). The brain must audit its own checks, not just trust them. This is the meta
layer: monitor each verifier's behaviour against ground truth + against wrong candidates,
and flag the irregular ones for demotion or pruning.

  SPURIOUS  : rejects a KNOWN-CORRECT output            -> demote (it blocks truth)
  USELESS   : never rejects ANY wrong candidate         -> prune (no discriminative power)
  HEALTHY   : catches wrong ones, never rejects correct -> keep

This closes the loop on self-made verifiers: mine -> validate -> admit -> USE -> AUDIT ->
demote/prune. A verifier only keeps its authority while it behaves.
"""

import invariant_miner as IM


def _passes_one(name, fn, probes):
    """Does fn pass this single invariant on the probes? (no oracle, just the check)."""
    ok, _ = IM.check(fn, probes, {name})
    return ok


def audit(invariants, correct_examples, wrong_fns, probes):
    """Return per-invariant health: (status, catches, spurious)."""
    report = {}
    for name in invariants:
        # SPURIOUS: does it reject any KNOWN-CORRECT output?
        if name == "monotonic_increasing":
            spurious = not IM._monotone(correct_examples)
        else:
            spurious = not IM._holds(name, correct_examples)
        # DISCRIMINATION: how many WRONG candidates does it catch (reject)?
        catches = sum(1 for fn in wrong_fns if not _passes_one(name, fn, probes))
        if spurious:
            status = "SPURIOUS — rejects a correct output -> DEMOTE"
        elif catches == 0:
            status = "USELESS — catches no wrong candidate -> PRUNE"
        else:
            status = "HEALTHY — catches %d/%d wrong" % (catches, len(wrong_fns))
        report[name] = (status, catches, spurious)
    return report


def _demo():
    import math
    print("=== verifier_monitor — audit the verifiers, flag the irregular ones ===\n")
    IM.PREDICATES["out_le_1000"] = lambda a, y: y <= 1000   # inject a spurious one to audit
    correct = [(0, 1), (1, 1), (4, 24), (5, 120), (6, 720), (7, 5040), (8, 40320)]  # factorial
    probes = [0, 1, 2, 5, 8]
    wrong = [lambda n: n * n, lambda n: n + 1, lambda n: -math.factorial(n), lambda n: 0]

    # a verifier set including a SPURIOUS one (out_le_1000 — fails on 5040) and a likely
    # USELESS one, mixed with healthy checks.
    invariants = {"out_positive", "out_ge_first", "monotonic_increasing",
                  "out_le_1000", "out_nonneg"}

    print("  auditing %d verifiers against %d correct cases + %d wrong candidates:\n"
          % (len(invariants), len(correct), len(wrong)))
    report = audit(invariants, correct, wrong, probes)
    for name in sorted(report, key=lambda n: report[n][0]):
        print("    %-22s %s" % (name, report[name][0]))

    demote = [n for n, (_, _, sp) in report.items() if sp]
    prune = [n for n, (s, c, sp) in report.items() if not sp and c == 0]
    print("\n  -> DEMOTE (spurious):", demote or "none")
    print("  -> PRUNE  (useless) :", prune or "none")
    print("\n  The brain audits its OWN checks: a verifier that blocks truth or catches nothing")
    print("  loses its authority. Self-made verifiers stay trustworthy only while they behave.")


if __name__ == "__main__":
    _demo()
