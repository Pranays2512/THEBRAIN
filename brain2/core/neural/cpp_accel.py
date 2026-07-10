#!/usr/bin/env python3
"""cpp_accel.py — optional C++ fast paths (the compiled brain2 .so) with a Python fallback.

The core primitives were ported to C++ and PROVEN equal to their Python reference in
harden_regress (brain2.<fn> == <fn>_py). This module is where those verified ports get wired
into the RUNTIME: a caller uses the C++ path when brain2 is importable, else the identical
Python path. Guarded, so a pure-Python environment (no compiled brain2) still runs.

Honest per-port assessment (why NOT all 9 are wired — a port being verified-equal does not make
it a safe drop-in):

  WIRED (clean signature, verified ==, and worth it):
    * law_error       — least-squares fit; float->float, not in a hot loop. Real compute.

  DELIBERATELY NOT WIRED (documented, not oversight):
    * cosine_map      — clean, but called per word-pair in tight vocab loops; pybind
                        marshalling overhead would SLOW it. Native Python wins at this size.
    * disc_weights /  — couples (weights feed feat_sim); must swap as a pair, trivial compute,
      feat_sim          no measurable gain. Left Python to avoid format-coupling risk.
    * inv_mine        — C++ is a SUBSET (lacks 'monotonic_increasing'); swapping would DROP a
                        real invariant -> behaviour regression. harden_regress line ~90 shows it.
    * refute_int1     — takes precomputed candidate/oracle arrays, not refute()'s (f, oracle)
                        shape; and has an honest 64-bit-output limit. Needs a refactor to wire.
    * eval_sexpr      — takes a SERIALIZED s-expression string; converting tree->string per call
                        would be slower than native tree eval in hot factorizer loops.
    * analogy_score   — returns only the score; callers (align/align_greedy) also need the
                        relation map (relmap) the Python _score returns. Partial output.

The real value of the ports is a correctness-proven C++ implementation READY for when data
scales (where marshalling overhead is amortized) — not a current speedup at these input sizes."""

try:
    import brain2 as _brain2
except Exception:                       # no compiled .so (e.g. pure-Python interpreter)
    _brain2 = None


def cpp():
    """The brain2 module if the compiled fast path is available, else None."""
    return _brain2
