#!/usr/bin/env python3
"""
curiosity_cross.py — curiosity into the ADJACENT POSSIBLE, not just data gaps (Novel #2).

The existing curiosity loop fills WITHIN-domain gaps: it promotes rules the data already
supports ("cannot invent a pattern that isn't there"). It misses the case where two
domains, each unremarkable alone, share an abstract structure that is only an insight when
you put them side by side.

This is that move, using the factorizer's anti-unification ACROSS domains: factor the
UNION of two domain libraries; a primitive that appears in BOTH is a cross-domain law —
the adjacent-possible insight neither domain shows on its own. Then it transfers: the
shared structure predicts the form of a quantity in one domain from the other.

Honest limit: still verified (the shared structure must actually generalize both
formulas); it discovers shared STRUCTURE, not brand-new physics. But "these two distant
things are the same shape" is exactly the cross-domain spark within-domain curiosity can't reach.
"""

from engines.math import factorizer as FZ

# Two distant domains. Each library alone shows nothing shared (one formula each).
DOMAIN_A = [("kinetic_energy", ("*", 0.5, ("*", "m", ("*", "v", "v"))))]   # ½·m·v²
DOMAIN_B = [("spring_energy",  ("*", 0.5, ("*", "k", ("*", "x", "x"))))]   # ½·k·x²


def cross_domain_laws(libraries, min_count=2, min_kept=1):
    """The reusable core (callable from the front). `libraries` is a list of (name, tree)
    from two+ domains. Factor their UNION via anti-unification: a skeleton that recurs
    across DISTINCT named laws is a shared cross-domain law. Returns the discovery plus a
    VERIFIED flag (the shared structure must reconstruct every input formula — membrane).

    Returns {"new": [(name,tree)], "prims": {...}, "discovery": (skel_name, skel),
             "verified": bool}. Empty `new` => no shared structure (honest miss)."""
    union = list(libraries)
    new, prims, disc = FZ.factor_au(union, min_count=min_count, min_kept=min_kept)
    verified = False
    if new:
        verified, _ = FZ._verify(union, new, prims)
    return {"new": new, "prims": prims, "discovery": disc, "verified": verified}


def _demo():
    print("=== curiosity_cross — find the law two domains share (the adjacent possible) ===\n")
    print("  domain A:", DOMAIN_A[0][0], "=", DOMAIN_A[0][1], "  (½·m·v²)")
    print("  domain B:", DOMAIN_B[0][0], "=", DOMAIN_B[0][1], "  (½·k·x²)")

    # within-domain: factoring one formula only COMPRESSES it (within-formula repeats) —
    # it never connects two DISTINCT named phenomena, because it only sees one.
    print("\n  within a single domain: curiosity can only compress what's already there —")
    print("  it never learns that a DIFFERENT field's law has the same shape.")

    # cross-domain curiosity: factor the UNION -> the shared abstract structure UNIFIES
    # two distinct named phenomena (kinetic vs spring) under one law.
    r = cross_domain_laws(DOMAIN_A + DOMAIN_B)
    new, disc, ok = r["new"], r["discovery"], r["verified"]
    print("\n  across A ∪ B: two distinct laws collapse to ONE shape =", disc[0], "=", disc[1])
    print("    (½·coeff·quantity² — kinetic energy and spring energy are the SAME law):")
    for n, t in new:
        print("      %-15s %s" % (n, t))

    print("\n  shared structure verified on both formulas:", ok)
    print("  -> the insight 'these two distant laws are the same shape' exists only when the")
    print("     domains are combined — within-domain curiosity could never surface it.")


if __name__ == "__main__":
    _demo()
