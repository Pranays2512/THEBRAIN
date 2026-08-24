#!/usr/bin/env python3
"""
brain3/finance/core/alpha_conviction.py

Canonical conviction -> win-probability mapping for THE BRAIN 3.0.

Single source of truth used by every engine that converts an alpha conviction
score into a win probability. Mirrors the C++ header
brain3/finance/core/alpha_conviction.hpp — keep the constants EXACTLY in sync:
    base = 0.55, gain = 0.20, output clamp = [0.50, 0.85]
"""

# Canonical constants (MUST match alpha_conviction.hpp exactly)
CANONICAL_BASE = 0.55
CANONICAL_GAIN = 0.20
CANONICAL_PROB_MIN = 0.50
CANONICAL_PROB_MAX = 0.85


def canonical_win_probability(alpha_score: float) -> float:
    """
    Map a raw alpha conviction signal to a win probability.

    Input is clamped to [0, 1]; result is clamped to [0.50, 0.85]:
        p = clamp(0.55 + 0.20 * clamp(alpha_score, 0, 1), 0.50, 0.85)
    """
    normalized = max(0.0, min(1.0, alpha_score))
    raw = CANONICAL_BASE + CANONICAL_GAIN * normalized
    return max(CANONICAL_PROB_MIN, min(CANONICAL_PROB_MAX, raw))
