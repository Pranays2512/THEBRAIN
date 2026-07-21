#!/usr/bin/env python3
"""
crisp_internal_router.py — CrispInternalRouter: the cognitive mode-switcher
for the PYTHON (crisp/symbolic) layer of the brain.

Mirrors the C++ InternalRouter in design but operates on crisp-layer signals:
- confidence        (0-1): how certain the last answer was
- verification_depth (int): how many reasoning hops were needed (0 = direct lookup)
- novelty           (0-1): fraction of tokens not seen before
- curiosity_error   (0-1): CuriosityLoop prediction error (gap in rules)
- solution_type     (str): "compute"|"factual"|"code"|"event"|"none"
- appraisal_type    (str): "question"|"command"|"greeting"|"statement"
- is_verified       (bool): did the symbolic solver confirm the answer?

Outputs a CrispRoutingDecision with:
- mode              (CrispMode enum): what the crisp layer should do next
- label             (str): human-readable reason
- trigger_teach     (bool): should we teach this fact to the fuzzy brain?
- trigger_propose   (bool): should the Proposer try a synthesis space?
- trigger_curiosity (bool): should the CuriosityLoop fire a tick?
- confidence_out    (float): adjusted confidence to report outward (to LLM/user)
- domain_hint       (str): "math"|"physics"|"code"|"factual"|"unknown"

Mode priority (checked top → bottom, first match wins):
  IDLE         — no input / greeting / nothing to do
  RETRIEVE     — high-confidence direct lookup, answer already known
  VERIFY       — answer found but needs checking (low confidence or novel)
  SYNTHESIZE   — no direct answer, synthesis pipeline needed (code/math)
  PROPOSE      — high curiosity error, Proposer should suggest a new space
  CURIOUS      — moderate curiosity, CuriosityLoop should tick
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class CrispMode(Enum):
    IDLE       = auto()  # dormant — nothing to act on
    RETRIEVE   = auto()  # high-confidence recall from KB
    VERIFY     = auto()  # answer found but uncertain — re-check
    SYNTHESIZE = auto()  # no answer — launch synthesis pipeline
    PROPOSE    = auto()  # high curiosity — Proposer picks synthesis space
    CURIOUS    = auto()  # mild curiosity — CuriosityLoop ticks


@dataclass
class CrispThresholds:
    # RETRIEVE: direct high-confidence recall
    retrieve_min_confidence: float = 0.85
    retrieve_max_depth: int = 1       # direct or 1-hop answers only

    # VERIFY: answer found but uncertain
    verify_min_confidence: float = 0.40
    verify_max_depth: int = 8

    # SYNTHESIZE: no answer at all (solution_type == "none") or code/math
    synthesize_types: frozenset = field(
        default_factory=lambda: frozenset({"compute", "code", "none"}))

    # PROPOSE: high curiosity / very high novelty — Proposer takes over
    propose_min_curiosity: float = 0.70
    propose_min_novelty: float = 0.65

    # CURIOUS: moderate curiosity
    curious_min_curiosity: float = 0.35

    # IDLE: greeting or empty input
    idle_types: frozenset = field(
        default_factory=lambda: frozenset({"greeting", "none_appraisal"}))


@dataclass
class CrispRoutingDecision:
    mode: CrispMode
    label: str = ""

    # action flags consumed by the caller
    trigger_teach: bool = False      # push verified fact → fuzzy brain
    trigger_propose: bool = False    # fire Proposer.rank()
    trigger_curiosity: bool = False  # fire CuriosityLoop.tick()

    confidence_out: float = 0.0
    domain_hint: str = "unknown"


class CrispInternalRouter:
    """
    Signal-based cognitive mode selector for the crisp (Python) brain layer.

    Stateless: each call to decide() is independent.  Thread-safe (no shared
    mutable state).  Thresholds are configurable at construction time.
    """

    def __init__(self, thresholds: Optional[CrispThresholds] = None):
        self.T = thresholds or CrispThresholds()

    # ── public API ────────────────────────────────────────────────────────────

    def decide(
        self,
        *,
        confidence: float = 0.0,
        verification_depth: int = 0,
        novelty: float = 0.0,
        curiosity_error: float = 0.0,
        solution_type: str = "none",
        appraisal_type: str = "statement",
        is_verified: bool = False,
    ) -> CrispRoutingDecision:
        """
        Compute the routing decision for this crisp-layer step.

        Parameters
        ----------
        confidence        0-1  solver certainty
        verification_depth  0  = direct lookup, >0 = multi-hop inference
        novelty           0-1  fraction of tokens not seen before
        curiosity_error   0-1  CuriosityLoop error (0 = all rules known)
        solution_type     "compute"|"factual"|"code"|"event"|"none"
        appraisal_type    "question"|"command"|"greeting"|"statement"
        is_verified       True iff the symbolic solver confirmed the answer
        """
        T = self.T
        d = CrispRoutingDecision(mode=CrispMode.IDLE)
        d.domain_hint = _domain_from_type(solution_type)

        # ── IDLE ─────────────────────────────────────────────────────────────
        # Greetings / empty / purely social input: nothing to reason about.
        if appraisal_type in ("greeting",) or solution_type == "greeting":
            d.mode = CrispMode.IDLE
            d.label = "IDLE(social)"
            d.confidence_out = 1.0
            return d

        # ── RETRIEVE ─────────────────────────────────────────────────────────
        # High-confidence, shallow, verified: answer is already in the KB.
        # Tell the fuzzy brain so it can align its SOM activation.
        if (is_verified
                and confidence >= T.retrieve_min_confidence
                and verification_depth <= T.retrieve_max_depth):
            d.mode = CrispMode.RETRIEVE
            d.label = f"RETRIEVE(conf={confidence:.2f}, depth={verification_depth})"
            d.trigger_teach = True   # push fact across the membrane
            d.confidence_out = confidence
            return d

        # ── VERIFY ───────────────────────────────────────────────────────────
        # Found something, but confidence is low or required deep chaining.
        # Re-verify before committing: trigger_teach only if VERY confident.
        if (solution_type in ("factual", "compute", "event")
                and confidence >= T.verify_min_confidence
                and verification_depth <= T.verify_max_depth):
            teach = is_verified and confidence >= 0.75
            d.mode = CrispMode.VERIFY
            d.label = f"VERIFY(conf={confidence:.2f}, depth={verification_depth})"
            d.trigger_teach = teach
            d.confidence_out = confidence * 0.9  # slight confidence penalty
            return d

        # ── PROPOSE ──────────────────────────────────────────────────────────
        # Very high curiosity OR novelty: the Proposer should pick a synthesis
        # space and the CuriosityLoop should get a tick.
        if (curiosity_error >= T.propose_min_curiosity
                or novelty >= T.propose_min_novelty):
            d.mode = CrispMode.PROPOSE
            d.label = (f"PROPOSE(curiosity={curiosity_error:.2f}, "
                       f"novelty={novelty:.2f})")
            d.trigger_propose = True
            d.trigger_curiosity = True
            d.confidence_out = max(0.0, 1.0 - novelty)
            return d

        # ── SYNTHESIZE ───────────────────────────────────────────────────────
        # No answer or a code/compute problem: launch the synthesis pipeline.
        if solution_type in T.synthesize_types:
            d.mode = CrispMode.SYNTHESIZE
            d.label = f"SYNTHESIZE(type={solution_type})"
            d.trigger_curiosity = curiosity_error >= T.curious_min_curiosity
            d.confidence_out = 0.0  # no answer yet
            return d

        # ── CURIOUS ──────────────────────────────────────────────────────────
        # Some gap in rules: CuriosityLoop should tick to fill it.
        if curiosity_error >= T.curious_min_curiosity:
            d.mode = CrispMode.CURIOUS
            d.label = f"CURIOUS(error={curiosity_error:.2f})"
            d.trigger_curiosity = True
            d.confidence_out = confidence
            return d

        # ── RETRIEVE (low-bar fallback) ───────────────────────────────────────
        # Had some answer, didn't meet strict RETRIEVE but still informative.
        if solution_type != "none" and confidence > 0:
            d.mode = CrispMode.RETRIEVE
            d.label = f"RETRIEVE(fallback, conf={confidence:.2f})"
            d.trigger_teach = is_verified
            d.confidence_out = confidence
            return d

        # ── IDLE (default) ───────────────────────────────────────────────────
        d.mode = CrispMode.IDLE
        d.label = "IDLE(no-signal)"
        d.confidence_out = 0.0
        return d


# ── helpers ───────────────────────────────────────────────────────────────────

def _domain_from_type(solution_type: str) -> str:
    return {
        "compute": "physics",
        "factual": "factual",
        "code":    "code",
        "event":   "factual",
        "none":    "unknown",
    }.get(solution_type, "unknown")
