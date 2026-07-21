#!/usr/bin/env python3
"""
crisp_external_router.py — CrispExternalRouter: the crisp→fuzzy membrane gate.

The EXTERNAL router is the OUTWARD membrane: it takes facts and policies that
the crisp (Python) layer has VERIFIED and pushes them across to the fuzzy
(C++) brain via brain.accept_fact() / brain.accept_policy().

It also packages the fuzzy brain's OutboundSignal (received when calling
brain.pack_outbound()) into a Python-friendly dict the crisp layer can act on.

─────────────────────────────────────────────────────────────────────────────
Data flow:

  Python crisp layer verifies fact/policy
          │
          ▼
  CrispExternalRouter.push_fact(fact, verified=True)
          │   ─── gate: only verified=True passes ───
          ▼
  brain.accept_fact(InboundFact)          ← C++ membrane gate (second check)
          │
          ▼
  brain.crisp_facts[entity, relation] = value   (stored in fuzzy brain)

  ─────────────────────────────────────────────────────────────────────────
  Also: reading the fuzzy brain's outbound signal back into Python:

  fuzzy OutboundSignal  ←  brain.pack_outbound(pr, mode)
          │
          ▼
  CrispExternalRouter.unpack_signal(signal_dict) → CrispInboundSignal
          │
          ▼
  CrispInternalRouter.decide(novelty=signal.novelty, ...) → routing decision

─────────────────────────────────────────────────────────────────────────────

Design principles:
  • Double-gate: crisp router checks verified=True; C++ ExternalRouter checks again
  • Quarantine: contradicted facts go to a quarantine dict, never into crisp_facts
  • Audit trail: every push/reject is logged with reason
  • Stateless core: the Brain reference is injected at construction
  • Graceful degradation: if brain2 is not available, facts are silently buffered
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class CrispFact:
    """A candidate fact from the crisp layer, ready to cross the membrane."""
    entity:   str
    relation: str
    value:    float
    verified: bool = False
    source:   str  = "crisp_reasoner"
    confidence: float = 1.0


@dataclass
class CrispPolicy:
    """A candidate policy from the crisp synthesis layer."""
    target:  str              # quantity to compute ("force", "energy", …)
    inputs:  Tuple[str, ...]  # input variable names
    expr:    Any              # expression tree (ExprPtr or Python tuple)
    verified: bool = False
    source:   str  = "crisp_synthesizer"
    confidence: float = 1.0


@dataclass
class PushResult:
    accepted: bool
    reason:   str
    fact:     Optional[CrispFact] = None
    policy:   Optional[CrispPolicy] = None
    ts:       float = field(default_factory=time.time)


@dataclass
class CrispInboundSignal:
    """Python-side interpretation of the fuzzy brain's OutboundSignal."""
    novelty:         float  # prediction error (surprise from SOM)
    valence:         float  # emotional valence
    arousal:         float  # emotional arousal
    bmu:             int    # best-matching unit in SOM
    gate_open:       bool   # fuzzy brain's attention gate open?
    confidence:      float  # fuzzy brain's confidence in its current state
    domain_hint:     str    # "LANGUAGE"|"MATH"|"PHYSICS"|"CODE"|"UNKNOWN"
    mode_name:       str    # fuzzy InternalRouter mode ("PERCEIVE", "REASON", …)
    episodic_stored: bool   # was an episode committed this step?
    # Extended fields (filled by pack_outbound, optional/default 0 otherwise)
    salience:        float = 0.0   # attention salience score
    wm_load:         float = 0.0   # working memory utilisation
    self_concept:    int   = -1    # SelfModel cluster
    gw_winner:       int   = -1    # GlobalWorkspace winner module id


# ── CrispExternalRouter ───────────────────────────────────────────────────────

class CrispExternalRouter:
    """
    Crisp→Fuzzy membrane gate.

    Inject a brain2.Brain instance (or None for offline mode).
    Call push_fact() / push_policy() to send verified knowledge across.
    Call unpack_signal() to decode the fuzzy brain's outbound signal dict.

    Example
    -------
    >>> import brain2
    >>> brain = brain2.Brain(som_rows=16, som_cols=16, n_dims=32)
    >>> router = CrispExternalRouter(brain)
    >>> result = router.push_fact(CrispFact("rocket", "mass", 1000.0, verified=True))
    >>> print(result.accepted, result.reason)
    True ok
    """

    def __init__(self, brain=None):
        """
        Parameters
        ----------
        brain : brain2.Brain or None
            The fuzzy C++ brain.  If None, router buffers facts for later
            injection (offline / test mode).
        """
        self._brain = brain
        self._audit: List[PushResult] = []          # full audit trail
        self._quarantine: List[CrispFact] = []      # contradicted facts
        self._buffer: List[CrispFact] = []          # offline buffer
        self._policy_buffer: List[CrispPolicy] = [] # offline policy buffer

        # stats
        self.facts_accepted = 0
        self.facts_rejected = 0
        self.policies_accepted = 0
        self.policies_rejected = 0

    # ── Brain injection ───────────────────────────────────────────────────────

    def set_brain(self, brain) -> None:
        """Inject (or replace) the fuzzy brain reference.  Thread-unsafe —
        call before concurrent use."""
        self._brain = brain
        # Drain the offline buffer
        if self._brain is not None:
            self._drain_buffer()

    # ── Fact gate ─────────────────────────────────────────────────────────────

    def push_fact(self, fact: CrispFact) -> PushResult:
        """
        Push a crisp fact through the membrane.

        Gate rules (in order):
          1. fact.verified must be True  (crisp-side invariant)
          2. fact.confidence >= 0.5      (low-confidence facts not promoted)
          3. brain.accept_fact() must return True (C++ double-gate)
        """
        # Gate 1: unverified facts NEVER cross
        if not fact.verified:
            r = PushResult(False, "unverified", fact=fact)
            self._log(r)
            self.facts_rejected += 1
            return r

        # Gate 2: confidence floor
        if fact.confidence < 0.5:
            r = PushResult(False, "low_confidence", fact=fact)
            self._log(r)
            self.facts_rejected += 1
            return r

        # Gate 3: push across membrane
        if self._brain is None:
            # Offline: buffer for later
            self._buffer.append(fact)
            r = PushResult(True, "buffered_offline", fact=fact)
            self._log(r)
            self.facts_accepted += 1
            return r

        try:
            # Call brain.accept_fact(entity, relation, value, verified, source)
            # — flat signature exposed by pybind (no InboundFact class needed in Python)
            accepted = self._brain.accept_fact(
                fact.entity, fact.relation, float(fact.value),
                True,         # verified: already checked above
                fact.source
            )
        except Exception as exc:
            r = PushResult(False, f"bridge_error:{exc}", fact=fact)
            self._log(r)
            self.facts_rejected += 1
            return r

        if accepted:
            self.facts_accepted += 1
            r = PushResult(True, "ok", fact=fact)
        else:
            # C++ gate rejected — likely a conflict; quarantine
            self._quarantine.append(fact)
            self.facts_rejected += 1
            r = PushResult(False, "fuzzy_rejected", fact=fact)
        self._log(r)
        return r

    # ── Policy gate ───────────────────────────────────────────────────────────

    def push_policy(self, policy: CrispPolicy) -> PushResult:
        """
        Push a verified crisp policy (expression rule) through the membrane.

        Gate rules mirror push_fact():
          1. policy.verified must be True
          2. policy.expr must not be None
          3. brain.accept_policy() must return True
        """
        if not policy.verified:
            r = PushResult(False, "unverified", policy=policy)
            self._log(r)
            self.policies_rejected += 1
            return r

        if policy.expr is None:
            r = PushResult(False, "null_expr", policy=policy)
            self._log(r)
            self.policies_rejected += 1
            return r

        if self._brain is None:
            self._policy_buffer.append(policy)
            r = PushResult(True, "buffered_offline", policy=policy)
            self._log(r)
            self.policies_accepted += 1
            return r

        try:
            # Call brain.accept_policy(target, inputs, expr, verified, source)
            accepted = self._brain.accept_policy(
                policy.target, list(policy.inputs), policy.expr,
                True,           # verified: already checked above
                policy.source
            )
        except Exception as exc:
            r = PushResult(False, f"bridge_error:{exc}", policy=policy)
            self._log(r)
            self.policies_rejected += 1
            return r

        if accepted:
            self.policies_accepted += 1
            r = PushResult(True, "ok", policy=policy)
        else:
            self.policies_rejected += 1
            r = PushResult(False, "fuzzy_rejected", policy=policy)
        self._log(r)
        return r

    # ── Inbound signal decoder ────────────────────────────────────────────────

    @staticmethod
    def unpack_signal(signal_dict: Dict[str, Any]) -> CrispInboundSignal:
        """
        Convert the dict produced by brain.pack_outbound() into a
        CrispInboundSignal the Python crisp layer can act on.

        Expected keys (all optional — missing keys default gracefully):
          novelty, valence, arousal, bmu, gate_open, confidence,
          domain_hint, mode_name, episodic_stored
        """
        return CrispInboundSignal(
            novelty         = float(signal_dict.get("novelty",         0.0)),
            valence         = float(signal_dict.get("valence",         0.0)),
            arousal         = float(signal_dict.get("arousal",         0.0)),
            bmu             = int(  signal_dict.get("bmu",             -1)),
            gate_open       = bool( signal_dict.get("gate_open",       False)),
            confidence      = float(signal_dict.get("confidence",      0.0)),
            domain_hint     = str(  signal_dict.get("domain_hint",     "UNKNOWN")),
            mode_name       = str(  signal_dict.get("mode_name",       "PERCEIVE")),
            episodic_stored = bool( signal_dict.get("episodic_stored", False)),
            # Extended fields from pack_outbound
            salience        = float(signal_dict.get("salience",        0.0)),
            wm_load         = float(signal_dict.get("wm_load",         0.0)),
            self_concept    = int(  signal_dict.get("self_concept",    -1)),
            gw_winner       = int(  signal_dict.get("gw_winner",       -1)),
        )

    # ── Convenience: push many facts at once ─────────────────────────────────

    def push_facts(self, facts: List[CrispFact]) -> List[PushResult]:
        """Push a list of facts and return results in the same order."""
        return [self.push_fact(f) for f in facts]

    def push_policies(self, policies: List[CrispPolicy]) -> List[PushResult]:
        """Push a list of policies and return results in the same order."""
        return [self.push_policy(p) for p in policies]

    # ── Stats + audit ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        return {
            "facts_accepted":    self.facts_accepted,
            "facts_rejected":    self.facts_rejected,
            "policies_accepted": self.policies_accepted,
            "policies_rejected": self.policies_rejected,
            "quarantined":       len(self._quarantine),
            "buffered":          len(self._buffer) + len(self._policy_buffer),
        }

    def audit_trail(self) -> List[PushResult]:
        """Full immutable audit trail (newest last)."""
        return list(self._audit)

    def quarantined(self) -> List[CrispFact]:
        """Facts that were rejected by the C++ membrane (possible conflicts)."""
        return list(self._quarantine)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _log(self, result: PushResult) -> None:
        self._audit.append(result)

    def _drain_buffer(self) -> None:
        """Flush the offline buffer now that a brain is available."""
        drained_facts, drained_policies = self._buffer[:], self._policy_buffer[:]
        self._buffer.clear()
        self._policy_buffer.clear()
        for f in drained_facts:
            self.push_fact(f)
        for p in drained_policies:
            self.push_policy(p)
