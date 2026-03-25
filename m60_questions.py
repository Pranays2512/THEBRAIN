"""
M60: QUESTION MEMORY — THE BRAIN'S UNRESOLVED CONFUSION
=========================================================

WHAT THIS IS
------------
Question memory is the bridge between confusion and directed thought.

Before M60, when the brain was confused — L2 wrong, L4 uncertain —
that confusion was felt as high arousal and curiosity_pull, but nothing
was recorded. The brain would raise epsilon, wander, and move on.
The confusion was never stored. It could never be returned to.

M60 changes that. Every time confusion reaches threshold, a Question
is born — a specific record of what the brain didn't understand:
  - Where it was (zone, freq_idx)
  - What it expected (predicted_bmu, predicted_zone)
  - What actually happened (actual_bmu, actual_zone)
  - What its internal state was (self-model snapshot)
  - When it happened (step)

Questions are stored and aged. When the brain returns to a similar
zone in a similar internal state, M60 flags it: "open question here."
M57 uses that flag to bias simulation toward the confusing zone —
the brain goes back deliberately, not randomly.

When prediction error drops at a zone, open questions there are marked
resolved. The brain has learned something — for the first time knowingly.

THREE THINGS M60 PROVIDES
--------------------------

1. QUESTION STORE
   A list of Question objects, each with:
     zone           int      — where confusion happened
     freq_idx       int      — frequency zone
     predicted_bmu  int      — what L2 expected
     actual_bmu     int      — what actually fired
     pe_at_creation float    — prediction error when question was born
     sm_snapshot    (8,)     — self-model state at creation
     created_at     int      — step number
     resolved       bool     — has PE dropped enough to mark resolved?
     revisit_count  int      — how many times brain returned to this zone
     pe_history     list     — PE at each revisit (shows learning)

2. DIRECTED EXPLORATION SIGNAL
   open_question_zones: set of zone indices with open questions.
   M57 receives a zone_pull dict: {zone_idx: pull_strength} based on
   question age (older = stronger pull) and PE delta (still confusing = stronger).
   This replaces random epsilon with targeted curiosity.

3. RESOLUTION TRACKING
   When PE at a zone drops below Q60_RESOLVE_THRESH relative to creation PE,
   the question is resolved. Brain.step() returns:
     questions_resolved_this_step: list of just-resolved questions
   This is the first time the brain can know it learned something specific.

HOW IT PLUGS IN
---------------
M60.step() is called inside Brain.step() after M59 (self-model), before M57.

It reads:
  prediction_error   — from L2, via brain feedback state
  freq_idx           — current zone
  bmu_idx            — current cortical neuron
  predicted_bmu      — what Thought expected (from thought_out)
  l4_top_prob        — L4 certainty (from l4_out)
  sm_state_vec       — self-model snapshot (from sm_out)
  sm_state_label     — current self label
  step               — brain.t

It writes into Brain return dict:
  q60_open_count         int     — number of open questions
  q60_zone_pull          dict    — {zone: pull_strength} for M57
  q60_active_question    bool    — is there an open question in current zone?
  q60_just_resolved      list    — questions resolved this step
  q60_total_resolved     int     — lifetime resolved count

M57 receives zone_pull and adds a directional bonus to sim_values for
actions that move toward high-pull zones — the brain is drawn back to
what confused it, not just away from familiarity.

TRIGGER CONDITION
-----------------
A question is created when ALL of these are true simultaneously:
  1. prediction_error > Q60_PE_THRESH          (L2 was meaningfully wrong)
  2. l4_top_prob < Q60_L4_CERTAINTY_THRESH     (L4 doesn't know where we are)
  3. sm_state_label in ('confused', 'curious') (self-model agrees we're lost)
  4. No open question already exists for this zone (no duplicates)
  5. step > Q60_WARMUP_STEPS                   (don't create during cold boot)

The triple-gate (L2 wrong + L4 uncertain + self-model aware) ensures
questions are only created for genuine unresolved confusion, not
routine prediction error during early learning.

RESOLUTION CONDITION
--------------------
A question at zone Z is resolved when:
  current PE at zone Z < Q60_RESOLVE_THRESH * question.pe_at_creation
  AND question.revisit_count >= 1 (brain actually returned)

Meaning: PE dropped by at least (1 - Q60_RESOLVE_THRESH) fraction
from when the question was born. The brain has learned the zone.

BIOLOGICAL BASIS
----------------
This module corresponds to the hippocampal novelty detection system:

  CA1 neurons fire when a prediction is violated — the "mismatch signal."
  This is well-documented: CA1 fires to unexpected events, not expected ones.
  The mismatch signal triggers encoding: the unexpected event gets stored
  in the hippocampus as an "episode" — a specific memory of what happened.

  Later, when the brain re-enters a similar context, hippocampal pattern
  completion (CA3 → CA1) reactivates the stored episode.
  This is recognition: "I've been confused here before."

  Directed re-exploration toward unresolved episodes corresponds to
  the hippocampal → prefrontal cortex projection that biases attention
  and action toward unresolved predictions.

  question.pe_history is analogous to place-cell remapping:
  as the brain learns a zone, the hippocampal representation becomes
  stable (low PE = stable map). Resolution = stable map.

PARAMETERS
----------
Q60_PE_THRESH           = 0.45  — prediction error must exceed this to trigger
                                   a question. Below 0.45 = routine noise.
                                   Above 0.45 = genuine prediction failure.

Q60_L4_CERTAINTY_THRESH = 0.40  — L4 top_prob must be BELOW this.
                                   If L4 is confident (>0.40), the brain knows
                                   where it is — confusion is mild, not worth
                                   storing as a question.

Q60_RESOLVE_THRESH      = 0.50  — PE at revisit must drop to 50% of creation PE.
                                   A 50% reduction means the brain has learned
                                   half of what confused it. Strict enough to
                                   require real learning, loose enough to resolve
                                   in a world with inherent stochasticity.

Q60_MAX_QUESTIONS       = 32    — max open questions at any time.
                                   Oldest questions are dropped when full.
                                   Prevents question accumulation from a cold,
                                   confused brain drowning in low-quality questions.

Q60_PULL_DECAY          = 0.995 — pull strength decays per step (tau ~200 steps).
                                   Questions that aren't revisited slowly lose
                                   pull — the brain stops obsessing over old
                                   confusion if new confusion is more recent.

Q60_MAX_AGE_STEPS       = 2000  — questions older than this are expired even if
                                   not resolved. Prevents forever-open questions
                                   from cluttering the store.

Q60_WARMUP_STEPS        = 200   — no questions before this step.
                                   Cold brain needs time to build L2 sequences
                                   and L4 beliefs before confusion is meaningful.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

Q60_PE_THRESH           = 0.45
Q60_L4_CERTAINTY_THRESH = 0.40
Q60_RESOLVE_THRESH      = 0.50
Q60_MAX_QUESTIONS       = 32
Q60_PULL_DECAY          = 0.995
Q60_MAX_AGE_STEPS       = 2000
Q60_WARMUP_STEPS        = 200
Q60_PULL_BASE           = 0.30   # base pull strength at question creation
Q60_PULL_AGE_BONUS      = 0.20   # older unresolved questions get stronger pull
Q60_REVISIT_WINDOW      = 3      # steps within which returning to a zone counts as revisit


# ═══════════════════════════════════════════════════════════════
# QUESTION DATACLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class Question:
    """
    A single stored confusion event.
    Created when L2 is wrong + L4 is uncertain + self-model says confused/curious.
    """
    zone:           int
    freq_idx:       int
    predicted_bmu:  int
    actual_bmu:     int
    pe_at_creation: float
    sm_snapshot:    np.ndarray        # (8,) self-state vec at creation
    sm_label:       str               # state label at creation
    created_at:     int               # step number

    resolved:       bool  = False
    resolved_at:    int   = -1
    revisit_count:  int   = 0
    pe_history:     list  = field(default_factory=list)  # PE at each revisit
    pull_strength:  float = Q60_PULL_BASE

    def age(self, current_step: int) -> int:
        return current_step - self.created_at

    def decay_pull(self):
        self.pull_strength = float(np.clip(
            self.pull_strength * Q60_PULL_DECAY, 0.0, 1.0))

    def boost_pull(self):
        """Called when brain returns to zone but PE still high — still confused."""
        self.pull_strength = float(np.clip(
            self.pull_strength + 0.10, 0.0, 1.0))

    def to_dict(self) -> dict:
        return {
            'zone':           self.zone,
            'freq_idx':       self.freq_idx,
            'predicted_bmu':  self.predicted_bmu,
            'actual_bmu':     self.actual_bmu,
            'pe_at_creation': self.pe_at_creation,
            'sm_label':       self.sm_label,
            'created_at':     self.created_at,
            'resolved':       self.resolved,
            'resolved_at':    self.resolved_at,
            'revisit_count':  self.revisit_count,
            'pull_strength':  self.pull_strength,
            'pe_history':     list(self.pe_history),
        }


# ═══════════════════════════════════════════════════════════════
# QUESTION MEMORY
# ═══════════════════════════════════════════════════════════════

class QuestionMemory:
    """
    M60 — the brain's store of unresolved confusions.

    Reads prediction_error + L4 certainty + self-model state each step.
    Creates Question objects when confusion threshold is met.
    Tracks revisits, resolution, and directed exploration pull.
    """

    def __init__(self, seed: int = 42):
        self._questions: List[Question] = []   # all questions (open + resolved)
        self._open: List[Question] = []        # open questions only (fast access)

        self._total_created  = 0
        self._total_resolved = 0

        # Per-zone PE tracking for resolution detection
        self._zone_pe_ema = {}           # {zone: ema_pe}
        self._zone_pe_alpha = 0.20

        # Recent zone visit tracking (for revisit detection)
        self._recent_zones = deque(maxlen=Q60_REVISIT_WINDOW + 1)

        self._last_out = {}
        self.t = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             prediction_error:   float,
             freq_idx:           int,
             bmu_idx:            int,
             predicted_bmu:      int,
             l4_top_prob:        float,
             sm_state_vec:       np.ndarray,
             sm_state_label:     str,
             reward:             float = 0.0,
             ) -> dict:
        """
        One question memory step.

        Parameters
        ----------
        prediction_error  : float  — L2 raw prediction error [0,1]
        freq_idx          : int    — current zone index
        bmu_idx           : int    — current cortical neuron (actual)
        predicted_bmu     : int    — what Thought/L2 expected
        l4_top_prob       : float  — L4 top node probability [0,1]
        sm_state_vec      : (8,)   — self-model state vector
        sm_state_label    : str    — self-model state label
        reward            : float  — reward this step (for resolution boost)

        Returns
        -------
        dict with keys:
          open_count          int   — number of open questions
          zone_pull           dict  — {zone_idx: pull_strength} for M57
          active_question     bool  — open question in current zone
          just_resolved       list  — questions resolved this step (dicts)
          total_resolved      int   — lifetime resolved count
          just_created        bool  — a question was created this step
          question_created    dict  — the new question if created, else None
        """
        pe  = float(np.clip(prediction_error, 0.0, 1.0))
        l4p = float(np.clip(l4_top_prob,      0.0, 1.0))
        fi  = int(freq_idx) if freq_idx is not None and freq_idx >= 0 else -1

        # ── 1. Update zone PE EMA ──────────────────────────────
        if fi >= 0:
            prev = self._zone_pe_ema.get(fi, pe)
            self._zone_pe_ema[fi] = ((1.0 - self._zone_pe_alpha) * prev
                                     + self._zone_pe_alpha * pe)

        # ── 2. Decay all open question pull strengths ──────────
        for q in self._open:
            q.decay_pull()

        # ── 3. Check for revisits ──────────────────────────────
        just_resolved = []
        if fi >= 0:
            for q in list(self._open):
                if q.zone == fi:
                    q.revisit_count += 1
                    q.pe_history.append(pe)

                    # Resolution check
                    if (pe < Q60_RESOLVE_THRESH * q.pe_at_creation
                            and q.revisit_count >= 1):
                        q.resolved   = True
                        q.resolved_at = self.t
                        self._open.remove(q)
                        self._total_resolved += 1
                        just_resolved.append(q.to_dict())
                    else:
                        # Still confused at this zone — boost pull
                        if pe > Q60_PE_THRESH * 0.8:
                            q.boost_pull()

        # ── 4. Expire old questions ────────────────────────────
        for q in list(self._open):
            if q.age(self.t) > Q60_MAX_AGE_STEPS:
                self._open.remove(q)

        # ── 5. Maybe create a new question ────────────────────
        just_created = False
        question_created = None

        if self.t >= Q60_WARMUP_STEPS and fi >= 0:
            already_open = any(q.zone == fi for q in self._open)
            if (pe > Q60_PE_THRESH
                    and l4p < Q60_L4_CERTAINTY_THRESH
                    and sm_state_label in ('confused', 'curious', 'alert')
                    and not already_open):

                # Capacity check — drop oldest if full
                if len(self._open) >= Q60_MAX_QUESTIONS:
                    self._open.sort(key=lambda q: q.created_at)
                    self._open.pop(0)

                q = Question(
                    zone=fi,
                    freq_idx=fi,
                    predicted_bmu=int(predicted_bmu),
                    actual_bmu=int(bmu_idx),
                    pe_at_creation=pe,
                    sm_snapshot=sm_state_vec.copy() if sm_state_vec is not None
                                else np.zeros(8),
                    sm_label=sm_state_label,
                    created_at=self.t,
                    pull_strength=Q60_PULL_BASE,
                )
                self._open.append(q)
                self._questions.append(q)
                self._total_created += 1
                just_created = True
                question_created = q.to_dict()

        # ── 6. Build zone_pull dict for M57 ───────────────────
        zone_pull = {}
        for q in self._open:
            # Age bonus: older unresolved = slightly more urgent to revisit
            age_factor = float(np.clip(q.age(self.t) / Q60_MAX_AGE_STEPS, 0.0, 1.0))
            pull = q.pull_strength + Q60_PULL_AGE_BONUS * age_factor
            zone_pull[q.zone] = float(np.clip(
                max(zone_pull.get(q.zone, 0.0), pull), 0.0, 1.0))

        # ── 7. Active question in current zone ────────────────
        active_question = fi >= 0 and any(q.zone == fi for q in self._open)

        # ── 8. Update recent zone buffer ──────────────────────
        if fi >= 0:
            self._recent_zones.append(fi)

        out = {
            'open_count':       len(self._open),
            'zone_pull':        zone_pull,
            'active_question':  active_question,
            'just_resolved':    just_resolved,
            'total_resolved':   self._total_resolved,
            'just_created':     just_created,
            'question_created': question_created,
        }
        self._last_out = out
        self.t += 1
        return out

    # ── Queries ───────────────────────────────────────────────

    def open_questions(self) -> List[Question]:
        """All currently open questions."""
        return list(self._open)

    def questions_for_zone(self, zone: int) -> List[Question]:
        """Open questions at a specific zone."""
        return [q for q in self._open if q.zone == zone]

    def most_urgent_zone(self) -> Optional[int]:
        """Zone with the highest pull strength, or None if no open questions."""
        if not self._open:
            return None
        return max(self._open, key=lambda q: q.pull_strength).zone

    def resolution_rate(self) -> float:
        """Fraction of created questions that have been resolved."""
        if self._total_created == 0:
            return 0.0
        return float(self._total_resolved / self._total_created)

    def avg_pe_at_creation(self) -> float:
        """Average PE when questions were born (how confused the brain typically is)."""
        if not self._questions:
            return 0.0
        return float(np.mean([q.pe_at_creation for q in self._questions]))

    def learning_trajectory(self, zone: int) -> list:
        """
        For a given zone, return the PE history across all revisits.
        A declining trajectory = the brain is learning that zone.
        """
        for q in self._questions:
            if q.zone == zone:
                return [q.pe_at_creation] + list(q.pe_history)
        return []

    # ── Diagnostics ───────────────────────────────────────────

    def summary(self):
        print(f"  QuestionMemory (M60) — step {self.t}")
        print(f"  Open:     {len(self._open)}/{Q60_MAX_QUESTIONS}")
        print(f"  Resolved: {self._total_resolved}/{self._total_created}"
              f"  (rate={self.resolution_rate():.0%})")
        if self._open:
            urgent = self.most_urgent_zone()
            print(f"  Most urgent zone: {urgent}")
            for q in sorted(self._open, key=lambda x: -x.pull_strength)[:3]:
                print(f"    zone={q.zone} pull={q.pull_strength:.2f}"
                      f" age={q.age(self.t)} pe={q.pe_at_creation:.2f}"
                      f" revisits={q.revisit_count} label={q.sm_label}")

    def get_state(self) -> dict:
        return {
            't':              self.t,
            'open_count':     len(self._open),
            'total_created':  self._total_created,
            'total_resolved': self._total_resolved,
            'resolution_rate': self.resolution_rate(),
            'most_urgent_zone': self.most_urgent_zone(),
            'open_questions': [q.to_dict() for q in self._open],
        }