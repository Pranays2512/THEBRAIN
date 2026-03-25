"""
M63: EPISODIC TRACE — THE BRAIN'S BEFORE/AFTER MEMORY
======================================================

WHAT THIS IS
------------
M63 is the first module that gives the brain a sense of time.

Every module so far operates on the current moment or an EMA of
recent moments. Even M60's question memory stores confusion but
doesn't record what happened *between* the confusion and the resolution.
The brain cannot yet answer: "The last time I was confused at zone 3,
what did I do that made it better?"

M63 closes that gap with the simplest possible episodic memory:
a ring buffer of *resolved episodes* — specific events where the brain
was confused, did something, and confusion dropped.

Each episode records:
  - The zone where confusion occurred
  - The self-model state at confusion time
  - The actions taken between confusion and resolution
  - The self-model state at resolution time
  - How many steps the resolution took
  - How much PE dropped (magnitude of learning)

When the brain enters a zone with a matching past episode, M63 fires
a "recognition signal" — the brain has a relevant memory. The episode's
action sequence is returned as a soft prior that M57 can optionally use.

THREE THINGS M63 PROVIDES
--------------------------

1. EPISODE STORE
   A ring buffer of the last M63_MAX_EPISODES resolved episodes.
   Each episode is a complete before/after record:
     zone              int      — where confusion happened
     confused_label    str      — self-model label at confusion
     resolved_label    str      — self-model label at resolution
     action_trace      list     — actions taken during the episode
     pe_at_confusion   float    — prediction error when episode began
     pe_at_resolution  float    — prediction error when resolved
     pe_drop           float    — magnitude of learning (pe_confusion - pe_resolved)
     duration          int      — steps from confusion to resolution
     created_at        int      — step number of confusion
     resolved_at       int      — step number of resolution

2. RECOGNITION SIGNAL
   When the brain enters a zone that has past episodes:
     match_strength    float [0,1] — how similar is the current state
                                     to the confused state in the best
                                     matching episode? (cosine similarity
                                     of self-model vectors)
     best_episode      Episode     — the episode with highest match_strength
     action_prior      (n_actions,) float — soft prior from the episode's
                                     action trace (which actions were taken
                                     during successful resolution)

3. TEMPORAL CONTEXT
   M63 tracks the current "open episode" — the brain is actively inside
   an unresolved confusion arc. When M60 creates a question, M63 opens
   an episode. When M60 resolves it, M63 closes the episode.
   During an open episode, M63 records each action taken.
   This gives the brain a causal trace: what did I do while confused?

HOW IT PLUGS IN
---------------
M63.step() is called inside Brain.step() after M62, before the return.

It reads:
  q60_just_created    — was a question created this step? (open episode)
  q60_just_resolved   — were questions resolved? (close episodes)
  q60_zone_pull       — current open question zones
  current_action      — what action was just taken (for action trace)
  freq_idx            — current zone (for recognition lookup)
  sm_state_vec        — current self-model vector (for similarity)
  sm_state_label      — current label

It writes into Brain return dict:
  m63_recognition       float [0,1]   — match strength to best past episode
  m63_action_prior      (n_actions,)  — soft action prior from best episode
  m63_episode_open      bool          — currently inside an open episode
  m63_episodes_total    int           — total resolved episodes
  m63_best_zone         int or None   — zone of best matching episode

EPISODE OPENING AND CLOSING
-----------------------------
Open:  when q60_just_created = True (M60 created a new question)
Close: when q60_just_resolved is non-empty (M60 resolved a question)
       The episode matching that zone is closed and stored.

During an open episode, every action taken is appended to action_trace.
The episode is associated with the zone of the open question.

BIOLOGICAL BASIS
----------------
M63 corresponds to hippocampal episodic memory:

  CA1/CA3 circuits encode "episodes" — sequences of events with a
  beginning and end, stored as patterns that can be pattern-completed
  when re-entering similar contexts.

  "Episodic future thinking" in humans: the hippocampus doesn't just
  recall the past, it uses past episodes to simulate the future. When
  you enter a familiar context, past episodes of that context are
  reactivated and influence action selection.

  Our recognition signal = CA3 pattern completion:
  partial cue (current zone + confused state) → reactivated episode

  action_prior = the hippocampal → prefrontal projection that biases
  PFC action selection based on past episode outcomes.

PARAMETERS
----------
M63_MAX_EPISODES       = 64   — ring buffer size. 64 episodes covers
                                 the full World5 grid many times over.
                                 Old episodes are dropped FIFO.

M63_MATCH_THRESH       = 0.60 — minimum cosine similarity to activate
                                 recognition signal. Below 0.60, the
                                 past episode is too different to be
                                 relevant. Prevents false memories.

M63_PRIOR_DECAY        = 0.80 — how strongly recent actions in the
                                 episode trace are weighted vs old ones.
                                 Recent actions matter more for the prior.

M63_MAX_EPISODE_LEN    = 200  — max steps in an open episode.
                                 If confusion isn't resolved in 200 steps,
                                 the episode is abandoned (the question
                                 may have expired from M60 too).

M63_PRIOR_WEIGHT       = 0.15 — how strongly action_prior influences M57.
                                 Kept low: M56 Q-values still dominate.
                                 M63 is a nudge, not an override.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

M63_MAX_EPISODES    = 64
M63_MATCH_THRESH    = 0.60
M63_PRIOR_DECAY     = 0.80
M63_MAX_EPISODE_LEN = 200
M63_PRIOR_WEIGHT    = 0.15


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """One resolved confusion arc."""
    zone:             int
    confused_label:   str
    confused_vec:     np.ndarray        # self-model vector at confusion
    action_trace:     List[int]         = field(default_factory=list)
    pe_at_confusion:  float             = 0.0
    pe_at_resolution: float             = 0.0
    duration:         int               = 0
    created_at:       int               = 0
    resolved_at:      int               = -1
    resolved_label:   str               = ''

    @property
    def pe_drop(self) -> float:
        return max(0.0, self.pe_at_confusion - self.pe_at_resolution)

    def action_prior(self, n_actions: int) -> np.ndarray:
        """
        Soft prior over actions from this episode's trace.
        Recent actions weighted more than early ones.
        Returns a normalised (n_actions,) float array.
        """
        prior = np.zeros(n_actions, dtype=np.float64)
        if not self.action_trace:
            prior[:] = 1.0 / n_actions
            return prior
        weight = 1.0
        for action in reversed(self.action_trace):
            if 0 <= action < n_actions:
                prior[action] += weight
            weight *= M63_PRIOR_DECAY
        s = prior.sum()
        if s > 1e-9:
            prior /= s
        else:
            prior[:] = 1.0 / n_actions
        return prior

    def similarity_to(self, vec: np.ndarray) -> float:
        """Cosine similarity between this episode's confused_vec and vec."""
        a = self.confused_vec
        b = vec
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))

    def to_dict(self) -> dict:
        return {
            'zone':             self.zone,
            'confused_label':   self.confused_label,
            'resolved_label':   self.resolved_label,
            'action_trace':     list(self.action_trace),
            'pe_at_confusion':  self.pe_at_confusion,
            'pe_at_resolution': self.pe_at_resolution,
            'pe_drop':          self.pe_drop,
            'duration':         self.duration,
            'created_at':       self.created_at,
            'resolved_at':      self.resolved_at,
        }


# ═══════════════════════════════════════════════════════════════
# EPISODIC TRACE
# ═══════════════════════════════════════════════════════════════

class EpisodicTrace:
    """
    M63 — episodic before/after memory.

    Tracks open confusion arcs and stores them when resolved.
    Provides recognition signals and action priors for familiar situations.
    """

    def __init__(self, n_actions: int = 4, seed: int = 42):
        self.n_actions = n_actions

        # Resolved episodes — FIFO ring buffer
        self._episodes: deque = deque(maxlen=M63_MAX_EPISODES)

        # Currently open episodes — keyed by zone
        # (multiple zones can be confused simultaneously)
        self._open_episodes: Dict[int, Episode] = {}

        # Step counter
        self.t = 0

        # Diagnostics
        self._total_opened   = 0
        self._total_resolved = 0
        self._total_abandoned = 0

        # Last outputs for diagnostics
        self._last_recognition  = 0.0
        self._last_action_prior = np.ones(n_actions, dtype=np.float64) / n_actions

    def step(self,
             q60_just_created:  bool,
             q60_just_resolved: list,
             q60_zone_pull:     dict,
             current_action:    int,
             freq_idx:          int,
             sm_state_vec:      np.ndarray,
             sm_state_label:    str,
             prediction_error:  float,
             ) -> dict:
        """
        One M63 step.

        Parameters
        ----------
        q60_just_created   : bool  — M60 created a new question this step
        q60_just_resolved  : list  — M60-resolved question dicts this step
        q60_zone_pull      : dict  — {zone: pull} open questions from M60
        current_action     : int   — action taken this step
        freq_idx           : int   — current zone index
        sm_state_vec       : ndarray (8,) — self-model state vector
        sm_state_label     : str   — self-model label
        prediction_error   : float — current PE

        Returns
        -------
        dict with keys:
          recognition       float [0,1]   — match strength to best past episode
          action_prior      (n_actions,)  — soft action prior from best episode
          episode_open      bool          — currently tracking an open episode
          episodes_total    int           — total resolved episodes
          best_zone         int or None   — zone of best matching episode
          just_closed       list          — episodes closed this step (dicts)
        """
        pe  = float(np.clip(prediction_error, 0.0, 1.0))
        fi  = int(freq_idx) if freq_idx is not None and freq_idx >= 0 else -1
        vec = sm_state_vec.copy() if sm_state_vec is not None else np.zeros(8)
        act = int(current_action) if current_action is not None else 0

        just_closed = []

        # ── 1. Open new episode when M60 creates a question ───
        if q60_just_created and fi >= 0 and fi not in self._open_episodes:
            ep = Episode(
                zone=fi,
                confused_label=sm_state_label,
                confused_vec=vec.copy(),
                pe_at_confusion=pe,
                created_at=self.t,
            )
            self._open_episodes[fi] = ep
            self._total_opened += 1

        # ── 2. Append current action to all open episodes ─────
        for ep in self._open_episodes.values():
            if len(ep.action_trace) < M63_MAX_EPISODE_LEN:
                ep.action_trace.append(act)
            ep.duration = self.t - ep.created_at

        # ── 3. Close episodes when M60 resolves questions ─────
        for resolved_q in q60_just_resolved:
            zone = resolved_q.get('zone', -1)
            if zone in self._open_episodes:
                ep = self._open_episodes.pop(zone)
                ep.pe_at_resolution = pe
                ep.resolved_at      = self.t
                ep.resolved_label   = sm_state_label
                ep.duration         = self.t - ep.created_at
                self._episodes.append(ep)
                self._total_resolved += 1
                just_closed.append(ep.to_dict())

        # ── 4. Abandon stale open episodes ────────────────────
        stale = [z for z, ep in self._open_episodes.items()
                 if self.t - ep.created_at > M63_MAX_EPISODE_LEN]
        for z in stale:
            self._open_episodes.pop(z)
            self._total_abandoned += 1

        # ── 5. Recognition — does current context match a past episode? ──
        recognition   = 0.0
        action_prior  = np.ones(self.n_actions, dtype=np.float64) / self.n_actions
        best_zone     = None

        if fi >= 0 and self._episodes:
            best_sim = 0.0
            best_ep  = None
            for ep in self._episodes:
                if ep.zone == fi:
                    sim = ep.similarity_to(vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_ep  = ep

            if best_ep is not None and best_sim >= M63_MATCH_THRESH:
                recognition  = float(best_sim)
                action_prior = best_ep.action_prior(self.n_actions)
                best_zone    = best_ep.zone

        self._last_recognition  = recognition
        self._last_action_prior = action_prior.copy()

        self.t += 1

        return {
            'recognition':    recognition,
            'action_prior':   action_prior,
            'episode_open':   len(self._open_episodes) > 0,
            'episodes_total': self._total_resolved,
            'best_zone':      best_zone,
            'just_closed':    just_closed,
        }

    # ── Diagnostics ───────────────────────────────────────────

    def episodes_for_zone(self, zone: int) -> List[Episode]:
        return [ep for ep in self._episodes if ep.zone == zone]

    def best_episode_for_zone(self, zone: int) -> Optional[Episode]:
        eps = self.episodes_for_zone(zone)
        if not eps: return None
        return max(eps, key=lambda e: e.pe_drop)

    def summary(self):
        print(f"  EpisodicTrace (M63) — step {self.t}")
        print(f"  Open episodes:  {len(self._open_episodes)}")
        print(f"  Resolved:       {self._total_resolved}/{self._total_opened}")
        print(f"  Abandoned:      {self._total_abandoned}")
        print(f"  Recognition:    {self._last_recognition:.3f}")
        if self._episodes:
            best = max(self._episodes, key=lambda e: e.pe_drop)
            print(f"  Best episode:   zone={best.zone} "
                  f"pe_drop={best.pe_drop:.2f} dur={best.duration}")

    def get_state(self) -> dict:
        return {
            't':              self.t,
            'open_count':     len(self._open_episodes),
            'resolved_total': self._total_resolved,
            'abandoned':      self._total_abandoned,
            'recognition':    self._last_recognition,
            'episodes':       [ep.to_dict() for ep in self._episodes],
        }