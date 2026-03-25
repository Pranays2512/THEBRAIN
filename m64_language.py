"""
M64: LANGUAGE SEED — THE BRAIN'S FIRST WORDS
=============================================

WHAT THIS IS
------------
M64 is the first module that gives the brain *symbolic handles* on its
own experience.

Every module below it operates in continuous numeric space:
  M59 asks: what kind of thing am I right now? → 8-float vector
  M63 asks: have I been confused here before?  → cosine similarity
  M57 asks: which action is best?              → sim_values array

None of them can produce the thought:
  "I am confused at zone 3. Last time this happened, East resolved it
   in 4 steps. I should try East."

That sentence requires three things:
  1. A NAME for the situation   ("confused at zone 3")
  2. A RECORD of past outcomes  ("East resolved it in 4 steps")
  3. COMPOSITION                (link name → record → action)

M64 provides all three. It is NOT a large language model. It does not
use external text. Its vocabulary is earned entirely from experience.

THREE THINGS M64 PROVIDES
--------------------------

1. SITUATION NAMING (SituationTag)
   M64 clusters recurring (state_label × zone × outcome_quality) triples
   into named situations. Each tag is a compound label like:
     "confused@z3"     — confused in zone 3
     "stuck@z1"        — stuck in zone 1
     "curious@z5"      — curious in zone 5
     "hunting@z2"      — hunting in zone 2
   Tags are assigned by matching: if current (state_label, zone) matches
   a known tag's prototype, that tag fires. New combinations become new tags.
   Maximum M64_MAX_TAGS tags; oldest are evicted FIFO when full.

2. OUTCOME MEMORY (per-tag action statistics)
   Each tag stores per-action outcome statistics:
     times_tried[action]   — how often action was taken in this situation
     times_resolved[action] — how often it led to PE drop (resolution)
     mean_duration[action]  — mean steps to resolve when it worked
   This is the brain's first *situation-specific* action memory —
   distinct from M56's Q-values (state-action globally) and M63's
   episode trace (single past episode).

3. COMPOSED NARRATIVE (for M57 / M61)
   Each step M64 returns:
     situation_tag     str or None  — the current situation's name
     tag_action_prior  (n_actions,) — soft prior from outcome memory
     tag_confidence    float [0,1]  — how well the current state matches
                                      the tag prototype (drives prior weight)
     narrative         str          — one-sentence description the brain
                                      can use in M61 thought loops
                                      e.g. "confused@z3: East worked 3/4 times"

HOW IT PLUGS IN
---------------
M64.step() is called inside Brain.step() after M63, before the return.

It reads:
  sm_state_label    str     — from M59 (state label)
  sm_state_vec      (8,)    — from M59 (for prototype matching)
  freq_idx          int     — current zone
  m63_just_closed   bool    — did M63 close an episode this step?
  m63_best_episode  dict    — the episode that just closed (for learning)
  current_action    int     — action taken this step
  prediction_error  float   — current PE (for outcome quality signal)
  q60_just_resolved list    — M60 resolved questions (confirms learning)

It writes into Brain return dict:
  m64_situation_tag      str or None  — current situation name
  m64_tag_action_prior   (n_actions,) — soft prior from outcome memory
  m64_tag_confidence     float        — tag match strength
  m64_narrative          str          — composed sentence
  m64_tags_total         int          — total distinct situations learned

M57 INTEGRATION
---------------
M64's tag_action_prior feeds into M57 as a second prior (alongside M63).
Unlike M63's prior (single best episode), M64's prior is an *aggregate*
across all times the brain was in this situation — more statistically
stable, less dependent on one lucky episode.

When both M63 and M64 priors are available, M57 blends them:
  combined_prior = M63_WEIGHT * m63_prior + M64_WEIGHT * m64_prior
  where M63_WEIGHT = 0.40, M64_WEIGHT = 0.60  (M64 more stable)

M61 INTEGRATION
---------------
M64's narrative is available to M61's thought loop as a seed sentence.
When M61 runs, it can prepend the narrative to its internal monologue —
the brain "thinks in words" about its current situation before simulating.
This is the first step toward language-mediated planning.

BIOLOGICAL BASIS
----------------
M64 corresponds to the language-self interface in the left hemisphere:

  Broca's area (BA44/45): produces compositional structure from symbols.
  Angular gyrus (BA39):   binds words to concepts and situations.
  Left PFC:               maintains situation frames during planning.

The key insight from cognitive neuroscience: language in the brain is
not primarily FOR communication. It is a COMPRESSION and RETRIEVAL tool.
Naming a situation makes it retrievable across contexts, generalisable
to new instances, and composable with other named concepts.

A rat cannot think "I've been confused here before and East worked."
It can only have the pattern-completed activation. Language lets the
brain *hold* a past episode long enough to reason about it explicitly.

PARAMETERS
----------
M64_MAX_TAGS           = 32    — maximum distinct situation tags.
                                  World5 has 8 zones × ~5 states = 40 possible
                                  combinations but many are rare. 32 covers the
                                  common ones and evicts stale edge cases.

M64_TAG_MATCH_THRESH   = 0.75  — cosine similarity of self-state vec to tag
                                  prototype before tag fires. High threshold:
                                  the brain only names a situation it's genuinely
                                  in, not one it's vaguely resembling.

M64_OUTCOME_EMA_ALPHA  = 0.20  — smoothing for per-tag action outcome stats.
                                  tau ~5 episodes per tag. Fast enough to
                                  update when new episodes close, slow enough
                                  not to flip on a single lucky action.

M64_PRIOR_WEIGHT       = 0.15  — maximum M57 sim_values bias from M64 prior.
                                  Same scale as M63_PRIOR_WEIGHT. Conservative:
                                  aggregate statistics guide but don't dictate.

M64_MIN_TRIES          = 3     — action must have been tried at least this many
                                  times in a tag before its outcome rate is
                                  trusted. Below this: prior contribution for
                                  that action is 0 (not enough evidence).
"""

import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

M64_MAX_TAGS          = 32
M64_TAG_MATCH_THRESH  = 0.75
M64_OUTCOME_EMA_ALPHA = 0.20
M64_PRIOR_WEIGHT      = 0.15
M64_MIN_TRIES         = 3

# State labels that qualify for tagging — all M59 labels
_TAGGABLE_LABELS = {'confused', 'focused', 'stuck', 'curious',
                    'satisfied', 'hunting', 'drifting', 'alert'}

# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class SituationTag:
    """
    One named situation the brain has learned to recognise.

    name        — compound label e.g. "confused@z3"
    label       — M59 state label component
    zone        — zone index component
    prototype   — mean self-state vec at creation + first N encounters
    n_actions   — number of available actions

    Per-action outcome statistics (updated when M63 closes an episode
    while this tag was active):
      times_tried[a]    — how often action a was taken while in this tag
      times_resolved[a] — how often it led to PE drop / M60 resolution
      mean_duration[a]  — EMA of steps to resolve when it worked

    created_at      — step number of tag creation
    last_seen_at    — step number of last activation
    total_activations — lifetime activations
    """
    name:              str
    label:             str
    zone:              int
    prototype:         np.ndarray          # (8,) float, normalised
    n_actions:         int
    created_at:        int

    times_tried:       np.ndarray = field(default_factory=lambda: None)
    times_resolved:    np.ndarray = field(default_factory=lambda: None)
    mean_duration:     np.ndarray = field(default_factory=lambda: None)
    last_seen_at:      int        = 0
    total_activations: int        = 0

    def __post_init__(self):
        if self.times_tried is None:
            self.times_tried    = np.zeros(self.n_actions, dtype=np.float64)
        if self.times_resolved is None:
            self.times_resolved = np.zeros(self.n_actions, dtype=np.float64)
        if self.mean_duration is None:
            self.mean_duration  = np.zeros(self.n_actions, dtype=np.float64)

    def match(self, state_vec: np.ndarray) -> float:
        """Cosine similarity between state_vec and prototype."""
        a = self.prototype
        b = state_vec
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))

    def update_prototype(self, state_vec: np.ndarray, alpha: float = 0.10):
        """Slowly drift prototype toward new observations."""
        self.prototype = ((1.0 - alpha) * self.prototype
                          + alpha * state_vec)
        # Re-normalise
        n = float(np.linalg.norm(self.prototype))
        if n > 1e-9:
            self.prototype = self.prototype / n

    def record_outcome(self, action: int, resolved: bool, duration: int):
        """Record what happened after action was taken in this situation."""
        self.times_tried[action] += 1.0
        if resolved:
            self.times_resolved[action] += 1.0
            old_dur = self.mean_duration[action]
            # EMA update for duration
            if old_dur < 1e-9:
                self.mean_duration[action] = float(duration)
            else:
                self.mean_duration[action] = (
                    (1.0 - M64_OUTCOME_EMA_ALPHA) * old_dur
                    + M64_OUTCOME_EMA_ALPHA * float(duration)
                )

    def action_prior(self) -> np.ndarray:
        """
        Soft prior over actions based on resolution rate.

        Actions tried fewer than M64_MIN_TRIES times contribute 0.
        Others contribute resolution_rate = resolved / tried.
        Result is normalised to sum=1.
        If no action has enough evidence, returns uniform.
        """
        prior = np.zeros(self.n_actions, dtype=np.float64)
        for a in range(self.n_actions):
            if self.times_tried[a] >= M64_MIN_TRIES:
                prior[a] = self.times_resolved[a] / self.times_tried[a]
        s = prior.sum()
        if s < 1e-9:
            return np.ones(self.n_actions, dtype=np.float64) / self.n_actions
        return prior / s

    def best_action(self) -> Optional[int]:
        """
        Action with highest resolution rate (min tries enforced).
        Returns None if no action has sufficient evidence.
        """
        prior = self.action_prior()
        # Uniform prior means no evidence — return None
        if np.allclose(prior, 1.0 / self.n_actions):
            return None
        return int(np.argmax(prior))

    def narrative_fragment(self) -> str:
        """
        Best-action evidence summary for this tag.
        e.g. "East worked 3/4 times" or "no data yet"
        """
        best = self.best_action()
        if best is None:
            return "no data yet"
        tried    = int(self.times_tried[best])
        resolved = int(self.times_resolved[best])
        dur      = self.mean_duration[best]
        action_names = {0: 'North', 1: 'East', 2: 'South', 3: 'West',
                        4: 'Stay'}
        name = action_names.get(best, f'a{best}')
        if dur > 0.5:
            return f"{name} worked {resolved}/{tried} times (~{dur:.0f} steps)"
        return f"{name} worked {resolved}/{tried} times"


# ═══════════════════════════════════════════════════════════════
# LANGUAGE SEED
# ═══════════════════════════════════════════════════════════════

class LanguageSeed:
    """
    M64 — situation naming and language-mediated planning.

    Learns named situation tags from (state_label × zone) combinations.
    Accumulates per-tag action outcome statistics from M63 episodes.
    Returns a tag_action_prior for M57 and a narrative string for M61.

    Parameters
    ----------
    n_actions : int  — number of available actions (must match M56/M57)
    seed      : int  — random seed (reserved for future stochastic features)
    """

    def __init__(self, n_actions: int = 4, seed: int = 42):
        self.n_actions = n_actions

        # Tag store — ordered by creation time (FIFO eviction when full)
        self._tags: list[SituationTag] = []
        self._tag_index: dict[str, SituationTag] = {}  # name → tag (fast lookup)

        # Current active tag (or None)
        self._active_tag:       Optional[SituationTag] = None
        self._active_tag_score: float = 0.0

        # Episode tracking — records action taken when a tag was active
        # so we can close the outcome when M63 reports resolution.
        # Stores the action taken while the current open-episode tag was active.
        self._open_episode_tag:    Optional[SituationTag] = None
        self._open_episode_action: int = -1
        self._open_episode_start:  int = -1

        # Step counter
        self.t = 0

        # Diagnostics
        self._last_narrative    = ""
        self._last_tag_name     = None
        self._last_tag_score    = 0.0
        self._n_outcomes_learned = 0

    # ── Main step ─────────────────────────────────────────────

    def step(self,
             sm_state_label:    str,
             sm_state_vec:      np.ndarray,
             freq_idx:          int,
             current_action:    int,
             prediction_error:  float,
             m63_just_closed:   bool          = False,
             m63_episode:       Optional[dict] = None,
             q60_just_resolved: list          = None,
             ) -> dict:
        """
        One M64 step.

        Parameters
        ----------
        sm_state_label   : str     — M59 state label ('confused', 'stuck', ...)
        sm_state_vec     : (8,)    — M59 self-state vector
        freq_idx         : int     — current zone index
        current_action   : int     — action just taken
        prediction_error : float   — current L2 PE
        m63_just_closed  : bool    — did M63 close an episode this step?
        m63_episode      : dict    — the M63 episode that just closed (or None)
        q60_just_resolved: list    — M60 resolved questions this step

        Returns
        -------
        dict with keys:
          situation_tag      str or None
          tag_action_prior   (n_actions,) float
          tag_confidence     float [0,1]
          narrative          str
          tags_total         int
        """
        vec = np.array(sm_state_vec, dtype=np.float64)

        # ── 1. Match current state to existing tags ────────────
        active_tag, match_score = self._match_tag(
            label=sm_state_label,
            zone=freq_idx,
            vec=vec,
        )

        # ── 2. If no match and zone is valid, create new tag ───
        if active_tag is None and freq_idx >= 0 and sm_state_label in _TAGGABLE_LABELS:
            active_tag = self._create_tag(
                label=sm_state_label,
                zone=freq_idx,
                vec=vec,
            )
            match_score = 1.0  # just created — perfect match

        # ── 3. Update prototype toward current observation ────
        if active_tag is not None:
            active_tag.update_prototype(vec)
            active_tag.last_seen_at    = self.t
            active_tag.total_activations += 1
            match_score = active_tag.match(vec)

        self._active_tag       = active_tag
        self._active_tag_score = float(match_score) if active_tag else 0.0

        # ── 4. Track open episode for outcome recording ───────
        # When a tag is active and M63 has an open episode, record the
        # tag and action so we can attribute the outcome when M63 closes.
        if active_tag is not None and self._open_episode_tag is None:
            self._open_episode_tag    = active_tag
            self._open_episode_action = int(current_action)
            self._open_episode_start  = self.t

        # ── 5. Record outcome when M63 closes an episode ──────
        if m63_just_closed and m63_episode is not None:
            self._record_episode_outcome(m63_episode)

        # Also gate on M60 resolution (belt-and-suspenders)
        if q60_just_resolved and self._open_episode_tag is not None:
            # M60 resolved something this step — count as success for
            # the open episode tag's action even if M63 didn't close.
            # This lets M64 learn from M60 resolutions that M63 didn't
            # explicitly track.
            duration = max(1, self.t - self._open_episode_start)
            self._open_episode_tag.record_outcome(
                action=self._open_episode_action,
                resolved=True,
                duration=duration,
            )
            self._n_outcomes_learned += 1
            self._open_episode_tag    = None
            self._open_episode_action = -1
            self._open_episode_start  = -1

        # ── 6. Compute prior and narrative ────────────────────
        if active_tag is not None and match_score >= M64_TAG_MATCH_THRESH:
            tag_prior   = active_tag.action_prior()
            tag_name    = active_tag.name
            fragment    = active_tag.narrative_fragment()
            narrative   = f"{tag_name}: {fragment}"
        else:
            tag_prior   = np.ones(self.n_actions, dtype=np.float64) / self.n_actions
            tag_name    = None
            narrative   = ""

        self._last_narrative = narrative
        self._last_tag_name  = tag_name
        self._last_tag_score = self._active_tag_score

        self.t += 1

        return {
            'situation_tag':     tag_name,
            'tag_action_prior':  tag_prior,
            'tag_confidence':    float(self._active_tag_score),
            'narrative':         narrative,
            'tags_total':        len(self._tags),
        }

    # ── Tag matching ──────────────────────────────────────────

    def _match_tag(self,
                   label: str,
                   zone:  int,
                   vec:   np.ndarray,
                   ) -> tuple[Optional[SituationTag], float]:
        """
        Find the best-matching existing tag for (label, zone, vec).

        Requires:
          - tag.label == label        (exact label match — tags are label-specific)
          - tag.zone  == zone         (exact zone match — tags are zone-specific)
          - cosine_sim >= M64_TAG_MATCH_THRESH

        Returns (tag, score) or (None, 0.0).
        """
        best_tag   = None
        best_score = 0.0

        for tag in self._tags:
            if tag.label != label or tag.zone != zone:
                continue
            score = tag.match(vec)
            if score > best_score:
                best_score = score
                best_tag   = tag

        if best_score >= M64_TAG_MATCH_THRESH:
            return best_tag, best_score
        return None, 0.0

    # ── Tag creation ──────────────────────────────────────────

    def _create_tag(self,
                    label: str,
                    zone:  int,
                    vec:   np.ndarray,
                    ) -> SituationTag:
        """
        Create a new situation tag and add it to the store.

        If at capacity (M64_MAX_TAGS), evict the least-recently-seen tag.
        """
        name = f"{label}@z{zone}"

        # Normalise prototype vector
        prototype = vec.copy()
        n = float(np.linalg.norm(prototype))
        if n > 1e-9:
            prototype = prototype / n

        tag = SituationTag(
            name=name,
            label=label,
            zone=zone,
            prototype=prototype,
            n_actions=self.n_actions,
            created_at=self.t,
        )

        if len(self._tags) >= M64_MAX_TAGS:
            # Evict least-recently-seen tag
            evict_idx = int(np.argmin([t.last_seen_at for t in self._tags]))
            evicted   = self._tags.pop(evict_idx)
            if evicted.name in self._tag_index:
                del self._tag_index[evicted.name]

        self._tags.append(tag)
        self._tag_index[name] = tag
        return tag

    # ── Outcome recording ─────────────────────────────────────

    def _record_episode_outcome(self, episode: dict):
        """
        When M63 closes an episode, attribute the outcome to the
        open episode tag (if one was active).
        """
        if self._open_episode_tag is None:
            return

        duration = max(1, self.t - self._open_episode_start)
        pe_drop  = float(episode.get('pe_drop', 0.0))
        resolved = pe_drop > 0.0  # any PE improvement = success

        self._open_episode_tag.record_outcome(
            action=self._open_episode_action,
            resolved=resolved,
            duration=duration,
        )
        self._n_outcomes_learned += 1

        # Close the open episode tracking
        self._open_episode_tag    = None
        self._open_episode_action = -1
        self._open_episode_start  = -1

    # ── Diagnostics ───────────────────────────────────────────

    def get_tag(self, name: str) -> Optional[SituationTag]:
        """Return a tag by name, or None if not found."""
        return self._tag_index.get(name, None)

    def all_tags(self) -> list[SituationTag]:
        """Return all current tags."""
        return list(self._tags)

    def summary(self):
        print(f"  LanguageSeed (M64) — step {self.t}")
        print(f"  Tags learned:     {len(self._tags)}/{M64_MAX_TAGS}")
        print(f"  Outcomes learned: {self._n_outcomes_learned}")
        if self._last_tag_name:
            print(f"  Active tag:       {self._last_tag_name}  "
                  f"(match={self._last_tag_score:.3f})")
            print(f"  Narrative:        {self._last_narrative}")
        else:
            print(f"  Active tag:       None")
        if self._tags:
            print(f"  Tag roster:")
            for tag in sorted(self._tags, key=lambda t: -t.total_activations)[:6]:
                best = tag.best_action()
                action_names = {0: 'N', 1: 'E', 2: 'S', 3: 'W', 4: 'Stay'}
                ba_str = action_names.get(best, f'a{best}') if best is not None else '?'
                print(f"    {tag.name:<18}  act={tag.total_activations:>4}  "
                      f"best={ba_str}  outcomes={self._n_outcomes_learned}")