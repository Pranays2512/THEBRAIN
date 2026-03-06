"""
M54 EXPERIENCE BUFFER
=====================
Sits between the cortex (M54) and Layer 2. Accumulates what the cortex
perceives into structured, queryable memory — the first thing the system
does with the data it computes.

WHAT THIS IS
------------
Every time the frequency changes (CUSUM fires), a new episode begins.
The buffer records each episode as an Experience — a compact summary
of what was perceived, how surprising it was, how quickly it became
familiar, and what came immediately before it.

The result is a log you can query:
  "Have I seen this before?"
  "What usually follows this frequency?"
  "Is this more surprising than usual?"
  "How long does it take me to learn this?"

This is the raw material Layer 2 needs for sequence memory. It does not
learn sequences itself — it records the transitions so Layer 2 can.

DESIGN PRINCIPLES
-----------------
1. Episode-based, not sample-based.
   The buffer thinks in episodes (stable frequency periods), not in
   individual timesteps. One episode = one coherent perception event.
   This matches how memory works: you remember "I heard 0.80 Hz for
   30 seconds" not "I received 350 individual samples."

2. Surprise-triggered richness.
   Onset surprise (qe at first encounter) is the primary salience signal.
   The buffer tracks how surprise decays within the episode — fast decay
   means fast learning; slow decay means the cortex is struggling.

3. Transition graph as a byproduct.
   Every episode records its predecessor (prev_bmu_idx). Over many
   episodes, this naturally builds a co-occurrence matrix: how often
   does frequency A precede frequency B? Layer 2 reads this directly.

4. Warmup awareness.
   The EMA baseline in M54 needs ~100 steps to settle. The buffer flags
   episodes that occurred during warmup so Layer 2 can discount them.

EPISODE LIFECYCLE
-----------------
  OPEN   → episode started, accumulating per-sample data
  CLOSED → CUSUM fired (or buffer reset), episode finalized

Per sample (inside an open episode):
  - append qe, eta, w, bmu_idx to running lists
  - track peak eta, min w, running qe mean

On episode close:
  - compute onset_qe   (mean of first N_ONSET_SAMPLES samples)
  - compute settled_qe (mean of last N_SETTLE_SAMPLES samples)
  - compute qe_decay   = onset_qe - settled_qe  (how much it adapted)
  - compute eta_peak   (max learning rate seen)
  - compute w_mean     (mean confidence)
  - compute duration   (seconds)
  - store prev_bmu_idx (predecessor, seed of sequence memory)
  - look up times_seen for this bmu_idx (familiarity counter)
  - store and increment

OUTPUTS PER EPISODE
-------------------
{
  'episode_id':   int,         # monotonic counter
  't_start':      float,       # time episode opened (seconds)
  't_end':        float,       # time episode closed (seconds)
  'duration':     float,       # t_end - t_start (seconds)

  'freq_est':     float,       # mean decoded frequency during episode
  'bmu_idx':      int,         # most common BMU (mode) during episode
  'bmu_pos':      (row, col),  # grid position of mode BMU

  'onset_qe':     float,       # mean QE over first N_ONSET_SAMPLES.
                               # First impression — includes transition noise.
                               # Use for novelty detection, not learning curves.
  'settled_qe':   float,       # mean QE over last N_SETTLE_SAMPLES
  'qe_decay':     float,       # onset_qe - settled_qe (adaptation speed)
  'eta_peak':     float,       # highest learning rate during episode
  'w_mean':       float,       # mean stability weight during episode

  'times_seen':   int,         # how many times this BMU has been the
                               # dominant BMU in a closed episode (before
                               # this one — so 0 = first encounter)
  'is_novel':     bool,        # onset_qe > NOVEL_THRESH
  'is_warmup':    bool,        # episode occurred during EMA warmup period

  'prev_bmu_idx': int | None,  # BMU of immediately preceding episode
                               # (None for the very first episode)
  'prev_freq':    float | None # freq_est of preceding episode
}

TRANSITION GRAPH
----------------
buffer.transitions  →  dict: (prev_bmu_idx, curr_bmu_idx) → count
buffer.freq_follows →  dict: bmu_idx → Counter of successor bmu_idx

Query: "what usually follows BMU 42?"
  buffer.most_likely_successor(42)  → (bmu_idx, probability)

Query: "have I seen BMU 42 before?"
  buffer.times_seen(42)  → int

Query: "show me all novel episodes"
  buffer.novel_episodes()  → list of Experience dicts

Query: "learning curve for BMU 42"
  buffer.qe_curve(42)  → list of (episode_id, onset_qe, settled_qe)
"""

import numpy as np
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

# How many samples at the START of an episode define "onset surprise"
N_ONSET_SAMPLES  = 5

# How many samples at the END of an episode define "settled surprise"
N_SETTLE_SAMPLES = 10

# QE threshold above which an episode is flagged as novel
# Matches M54's SURPRISE_THRESH = 0.15, but applied to onset mean
NOVEL_THRESH = 0.15

# After this many cortex steps the EMA baseline is considered settled
# QE_EMA_ALPHA=0.01 → τ=100 steps. Use 2τ=200 as conservative threshold.
WARMUP_STEPS = 200

# Minimum episode duration in samples to be worth storing
# Avoids spurious micro-episodes from double-fires or noise
MIN_EPISODE_SAMPLES = 3

# Maximum episodes to keep in memory
# Oldest are dropped when full (ring buffer behaviour)
MAX_EPISODES = 10_000


# ═══════════════════════════════════════════════════════════════
# OPEN EPISODE  (accumulator — not stored directly)
# ═══════════════════════════════════════════════════════════════

class _OpenEpisode:
    """
    Accumulates per-sample data for the currently-active episode.
    Closed and converted to a plain dict when the frequency changes.
    Not part of the public API.
    """
    __slots__ = [
        't_start', 'cortex_step_start',
        'qe_samples', 'eta_samples', 'w_samples',
        'bmu_idx_samples', 'freq_samples',
    ]

    def __init__(self, t_start, cortex_step_start):
        self.t_start            = t_start
        self.cortex_step_start  = cortex_step_start
        self.qe_samples         = []
        self.eta_samples        = []
        self.w_samples          = []
        self.bmu_idx_samples    = []
        self.freq_samples       = []

    def push(self, qe, eta, w, bmu_idx, freq):
        self.qe_samples.append(qe)
        self.eta_samples.append(eta)
        self.w_samples.append(w)
        self.bmu_idx_samples.append(bmu_idx)
        self.freq_samples.append(freq)

    @property
    def n_samples(self):
        return len(self.qe_samples)


# ═══════════════════════════════════════════════════════════════
# EXPERIENCE BUFFER
# ═══════════════════════════════════════════════════════════════

class ExperienceBuffer:
    """
    Surprise-triggered episodic memory for the M54 cortex.

    Usage
    -----
    buf = ExperienceBuffer()

    # In your processing loop, after calling cortex.step() and cusum.update():
    buf.push(
        t           = current_time,
        cortex_out  = cortex.step(...),   # full dict from CortexM54.step()
        decoded_freq= fused_freq,         # float Hz
        stability_w = w,                  # float [0,1]
        transition  = is_novel,           # bool from DivergenceCUSUM.update()
        cortex_step = cortex.t,           # cortex.t after step()
    )

    # Query
    ep = buf.last_episode()
    seq = buf.most_likely_successor(ep['bmu_idx'])
    """

    def __init__(self):
        self._episodes      = []          # list of closed episode dicts
        self._episode_id    = 0           # monotonic counter

        # Per-BMU familiarity: how many CLOSED episodes had this BMU as mode
        self._bmu_seen_count = Counter()  # bmu_idx → int

        # Transition counts: (prev_bmu, curr_bmu) → count
        self.transitions    = Counter()

        # Successor map: bmu_idx → Counter of following bmu_idx
        self.freq_follows   = defaultdict(Counter)

        # Currently open episode
        self._open          = None        # _OpenEpisode | None
        self._prev_bmu_idx  = None        # mode BMU of last closed episode
        self._prev_freq     = None        # freq_est of last closed episode

    # ── Public push interface ──────────────────────────────────

    def push(self, t, cortex_out, decoded_freq, stability_w,
             transition, cortex_step):
        """
        Feed one timestep of cortex output into the buffer.

        Parameters
        ----------
        t            : float  — current simulation time (seconds)
        cortex_out   : dict   — return value of CortexM54.step()
        decoded_freq : float  — fused frequency estimate (Hz)
        stability_w  : float  — PLV stability weight [0, 1]
        transition   : bool   — True if CUSUM fired this step
        cortex_step  : int    — cortex.t (used for warmup detection)
        """
        qe      = cortex_out['qe']
        eta     = cortex_out['eta']
        bmu_idx = cortex_out['bmu_idx']

        # Open a new episode if none is active
        if self._open is None:
            self._open = _OpenEpisode(t_start=t,
                                      cortex_step_start=cortex_step)

        # Accumulate into open episode
        self._open.push(qe=qe, eta=eta, w=stability_w,
                        bmu_idx=bmu_idx, freq=decoded_freq)

        # Close episode on transition (or if this is the first push and
        # we already have a transition — handles edge case at t=0)
        if transition and self._open.n_samples >= MIN_EPISODE_SAMPLES:
            self._close_episode(t_end=t,
                                cortex_step_end=cortex_step)
            # Immediately open a fresh episode at this boundary
            self._open = _OpenEpisode(t_start=t,
                                      cortex_step_start=cortex_step)
            # Push the current sample into the new episode too
            # (the transition sample belongs to both the end of old
            # and the start of new — we put it in the new one)
            self._open.push(qe=qe, eta=eta, w=stability_w,
                            bmu_idx=bmu_idx, freq=decoded_freq)

    def flush(self, t_end, cortex_step):
        """
        Close any open episode at end of stream / session.
        Call this when you're done feeding data.
        """
        if self._open is not None and \
                self._open.n_samples >= MIN_EPISODE_SAMPLES:
            self._close_episode(t_end=t_end,
                                cortex_step_end=cortex_step)
        self._open = None

    # ── Episode close logic ────────────────────────────────────

    def _close_episode(self, t_end, cortex_step_end):
        ep = self._open

        # Mode BMU = the neuron that won most during this episode
        bmu_mode = int(Counter(ep.bmu_idx_samples).most_common(1)[0][0])

        # Convert bmu_idx to (row, col) — works for any 8×8 grid
        # We import GRID_W lazily to avoid circular imports
        from m54_cortex import GRID_W, GRID_H
        bmu_pos = (bmu_mode // GRID_W, bmu_mode % GRID_W)

        # Onset QE: first impression of the new frequency.
        # Includes transition noise by design — this is what the cortex
        # actually experienced first. Useful for novelty detection and
        # surprise flagging. NOT the right metric for a learning curve
        # (use settled_qe for that — see qe_curve()).
        onset_slice = ep.qe_samples[:N_ONSET_SAMPLES]
        onset_qe    = float(np.mean(onset_slice))

        # Settled QE: last N_SETTLE_SAMPLES (how well adapted by end)
        settle_slice  = ep.qe_samples[-N_SETTLE_SAMPLES:]
        settled_qe    = float(np.mean(settle_slice))

        qe_decay      = float(onset_qe - settled_qe)
        eta_peak      = float(max(ep.eta_samples))
        w_mean        = float(np.mean(ep.w_samples))
        freq_est      = float(np.mean(ep.freq_samples))
        duration      = float(t_end - ep.t_start)

        # Warmup: episode started before EMA settled
        is_warmup = (ep.cortex_step_start < WARMUP_STEPS)

        # Familiarity: how many prior closed episodes had this BMU
        times_seen = self._bmu_seen_count[bmu_mode]
        is_novel   = (onset_qe > NOVEL_THRESH)

        record = {
            'episode_id':    self._episode_id,
            't_start':       float(ep.t_start),
            't_end':         float(t_end),
            'duration':      duration,

            'freq_est':      freq_est,
            'bmu_idx':       bmu_mode,
            'bmu_pos':       bmu_pos,

            'onset_qe':      onset_qe,
            'settled_qe':    settled_qe,
            'qe_decay':      qe_decay,
            'eta_peak':      eta_peak,
            'w_mean':        w_mean,

            'times_seen':    times_seen,
            'is_novel':      is_novel,
            'is_warmup':     is_warmup,

            'prev_bmu_idx':  self._prev_bmu_idx,
            'prev_freq':     self._prev_freq,

            # Internals for Layer 2 / diagnostics
            '_qe_series':          list(ep.qe_samples),
            '_eta_series':         list(ep.eta_samples),
            '_n_samples':          ep.n_samples,
            '_cortex_step_start':  ep.cortex_step_start,
        }

        # Store (ring buffer: drop oldest if full)
        if len(self._episodes) >= MAX_EPISODES:
            self._episodes.pop(0)
        self._episodes.append(record)

        # Update familiarity counter AFTER storing (times_seen = before)
        self._bmu_seen_count[bmu_mode] += 1

        # Update transition graph
        if self._prev_bmu_idx is not None:
            key = (self._prev_bmu_idx, bmu_mode)
            self.transitions[key] += 1
            self.freq_follows[self._prev_bmu_idx][bmu_mode] += 1

        # Advance state
        self._prev_bmu_idx = bmu_mode
        self._prev_freq    = freq_est
        self._episode_id  += 1

    # ── Query API ─────────────────────────────────────────────

    def last_episode(self):
        """Most recently closed episode, or None."""
        return self._episodes[-1] if self._episodes else None

    def n_episodes(self):
        """Total closed episodes."""
        return len(self._episodes)

    def times_seen(self, bmu_idx):
        """How many closed episodes had bmu_idx as their mode BMU."""
        return self._bmu_seen_count[bmu_idx]

    def novel_episodes(self):
        """All closed episodes flagged as novel (onset_qe > NOVEL_THRESH)."""
        return [e for e in self._episodes if e['is_novel']]

    def episodes_for_bmu(self, bmu_idx):
        """All closed episodes whose mode BMU matches bmu_idx."""
        return [e for e in self._episodes if e['bmu_idx'] == bmu_idx]

    def most_likely_successor(self, bmu_idx):
        """
        Given a BMU index, return (successor_bmu_idx, probability).
        Returns (None, 0.0) if no transitions recorded for this BMU.
        """
        counts = self.freq_follows.get(bmu_idx)
        if not counts:
            return None, 0.0
        total = sum(counts.values())
        best  = counts.most_common(1)[0]
        return best[0], best[1] / total

    def qe_curve(self, bmu_idx):
        """
        Learning curve for a BMU: how onset and settled QE change
        across repeated encounters.
        Returns list of (episode_id, onset_qe, settled_qe).
        """
        return [
            (e['episode_id'], e['onset_qe'], e['settled_qe'])
            for e in self._episodes
            if e['bmu_idx'] == bmu_idx
        ]

    def transition_matrix(self, n_neurons=64):
        """
        Return an (n_neurons × n_neurons) numpy array where
        M[i, j] = number of times BMU i was followed by BMU j.
        This is the raw input for Layer 2's sequence learner.
        """
        M = np.zeros((n_neurons, n_neurons), dtype=np.int32)
        for (i, j), count in self.transitions.items():
            if i < n_neurons and j < n_neurons:
                M[i, j] = count
        return M

    def surprise_baseline(self):
        """
        Mean and std of onset_qe across all non-warmup closed episodes.
        Useful for calibrating what counts as "truly novel" vs routine.
        Returns (mean, std) or (None, None) if no data.
        """
        vals = [e['onset_qe'] for e in self._episodes if not e['is_warmup']]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    def familiarity_map(self, grid_h=8, grid_w=8):
        """
        Return (grid_h × grid_w) array of times_seen counts.
        Shows which parts of the cortical map have been experienced most.
        """
        fmap = np.zeros((grid_h, grid_w), dtype=np.int32)
        for bmu_idx, count in self._bmu_seen_count.items():
            r = bmu_idx // grid_w
            c = bmu_idx %  grid_w
            if 0 <= r < grid_h and 0 <= c < grid_w:
                fmap[r, c] = count
        return fmap

    def summary(self):
        """
        Print a human-readable summary of buffer state.
        """
        n = len(self._episodes)
        if n == 0:
            print("  ExperienceBuffer: empty")
            return

        novel  = sum(1 for e in self._episodes if e['is_novel'])
        warmup = sum(1 for e in self._episodes if e['is_warmup'])
        mean_dur = np.mean([e['duration'] for e in self._episodes])
        mean_onset = np.mean([e['onset_qe'] for e in self._episodes])
        mean_decay = np.mean([e['qe_decay'] for e in self._episodes])
        n_trans = sum(self.transitions.values())
        n_bmus  = len(self._bmu_seen_count)

        print(f"  ExperienceBuffer: {n} episodes  "
              f"({novel} novel, {warmup} warmup)")
        print(f"  Mean duration:   {mean_dur:.1f}s")
        print(f"  Mean onset QE:   {mean_onset:.4f}")
        print(f"  Mean QE decay:   {mean_decay:.4f}  "
              f"(+ve = cortex adapted, -ve = grew more surprised)")
        print(f"  Unique BMUs:     {n_bmus}/64")
        print(f"  Transitions:     {n_trans} total, "
              f"{len(self.transitions)} unique pairs")

        # Top 5 most-seen BMUs
        if self._bmu_seen_count:
            top = self._bmu_seen_count.most_common(5)
            print(f"  Most-seen BMUs:  "
                  + "  ".join(f"BMU{i}×{c}" for i, c in top))

        # Any strong transition predictions?
        strong = []
        for bmu_idx in self._bmu_seen_count:
            succ, prob = self.most_likely_successor(bmu_idx)
            if succ is not None and prob >= 0.60:
                strong.append((bmu_idx, succ, prob))
        if strong:
            print(f"  Strong transitions (p≥0.60):")
            for src, dst, p in sorted(strong, key=lambda x: -x[2])[:5]:
                print(f"    BMU{src:02d} → BMU{dst:02d}  p={p:.2f}")