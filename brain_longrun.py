"""
BRAIN LONG RUN — Continuous audio learning over 50,000 steps
=============================================================

One Brain instance. Never resets. Listens to a structured
audio environment for 50,000 steps and builds up real memory.

THE AUDIO ENVIRONMENT
---------------------
8 frequencies with a simple transition grammar — like a language.
Some transitions are common (high probability), some rare, some
never happen. After enough experience the brain should have an
internal model that reflects this structure.

Frequencies (Hz):   0.5  0.7  0.9  1.1  1.3  1.5  1.7  2.0
Labels:              A    B    C    D    E    F    G    H

Transition grammar (what follows what, with probability):
  A → B (0.7)  C (0.3)
  B → C (0.6)  D (0.4)
  C → A (0.5)  D (0.5)
  D → E (0.8)  B (0.2)
  E → F (0.6)  G (0.4)
  F → E (0.5)  H (0.4)  G (0.1)
  G → H (0.7)  E (0.3)
  H → A (0.6)  G (0.4)

This creates natural "loops" (A→B→C→A, D→E→F→E, G→H→A)
which L2 can learn if it runs long enough.

WHAT THIS MEASURES
------------------
Every 5000 steps a snapshot is taken:

  1. Familiarity map    — which frequencies feel known vs novel
  2. Transition probing — for each frequency, what does L2 predict comes next?
                          Does it match the grammar?
  3. Salience profile   — does the brain attend more to rare transitions?
  4. Planning rate      — how often does M57 override M56?
  5. Prediction accuracy at cluster level — does top prediction land
                          in the correct frequency zone?

EXPECTED BEHAVIOUR over 50,000 steps
--------------------------------------
Steps     0–5000:   Brain is new. Everything is novel. High salience,
                    high curiosity, low familiarity everywhere. L2 has
                    no model yet. Planning barely engages.

Steps 5000–15000:   Familiarity starts differentiating. Common
                    frequencies (those visited most) get higher
                    familiarity. L2 starts building transition stats.

Steps 15000–30000:  L2's P matrix starts reflecting the grammar.
                    High-probability transitions (A→B, D→E, G→H)
                    become predictable. Planning weight grows on
                    these transitions.

Steps 30000–50000:  Clear familiarity landscape. Rare transitions
                    (F→G: 0.1) still spike salience. Common ones
                    are calm. L2 predictions increasingly correct
                    at cluster level. M57 engages strongly on
                    well-learned sequences.

RUN TIME
--------
M50 calibration:  ~3-5 minutes
50,000 brain steps: ~5-10 minutes (no oscillator sim — we reuse
                    calibrated M50 outputs via a fast decode loop)

Total: ~10-15 minutes.
"""

import numpy as np
import json
import os
import time
from collections import deque, defaultdict

from m50_neuron import (
    run_sim, make_blocks,
    decode_resonance, build_reverse_lookup,
    DivergenceCUSUM, compute_stability_plv,
    PLV_STAB_WINDOW, stabilization_time, dt,
)
from brain import Brain
from l3_concepts import ConceptLayer, N_ZONES, ZONE_UPDATE_INTERVAL, ZONE_UPDATE_WARMUP


# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════

FREQS = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
FREQ_LABEL = {f: l for f, l in zip(FREQS, LABELS)}

# Transition grammar — (from_idx, to_idx, probability)
# Rows sum to 1.0
GRAMMAR = {
    0: [(1, 0.7), (2, 0.3)],           # A → B(0.7) C(0.3)
    1: [(2, 0.6), (3, 0.4)],           # B → C(0.6) D(0.4)
    2: [(0, 0.5), (3, 0.5)],           # C → A(0.5) D(0.5)
    3: [(4, 0.8), (1, 0.2)],           # D → E(0.8) B(0.2)
    4: [(5, 0.6), (6, 0.4)],           # E → F(0.6) G(0.4)
    5: [(4, 0.5), (7, 0.4), (6, 0.1)], # F → E(0.5) H(0.4) G(0.1)
    6: [(7, 0.7), (4, 0.3)],           # G → H(0.7) E(0.3)
    7: [(0, 0.6), (6, 0.4)],           # H → A(0.6) G(0.4)
}

TOTAL_STEPS    = 50_000
BLOCK_DUR_S    = 30.0       # seconds per tone block (M50 timescale)
SNAPSHOT_EVERY = 5_000      # probe and print every N brain steps
CAL_BLOCK_DUR  = 40.0       # longer blocks for calibration accuracy

# ── SOM warmup curriculum ──────────────────────────────────────────────────
# Root cause fix: the grammar-structured stream is biased toward A,B,C,D early.
# G (1.7Hz) can't appear until the chain traverses D→E→G or H→G, taking
# 720–5280 steps depending on rng. L3's first clustering fires at
# CLUSTER_WARMUP=2000 steps — if G hasn't appeared yet it gets mis-clustered,
# producing an unstable BMU that corrupts zone assignments for the entire run.
#
# Fix: prepend WARMUP_STEPS round-robin steps (all 8 freqs equally) BEFORE
# the grammar walk. Every freq appears ~1000 times before L3 first clusters.
# Grammar phase still runs for (TOTAL_STEPS - WARMUP_STEPS) = 42,000 steps.
WARMUP_STEPS   = 0       # Fix 4: removed — round-robin contaminates L2 with fake sequential grammar


# ═══════════════════════════════════════════════════════════════
# STEP 1: CALIBRATE M50
# ═══════════════════════════════════════════════════════════════

def calibrate():
    print("=" * 64)
    print("  CALIBRATING M50 EAR")
    print("=" * 64)
    print(f"\n  {len(FREQS)} frequencies: {FREQS}")
    print(f"  Block duration: {CAL_BLOCK_DUR}s per frequency")

    sig, _ = make_blocks(FREQS, block_dur=CAL_BLOCK_DUR)
    total  = stabilization_time + 2 * len(FREQS) * CAL_BLOCK_DUR + 10.0
    print(f"  Sim time: {total:.0f}s  ({int(total/dt):,} oscillator steps)")
    print(f"  Running...", end="", flush=True)

    t0 = time.time()
    np.random.seed(1)
    data = run_sim(sig, total_time=total,
                   sweep_mode=False, dynamic_settle=True,
                   verbose=False, collect_calib=True)
    print(f" done ({time.time()-t0:.1f}s)")

    rx_slow, ry_slow = build_reverse_lookup(
        sorted(data['calib_plv_slow'].keys()),
        data['calib_plv_slow'], data['calib_energy_slow'])
    rx_fast, ry_fast = build_reverse_lookup(
        sorted(data['calib_plv_fast'].keys()),
        data['calib_plv_fast'], data['calib_energy_fast'])

    print(f"  Slow lookup: {len(rx_slow)} pts")
    print(f"  Fast lookup: {len(rx_fast)} pts")
    return rx_slow, ry_slow, rx_fast, ry_fast


# ═══════════════════════════════════════════════════════════════
# STEP 2: BUILD M50 SIGNAL LIBRARY
# ═══════════════════════════════════════════════════════════════
# Pre-simulate each frequency as a stable block.
# Then during the long run we just replay these pre-computed
# signals in grammar order — fast, no re-simulating each block.

N_RUNS_PER_FREQ = 2   # independent M50 seeds per frequency

def build_signal_library(rx_slow, ry_slow, rx_fast, ry_fast):
    """
    For each frequency, simulate N_RUNS_PER_FREQ independent runs with
    different RNG seeds. Each visit to a frequency in the audio stream
    cycles through these runs round-robin, so L3 sees genuine variation
    across visits rather than the identical BMU sequence every time.

    Returns dict: freq → list-of-runs, where each run is a list of
    (decoded_freq, stability_w, novelty_flag, plv_vector) tuples.
    """
    print("\n" + "=" * 64)
    print(f"  BUILDING SIGNAL LIBRARY ({N_RUNS_PER_FREQ} runs per frequency)")
    print("=" * 64)

    library = {}

    for fi, freq in enumerate(FREQS):
        label = LABELS[fi]
        runs  = []
        for run in range(N_RUNS_PER_FREQ):
            seed = 100 + fi * 20 + run
            print(f"  [{label}] {freq:.1f} Hz  seed={seed}...", end="", flush=True)

            sig, _ = make_blocks([freq], block_dur=BLOCK_DUR_S)
            total  = stabilization_time + 3 * BLOCK_DUR_S + 5.0
            np.random.seed(seed)
            data   = run_sim(sig, total_time=total,
                             sweep_mode=False, dynamic_settle=False,
                             verbose=False)

            plv_hist = deque(maxlen=PLV_STAB_WINDOW)
            cusum    = DivergenceCUSUM()
            steps    = []

            for i in range(len(data['Y'])):
                max_plv = float(np.max(data['plv_slow'][i]))
                plv_hist.append(max_plv)
                w = compute_stability_plv(plv_hist)

                df = decode_resonance(data['plv_fast'][i], data['energy_fast'][i],
                                      rx_fast, ry_fast)
                ds = decode_resonance(data['plv_slow'][i], data['energy_slow'][i],
                                      rx_slow, ry_slow)
                _, is_novel = cusum.update(df, ds, data['T'][i], w=w)
                if is_novel:
                    w = 0.0
                decoded = float(w * ds + (1.0 - w) * df)

                if w > 0.5:
                    steps.append((
                        decoded,
                        float(w),
                        float(is_novel),
                        data['plv_slow'][i].copy(),
                    ))

            runs.append(steps)
            print(f" {len(steps)} stable steps")

        library[freq] = runs

    return library


# ═══════════════════════════════════════════════════════════════
# STEP 3: BUILD THE LONG AUDIO STREAM
# ═══════════════════════════════════════════════════════════════

def build_audio_stream(library, total_steps, rng):
    """
    Build a sequence of (freq_idx, step_data) pairs totalling total_steps.

    Phase 1 — WARMUP (WARMUP_STEPS steps, round-robin):
      All 8 frequencies presented in round-robin order before the grammar starts.
      Guarantees every frequency appears ~WARMUP_STEPS/8 times before L3's
      first clustering (CLUSTER_WARMUP=2000 steps). Without this, grammar-late
      frequencies (G=1.7Hz can't appear until step 720–5280 depending on seed)
      miss the SOM organisation window, producing unstable BMUs that corrupt
      L3 zone assignments for the entire run.

    Phase 2 — GRAMMAR (remaining steps, Markov walk):
      Walk the transition grammar as before. Grammar structure is fully
      preserved over (TOTAL_STEPS - WARMUP_STEPS) = 42,000 steps.

    Each visit cycles through pre-simulated runs round-robin.
    """
    from collections import defaultdict as _dd
    stream      = []
    transitions = []
    run_counter = _dd(int)   # visit count per freq → selects run index

    # ── Phase 1: Uniform warmup ────────────────────────────────────────────
    warmup_cycle = 0
    while len(stream) < WARMUP_STEPS:
        freq_idx = warmup_cycle % len(FREQS)
        warmup_cycle += 1

        freq = FREQS[freq_idx]
        runs = library[freq]
        if not runs or not runs[0]:
            continue

        run_idx = run_counter[freq_idx] % len(runs)
        run_counter[freq_idx] += 1
        steps = runs[run_idx]

        n_use = min(len(steps), int(BLOCK_DUR_S / dt * 0.4))
        start = rng.randint(0, max(1, len(steps) - n_use))
        block = steps[start: start + n_use]

        for s in block:
            if len(stream) >= WARMUP_STEPS:
                break
            stream.append((freq_idx,) + s)

    # ── Phase 2: Grammar walk ─────────────────────────────────────────────
    freq_idx = 0   # start grammar from freq A

    while len(stream) < total_steps:
        freq  = FREQS[freq_idx]
        runs  = library[freq]
        if not runs or not runs[0]:
            freq_idx = _next_freq(freq_idx, rng)
            continue

        # Cycle through runs round-robin
        run_idx = run_counter[freq_idx] % len(runs)
        run_counter[freq_idx] += 1
        steps = runs[run_idx]

        n_use = min(len(steps), int(BLOCK_DUR_S / dt * 0.4))
        start = rng.randint(0, max(1, len(steps) - n_use))
        block = steps[start: start + n_use]

        for s in block:
            if len(stream) >= total_steps:
                break
            stream.append((freq_idx,) + s)

        prev_idx = freq_idx
        freq_idx = _next_freq(freq_idx, rng)
        transitions.append((len(stream), prev_idx, freq_idx))

    return stream, transitions


def _next_freq(current_idx, rng):
    choices = GRAMMAR[current_idx]
    probs   = [p for _, p in choices]
    r       = rng.random()
    cumsum  = 0.0
    for idx, p in choices:
        cumsum += p
        if r < cumsum:
            return idx
    return choices[-1][0]


# ═══════════════════════════════════════════════════════════════
# STEP 4: PROBE BRAIN STATE (snapshot)
# ═══════════════════════════════════════════════════════════════

def probe_brain(brain, l3, freq_bmu_counters, freq_pe_ema,
                freq_fam_ema, freq_sal_ema, freq_pw_ema, freq_tc_ema):
    """
    Fix 1: Non-destructive probe — reads state accumulated during training.
    NO brain.step() calls. Uses per-frequency modal BMUs tracked during
    the training loop and per-frequency running EMAs of PE/fam/sal/pw/tc.

    Returns dict of per-frequency metrics.
    """
    results = {}

    for fi, freq in enumerate(FREQS):
        label = LABELS[fi]

        # Determine modal BMU for this frequency from training observations
        counter = freq_bmu_counters[fi]
        if len(counter) == 0:
            continue
        bmu_counts = np.zeros(64, dtype=np.int32)
        for bmu, cnt in counter.items():
            if 0 <= bmu < 64:
                bmu_counts[bmu] = cnt
        bmu_mode = int(bmu_counts.argmax())

        # Fix 2: Read PE from training-stream EMA (honest signal)
        pe_vals  = list(freq_pe_ema[fi])
        fam_vals = list(freq_fam_ema[fi])
        sal_vals = list(freq_sal_ema[fi])
        pw_vals  = list(freq_pw_ema[fi])
        tc_vals  = list(freq_tc_ema[fi])

        mean_pe  = float(np.mean(pe_vals))  if pe_vals  else 0.5
        mean_fam = float(np.mean(fam_vals)) if fam_vals else 0.0
        mean_sal = float(np.mean(sal_vals)) if sal_vals else 0.5
        mean_pw  = float(np.mean(pw_vals))  if pw_vals  else 0.0
        mean_tc  = float(np.mean(tc_vals))  if tc_vals  else 0.0

        # Read familiarity read-only (no state write)
        recall = brain.memory.recall(bmu_mode)
        fam_stored = float(recall['familiarity'])

        # L2 top predictions from modal BMU — read-only
        modal_preds = brain.pred.top_predictions(bmu_mode, k=5)

        results[label] = {
            'freq':      freq,
            'fam':       mean_fam,
            'fam_stored': fam_stored,
            'sal':       mean_sal,
            'cur':       0.0,   # curiosity not tracked per-freq, kept for schema compat
            'pe':        mean_pe,
            'tc':        mean_tc,
            'pw':        mean_pw,
            'bmu_mode':  bmu_mode,
            'top_preds': modal_preds,
        }

    # freq_to_zone: zone index = freq index (by design in direct ownership L3)
    # Do NOT use modal BMU lookup — multiple freqs can share a modal BMU,
    # causing them all to map to the same zone even when they are distinct.
    freq_to_zone = {fi2: fi2 for fi2 in range(len(LABELS))}
    for label2 in results:
        results[label2]['freq_to_zone'] = freq_to_zone

    return results


def _bmu_grid_dist(a, b, grid_w=8):
    """Euclidean distance between BMU a and b on the 8×8 SOM grid."""
    ra, ca = a // grid_w, a % grid_w
    rb, cb = b // grid_w, b % grid_w
    return float(((ra - rb)**2 + (ca - cb)**2) ** 0.5)


def direct_l2_accuracy(probe_results, freq_bmu_counters,
                        proximity_thresh=2.0):
    """
    Direct L2 accuracy: for each frequency's modal BMU, check if ANY of L2's
    top-5 predicted BMUs lie within proximity_thresh grid cells of ANY of the
    top-3 most-visited BMUs of each grammar-valid target frequency.

    Using top-3 target BMUs (not just modal) makes the metric robust to SOM
    drift: as cortex keeps learning online, the modal BMU can shift by 1-2
    cells but the top-3 set remains stable and covers the true region.
    """
    scores = {}

    # Build fi -> top-3 BMUs from visit counters (robust to SOM drift)
    fi_to_top_bmus = {}
    for fi in range(len(LABELS)):
        counter = freq_bmu_counters[fi]
        if counter:
            sorted_bmus = sorted(counter.items(), key=lambda x: -x[1])
            fi_to_top_bmus[fi] = [b for b, _ in sorted_bmus[:3]]
        else:
            fi_to_top_bmus[fi] = []

    for label, res in probe_results.items():
        fi        = LABELS.index(label)
        src_bmu   = res['bmu_mode']
        top5      = res['top_preds']
        pred_bmus = [b for b, _ in top5]

        target_fis  = [t for t, _ in GRAMMAR[fi]]
        target_bmus = []
        for tfi in target_fis:
            target_bmus.extend(fi_to_top_bmus.get(tfi, []))

        hit = False
        best_dist = 999.0
        for pb in pred_bmus:
            for tb in target_bmus:
                d = _bmu_grid_dist(pb, tb)
                if d < best_dist:
                    best_dist = d
                if d <= proximity_thresh:
                    hit = True

        # Display: show top-1 target BMU per target freq
        modal_targets = [fi_to_top_bmus.get(tfi, [None])[0]
                         for tfi in target_fis if fi_to_top_bmus.get(tfi)]
        modal_targets = sorted([b for b in modal_targets if b is not None])

        scores[label] = {
            'correct':   int(hit),
            'rate':      float(hit),
            'src_bmu':   src_bmu,
            'pred_bmus': sorted(pred_bmus),
            'target_bmus': modal_targets,
            'targets':   [LABELS[t] for t in target_fis],
            'best_dist': round(best_dist, 2),
        }

    return scores


def score_transition_prediction(probe_results, l3):
    """
    Three-metric transition evaluation:

    1. CALIBRATION (primary) — grammar-probability-weighted mass:
         calib = sum_j( Z_norm[src,j] * grammar_prob[src,j] )
       Range 0–1. A perfectly calibrated Z scores 1.0 regardless of
       whether any grammar transition has a tie. A random Z scores
       ~(sum of grammar probs^2) ≈ 0.3–0.5 depending on n_successors.
       A 50/50 grammar zone with Z=[0.5,0.5] scores 0.5 — correct.
       This is the honest metric: it rewards putting mass where the
       grammar says mass should be.

    2. STRICT (top-1 vs grammar-top) — kept for reference.
       Biased: ties in grammar are broken by list order, so a 50/50
       zone is always "wrong" half the time regardless of Z quality.

    3. LOOSE (top-1 in any valid successor) — kept for reference.

    Primary reported metric is CALIBRATION.
    """
    scores = {}

    for label, res in probe_results.items():
        fi              = LABELS.index(label)
        grammar_targets = GRAMMAR[fi]           # [(tfi, prob), ...]
        src_zone        = fi

        if src_zone < 0:
            scores[label] = {
                'calib': 0.0, 'rate': 0.0, 'rate_loose': 0.0,
                'targets': [LABELS[t] for t, _ in grammar_targets],
                'src_zone': -1, 'pred_zone': -1, 'pred_conf': 0.0,
                'valid_mass': 0.0, 'top_target': '?',
            }
            continue

        z_probs   = l3.get_zone_probs(src_zone)
        pred_zone = int(np.argmax(z_probs))
        pred_conf = float(z_probs[pred_zone])

        # ── Calibration score ─────────────────────────────────────────
        # dot product of Z distribution with grammar distribution
        # perfect Z → score = sum(grammar_prob^2)  (upper bound given grammar)
        # but we normalise so perfect = 1.0:
        #   calib = dot(z, g) / sum(g^2)   where g is the grammar prob vector
        # This way a zone that perfectly mirrors the grammar always scores 1.0
        g_vec = np.zeros(len(z_probs))
        for tfi, p in grammar_targets:
            g_vec[tfi] = p
        g_sum_sq = float(np.dot(g_vec, g_vec))   # sum of squares of grammar probs
        raw_dot  = float(np.dot(z_probs, g_vec))
        calib    = raw_dot / g_sum_sq if g_sum_sq > 1e-9 else 0.0

        # ── Strict / loose (kept for reference) ───────────────────────
        top_target_fi  = max(grammar_targets, key=lambda x: x[1])[0]
        all_target_fis = [t for t, _ in grammar_targets]
        correct_strict = int(pred_zone == top_target_fi)
        correct_loose  = int(pred_zone in all_target_fis)
        valid_mass     = float(sum(z_probs[tfi] for tfi in all_target_fis))

        # Tie flag: grammar has multiple successors sharing the max prob
        max_prob  = max(p for _, p in grammar_targets)
        n_tied    = sum(1 for _, p in grammar_targets if p == max_prob)
        is_tied   = n_tied > 1

        scores[label] = {
            'calib':          round(calib, 3),
            'rate':           round(calib, 3),      # primary = calibration
            'correct_strict': correct_strict,
            'rate_strict':    float(correct_strict),
            'correct_loose':  correct_loose,
            'rate_loose':     float(correct_loose),
            'targets':        [LABELS[t] for t in all_target_fis],
            'top_target':     LABELS[top_target_fi],
            'src_zone':       src_zone,
            'pred_zone':      pred_zone,
            'pred_conf':      round(pred_conf, 3),
            'valid_mass':     round(valid_mass, 3),
            'is_tied':        is_tied,
        }

    return scores


# ═══════════════════════════════════════════════════════════════
# STEP 5: PRINT SNAPSHOT
# ═══════════════════════════════════════════════════════════════

def print_snapshot(step, probe_results, transition_scores, l2_scores,
                   freq_visit_counts, transition_counts):
    print(f"\n{'─'*64}")
    print(f"  SNAPSHOT @ step {step:,}")
    print(f"{'─'*64}")

    # Visit counts
    total_visits = sum(freq_visit_counts.values())
    print(f"\n  VISIT DISTRIBUTION ({total_visits:,} total steps):")
    for fi, freq in enumerate(FREQS):
        label = LABELS[fi]
        count = freq_visit_counts.get(fi, 0)
        pct   = 100 * count / max(1, total_visits)
        bar   = '█' * int(pct / 3)
        print(f"    {label} ({freq:.1f}Hz)  {bar:<20}  {count:5d}  ({pct:4.1f}%)")

    # Per-frequency brain state
    print(f"\n  BRAIN STATE PER FREQUENCY:")
    print(f"  {'Freq':>4}  {'FAM':>6}  {'SAL':>6}  {'PE':>6}  {'TC':>6}  {'PW':>8}  {'ZONE':>9}  PREDICTS→")
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*20}")

    for label in LABELS:
        if label not in probe_results:
            continue
        r  = probe_results[label]
        ts = transition_scores.get(label, {})
        calib       = ts.get('calib', 0.0)
        rate_strict = ts.get('rate_strict', 0.0)
        rate_loose  = ts.get('rate_loose', 0.0)
        valid_mass  = ts.get('valid_mass', 0.0)
        top_target  = ts.get('top_target', '?')
        is_tied     = ts.get('is_tied', False)
        src_zone    = ts.get('src_zone', -1)
        pred_zone   = ts.get('pred_zone', -1)
        pred_conf   = ts.get('pred_conf', 0.0)
        zone_str    = f"Z{src_zone}→Z{pred_zone}({pred_conf:.2f})" if src_zone >= 0 else "unstable"
        # Calibration bar (0-1 shown as 10-char block)
        calib_bar   = '█' * int(calib * 10)
        strict_sym  = "✓" if rate_strict else ("~" if rate_loose else "✗")
        tie_note    = "[tie]" if is_tied else ""
        rate_str    = f"calib={calib:.2f} {calib_bar:<10} {strict_sym}strict  {tie_note}"
        print(f"  {label:>4}  {r['fam']:6.3f}  {r['sal']:6.3f}  "
              f"{r['pe']:6.3f}  {r['tc']:6.3f}  {r['pw']:8.5f}  "
              f"{zone_str:>9}  {rate_str}")

    # Familiarity ranking
    sorted_by_fam = sorted(probe_results.items(),
                           key=lambda x: x[1]['fam'], reverse=True)
    most_familiar  = sorted_by_fam[0][0]
    least_familiar = sorted_by_fam[-1][0]
    print(f"\n  Most familiar:  {most_familiar} ({sorted_by_fam[0][1]['fam']:.3f})")
    print(f"  Least familiar: {least_familiar} ({sorted_by_fam[-1][1]['fam']:.3f})")

    # Overall prediction accuracy — zone-based (L3)
    all_calib  = [ts.get('calib', 0) for ts in transition_scores.values()]
    all_strict = [ts.get('rate_strict', 0) for ts in transition_scores.values()]
    all_loose  = [ts.get('rate_loose', 0) for ts in transition_scores.values()]
    mean_calib  = np.mean(all_calib)  if all_calib  else 0.0
    mean_strict = np.mean(all_strict) if all_strict else 0.0
    mean_loose  = np.mean(all_loose)  if all_loose  else 0.0
    print(f"\n  L3 calibration score:  {mean_calib*100:.1f}%  "
          f"(strict={mean_strict*100:.1f}%  loose={mean_loose*100:.1f}%)")
    print(f"  calibration = how well Z probabilities match grammar probabilities")

    # Fix 3: Direct L2 accuracy — bypasses L3 zones entirely
    l2_rates = [s.get('rate', 0) for s in l2_scores.values()]
    mean_l2  = np.mean(l2_rates) if l2_rates else 0.0
    print(f"\n  Direct L2 accuracy (P-matrix, spatial d≤2): {mean_l2*100:.1f}%")
    print(f"  {'Freq':>4}  {'Hit':>3}  {'BestDist':>8}  PredBMUs → TargetBMUs")
    for lbl in LABELS:
        if lbl not in l2_scores:
            continue
        s = l2_scores[lbl]
        hit_str  = "✓" if s['correct'] else "✗"
        pb_str   = str(s['pred_bmus'][:3])[1:-1]
        tb_str   = str(s['target_bmus'])[1:-1]
        print(f"  {lbl:>4}   {hit_str}    {s['best_dist']:>6.1f}   [{pb_str}] → [{tb_str}]")

    # Transition counts
    print(f"\n  TRANSITION COUNTS (grammar learning evidence):")
    for (fi, ti), count in sorted(transition_counts.items(),
                                   key=lambda x: -x[1])[:8]:
        fl, tl = LABELS[fi], LABELS[ti]
        prob   = dict(GRAMMAR[fi]).get(ti, 0)
        expected_pct = prob * 100
        actual_pct   = 100 * count / max(1, sum(
            v for (f2, _), v in transition_counts.items() if f2 == fi))
        print(f"    {fl}→{tl}  expected={expected_pct:.0f}%  "
              f"actual={actual_pct:.0f}%  count={count}")


# ═══════════════════════════════════════════════════════════════
# STEP 6: THE LONG RUN
# ═══════════════════════════════════════════════════════════════

def long_run(library, rx_slow, ry_slow, rx_fast, ry_fast):
    global _brain_ref

    print("\n" + "=" * 64)
    print("  LONG RUN — 50,000 CONTINUOUS STEPS")
    print("=" * 64)
    print(f"\n  One Brain. No resets. Grammar-structured audio.")
    print(f"  Snapshot every {SNAPSHOT_EVERY:,} steps.\n")

    brain      = Brain(seed=42)
    l3         = ConceptLayer()

    rng = np.random.RandomState(99)

    # Build the full audio stream upfront
    print(f"  Building audio stream ({WARMUP_STEPS:,} warmup + {TOTAL_STEPS-WARMUP_STEPS:,} grammar)...",
          end="", flush=True)
    stream, transitions = build_audio_stream(library, TOTAL_STEPS, rng)
    print(f" {len(stream):,} steps, {len(transitions)} grammar transitions")

    # Tracking
    freq_visit_counts  = defaultdict(int)
    transition_counts  = defaultdict(int)
    snapshots          = []

    # Running stats
    step_fam  = deque(maxlen=500)
    step_sal  = deque(maxlen=500)
    step_pe   = deque(maxlen=500)
    step_pw   = deque(maxlen=500)

    # Fix 2: Per-frequency training-stream EMAs (honest PE signal)
    from collections import Counter as _Counter
    freq_bmu_counters = [_Counter() for _ in range(len(FREQS))]
    freq_pe_ema  = {fi: deque(maxlen=500) for fi in range(len(FREQS))}
    freq_fam_ema = {fi: deque(maxlen=500) for fi in range(len(FREQS))}
    freq_sal_ema = {fi: deque(maxlen=500) for fi in range(len(FREQS))}
    freq_pw_ema  = {fi: deque(maxlen=500) for fi in range(len(FREQS))}
    freq_tc_ema  = {fi: deque(maxlen=500) for fi in range(len(FREQS))}

    t_start = time.time()
    last_freq_idx = None

    print(f"\n  Running...\n")

    for step, entry in enumerate(stream):
        freq_idx, decoded, stab, nov, plv = entry

        # Track visits and transitions
        freq_visit_counts[freq_idx] += 1
        if last_freq_idx is not None and last_freq_idx != freq_idx:
            transition_counts[(last_freq_idx, freq_idx)] += 1
        last_freq_idx = freq_idx

        # Brain step
        out = brain.step(
            decoded_freq = decoded,
            stability_w  = stab,
            novelty_flag = nov,
            plv_vector   = plv,
        )

        # L3 step — direct ownership zones + inter-freq transition learning
        # Pass freq_idx directly so Z-matrix uses ground-truth zone labels,
        # not bmu_to_zone lookups which can drift during reassignment.
        l3.step(
            bmu_idx     = out['bmu_idx'],
            l2_scores   = brain.pred._last_scores,
            familiarity = out['familiarity'],
            freq_idx    = freq_idx,
        )

        step_fam.append(out['familiarity'])
        step_sal.append(out['salience'])
        step_pe.append(out['prediction_error'])
        step_pw.append(out['planning_weight'])

        # Track per-frequency BMU visits (used for zone assignment)
        freq_bmu_counters[freq_idx][out['bmu_idx']] += 1

        # Periodic zone assignment: zone[bmu] = argmax_freq(visit_count)
        # Wait for warmup so all 8 frequencies have accumulated data
        if (step + 1) >= ZONE_UPDATE_WARMUP and (step + 1) % ZONE_UPDATE_INTERVAL == 0:
            l3.assign_zones_from_counters(freq_bmu_counters)
        freq_pe_ema[freq_idx].append(out['prediction_error'])
        freq_fam_ema[freq_idx].append(out['familiarity'])
        freq_sal_ema[freq_idx].append(out['salience'])
        freq_pw_ema[freq_idx].append(out['planning_weight'])
        freq_tc_ema[freq_idx].append(out['thought_confidence'])

        # Snapshot
        if (step + 1) % SNAPSHOT_EVERY == 0 or step == len(stream) - 1:
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / elapsed
            eta = (len(stream) - step - 1) / steps_per_sec

            print(f"  Step {step+1:>6,} / {len(stream):,}  "
                  f"({100*(step+1)/len(stream):.0f}%)  "
                  f"{steps_per_sec:.0f} steps/s  "
                  f"ETA {eta:.0f}s  "
                  f"L3={'stable' if l3._zones_stable else 'learning'}({l3._n_assignments})")
            print(f"           fam={np.mean(step_fam):.3f}  "
                  f"sal={np.mean(step_sal):.3f}  "
                  f"pe={np.mean(step_pe):.3f}  "
                  f"pw={np.mean(step_pw):.5f}")

            # Fix 1: Non-destructive probe — no brain.step() calls
            probe    = probe_brain(brain, l3,
                                   freq_bmu_counters,
                                   freq_pe_ema, freq_fam_ema,
                                   freq_sal_ema, freq_pw_ema, freq_tc_ema)
            t_scores  = score_transition_prediction(probe, l3)
            l2_scores = direct_l2_accuracy(probe, freq_bmu_counters)
            print_snapshot(step + 1, probe, t_scores, l2_scores,
                           freq_visit_counts, transition_counts)

            snapshots.append({
                'step':         step + 1,
                'probe':        {k: {kk: float(vv) if isinstance(vv, (float, np.floating))
                                     else vv
                                     for kk, vv in v.items()
                                     if kk != 'top_preds'}
                                 for k, v in probe.items()},
                'transition_accuracy': {k: float(v.get('calib', 0))
                                        for k, v in t_scores.items()},
                'strict_accuracy':     {k: float(v.get('rate_strict', 0))
                                        for k, v in t_scores.items()},
                'l2_accuracy':  {k: float(v.get('rate', 0))
                                  for k, v in l2_scores.items()},
                'mean_fam':     float(np.mean(step_fam)),
                'mean_sal':     float(np.mean(step_sal)),
                'mean_pe':      float(np.mean(step_pe)),
                'mean_pw':      float(np.mean(step_pw)),
            })

    total_time = time.time() - t_start
    print(f"\n{'='*64}")
    print(f"  LONG RUN COMPLETE")
    print(f"  Total time: {total_time:.1f}s  "
          f"({TOTAL_STEPS/total_time:.0f} steps/s)")
    print(f"{'='*64}")

    return brain, l3, snapshots, freq_visit_counts, transition_counts


# ═══════════════════════════════════════════════════════════════
# STEP 7: FINAL ANALYSIS
# ═══════════════════════════════════════════════════════════════

def final_analysis(snapshots):
    print(f"\n{'='*64}")
    print(f"  LEARNING TRAJECTORY ACROSS 50,000 STEPS")
    print(f"{'='*64}")

    print(f"\n  {'Step':>8}  {'Fam':>6}  {'Sal':>6}  {'PE':>6}  "
          f"{'PW':>8}  {'Calib%':>7}  {'Strict%':>8}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*8}")

    for s in snapshots:
        mean_calib  = np.mean(list(s['transition_accuracy'].values()))
        mean_strict = np.mean(list(s.get('strict_accuracy', s['transition_accuracy']).values()))
        print(f"  {s['step']:>8,}  "
              f"{s['mean_fam']:>6.3f}  "
              f"{s['mean_sal']:>6.3f}  "
              f"{s['mean_pe']:>6.3f}  "
              f"{s['mean_pw']:>8.5f}  "
              f"{mean_calib*100:>6.1f}%  "
              f"{mean_strict*100:>7.1f}%")

    # Show growth
    if len(snapshots) >= 2:
        first = snapshots[0]
        last  = snapshots[-1]
        print(f"\n  GROWTH (first snapshot → last snapshot):")
        print(f"    Familiarity:     {first['mean_fam']:.3f} → {last['mean_fam']:.3f}"
              f"  ({last['mean_fam']-first['mean_fam']:+.3f})")
        print(f"    Salience:        {first['mean_sal']:.3f} → {last['mean_sal']:.3f}"
              f"  ({last['mean_sal']-first['mean_sal']:+.3f})")
        print(f"    Pred error:      {first['mean_pe']:.3f} → {last['mean_pe']:.3f}"
              f"  ({last['mean_pe']-first['mean_pe']:+.3f})")
        print(f"    Plan weight:     {first['mean_pw']:.5f} → {last['mean_pw']:.5f}"
              f"  ({last['mean_pw']-first['mean_pw']:+.5f})")

        first_calib  = np.mean(list(first['transition_accuracy'].values()))
        last_calib   = np.mean(list(last['transition_accuracy'].values()))
        first_strict = np.mean(list(first.get('strict_accuracy', first['transition_accuracy']).values()))
        last_strict  = np.mean(list(last.get('strict_accuracy', last['transition_accuracy']).values()))
        print(f"    Calibration score:   {first_calib*100:.1f}% → {last_calib*100:.1f}%"
              f"  ({last_calib-first_calib:+.1%})")
        print(f"    Strict top-1:        {first_strict*100:.1f}% → {last_strict*100:.1f}%"
              f"  ({last_strict-first_strict:+.1%})")

        if 'l2_accuracy' in first and 'l2_accuracy' in last:
            first_l2 = np.mean(list(first['l2_accuracy'].values()))
            last_l2  = np.mean(list(last['l2_accuracy'].values()))
            print(f"    Direct L2 accuracy:  {first_l2*100:.1f}% → {last_l2*100:.1f}%"
                  f"  ({last_l2-first_l2:+.1%})")

    # Per-frequency familiarity at end
    if snapshots:
        last_probe = snapshots[-1]['probe']
        print(f"\n  FINAL FAMILIARITY LANDSCAPE:")
        sorted_fam = sorted(last_probe.items(),
                            key=lambda x: x[1].get('fam', 0), reverse=True)
        for label, vals in sorted_fam:
            fam  = vals.get('fam', 0)
            freq = vals.get('freq', 0)
            bar  = '█' * int(fam * 20)
            print(f"    {label} ({freq:.1f}Hz)  {bar:<20}  {fam:.3f}")

    # Per-frequency prediction accuracy at end
    if snapshots:
        last_acc = snapshots[-1]['transition_accuracy']
        print(f"\n  FINAL TRANSITION PREDICTION ACCURACY:")
        for label in LABELS:
            if label not in last_acc:
                continue
            rate = last_acc[label]
            fi   = LABELS.index(label)
            tgts = [LABELS[t] for t, _ in GRAMMAR[fi]]
            bar  = '█' * int(rate * 20)
            print(f"    {label}→{','.join(tgts):8s}  {bar:<20}  {rate*100:.0f}%")


# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════

def save_results(snapshots, freq_visit_counts, transition_counts):
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):   return int(obj)
        if isinstance(obj, np.integer):          return int(obj)
        if isinstance(obj, np.floating):         return float(obj)
        if isinstance(obj, np.ndarray):          return obj.tolist()
        if isinstance(obj, defaultdict):         return dict(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    out = {
        'snapshots':         snapshots,
        'freq_visit_counts': {int(k): int(v)
                              for k, v in freq_visit_counts.items()},
        'transition_counts': {f"{LABELS[k[0]]}->{LABELS[k[1]]}": int(v)
                              for k, v in transition_counts.items()},
        'grammar':           {LABELS[k]: [(LABELS[t], p) for t, p in v]
                              for k, v in GRAMMAR.items()},
        'freqs':             FREQS,
        'labels':            LABELS,
        'total_steps':       TOTAL_STEPS,
    }

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'brain_longrun_results.json'
    )
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=convert)

    print(f"\n  Results saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         BRAIN LONG RUN — 50,000 Continuous Steps            ║")
    print("║         One brain. Real audio. Grammar-structured.          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    t_total = time.time()

    # 1. Calibrate M50
    rx_slow, ry_slow, rx_fast, ry_fast = calibrate()

    # 2. Build signal library (one stable block per frequency)
    library = build_signal_library(rx_slow, ry_slow, rx_fast, ry_fast)

    # 3. Run
    brain, l3, snapshots, visit_counts, trans_counts = long_run(
        library, rx_slow, ry_slow, rx_fast, ry_fast
    )

    # 4. Final analysis
    final_analysis(snapshots)

    # 5. Save
    save_results(snapshots, visit_counts, trans_counts)

    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")
    print()