"""
BRAIN CONFIRMATION TEST — A–H Training + I–P Generalisation
=============================================================

Two-phase experiment to test whether L3 genuinely learns grammar
structure, or just memorises 8 specific zone transitions.

PHASE 1 — TRAIN (50,000 steps, A–H, chain grammar)
  Same as brain_longrun.py. After this, L3 knows the A–H grammar.
  Z matrix is an 8×8 counter. Brain is fully trained.

PHASE 2 — CONFIRMATION (30,000 steps, I–P, hub-and-spoke grammar)
  NEW frequencies (2.3–3.7 Hz), DIFFERENT grammar topology.
  L3 is expanded to 16 zones. Z grows to 16×16.
  The A–H quadrant of Z is frozen (no new transitions written there).
  Only zones 8–15 accumulate new transitions.
  We watch whether L3 learns the I–P grammar as fast/accurately as A–H.

FREEZE-PROBE
  After Phase 2, Z is frozen. We probe:
    - Can the brain still predict A–H transitions? (retention)
    - Can it predict I–P transitions? (new learning)
    - Does it correctly NOT predict cross-zone transitions
      (A→I, I→A, etc. that never appeared)?

GRAMMAR COMPARISON
  A–H: Linear chain with branches
    A→B(0.7)C(0.3), B→C(0.6)D(0.4), C→A(0.5)D(0.5),
    D→E(0.8)B(0.2), E→F(0.6)G(0.4), F→E(0.5)H(0.4)G(0.1),
    G→H(0.7)E(0.3), H→A(0.6)G(0.4)

  I–P: Hub-and-spoke with two clusters + rare bridges
    Cluster 1: I,J,K,L   hub=I
    Cluster 2: M,N,O,P   hub=M
    I→J(0.5)K(0.3)M(0.2)  ← hub: bridges to cluster 2
    J→I(0.6)L(0.4)
    K→J(0.7)L(0.3)
    L→I(0.8)K(0.2)
    M→N(0.5)O(0.5)         ← hub: no outgoing bridge
    N→M(0.4)O(0.4)P(0.2)
    O→P(0.6)M(0.4)
    P→M(0.5)I(0.5)         ← bridges back to cluster 1

KEY QUESTIONS
  Q1. Does L3 learn I–P grammar at all? (basic capacity)
  Q2. Does it learn faster than A–H did? (warm-start benefit)
  Q3. Does freeze-probe show correct A–H AND I–P predictions? (no catastrophic forgetting)
  Q4. Does it correctly assign near-zero probability to cross-zone transitions?
"""

import numpy as np
import json
import os
import time
from collections import deque, defaultdict, Counter

from m50_neuron import (
    run_sim, make_blocks,
    decode_resonance, build_reverse_lookup,
    DivergenceCUSUM, compute_stability_plv,
    PLV_STAB_WINDOW, stabilization_time, dt,
)
from brain import Brain
from l3_concepts import ConceptLayer, ZONE_UPDATE_INTERVAL, ZONE_UPDATE_WARMUP


# ═══════════════════════════════════════════════════════════════
# PHASE 1 — A–H environment (same as brain_longrun.py)
# ═══════════════════════════════════════════════════════════════

FREQS_AH   = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]
LABELS_AH  = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

GRAMMAR_AH = {
    0: [(1, 0.7), (2, 0.3)],            # A → B(0.7) C(0.3)
    1: [(2, 0.6), (3, 0.4)],            # B → C(0.6) D(0.4)
    2: [(0, 0.5), (3, 0.5)],            # C → A(0.5) D(0.5)
    3: [(4, 0.8), (1, 0.2)],            # D → E(0.8) B(0.2)
    4: [(5, 0.6), (6, 0.4)],            # E → F(0.6) G(0.4)
    5: [(4, 0.5), (7, 0.4), (6, 0.1)],  # F → E(0.5) H(0.4) G(0.1)
    6: [(7, 0.7), (4, 0.3)],            # G → H(0.7) E(0.3)
    7: [(0, 0.6), (6, 0.4)],            # H → A(0.6) G(0.4)
}

# ═══════════════════════════════════════════════════════════════
# PHASE 2 — I–P environment (new frequencies, hub-and-spoke)
# ═══════════════════════════════════════════════════════════════

FREQS_IP   = [0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.10]
LABELS_IP  = ['I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

# Zone indices for I–P are 8–15 (offset from A–H's 0–7)
IP_OFFSET  = 8

GRAMMAR_IP = {
    # local indices 0–7, representing I–P
    0: [(1, 0.5), (2, 0.3), (4, 0.2)],  # I → J(0.5) K(0.3) M(0.2)  [hub+bridge]
    1: [(0, 0.6), (3, 0.4)],             # J → I(0.6) L(0.4)
    2: [(1, 0.7), (3, 0.3)],             # K → J(0.7) L(0.3)
    3: [(0, 0.8), (2, 0.2)],             # L → I(0.8) K(0.2)
    4: [(5, 0.5), (6, 0.5)],             # M → N(0.5) O(0.5)          [hub]
    5: [(4, 0.4), (6, 0.4), (7, 0.2)],  # N → M(0.4) O(0.4) P(0.2)
    6: [(7, 0.6), (4, 0.4)],             # O → P(0.6) M(0.4)
    7: [(4, 0.5), (0, 0.5)],             # P → M(0.5) I(0.5)          [bridge back]
}

# Combined for 16-zone analysis
FREQS_ALL  = FREQS_AH  + FREQS_IP
LABELS_ALL = LABELS_AH + LABELS_IP

# ── Run parameters ───────────────────────────────────────────────────────────
STEPS_PHASE1   = 50_000
STEPS_PHASE2   = 30_000
SNAPSHOT_EVERY =  5_000
CAL_BLOCK_DUR  =   40.0
BLOCK_DUR_S    =   30.0
N_RUNS_PER_FREQ = 2


# ═══════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════

def calibrate(freqs, label):
    print("=" * 64)
    print(f"  CALIBRATING M50 EAR  [{label}]")
    print("=" * 64)
    print(f"\n  {len(freqs)} frequencies: {freqs}")

    sig, _ = make_blocks(freqs, block_dur=CAL_BLOCK_DUR)
    total  = stabilization_time + 2 * len(freqs) * CAL_BLOCK_DUR + 10.0
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
# SIGNAL LIBRARY
# ═══════════════════════════════════════════════════════════════

def build_signal_library(freqs, labels, rx_slow, ry_slow, rx_fast, ry_fast,
                          freq_offset=0):
    print("\n" + "=" * 64)
    print(f"  BUILDING SIGNAL LIBRARY  [{labels[0]}–{labels[-1]}]  "
          f"({N_RUNS_PER_FREQ} runs per freq)")
    print("=" * 64)

    library = {}
    for fi, freq in enumerate(freqs):
        label = labels[fi]
        runs  = []
        for run in range(N_RUNS_PER_FREQ):
            seed = 100 + (fi + freq_offset) * 20 + run
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
# AUDIO STREAM BUILDER
# ═══════════════════════════════════════════════════════════════

def build_stream(library, freqs, grammar, total_steps, rng):
    """Build a grammar-walk stream for any set of frequencies."""
    stream      = []
    transitions = []
    run_counter = defaultdict(int)
    freq_idx    = 0

    while len(stream) < total_steps:
        freq = freqs[freq_idx]
        runs = library[freq]
        if not runs or not runs[0]:
            freq_idx = _next(freq_idx, grammar, rng)
            continue

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

        prev = freq_idx
        freq_idx = _next(freq_idx, grammar, rng)
        transitions.append((len(stream), prev, freq_idx))

    return stream, transitions


def _next(current, grammar, rng):
    choices = grammar[current]
    r       = rng.random()
    cumsum  = 0.0
    for idx, p in choices:
        cumsum += p
        if r < cumsum:
            return idx
    return choices[-1][0]


# ═══════════════════════════════════════════════════════════════
# SNAPSHOT HELPERS
# ═══════════════════════════════════════════════════════════════

def score_l3(l3, labels, grammar, zone_offset=0):
    """
    Three-metric evaluation per zone.

    PRIMARY — calibration score:
        calib = dot(z_probs, g_vec) / dot(g_vec, g_vec)
    where g_vec is the grammar probability vector.
    Normalised so a Z that perfectly mirrors the grammar = 1.0.
    A 50/50 grammar zone with Z=[0.5,0.5] scores 1.0 — correct.
    A random Z on an 8-way uniform grammar scores ~0.125/0.375 ≈ 0.33.

    ALSO reported: strict top-1 and loose (for reference).
    """
    scores = {}
    for fi, label in enumerate(labels):
        z_src           = fi + zone_offset
        grammar_targets = grammar[fi]

        z_probs = l3.get_zone_probs(z_src)
        pred_zone = int(np.argmax(z_probs))
        pred_conf = float(z_probs[pred_zone])

        # grammar probability vector (full zone space)
        g_vec = np.zeros(len(z_probs))
        for local_fi, p in grammar_targets:
            g_vec[local_fi + zone_offset] = p
        g_sum_sq = float(np.dot(g_vec, g_vec))
        raw_dot  = float(np.dot(z_probs, g_vec))
        calib    = raw_dot / g_sum_sq if g_sum_sq > 1e-9 else 0.0

        top_local_fi     = max(grammar_targets, key=lambda x: x[1])[0]
        top_zone         = top_local_fi + zone_offset
        all_local_fis    = [t for t, _ in grammar_targets]
        all_target_zones = [t + zone_offset for t in all_local_fis]
        all_target_labels = [labels[t] for t in all_local_fis]

        correct_strict = int(pred_zone == top_zone)
        correct_loose  = int(pred_zone in all_target_zones)
        valid_mass     = float(sum(z_probs[z] for z in all_target_zones))

        max_prob = max(p for _, p in grammar_targets)
        n_tied   = sum(1 for _, p in grammar_targets if p == max_prob)

        scores[label] = {
            'calib':          round(calib, 3),
            'rate':           round(calib, 3),
            'correct_strict': correct_strict,
            'rate_strict':    float(correct_strict),
            'correct_loose':  correct_loose,
            'rate_loose':     float(correct_loose),
            'top_target':     labels[top_local_fi],
            'targets':        all_target_labels,
            'src_zone':       z_src,
            'pred_zone':      pred_zone,
            'pred_conf':      round(pred_conf, 3),
            'valid_mass':     round(valid_mass, 3),
            'is_tied':        n_tied > 1,
        }
    return scores


def print_scores(title, scores, labels):
    mean_calib  = np.mean([s['calib'] for s in scores.values()])
    mean_strict = np.mean([s['rate_strict'] for s in scores.values()])
    mean_loose  = np.mean([s['rate_loose'] for s in scores.values()])
    print(f"\n  {title}")
    print(f"  calib={mean_calib*100:.1f}%  strict={mean_strict*100:.1f}%  loose={mean_loose*100:.1f}%")
    print(f"  {'Freq':>4}  {'Zone':>12}  {'Calib':>6}  {'Bar':<10}  {'S':>2}  Note")
    print(f"  {'────':>4}  {'────────────':>12}  {'─────':>6}  {'─'*10}  {'─':>2}  ────")
    for label in labels:
        if label not in scores:
            continue
        s       = scores[label]
        z_str   = f"Z{s['src_zone']}→Z{s['pred_zone']}({s['pred_conf']:.2f})"
        bar     = '█' * int(s['calib'] * 10)
        strict_sym = "✓" if s['rate_strict'] else ("~" if s['rate_loose'] else "✗")
        tie_note   = "[tie]" if s['is_tied'] else ""
        print(f"  {label:>4}  {z_str:>12}  {s['calib']:>5.2f}  {bar:<10}  {strict_sym:>2}  {tie_note}")


def print_visit_bar(visit_counts, labels, total):
    print(f"\n  VISIT DISTRIBUTION ({total:,} total steps):")
    for fi, label in enumerate(labels):
        n   = visit_counts.get(fi, 0)
        pct = n / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        freq = FREQS_AH[fi] if fi < 8 else FREQS_IP[fi - 8]
        print(f"    {label} ({freq:.1f}Hz)  {bar:<25}  {n:5d}  ({pct:4.1f}%)")


def cross_zone_probe(l3, n_ah=8, n_ip=8):
    """
    Check that Z assigns near-zero probability to cross-zone transitions.
    A–H zones 0–7 should never predict I–P zones 8–15, and vice versa.
    """
    print("\n  CROSS-ZONE LEAKAGE PROBE")
    print("  (should be ~0% — these transitions never happened)")
    print(f"  {'Zone':>6}  {'Cross-mass':>12}  {'Valid-mass':>12}")
    print(f"  {'──────':>6}  {'──────────':>12}  {'──────────':>12}")

    for z in range(n_ah + n_ip):
        z_probs = l3.get_zone_probs(z)
        if z < n_ah:
            # A–H zone: valid targets are in 0–7, cross = 8–15
            valid_mass = float(z_probs[:n_ah].sum())
            cross_mass = float(z_probs[n_ah:].sum())
            label      = LABELS_AH[z]
        else:
            # I–P zone: valid targets are in 8–15, cross = 0–7
            valid_mass = float(z_probs[n_ah:].sum())
            cross_mass = float(z_probs[:n_ah].sum())
            label      = LABELS_IP[z - n_ah]
        cross_flag = "⚠" if cross_mass > 0.05 else "✓"
        print(f"  {label}(Z{z:2d})  {cross_mass:>10.3f}  {valid_mass:>10.3f}  {cross_flag}")


# ═══════════════════════════════════════════════════════════════
# PHASE 1: TRAIN ON A–H
# ═══════════════════════════════════════════════════════════════

def phase1_train(library_ah, rx_slow, ry_slow, rx_fast, ry_fast):
    print("\n" + "═" * 64)
    print("  PHASE 1 — TRAIN ON A–H  (50,000 steps)")
    print("═" * 64)
    print("  Chain grammar. One brain. No resets.")
    print(f"  Snapshot every {SNAPSHOT_EVERY:,} steps.\n")

    brain = Brain(seed=42)
    l3    = ConceptLayer(n_zones=8)   # start with 8 zones for A–H

    rng = np.random.RandomState(99)

    print(f"  Building A–H stream ({STEPS_PHASE1:,} steps)...", end="", flush=True)
    stream, _ = build_stream(library_ah, FREQS_AH, GRAMMAR_AH, STEPS_PHASE1, rng)
    print(f" done ({len(stream):,} steps)")

    freq_visit  = defaultdict(int)
    trans_count = defaultdict(int)
    freq_bmu    = [Counter() for _ in range(8)]

    step_fam = deque(maxlen=500)
    step_sal = deque(maxlen=500)
    step_pe  = deque(maxlen=500)

    snapshots = []
    t0 = time.time()
    last_fi = None

    print("\n  Running...\n")

    for step, entry in enumerate(stream):
        fi, decoded, stab, nov, plv = entry

        freq_visit[fi] += 1
        if last_fi is not None and last_fi != fi:
            trans_count[(last_fi, fi)] += 1
        last_fi = fi

        out = brain.step(decoded_freq=decoded, stability_w=stab,
                         novelty_flag=nov, plv_vector=plv)
        l3.step(bmu_idx=out['bmu_idx'], l2_scores=brain.pred._last_scores,
                familiarity=out['familiarity'], freq_idx=fi)

        freq_bmu[fi][out['bmu_idx']] += 1
        step_fam.append(out['familiarity'])
        step_sal.append(out['salience'])
        step_pe.append(out['prediction_error'])

        if (step + 1) >= ZONE_UPDATE_WARMUP and (step + 1) % ZONE_UPDATE_INTERVAL == 0:
            l3.assign_zones_from_counters(freq_bmu)

        if (step + 1) % SNAPSHOT_EVERY == 0 or step == len(stream) - 1:
            elapsed = time.time() - t0
            sps     = (step + 1) / elapsed
            eta     = (len(stream) - step - 1) / sps

            print(f"  Step {step+1:>6,} / {len(stream):,}  "
                  f"({100*(step+1)/len(stream):.0f}%)  "
                  f"{sps:.0f} steps/s  ETA {eta:.0f}s")
            print(f"         fam={np.mean(step_fam):.3f}  "
                  f"sal={np.mean(step_sal):.3f}  "
                  f"pe={np.mean(step_pe):.3f}")

            scores = score_l3(l3, LABELS_AH, GRAMMAR_AH, zone_offset=0)
            print_scores("A–H L3 accuracy:", scores, LABELS_AH)
            print_visit_bar(freq_visit, LABELS_AH, step + 1)

            # Transition counts summary
            print(f"\n  TOP TRANSITIONS (A–H):")
            for (f1, f2), cnt in sorted(trans_count.items(), key=lambda x: -x[1])[:6]:
                expected = dict(GRAMMAR_AH[f1]).get(f2, 0)
                actual   = cnt / max(1, sum(v for (ff, _), v in trans_count.items() if ff == f1))
                print(f"    {LABELS_AH[f1]}→{LABELS_AH[f2]}  "
                      f"expected={expected*100:.0f}%  actual={actual*100:.0f}%  count={cnt}")

            strict = np.mean([s['rate'] for s in scores.values()])
            snapshots.append({'step': step + 1,
                               'strict_ah': float(strict),
                               'fam': float(np.mean(step_fam)),
                               'sal': float(np.mean(step_sal)),
                               'pe':  float(np.mean(step_pe))})

    print(f"\n  Phase 1 complete: {time.time()-t0:.1f}s")
    return brain, l3, freq_bmu, snapshots


# ═══════════════════════════════════════════════════════════════
# PHASE 2: RUN ON I–P (expand L3 to 16 zones)
# ═══════════════════════════════════════════════════════════════

def phase2_confirm(brain, l3, freq_bmu_ah, library_ip,
                   rx_slow, ry_slow, rx_fast, ry_fast):
    print("\n" + "═" * 64)
    print("  PHASE 2 — CONFIRMATION ON I–P  (30,000 steps)")
    print("═" * 64)
    print("  Hub-and-spoke grammar. SAME brain. L3 expanded to 16 zones.")
    print("  A–H quadrant of Z is FROZEN — only Z[8:,8:] accumulates.\n")

    # ── Expand L3 from 8→16 zones ──────────────────────────────────────────
    # Copy existing Z (8×8) into top-left of new Z (16×16)
    old_Z    = l3._Z.copy()              # 8×8
    old_bmu  = l3._bmu_to_zone.copy()   # 64 entries, values 0–7

    l3_new        = ConceptLayer(n_zones=16)
    l3_new._Z[:8, :8] = old_Z           # preserve A–H transition counts
    l3_new._bmu_to_zone[:] = old_bmu    # preserve cortex BMU→zone map
    l3_new._zones_stable = True          # A–H already stable
    l3_new._n_assignments = l3._n_assignments

    # Freeze flag: when writing to Z, skip if both zones < 8
    # Implemented via a subclass wrapper — simpler: patch _Z_ctx logic
    # We'll handle this in the loop by only calling l3_new.step() for I–P
    # steps and manually accumulating A–H zone context between phases
    l3_new._Z_ctx = -1                   # reset context on phase boundary

    print(f"  A–H Z matrix preserved (sum={old_Z.sum():.0f} transitions)")
    print(f"  I–P zones 8–15 start empty\n")

    rng = np.random.RandomState(77)

    print(f"  Building I–P stream ({STEPS_PHASE2:,} steps)...", end="", flush=True)
    stream, _ = build_stream(library_ip, FREQS_IP, GRAMMAR_IP, STEPS_PHASE2, rng)
    print(f" done ({len(stream):,} steps)")

    freq_visit_ip  = defaultdict(int)   # local fi 0–7 for I–P
    trans_count_ip = defaultdict(int)
    freq_bmu_ip    = [Counter() for _ in range(8)]

    step_fam = deque(maxlen=500)
    step_sal = deque(maxlen=500)
    step_pe  = deque(maxlen=500)

    snapshots = []
    t0        = time.time()
    last_fi   = None

    print("\n  Running...\n")

    for step, entry in enumerate(stream):
        local_fi, decoded, stab, nov, plv = entry
        global_fi = local_fi + IP_OFFSET   # zone 8–15 in the 16-zone Z

        freq_visit_ip[local_fi] += 1
        if last_fi is not None and last_fi != local_fi:
            trans_count_ip[(last_fi, local_fi)] += 1
        last_fi = local_fi

        out = brain.step(decoded_freq=decoded, stability_w=stab,
                         novelty_flag=nov, plv_vector=plv)

        # L3 step with global zone index (8–15)
        # This writes to Z[8:, 8:] only since freq_idx=global_fi
        l3_new.step(bmu_idx=out['bmu_idx'],
                    l2_scores=brain.pred._last_scores,
                    familiarity=out['familiarity'],
                    freq_idx=global_fi)

        freq_bmu_ip[local_fi][out['bmu_idx']] += 1
        step_fam.append(out['familiarity'])
        step_sal.append(out['salience'])
        step_pe.append(out['prediction_error'])

        # Zone assignment for I–P (offset freq_bmu by IP_OFFSET)
        # We pass a combined 16-entry list: first 8 unchanged (A–H),
        # next 8 from freq_bmu_ip
        if (step + 1) >= ZONE_UPDATE_WARMUP and (step + 1) % ZONE_UPDATE_INTERVAL == 0:
            combined = freq_bmu_ah + freq_bmu_ip
            l3_new.assign_zones_from_counters(combined)

        if (step + 1) % SNAPSHOT_EVERY == 0 or step == len(stream) - 1:
            elapsed = time.time() - t0
            sps     = (step + 1) / elapsed
            eta     = (len(stream) - step - 1) / sps

            print(f"  Step {step+1:>6,} / {len(stream):,}  "
                  f"({100*(step+1)/len(stream):.0f}%)  "
                  f"{sps:.0f} steps/s  ETA {eta:.0f}s")
            print(f"         fam={np.mean(step_fam):.3f}  "
                  f"sal={np.mean(step_sal):.3f}  "
                  f"pe={np.mean(step_pe):.3f}")

            # Score I–P (zone_offset=8)
            scores_ip = score_l3(l3_new, LABELS_IP, GRAMMAR_IP, zone_offset=IP_OFFSET)
            print_scores("I–P L3 accuracy:", scores_ip, LABELS_IP)

            # Also check A–H retention (zone_offset=0, using frozen Z)
            scores_ah = score_l3(l3_new, LABELS_AH, GRAMMAR_AH, zone_offset=0)
            strict_ah = np.mean([s['rate'] for s in scores_ah.values()])
            strict_ip = np.mean([s['rate'] for s in scores_ip.values()])
            print(f"\n  A–H RETENTION: {strict_ah*100:.1f}%  "
                  f"(should stay ~75-87%; Z[0:8,0:8] frozen)")

            print_visit_bar(freq_visit_ip, LABELS_IP, step + 1)

            calib_ip_val  = float(np.mean([s['calib']       for s in scores_ip.values()]))
            strict_ip_val = float(np.mean([s['rate_strict'] for s in scores_ip.values()]))
            calib_ah_val  = float(np.mean([s['calib']       for s in scores_ah.values()]))
            snapshots.append({'step':      step + 1,
                               'calib_ip':  calib_ip_val,
                               'strict_ip': strict_ip_val,
                               'calib_ah':  calib_ah_val,
                               'strict_ah': float(strict_ah),
                               'fam':       float(np.mean(step_fam)),
                               'pe':        float(np.mean(step_pe))})

    print(f"\n  Phase 2 complete: {time.time()-t0:.1f}s")
    return l3_new, snapshots


# ═══════════════════════════════════════════════════════════════
# FREEZE-PROBE
# ═══════════════════════════════════════════════════════════════

def freeze_probe(l3):
    print("\n" + "═" * 64)
    print("  FREEZE-PROBE — Z MATRIX LOCKED, FULL EVALUATION")
    print("═" * 64)
    print("  No more training. Pure prediction accuracy test.\n")

    print("  ── A–H (trained in Phase 1) ──────────────────────────")
    scores_ah = score_l3(l3, LABELS_AH, GRAMMAR_AH, zone_offset=0)
    print_scores("A–H strict prediction:", scores_ah, LABELS_AH)

    print("\n  ── I–P (trained in Phase 2) ──────────────────────────")
    scores_ip = score_l3(l3, LABELS_IP, GRAMMAR_IP, zone_offset=IP_OFFSET)
    print_scores("I–P strict prediction:", scores_ip, LABELS_IP)

    print("\n  ── CROSS-ZONE LEAKAGE ────────────────────────────────")
    cross_zone_probe(l3, n_ah=8, n_ip=8)

    # Summary Z matrix
    print("\n  ── Z MATRIX OVERVIEW (normalised) ───────────────────")
    Z = l3._Z.copy()
    row_sums = Z.sum(axis=1, keepdims=True)
    Z_norm   = np.where(row_sums > 0, Z / row_sums, 0.0)

    header = "     " + "".join(f" {LABELS_ALL[j]:>4}" for j in range(16))
    print(f"  {header}")
    for i in range(16):
        row_str = "".join(f" {Z_norm[i,j]:>4.2f}" for j in range(16))
        label   = LABELS_ALL[i]
        print(f"  {label}  {row_str}")

    calib_ah  = np.mean([s['calib'] for s in scores_ah.values()])
    calib_ip  = np.mean([s['calib'] for s in scores_ip.values()])
    strict_ah = np.mean([s['rate_strict'] for s in scores_ah.values()])
    strict_ip = np.mean([s['rate_strict'] for s in scores_ip.values()])
    loose_ah  = np.mean([s['rate_loose'] for s in scores_ah.values()])
    loose_ip  = np.mean([s['rate_loose'] for s in scores_ip.values()])

    print(f"\n  ═══════════════════════════════════")
    print(f"  FINAL SUMMARY")
    print(f"  ═══════════════════════════════════")
    print(f"  A–H  calib={calib_ah*100:.1f}%  strict={strict_ah*100:.1f}%  loose={loose_ah*100:.1f}%")
    print(f"  I–P  calib={calib_ip*100:.1f}%  strict={strict_ip*100:.1f}%  loose={loose_ip*100:.1f}%")
    print(f"")
    print(f"  Interpretation (calibration = primary metric):")
    if calib_ip >= 0.85:
        print(f"  ✓ L3 learned I–P grammar well (calib={calib_ip*100:.0f}%) — genuine generalisation")
    elif calib_ip >= 0.65:
        print(f"  ~ L3 partially learned I–P (calib={calib_ip*100:.0f}%) — some generalisation")
    else:
        print(f"  ✗ L3 failed on I–P (calib={calib_ip*100:.0f}%) — did not generalise")

    if calib_ah >= 0.85:
        print(f"  ✓ A–H knowledge retained (calib={calib_ah*100:.0f}%) — no catastrophic forgetting")
    elif calib_ah >= 0.65:
        print(f"  ~ A–H partially retained (calib={calib_ah*100:.0f}%)")
    else:
        print(f"  ✗ A–H forgotten (calib={calib_ah*100:.0f}%) — catastrophic forgetting occurred")

    # Cross-zone leakage summary
    max_cross = 0.0
    for z in range(16):
        probs = l3.get_zone_probs(z)
        if z < 8:
            cross = float(probs[8:].sum())
        else:
            cross = float(probs[:8].sum())
        max_cross = max(max_cross, cross)
    if max_cross < 0.05:
        print(f"  ✓ No cross-zone leakage (max={max_cross:.3f}) — Z correctly separated")
    else:
        print(f"  ⚠ Cross-zone leakage detected (max={max_cross:.3f})")

    return scores_ah, scores_ip


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_trajectory(snapshots_p1, snapshots_p2):
    print("\n" + "═" * 64)
    print("  LEARNING TRAJECTORY")
    print("═" * 64)

    print(f"\n  PHASE 1 — A–H training")
    print(f"  {'Step':>8}  {'Strict%':>8}  {'Fam':>6}  {'PE':>6}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")
    for s in snapshots_p1:
        print(f"  {s['step']:>8,}  {s['strict_ah']*100:>7.1f}%"
              f"  {s['fam']:>6.3f}  {s['pe']:>6.3f}")

    print(f"\n  PHASE 2 — I–P confirmation")
    print(f"  {'Step':>8}  {'I–P calib':>10}  {'I–P strict':>11}  {'A–H calib':>10}  {'Fam':>6}  {'PE':>6}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*11}  {'─'*10}  {'─'*6}  {'─'*6}")
    for s in snapshots_p2:
        print(f"  {s['step']:>8,}"
              f"  {s['calib_ip']*100:>9.1f}%"
              f"  {s['strict_ip']*100:>10.1f}%"
              f"  {s['calib_ah']*100:>9.1f}%"
              f"  {s['fam']:>6.3f}  {s['pe']:>6.3f}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   BRAIN CONFIRMATION TEST — A–H Training + I–P Validation   ║")
    print("║   Same brain. Different grammar. Does L3 generalise?        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    t_total = time.time()

    # ── Calibrate for A–H frequencies ─────────────────────────────────────
    rx_slow_ah, ry_slow_ah, rx_fast_ah, ry_fast_ah = calibrate(FREQS_AH, "A–H")

    # ── Build A–H signal library ───────────────────────────────────────────
    lib_ah = build_signal_library(
        FREQS_AH, LABELS_AH,
        rx_slow_ah, ry_slow_ah, rx_fast_ah, ry_fast_ah,
        freq_offset=0
    )

    # ── Calibrate for I–P frequencies ─────────────────────────────────────
    rx_slow_ip, ry_slow_ip, rx_fast_ip, ry_fast_ip = calibrate(FREQS_IP, "I–P")

    # ── Build I–P signal library ───────────────────────────────────────────
    lib_ip = build_signal_library(
        FREQS_IP, LABELS_IP,
        rx_slow_ip, ry_slow_ip, rx_fast_ip, ry_fast_ip,
        freq_offset=8
    )

    # ── Phase 1: train on A–H ─────────────────────────────────────────────
    brain, l3_ah, freq_bmu_ah, snaps_p1 = phase1_train(
        lib_ah, rx_slow_ah, ry_slow_ah, rx_fast_ah, ry_fast_ah
    )

    # ── Phase 2: confirm on I–P ───────────────────────────────────────────
    l3_16, snaps_p2 = phase2_confirm(
        brain, l3_ah, freq_bmu_ah,
        lib_ip, rx_slow_ip, ry_slow_ip, rx_fast_ip, ry_fast_ip
    )

    # ── Freeze-probe ──────────────────────────────────────────────────────
    scores_ah, scores_ip = freeze_probe(l3_16)

    # ── Trajectory summary ────────────────────────────────────────────────
    print_trajectory(snaps_p1, snaps_p2)

    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")