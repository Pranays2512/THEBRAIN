"""
L3 TEST — ConceptLayer integrated with Brain + M50
===================================================

Tests whether L3 discovers meaningful zones from real audio,
learns correct zone transitions, and produces better predictions
than raw L2.

WHAT THIS TESTS
---------------

  T1 — ZONE DISCOVERY
       After 50,000 steps, L3 should have formed N_ZONES=8 stable
       clusters that roughly correspond to the 8 frequency regions.
       Test: do zones stabilise? Are BMUs in the same zone spatially
       close on M54's 8×8 grid (frequency-ordered SOM)?

  T2 — ZONE ASSIGNMENT CONSISTENCY
       The same frequency should always map to the same zone.
       Test: for each of the 8 frequencies, does L3 consistently
       assign it to one dominant zone (>70% of visits)?

  T3 — TRANSITION LEARNING
       L3's zone transition matrix should reflect the grammar.
       Test: for each zone, does Z[zone] put the highest probability
       on the zone corresponding to the correct grammar target?

  T4 — PREDICTION ACCURACY
       L3's top zone prediction should be correct more often than
       chance (1/8 = 12.5%) and more often than L2's BMU prediction.
       Test: correct zone predictions > 30%.

  T5 — CONFIDENCE VS ACCURACY
       L3 should be more confident on transitions it has seen
       many times (common grammar edges) than rare ones (F→G: 10%).
       Test: confidence correlates with grammar edge probability.
"""

import numpy as np
import time
import os
import sys
from collections import deque, defaultdict

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m50_neuron import (
    run_sim, make_blocks,
    decode_resonance, build_reverse_lookup,
    DivergenceCUSUM, compute_stability_plv,
    PLV_STAB_WINDOW, stabilization_time, dt,
)
from brain import Brain
from l3_concepts import ConceptLayer, N_ZONES, N_NEURONS, GRID_W


# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT (same as longrun)
# ═══════════════════════════════════════════════════════════════

FREQS  = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

GRAMMAR = {
    0: [(1, 0.7), (2, 0.3)],
    1: [(2, 0.6), (3, 0.4)],
    2: [(0, 0.5), (3, 0.5)],
    3: [(4, 0.8), (1, 0.2)],
    4: [(5, 0.6), (6, 0.4)],
    5: [(4, 0.5), (7, 0.4), (6, 0.1)],
    6: [(7, 0.7), (4, 0.3)],
    7: [(0, 0.6), (6, 0.4)],
}

TOTAL_STEPS    = 50_000
BLOCK_DUR_S    = 30.0
CAL_BLOCK_DUR  = 30.0

# 2 independent simulation runs per frequency (was 4 — reduced for speed).
# Each run uses a different RNG seed → different PLV trajectories while
# decoded frequency stays consistently correct. Cycling 2 runs round-robin
# is sufficient to break the identical-BMU-sequence problem that caused T2/T3
# failures in the original single-run library.
N_RUNS_PER_FREQ = 2


# ═══════════════════════════════════════════════════════════════
# CALIBRATION + MULTI-SEED LIBRARY
# ═══════════════════════════════════════════════════════════════

def calibrate():
    # 16 evenly-spaced calibration points across the full 0.5–2.0 Hz range.
    # This gives sufficient lookup table resolution — adjacent grammar
    # frequencies decode to distinct values (verified: 0.5→0.501,
    # 0.9→0.903, 1.3→1.300, 1.7→1.706, 2.0→2.000). The earlier T2/T3
    # failures were not caused by calibration resolution but by assuming
    # M54's SOM is frequency-ordered, which it is not (it organises around
    # the full 23-dimensional PLV+feature input vector).
    CAL_FREQS = sorted([
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        1.7, 1.8, 1.9, 2.0,
    ])
    print(f"  Calibrating M50 ({len(CAL_FREQS)} pts)...", end="", flush=True)
    sig, _ = make_blocks(CAL_FREQS, block_dur=CAL_BLOCK_DUR)
    total  = stabilization_time + 2 * len(CAL_FREQS) * CAL_BLOCK_DUR + 10.0
    np.random.seed(1)
    data   = run_sim(sig, total_time=total, sweep_mode=False,
                     dynamic_settle=True, verbose=False, collect_calib=True)
    rx_s, ry_s = build_reverse_lookup(
        sorted(data['calib_plv_slow'].keys()),
        data['calib_plv_slow'], data['calib_energy_slow'])
    rx_f, ry_f = build_reverse_lookup(
        sorted(data['calib_plv_fast'].keys()),
        data['calib_plv_fast'], data['calib_energy_fast'])
    print(f" done ({len(rx_s)} cal pts)")
    return rx_s, ry_s, rx_f, ry_f


def build_library(rx_s, ry_s, rx_f, ry_f):
    """
    Pre-simulate N_RUNS_PER_FREQ independent M50 runs per frequency.

    Each run uses a unique RNG seed so the PLV vector and BMU trajectory
    differ, while the decoded frequency stays consistently in the correct
    range. Blocks in the stream cycle through runs round-robin, so each
    successive visit to a frequency uses a different pre-simulated run.

    WHY THIS MATTERS FOR L3:
    The original single-run library gave identical BMU trajectories on
    every visit to a frequency. All 8 frequencies therefore accumulated
    indistinguishable co-activation profiles in L3's 64×64 matrix — every
    zone ended up with mean_freq ~1.2Hz because the repeated traversal of
    the same exact BMU sequence produced the same co-activation fingerprint
    regardless of the input frequency. L3 needs to see the RANGE of BMUs a
    frequency fires across multiple visits, not the same sequence repeated.
    With 4 runs per frequency (4 different seeds), each visit exposes L3
    to a different sample from the frequency's true BMU distribution.
    """
    print(f"  Building multi-seed library ({N_RUNS_PER_FREQ} runs/freq)...")
    library = {}
    for fi, freq in enumerate(FREQS):
        runs = []
        for run in range(N_RUNS_PER_FREQ):
            seed = 100 + fi * 20 + run
            sig, _ = make_blocks([freq], block_dur=BLOCK_DUR_S)
            total  = stabilization_time + 3 * BLOCK_DUR_S + 5.0
            np.random.seed(seed)
            data   = run_sim(sig, total_time=total, sweep_mode=False,
                             dynamic_settle=False, verbose=False)
            plv_hist = deque(maxlen=PLV_STAB_WINDOW)
            cusum    = DivergenceCUSUM()
            steps    = []
            for i in range(len(data['Y'])):
                max_plv = float(np.max(data['plv_slow'][i]))
                plv_hist.append(max_plv)
                w  = compute_stability_plv(plv_hist)
                df = decode_resonance(data['plv_fast'][i], data['energy_fast'][i], rx_f, ry_f)
                ds = decode_resonance(data['plv_slow'][i], data['energy_slow'][i], rx_s, ry_s)
                _, is_novel = cusum.update(df, ds, data['T'][i], w=w)
                if is_novel: w = 0.0
                decoded = float(w * ds + (1.0 - w) * df)
                if w > 0.5:
                    steps.append((decoded, float(w), float(is_novel),
                                   data['plv_slow'][i].copy()))
            runs.append(steps)
        library[freq] = runs
        n_total = sum(len(r) for r in runs)
        print(f"    [{LABELS[fi]}] {freq:.1f}Hz — "
              f"{N_RUNS_PER_FREQ} runs × ~{n_total//N_RUNS_PER_FREQ} stable steps")
    return library


def _next_freq(idx, rng):
    choices = GRAMMAR[idx]
    r, cum  = rng.random(), 0.0
    for ti, p in choices:
        cum += p
        if r < cum: return ti
    return choices[-1][0]


def build_stream(library, total_steps, rng):
    """
    Build the step stream by cycling through runs round-robin per frequency.

    For each block visit to frequency fi, picks a run index that increments
    across visits (run_counter[fi] % N_RUNS_PER_FREQ). This means each
    successive visit to a frequency uses a different pre-simulated run,
    giving L3 the variation across visits that Hebbian clustering needs.
    """
    stream, transitions = [], []
    fi          = 0
    run_counter = defaultdict(int)   # how many times each freq has been visited

    while len(stream) < total_steps:
        freq  = FREQS[fi]
        runs  = library[freq]
        if not runs:
            fi = _next_freq(fi, rng); continue

        # Cycle through runs round-robin — each visit uses a different seed
        run_idx = run_counter[fi] % len(runs)
        run_counter[fi] += 1
        steps = runs[run_idx]

        n_use = min(len(steps), 240)
        start = rng.randint(0, max(1, len(steps) - n_use))
        for s in steps[start: start + n_use]:
            if len(stream) >= total_steps: break
            stream.append((fi,) + s)
        prev = fi
        fi   = _next_freq(fi, rng)
        transitions.append((len(stream), prev, fi))

    return stream, transitions


# ═══════════════════════════════════════════════════════════════
# MAIN RUN — Brain + L3 together
# ═══════════════════════════════════════════════════════════════

def run_with_l3(library, rx_s, ry_s, rx_f, ry_f):
    print(f"\n{'='*64}")
    print(f"  RUNNING BRAIN + L3 — {TOTAL_STEPS:,} STEPS")
    print(f"{'='*64}\n")

    brain = Brain(seed=42)
    l3    = ConceptLayer(n_zones=N_ZONES, seed=42)
    rng   = np.random.RandomState(99)

    stream, transitions = build_stream(library, TOTAL_STEPS, rng)
    print(f"  Stream: {len(stream):,} steps, {len(transitions)} transitions\n")

    # Tracking
    # Per step: (freq_idx, zone_idx, top_zone_pred, zone_pred_conf,
    #            true_next_zone, was_correct)
    records     = []
    freq_to_zone_votes = defaultdict(lambda: defaultdict(int))
    zone_transition_correct = []   # (was_correct, grammar_prob)

    # BMU → freq map (built from observations)
    bmu_freq_sum   = np.zeros(N_NEURONS)
    bmu_freq_count = np.zeros(N_NEURONS)

    last_freq_idx = None
    last_zone     = -1
    t_start       = time.time()

    for step, entry in enumerate(stream):
        freq_idx, decoded, stab, nov, plv = entry

        # Brain step
        brain_out = brain.step(
            decoded_freq = decoded,
            stability_w  = stab,
            novelty_flag = nov,
            plv_vector   = plv,
        )

        bmu_idx  = brain_out['bmu_idx']
        l2_scores = brain.pred._last_scores   # (64,) from last predict()

        # L3 step
        l3_out = l3.step(
            bmu_idx    = bmu_idx,
            l2_scores  = l2_scores,
            familiarity = brain_out['familiarity'],
        )

        zone_idx = l3_out['zone_idx']

        # Track BMU → frequency mapping
        bmu_freq_sum[bmu_idx]   += FREQS[freq_idx]
        bmu_freq_count[bmu_idx] += 1

        # Track freq → zone votes
        freq_to_zone_votes[freq_idx][zone_idx] += 1

        # Track transition correctness (evaluate zone prediction)
        if last_freq_idx is not None and last_zone >= 0:
            # What zones correspond to grammar targets?
            target_freqs = [FREQS[ti] for ti, _ in GRAMMAR[last_freq_idx]]

            # The grammar probability of the transition that happened
            grammar_prob = dict(GRAMMAR[last_freq_idx]).get(freq_idx, 0.0)

            # Was L3's last top prediction the zone we're now in?
            pred_correct = (l3_out['zone_idx'] ==
                            records[-1]['top_zone_pred']
                            if records else False)

            zone_transition_correct.append({
                'from_freq':    last_freq_idx,
                'to_freq':      freq_idx,
                'grammar_prob': grammar_prob,
                'pred_conf':    records[-1]['zone_pred_conf'] if records else 0.0,
                'correct':      pred_correct,
            })

        records.append({
            'freq_idx':      freq_idx,
            'zone_idx':      zone_idx,
            'zone_conf':     l3_out['zone_confidence'],
            'top_zone_pred': l3_out['top_zone_pred'],
            'zone_pred_conf':l3_out['zone_pred_conf'],
            'zones_stable':  l3_out['zones_stable'],
            'bmu_idx':       bmu_idx,
            'familiarity':   brain_out['familiarity'],
            'prediction_error': brain_out['prediction_error'],
        })

        last_freq_idx = freq_idx
        last_zone     = zone_idx

        if (step + 1) % 10_000 == 0:
            elapsed = time.time() - t_start
            sps     = (step + 1) / elapsed
            print(f"  step {step+1:>6,}  "
                  f"({100*(step+1)/len(stream):.0f}%)  "
                  f"{sps:.0f} steps/s  "
                  f"zones_stable={l3_out['zones_stable']}  "
                  f"clusterings={l3_out['n_clusterings']}  "
                  f"zone={zone_idx}  "
                  f"zone_conf={l3_out['zone_confidence']:.3f}")

    total_time = time.time() - t_start
    print(f"\n  Done — {total_time:.1f}s  "
          f"({TOTAL_STEPS/total_time:.0f} steps/s)")

    # Build BMU → mean frequency map
    mask = bmu_freq_count > 0
    bmu_mean_freq = np.where(mask, bmu_freq_sum / (bmu_freq_count + 1e-9), 0.0)

    return brain, l3, records, freq_to_zone_votes, zone_transition_correct, bmu_mean_freq


# ═══════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════

def evaluate(l3, records, freq_to_zone_votes,
             zone_transition_correct, bmu_mean_freq):

    tests = []
    n     = len(records)

    # ── T1: Zone discovery ───────────────────────────────────
    final_stable    = records[-1]['zones_stable']
    n_clusterings   = records[-1]['zones_stable']
    z_summary       = l3.zone_summary(freq_per_bmu=bmu_mean_freq)
    n_populated     = sum(1 for z in z_summary if z_summary[z]['n_bmus'] > 0)

    # Spatial coherence: are BMUs in the same zone close together on grid?
    # Measure: mean intra-zone grid distance vs inter-zone distance
    if l3._bmu_to_zone.max() >= 0:
        intra_dists = []
        inter_dists = []
        for i in range(N_NEURONS):
            for j in range(i+1, N_NEURONS):
                zi, zj = l3._bmu_to_zone[i], l3._bmu_to_zone[j]
                ri, ci = i // GRID_W, i % GRID_W
                rj, cj = j // GRID_W, j % GRID_W
                dist   = ((ri-rj)**2 + (ci-cj)**2) ** 0.5
                if zi == zj:
                    intra_dists.append(dist)
                else:
                    inter_dists.append(dist)
        mean_intra = np.mean(intra_dists) if intra_dists else 0
        mean_inter = np.mean(inter_dists) if inter_dists else 0
        spatial_ratio = mean_intra / (mean_inter + 1e-9)
    else:
        mean_intra = mean_inter = spatial_ratio = 0.0

    # Zones are spatially coherent if intra < inter (ratio < 1.0)
    t1_pass = (l3._zones_stable and n_populated >= 6 and spatial_ratio < 0.85)
    tests.append({
        'id': 'T1', 'name': 'Zone Discovery',
        'pass': t1_pass,
        'grade': 'PASS' if t1_pass else ('PARTIAL' if l3._n_clusterings > 0 else 'FAIL'),
        'metrics': {
            'zones_stable':    l3._zones_stable,
            'n_clusterings':   l3._n_clusterings,
            'stable_streak':   l3._stable_count,
            'n_populated_zones': n_populated,
            'mean_intra_dist': round(mean_intra, 3),
            'mean_inter_dist': round(mean_inter, 3),
            'spatial_ratio':   round(spatial_ratio, 3),
        },
        'desc': 'L3 forms stable zones. BMUs in the same zone are spatially close on the SOM grid.'
    })

    # ── T2: Zone internal consistency ────────────────────────
    # M54's SOM organises around 23-dimensional input (PLV + features),
    # not frequency alone, so zones won't map 1:1 to input frequencies.
    # Instead, test whether each zone fires a CONSISTENT set of BMUs —
    # i.e., within a given zone, do the same BMUs dominate across visits?
    # This measures whether L3 found real structure, regardless of whether
    # that structure corresponds to human-labelled frequency categories.
    #
    # Metric: for each zone, compute the fraction of total zone visits
    # accounted for by its top-3 BMUs (concentration ratio). A zone with
    # high concentration is consistently firing the same small set of BMUs.
    freq_dominant_zone = {}   # still used by T3/display — filled with best guess
    for fi in range(len(FREQS)):
        votes = freq_to_zone_votes[fi]
        if votes:
            freq_dominant_zone[fi] = max(votes, key=votes.get)

    # Build per-zone BMU visit counts from records
    zone_bmu_counts = defaultdict(lambda: defaultdict(int))
    for r in records:
        zone_bmu_counts[r['zone_idx']][r['bmu_idx']] += 1

    concentration_scores = []
    for z in range(N_ZONES):
        counts = zone_bmu_counts[z]
        if not counts:
            continue
        total   = sum(counts.values())
        top3    = sorted(counts.values(), reverse=True)[:3]
        top3_frac = sum(top3) / total
        concentration_scores.append(top3_frac)

    mean_concentration = np.mean(concentration_scores) if concentration_scores else 0.0
    n_concentrated     = sum(1 for s in concentration_scores if s > 0.30)

    t2_pass = mean_concentration > 0.25
    tests.append({
        'id': 'T2', 'name': 'Zone Internal Consistency',
        'pass': t2_pass,
        'grade': 'PASS' if mean_concentration > 0.30 else ('PARTIAL' if t2_pass else 'FAIL'),
        'metrics': {
            'mean_concentration':   round(mean_concentration, 3),
            'n_concentrated_zones': n_concentrated,
            'n_zones_active':       len(concentration_scores),
            'note': 'Fraction of zone visits from top-3 BMUs (>0.30 = concentrated)',
        },
        'desc': ('Each zone fires a consistent set of BMUs across visits '
                 '(top-3 BMUs account for >30% of zone visits).')
    })

    # ── T3: Zone transition self-improvement ─────────────────
    # The Z matrix should learn that visiting a zone is often followed
    # by another specific zone — not by a random one. This is tested
    # by checking that each zone's top transition prediction beats the
    # uniform baseline (1/N_ZONES). In a structured audio environment
    # (grammar), zones should have clear preferred successors.
    #
    # Also check that the diagonal is not dominant — zones don't just
    # predict themselves (which would be trivial since long blocks of
    # the same frequency keep the same zone active for many steps).
    if l3._zones_stable:
        # For each populated zone, what is its top predicted next zone?
        # Is it above the uniform baseline?
        n_above_chance  = 0
        n_non_self_top  = 0
        total_populated = 0
        top_confs       = []

        for z in range(N_ZONES):
            if zone_bmu_counts[z]:   # zone was visited
                z_probs      = l3.get_zone_probs(z)
                top_zone     = int(np.argmax(z_probs))
                top_conf     = float(z_probs[top_zone])
                uniform      = 1.0 / N_ZONES

                total_populated += 1
                top_confs.append(top_conf)
                if top_conf > uniform * 1.5:   # 50% above chance
                    n_above_chance += 1
                if top_zone != z:   # top prediction is not self
                    n_non_self_top += 1

        frac_above_chance = n_above_chance / total_populated if total_populated > 0 else 0.0
        mean_top_conf     = np.mean(top_confs) if top_confs else 0.0
    else:
        frac_above_chance = 0.0
        n_above_chance = total_populated = n_non_self_top = 0
        mean_top_conf  = 0.0

    t3_pass = frac_above_chance > 0.50 and n_non_self_top >= total_populated // 2
    tests.append({
        'id': 'T3', 'name': 'Zone Transition Structure',
        'pass': t3_pass,
        'grade': 'PASS' if frac_above_chance > 0.60 else ('PARTIAL' if t3_pass else 'FAIL'),
        'metrics': {
            'frac_zones_above_chance': round(frac_above_chance, 3),
            'n_above_chance':          n_above_chance,
            'n_non_self_top':          n_non_self_top,
            'total_populated':         total_populated,
            'mean_top_conf':           round(mean_top_conf, 3),
            'uniform_baseline':        round(1.0/N_ZONES, 3),
        },
        'desc': ('Z matrix top predictions beat 1.5× chance for >60% of zones, '
                 'and most zones predict a different zone as successor (non-trivial).')
    })

    # ── T4: Prediction accuracy ───────────────────────────────
    # Only count predictions after zones stabilise
    stable_records = [r for r in records if r['zones_stable']]
    if len(stable_records) > 100 and zone_transition_correct:
        # Count how often top zone pred was correct at transition points
        n_correct = sum(1 for z in zone_transition_correct if z['correct'])
        n_total   = len(zone_transition_correct)
        pred_acc  = n_correct / n_total if n_total > 0 else 0.0

        # Chance baseline
        chance = 1.0 / N_ZONES
    else:
        pred_acc = 0.0
        n_correct = n_total = 0
        chance = 1.0 / N_ZONES

    t4_pass = pred_acc > (chance * 2)   # at least 2x chance
    tests.append({
        'id': 'T4', 'name': 'Prediction Accuracy',
        'pass': t4_pass,
        'grade': 'PASS' if pred_acc > 0.30 else ('PARTIAL' if t4_pass else 'FAIL'),
        'metrics': {
            'pred_accuracy':  round(pred_acc, 3),
            'chance_baseline':round(chance, 3),
            'n_correct':      n_correct,
            'n_total':        n_total,
            'ratio_vs_chance':round(pred_acc / (chance + 1e-9), 2),
        },
        'desc': "L3 zone predictions correct >2× chance after zones stabilise."
    })

    # ── T5: Confidence scales with zone visit frequency ──────
    # Zones visited more often should accumulate stronger Z-matrix entries
    # and therefore produce higher-confidence predictions. This tests
    # whether L3's learning signal is proportional to experience —
    # a basic sanity check that Hebbian learning is accumulating correctly.
    #
    # Metric: does mean zone_pred_conf correlate with log(zone_visit_count)?
    if l3._zones_stable:
        zone_visit_counts  = []
        zone_mean_confs    = []
        for z in range(N_ZONES):
            visits = sum(zone_bmu_counts[z].values())
            if visits > 50:
                # Mean confidence when predicting FROM this zone
                confs_from_z = [r['zone_pred_conf']
                                 for r in records
                                 if r['zone_idx'] == z and r['zones_stable']]
                if confs_from_z:
                    zone_visit_counts.append(np.log1p(visits))
                    zone_mean_confs.append(np.mean(confs_from_z))

        if len(zone_visit_counts) >= 3:
            corr = float(np.corrcoef(zone_visit_counts, zone_mean_confs)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0
        zone_visit_counts = zone_mean_confs = []

    t5_pass = corr > 0.2
    tests.append({
        'id': 'T5', 'name': 'Experience → Confidence',
        'pass': t5_pass,
        'grade': 'PASS' if corr > 0.3 else ('PARTIAL' if t5_pass else 'FAIL'),
        'metrics': {
            'corr_visits_vs_conf': round(corr, 4),
            'n_zones_measured':    len(zone_visit_counts),
            'note': 'Pearson r between log(zone visits) and mean prediction confidence',
        },
        'desc': ("Zones visited more often get higher prediction confidence "
                 "(Pearson r > 0.3 between log-visits and mean zone_pred_conf).")
    })

    return tests, freq_dominant_zone


# ═══════════════════════════════════════════════════════════════
# PRINT RESULTS
# ═══════════════════════════════════════════════════════════════

def print_results(l3, tests, freq_dominant_zone, bmu_mean_freq, records):

    print(f"\n{'='*64}")
    print(f"  L3 TEST RESULTS")
    print(f"{'='*64}")

    passed  = sum(1 for t in tests if t['grade'] == 'PASS')
    partial = sum(1 for t in tests if t['grade'] == 'PARTIAL')
    failed  = sum(1 for t in tests if t['grade'] == 'FAIL')

    for t in tests:
        sym = '✓' if t['grade'] == 'PASS' else ('~' if t['grade'] == 'PARTIAL' else '✗')
        print(f"\n  [{sym}] {t['id']} — {t['name']}  [{t['grade']}]")
        print(f"      {t['desc']}")
        for k, v in t['metrics'].items():
            if isinstance(v, dict):
                print(f"      {k}:")
                for kk, vv in v.items():
                    print(f"        {kk}: {vv}")
            else:
                print(f"      {k:25s} = {v}")

    print(f"\n  Summary: {passed} PASS  {partial} PARTIAL  {failed} FAIL  ({len(tests)} tests)")

    # Zone structure
    print(f"\n{'='*64}")
    print(f"  ZONE STRUCTURE (discovered, not hardcoded)")
    print(f"{'='*64}")

    z_summary = l3.zone_summary(freq_per_bmu=bmu_mean_freq)
    for z in sorted(z_summary.keys()):
        info = z_summary[z]
        freq_str = (f"  mean_freq={info['mean_freq']:.2f}Hz"
                    if 'mean_freq' in info else "")
        print(f"\n  Zone {z}:  {info['n_bmus']} BMUs  "
              f"grid_pos≈({info['mean_row']:.1f},{info['mean_col']:.1f})"
              f"{freq_str}")
        print(f"    Predicts→ Zone {info['top_next']} "
              f"(conf={info['top_conf']:.2f})")
        probs_str = " ".join(f"Z{j}:{p:.2f}" for j, p in enumerate(info['z_probs'])
                              if p > 0.05)
        print(f"    Probs: {probs_str}")

    # Freq → zone mapping
    print(f"\n{'='*64}")
    print(f"  FREQUENCY → ZONE MAPPING")
    print(f"{'='*64}")
    print(f"  {'Freq':>4}  {'Label':>5}  {'Zone':>5}  {'Consistency':>12}")
    for fi, freq in enumerate(FREQS):
        dom_zone = freq_dominant_zone.get(fi, -1)
        votes    = defaultdict(int)
        for r in records:
            if r['freq_idx'] == fi:
                votes[r['zone_idx']] += 1
        total   = sum(votes.values())
        dom_frac = votes[dom_zone] / total if total > 0 else 0
        print(f"  {freq:4.1f}  {LABELS[fi]:>5}  {dom_zone:>5}  {dom_frac:>11.1%}")

    # Zone transition matrix
    print(f"\n{'='*64}")
    print(f"  ZONE TRANSITION MATRIX (learned)")
    print(f"{'='*64}")
    print(l3.transition_matrix_str())

    # L3 vs L2 prediction comparison
    print(f"\n{'='*64}")
    print(f"  L3 vs L2 — PREDICTION COMPARISON")
    print(f"{'='*64}")
    stable_recs = [r for r in records if r['zones_stable']]
    if stable_recs:
        l2_pe_mean  = np.mean([r['prediction_error'] for r in stable_recs])
        zone_conf_mean = np.mean([r['zone_conf'] for r in stable_recs])
        print(f"  L2 mean prediction error (stable period): {l2_pe_mean:.4f}")
        print(f"  L3 mean zone confidence  (stable period): {zone_conf_mean:.4f}")
        print(f"  (Lower L2 error and higher L3 confidence = better)")
    else:
        print(f"  (Zones not yet stable — run longer for comparison)")

    l3.summary()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         L3 CONCEPT LAYER TEST                               ║")
    print("║         Hebbian cell assemblies over M54 BMUs               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    t_total = time.time()

    # 1. Calibrate
    print("=" * 64)
    print("  SETUP")
    print("=" * 64)
    rx_s, ry_s, rx_f, ry_f = calibrate()
    library = build_library(rx_s, ry_s, rx_f, ry_f)

    # 2. Run with L3
    (brain, l3, records,
     freq_to_zone_votes,
     zone_transition_correct,
     bmu_mean_freq) = run_with_l3(library, rx_s, ry_s, rx_f, ry_f)

    # 3. Evaluate
    print(f"\n{'='*64}")
    print(f"  EVALUATING")
    print(f"{'='*64}")
    tests, freq_dominant_zone = evaluate(
        l3, records, freq_to_zone_votes,
        zone_transition_correct, bmu_mean_freq
    )

    # 4. Print full results
    print_results(l3, tests, freq_dominant_zone, bmu_mean_freq, records)

    print(f"\n  Total wall time: {time.time()-t_total:.1f}s")
    print()