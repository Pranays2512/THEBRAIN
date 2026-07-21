#!/usr/bin/env python3
"""
stress_test.py — Comprehensive stress test of the trained Brain2 against its own training data.

Tests:
  1. FACT RECALL: Query facts the brain was explicitly trained on
  2. ISA CHAINS: Test transitive taxonomy reasoning (dog → mammal → animal → living thing)
  3. EVENT COMPREHENSION: Feed declarative sentences and check the membrane admits/rejects correctly
  4. COMPARISON QUERIES: Rich comparisons between entities
  5. NEURAL LM: Generate samples and check coherence
  6. EDGE CASES: Unseen/nonsense queries → should say "I don't know"

Prints a final scorecard.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import traceback

# ── Setup ────────────────────────────────────────────────────────────────────
def load():
    from engines.neural.neural_lm_torch import NeuralLMTorch
    lm = NeuralLMTorch.load("trained/owned_lm_auto.pt")
    print(f"  LM: {lm.param_count()} params on {lm.device}", flush=True)

    from faculties.whole_brain import WholeBrain
    wb = WholeBrain()
    wb.brain = None  # avoid OMP conflict
    print(f"  WholeBrain loaded.", flush=True)
    return wb, lm


# ── Test Categories ──────────────────────────────────────────────────────────

def test_fact_recall(wb):
    """Test: can the brain recall facts it was trained on?"""
    tests = [
        # (query, expected_kind, expected_substring_in_answer)
        ("what is the force on the rocket?", "compute", "1.2e+04"),
        ("what is the momentum of the rocket?", "compute", "3e+05"),
        ("what is the density of the sample?", "compute", "4"),
        ("what is the energy of the sample?", "compute", "900"),
        ("what is the momentum of the sample?", "compute", "60"),
        ("what is the density of the rocket?", "compute", "500"),
        ("what is the energy of the rocket?", "compute", "4.5e+07"),
        ("what is the force on the sample?", "compute", "19.6"),
    ]
    return _run_tests("FACT RECALL (Physics Policies)", wb, tests)


def test_isa_chains(wb):
    """Test: transitive ISA reasoning over trained taxonomy."""
    tests = [
        # Direct ISA
        ("is a dog an animal?", "factual", "Yes"),
        ("is a cat an animal?", "factual", "Yes"),
        # Multi-hop ISA (dog → mammal → animal)
        ("is a dog a mammal?", "factual", "Yes"),
        # Abilities inherited via ISA
        ("what can a bird do?", "factual", "fly"),
        ("what can a fish do?", "factual", "swim"),
    ]
    return _run_tests("ISA CHAINS (Taxonomy Reasoning)", wb, tests)


def test_event_comprehension(wb):
    """Test: event reader parses declaratives and the membrane disposes."""
    tests = [
        # Events that resemble training data patterns
        ("the cat chased the mouse", "event", None),
        ("elephants live in forests", "event", None),
        ("the students explored the world", "event", None),
        ("patterns occur in nature", "event", None),
        ("the butterfly fluttered across the garden", "event", None),
        ("the dog ate the fish", "event", None),
        ("the monkey climbed the tree", "event", None),
        ("scientists discovered new elements", "event", None),
        # Factual statements the brain should just accept
        ("science helps sustainable world", "event", None),
        ("mathematics explains patterns", "event", None),
    ]
    return _run_tests("EVENT COMPREHENSION (Reader + Membrane)", wb, tests)


def test_comparisons(wb):
    """Test: rich comparison queries between trained entities."""
    tests = [
        ("which is heavier, the rocket or the sample?", "compute", "rocket"),
        ("which is faster, the rocket or the sample?", "compute", "rocket"),
        ("which is denser, the rocket or the sample?", "compute", "rocket"),
    ]
    return _run_tests("COMPARISON QUERIES", wb, tests)


def test_unknown(wb):
    """Test: queries about things the brain was NOT trained on → should say 'I don't know'."""
    tests = [
        ("what is the meaning of life?", "none", "don't know"),
        ("what is the speed of a unicorn?", "none", "don't know"),
        ("tell me about quantum entanglement", "none", None),
    ]
    return _run_tests("EDGE CASES (Unknown/OOD)", wb, tests)


def test_lm_generation(lm):
    """Test: neural LM generates coherent text."""
    print(f"\n{'─' * 60}")
    print(f"  NEURAL LM GENERATION (9.6M Transformer)")
    print(f"{'─' * 60}")
    passed, total = 0, 0
    for seed in range(10):
        total += 1
        try:
            words = lm.generate(seed=seed, max_len=25)
            text = " ".join(words)
            # Basic sanity: generated at least 3 words and contains some real words
            ok = len(words) >= 3 and any(len(w) > 2 for w in words)
            status = "✅" if ok else "❌"
            if ok:
                passed += 1
            print(f"  {status} seed={seed:>3d}: {text[:90]}")
        except Exception as e:
            print(f"  ❌ seed={seed:>3d}: ERROR {e}")
    print(f"\n  Result: {passed}/{total} passed")
    return passed, total


def test_bulk_events(wb):
    """Test: feed a large batch of diverse declarative sentences rapidly."""
    sentences = [
        "the sun rises in the east",
        "water flows downhill",
        "plants need sunlight to grow",
        "the earth revolves around the sun",
        "metals conduct electricity",
        "birds build nests in trees",
        "rain falls from clouds",
        "fire needs oxygen to burn",
        "sound travels through air",
        "ice melts at zero degrees",
        "the moon reflects sunlight",
        "seeds germinate in soil",
        "rivers flow to the sea",
        "magnets attract iron",
        "light travels in straight lines",
        "friction slows moving objects",
        "gravity pulls objects downward",
        "insects have six legs",
        "diamonds are the hardest natural substance",
        "the brain controls the body",
        "photosynthesis produces oxygen",
        "blood carries oxygen to cells",
        "earthquakes occur along fault lines",
        "volcanoes erupt molten lava",
        "fossils form in sedimentary rock",
        "antibiotics kill bacteria",
        "vaccines prevent diseases",
        "DNA carries genetic information",
        "telescopes observe distant stars",
        "microscopes reveal tiny organisms",
    ]
    print(f"\n{'─' * 60}")
    print(f"  BULK EVENT PROCESSING ({len(sentences)} sentences)")
    print(f"{'─' * 60}")

    passed, total = 0, len(sentences)
    t0 = time.time()
    for s in sentences:
        try:
            result = wb.sense(s)
            kind = result["answer"]["kind"]
            msg = result["answer"]["msg"][:60]
            # Any non-crash response is a pass for bulk
            passed += 1
            status = "✅" if kind == "event" else "⚠️"
            print(f"  {status} [{kind:8s}] {s[:45]:45s} → {msg}")
        except Exception as e:
            print(f"  ❌ {s[:45]:45s} → ERROR: {e}")
    dt = time.time() - t0
    print(f"\n  Result: {passed}/{total} processed in {dt:.1f}s ({total/dt:.1f} queries/sec)")
    return passed, total


def test_rapid_fire(wb):
    """Test: many rapid-fire factual + compute queries."""
    queries = [
        "what is the force on the rocket?",
        "what is the force on the sample?",
        "what is the momentum of the rocket?",
        "what is the momentum of the sample?",
        "what is the density of the rocket?",
        "what is the density of the sample?",
        "what is the energy of the rocket?",
        "what is the energy of the sample?",
        "is a dog an animal?",
        "is a cat an animal?",
        "what can a bird do?",
        "what is the force on the rocket?",
        "what is the momentum of the sample?",
        "what is the density of the rocket?",
        "what is the energy of the sample?",
    ]
    print(f"\n{'─' * 60}")
    print(f"  RAPID FIRE ({len(queries)} queries)")
    print(f"{'─' * 60}")
    t0 = time.time()
    passed = 0
    for q in queries:
        try:
            result = wb.sense(q)
            passed += 1
        except:
            pass
    dt = time.time() - t0
    print(f"  {passed}/{len(queries)} passed in {dt:.2f}s ({len(queries)/dt:.0f} q/s)")
    return passed, len(queries)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_tests(category, wb, tests):
    print(f"\n{'─' * 60}")
    print(f"  {category}")
    print(f"{'─' * 60}")
    passed, total = 0, len(tests)
    for q, expected_kind, expected_substr in tests:
        try:
            result = wb.sense(q)
            ans = result["answer"]
            kind = ans["kind"]
            msg = ans["msg"]

            kind_ok = (kind == expected_kind)
            substr_ok = (expected_substr is None) or (expected_substr.lower() in msg.lower())
            ok = kind_ok and substr_ok

            status = "✅" if ok else "❌"
            if ok:
                passed += 1
            print(f"  {status} {q}")
            print(f"      [{kind:8s}] {msg[:70]}")
            if not ok:
                if not kind_ok:
                    print(f"      expected kind={expected_kind}, got={kind}")
                if not substr_ok:
                    print(f"      expected '{expected_substr}' in answer")
        except Exception as e:
            print(f"  ❌ {q}")
            print(f"      ERROR: {e}")
    print(f"\n  Result: {passed}/{total} passed")
    return passed, total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Brain2 — STRESS TEST")
    print("=" * 60, flush=True)

    wb, lm = load()

    scorecard = {}
    t0 = time.time()

    scorecard["Fact Recall"] = test_fact_recall(wb)
    scorecard["ISA Chains"] = test_isa_chains(wb)
    scorecard["Event Comprehension"] = test_event_comprehension(wb)
    scorecard["Comparisons"] = test_comparisons(wb)
    scorecard["Edge Cases"] = test_unknown(wb)
    scorecard["LM Generation"] = test_lm_generation(lm)
    scorecard["Bulk Events"] = test_bulk_events(wb)
    scorecard["Rapid Fire"] = test_rapid_fire(wb)

    total_time = time.time() - t0

    # ── Final Scorecard ──
    print("\n" + "=" * 60)
    print("  FINAL SCORECARD")
    print("=" * 60)
    grand_passed, grand_total = 0, 0
    for cat, (p, t) in scorecard.items():
        pct = 100 * p / t if t > 0 else 0
        bar_len = 20
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        status = "✅" if pct >= 80 else "⚠️" if pct >= 50 else "❌"
        print(f"  {status} {cat:25s} {bar} {p:>3d}/{t:<3d} ({pct:.0f}%)")
        grand_passed += p
        grand_total += t

    pct = 100 * grand_passed / grand_total
    print(f"\n  {'─' * 50}")
    print(f"  TOTAL: {grand_passed}/{grand_total} ({pct:.1f}%)")
    print(f"  Time:  {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
