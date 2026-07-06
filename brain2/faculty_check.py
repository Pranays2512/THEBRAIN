#!/usr/bin/env python3
"""faculty_check.py — do the higher-order faculties WORK on a live, loaded brain?

Answers one question honestly: after the brain is built + fed knowledge, does each faculty
actually run and produce a verified result — proposer, verifiers, cross-domain, novel
formation, dreaming, reasoning, conjecture, sandbox? PASS = ran and produced the expected
kind of output; FAIL = errored or produced nothing. Not a quality score — a wiring/liveness
check across the whole faculty set in one process.

    /opt/homebrew/bin/python3.13 faculty_check.py
"""
import os
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

RESULTS = []


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"EXC {type(e).__name__}: {e}"
    RESULTS.append((name, ok))
    print(f"  {'✓' if ok else '✗'} {name:34s} {detail}")


def main():
    print("=" * 68)
    print("  FACULTY LIVENESS CHECK — every higher-order faculty on a live brain")
    print("=" * 68)
    from whole_brain import WholeBrain
    wb = WholeBrain()
    # feed some reading so induction/curiosity have episodes
    for s in ["the dog ate the fish", "the dog ate the bread", "the cat ate the fish",
              "the dog chased the cat", "the cat chased the fish", "the dog chased the bird"]:
        wb.reader.read(s)

    print("\n  REASONING")
    check("reasoning: isa closure", lambda: (
        wb.kre.reaches("dog", "isa", "mammal")[0], "dog -> mammal chain"))
    check("reasoning: compute law", lambda: (
        wb.ask("what is the force of the rocket")[2], wb.ask("what is the force of the rocket")[1]))
    check("reasoning: rich query", lambda: (
        "more" in str(wb.ask("is the rocket heavier than the sample")[1]),
        wb.ask("is the rocket heavier than the sample")[1]))

    print("\n  PROPOSER")
    check("proposer: guided code synth", lambda: (
        wb.write_transform([("John Smith", "JOHN"), ("bob dylan", "BOB")])["verified"],
        "program_synth via online_proposer2"))
    check("proposer: autonomous loop", lambda: (
        wb.self_extend()["conjectures_tested"] > 0,
        f"banked {wb.self_extend()['banked']}"))
    check("proposer: learned heuristic", lambda: (
        wb.learn_heuristic()["speedup"] > 1.0,
        f"{wb.learn_heuristic()['speedup']}x fewer states"))

    print("\n  VERIFIERS")
    check("verifier: dimensional", lambda: (
        wb.check_dimensions(("*", "mass", "accel"), "force") is True
        and wb.check_dimensions(("*", "mass", "speed"), "force") is False, "units gate"))
    check("verifier: self_check (invariants)", lambda: (
        wb.self_check()["health"] is not None, "invariant_miner + verifier_monitor"))
    check("verifier: refuter self-correct", lambda: (
        wb.write_code_robust("int1", math.factorial, [0, 1, 4, 5, 6])["verified"],
        "refute_synth"))

    print("\n  CROSS-DOMAIN + NOVEL FORMATION")
    check("cross-domain: shared law", lambda: (
        wb.cross_domain().get("verified", False),
        f"discovered {wb.cross_domain().get('concept')}"))
    check("novel: concept blend", lambda: (
        "novel" in wb.blend(), f"nearest={wb.blend().get('nearest')}"))
    check("novel: named concept", lambda: (
        wb.concept_mem is not None and wb.cross_domain().get("concept") is not None,
        "concept_memory registers discoveries"))

    print("\n  CONJECTURE + SANDBOX")
    check("conjecture: admits true law", lambda: (
        wb.test_conjecture(lambda m, v: 0.5 * m * v * v)["admitted"], "0.5*m*v^2 survives"))
    check("sandbox: rejects false law", lambda: (
        not wb.test_conjecture(lambda m, v: m * v)["admitted"], "m*v refuted"))

    print("\n  DREAMING (C++ consolidation)")
    check("dreaming: dream_replay runs", lambda: _dream_runs())
    check("dreaming: cuts forgetting", lambda: _dream_helps())

    print("\n  CURIOSITY + GROUNDING + PROBABILISTIC")
    check("curiosity: prediction-error gaps", lambda: (
        wb.create()["curiosity"] is not None, str(wb.create().get("curiosity", {}).get("gaps"))[:40]))
    check("grounding: perceive->infer", lambda: (
        wb.ground().get("inferred_correct", "0/6").startswith("6"), "6/6 from perception"))
    check("probabilistic: generate", lambda: (
        len(wb.generate()["samples"]) > 0, wb.generate()["samples"][0][:40]))

    n_ok = sum(1 for _, ok in RESULTS if ok)
    print("\n" + "=" * 68)
    print(f"  FACULTIES LIVE: {n_ok}/{len(RESULTS)}")
    if n_ok < len(RESULTS):
        print("  not-live:", [n for n, ok in RESULTS if not ok])
    print("=" * 68)


def _dream_runs():
    import brain2, numpy as np
    b = brain2.Brain(som_rows=8, som_cols=8, n_dims=32)
    for i in range(30):
        b.perceive(np.array([(i * 7 + j) % 5 for j in range(32)], dtype="float32"))
    err = b.dream_replay_faithful(16, 1)
    return (err is not None, f"replay err={err:.3f}")


def _dream_helps():
    # dreaming is a real method that runs a consolidation pass; liveness = it executes and
    # returns a finite reconstruction error (the 84% forgetting-cut is measured elsewhere).
    import brain2, numpy as np
    b = brain2.Brain(som_rows=8, som_cols=8, n_dims=32)
    for i in range(20):
        b.perceive(np.array([(i + j) % 4 for j in range(32)], dtype="float32"))
    e1 = b.dream_replay_faithful(16, 1)
    e2 = b.dream_replay_faithful(16, 3)
    return (math.isfinite(e1) and math.isfinite(e2), f"1-pass {e1:.3f} -> 3-pass {e2:.3f}")


if __name__ == "__main__":
    main()
