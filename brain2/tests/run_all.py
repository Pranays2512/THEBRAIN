#!/usr/bin/env python3
"""
run_all.py — run every component test and summarize.

Each tests/test_*.py is a self-contained script that prints PASS/FAIL lines
and exits nonzero on failure. Tests listed in KNOWN_STALE target APIs that
were removed/changed (e.g. the old MDN-era Predictor.step(vec, vec)); they
are reported separately and don't fail the run until rewritten.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
TIMEOUT = 120

# Tests that exercise removed/changed APIs and need a rewrite (tracked work).
KNOWN_STALE = {
    "test_predictor.py",  # old MDN API: step(vec, target_vec); now LM-head step(vec, word_id)
}

# Not test scripts.
EXCLUDE = {"run_all.py", "generate_hardened_suite.py", "run_hardened_suite.py"}


def main():
    tests = sorted(f for f in os.listdir(HERE)
                   if f.startswith("test_") and f.endswith(".py")
                   and f not in EXCLUDE)
    passed, failed, stale = [], [], []

    for t in tests:
        try:
            r = subprocess.run([PYTHON, os.path.join(HERE, t)],
                               capture_output=True, text=True, timeout=TIMEOUT,
                               cwd=HERE)
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        if t in KNOWN_STALE:
            stale.append(t)
            status = "STALE"
        elif ok:
            passed.append(t)
            status = "PASS"
        else:
            failed.append(t)
            status = "FAIL"
        print(f"[{status:5s}] {t}")

    print(f"\n{len(passed)} passed, {len(failed)} failed, {len(stale)} known-stale "
          f"of {len(tests)} total")
    if failed:
        print("Failed:", ", ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
