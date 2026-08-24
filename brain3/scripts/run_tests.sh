#!/usr/bin/env bash
# run_tests.sh — the ONLY sanctioned way to run the brain3 test suite.
#
# Rationale: ctest happily executes STALE binaries from old builds and
# reports PASS against code that no longer compiles (this exact failure
# shipped: test_phase6_synth was broken for weeks while ctest stayed green).
# Always build fresh, then test.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== [1/2] Fresh full build (warnings visible, errors fatal) =="
cmake -S . -B build_cmake >/dev/null
cmake --build build_cmake -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo "== [2/2] ctest =="
ctest --test-dir build_cmake --output-on-failure "$@"
