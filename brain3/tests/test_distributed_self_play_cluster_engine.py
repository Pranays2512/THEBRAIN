#!/usr/bin/env python3
"""
brain3/tests/test_distributed_self_play_cluster_engine.py

Unit tests for The Brain's Distributed 24/7 Self-Play Cluster Engine.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"

CPP_TEST_HARNESS = """
#include <iostream>
#include <cassert>
#include "core/distributed_self_play_cluster_engine.hpp"
#include "crisp/engines/math/universal_axiomatic_knowledge_vault.hpp"

using namespace thebrain;

int main() {
    knowledge_vault::UniversalAxiomaticKnowledgeVault vault;
    distributed_self_play::DistributedSelfPlayClusterEngine cluster(vault);

    // 1. Run full distributed discovery cycle
    auto metrics = cluster.run_discovery_cycle(6);

    assert(metrics.total_conjectures_generated == 6);
    assert(metrics.counterexamples_found > 0);
    assert(metrics.surviving_conjectures > 0);
    assert(metrics.theorems_proven > 0);
    assert(metrics.lemmas_committed_to_vault > 0);

    std::cout << "[PASSED] Discovery Cycle Completed in " << metrics.total_runtime_ms << " ms\\n";
    std::cout << "   • Conjectures Generated : " << metrics.total_conjectures_generated << "\\n";
    std::cout << "   • Counterexamples Found : " << metrics.counterexamples_found << "\\n";
    std::cout << "   • Surviving Conjectures : " << metrics.surviving_conjectures << "\\n";
    std::cout << "   • Theorems Proven       : " << metrics.theorems_proven << "\\n";
    std::cout << "   • Committed to Vault    : " << metrics.lemmas_committed_to_vault << "\\n";

    // 2. Verify vault nodes were registered
    assert(vault.size() >= metrics.lemmas_committed_to_vault);
    std::cout << "[PASSED] Verified new lemmas stored inside Knowledge Vault DAG\\n";

    std::cout << "ALL DISTRIBUTED SELF-PLAY CLUSTER TESTS PASSED\\n";
    return 0;
}
"""

class TestDistributedSelfPlayClusterEngine(unittest.TestCase):
    def test_cluster_engine_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_file = Path(tmpdir) / "test_cluster.cpp"
            bin_file = Path(tmpdir) / "test_cluster_bin"
            cpp_file.write_text(CPP_TEST_HARNESS)

            cmd = [
                "clang++", "-std=c++17", "-O3",
                "-I.", "-I..", "-I../brain2",
                "-Wno-deprecated-declarations",
                "-o", str(bin_file),
                str(cpp_file)
            ]
            res = subprocess.run(cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Compilation failed: {res.stderr}")

            run_res = subprocess.run([str(bin_file)], cwd=str(BRAIN3_DIR), capture_output=True, text=True)
            self.assertEqual(run_res.returncode, 0, f"Execution failed: {run_res.stderr}")
            self.assertIn("ALL DISTRIBUTED SELF-PLAY CLUSTER TESTS PASSED", run_res.stdout)

if __name__ == "__main__":
    unittest.main()
