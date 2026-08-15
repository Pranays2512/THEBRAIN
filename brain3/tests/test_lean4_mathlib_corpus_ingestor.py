#!/usr/bin/env python3
"""
brain3/tests/test_lean4_mathlib_corpus_ingestor.py

Unit tests for The Brain's Lean 4 / Mathlib Formal Corpus Ingestion Engine.
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
#include "crisp/engines/math/lean4_mathlib_corpus_ingestor.hpp"
#include "crisp/engines/math/universal_axiomatic_knowledge_vault.hpp"

using namespace thebrain;

int main() {
    lean4_ingestor::Lean4MathlibCorpusIngestor ingestor;
    knowledge_vault::UniversalAxiomaticKnowledgeVault vault;

    // 1. Verify default declarations loaded
    const auto& decls = ingestor.get_all_declarations();
    assert(decls.size() >= 8);
    std::cout << "[PASSED] Parsed " << decls.size() << " formal Mathlib declarations\\n";

    // 2. Query by module prefix
    auto sobolev_decls = ingestor.query_by_module("Mathlib.Analysis.Sobolev");
    assert(!sobolev_decls.empty());
    std::cout << "[PASSED] Found " << sobolev_decls.size() << " Sobolev declarations\\n";

    // 3. Query premises for goal
    auto zeta_decls = ingestor.query_premises_for_goal("zeta");
    assert(!zeta_decls.empty());
    std::cout << "[PASSED] Found " << zeta_decls.size() << " Zeta-related premise declarations\\n";

    // 4. Transfer to Universal Axiomatic Knowledge Vault
    auto stats = ingestor.transfer_to_vault(vault);
    assert(stats.theorems_ingested > 0);
    assert(stats.edges_created > 0);
    std::cout << "[PASSED] Transferred " << stats.theorems_ingested << " theorems and " 
              << stats.edges_created << " dependency edges to Vault in " << stats.ingestion_time_ms << " ms\\n";

    std::cout << "ALL LEAN4 MATHLIB INGESTION TESTS PASSED\\n";
    return 0;
}
"""

class TestLean4MathlibCorpusIngestor(unittest.TestCase):
    def test_ingestor_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_file = Path(tmpdir) / "test_ingestor.cpp"
            bin_file = Path(tmpdir) / "test_ingestor_bin"
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
            self.assertIn("ALL LEAN4 MATHLIB INGESTION TESTS PASSED", run_res.stdout)

if __name__ == "__main__":
    unittest.main()
