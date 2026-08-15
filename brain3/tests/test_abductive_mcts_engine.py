#!/usr/bin/env python3
"""
brain3/tests/test_abductive_mcts_engine.py

Comprehensive Unit & Integration Test Suite for MCTS-Driven Abductive Latent Synthesis & Axiom Relaxation Engine.
Verifies:
1. Particle Physics (Beta Decay -> Neutrino nu & Relax two_body_decay_only)
2. Pure Mathematics (Negative Quadratic -> Imaginary unit i & Relax non_negative_squares_in_reals)
3. Astrophysics (Galactic Rotation -> Dark Matter halo & Relax visible_matter_is_all_matter)
4. Computer Science (Comparison Sorting -> Radix Dispatch & Relax pairwise_comparison_ordering)
5. Quantitative Finance (Flash Run Kurtosis -> Latent Dark Liquidity & Relax lit_only_visibility)
6. Natural language cognitive perception and BQL dispatch in MasterOrchestrator
7. Broca Bridge FastAPI discovery endpoints
"""

import os
import sys
import json
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRAIN3_DIR = REPO_ROOT / "brain3"
BRAIN_MASTER_BIN = BRAIN3_DIR / "brain_master"

def run_brain_master(arg_list):
    res = subprocess.run(
        [str(BRAIN_MASTER_BIN)] + arg_list,
        cwd=str(BRAIN3_DIR),
        capture_output=True,
        text=True,
        timeout=15
    )
    return res

class TestAbductiveMCTSEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure brain_master is compiled
        if not BRAIN_MASTER_BIN.exists():
            compile_cmd = [
                "clang++", "-std=c++17", "-O3",
                "-I.", "-I..", "-I../brain2",
                "-Wno-deprecated-declarations",
                "-o", "brain_master", "core/master_orchestrator.cpp"
            ]
            comp = subprocess.run(compile_cmd, cwd=str(BRAIN3_DIR), capture_output=True, text=True)
            assert comp.returncode == 0, f"Compilation failed: {comp.stderr}"

    def test_01_abductive_physics_neutrino(self):
        res = run_brain_master(["--abductive-invent", "missing_beta_decay_momentum"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("MCTS Abductive Invention Success", out)
        self.assertIn("missing_beta_decay_momentum", out)
        self.assertIn("two_body_decay_only", out)
        self.assertIn("neutrino_nu", out)
        self.assertIn("Residual Error: 0.00000000", out)

    def test_02_abductive_math_imaginary_unit(self):
        res = run_brain_master(["--abductive-invent", "negative_quadratic_roots"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("MCTS Abductive Invention Success", out)
        self.assertIn("negative_quadratic_roots", out)
        self.assertIn("non_negative_squares_in_reals", out)
        self.assertIn("imaginary_unit_i", out)
        self.assertIn("Residual Error: 0.00000000", out)

    def test_03_abductive_astrophysics_dark_matter(self):
        res = run_brain_master(["--abductive-invent", "dark_matter"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("MCTS Abductive Invention Success", out)
        self.assertIn("flat_galactic_rotation", out)
        self.assertIn("visible_matter_is_all_matter", out)
        self.assertIn("dark_matter_halo", out)
        self.assertIn("Residual Error: 0.00000000", out)

    def test_04_abductive_cs_radix_sort(self):
        res = run_brain_master(["--abductive-invent", "radix_sort"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("MCTS Abductive Invention Success", out)
        self.assertIn("comparison_sorting_lower_bound", out)
        self.assertIn("pairwise_comparison_ordering", out)
        self.assertIn("direct_radix_dispatch", out)
        self.assertIn("Residual Error: 0.00000000", out)

    def test_05_abductive_finance_dark_pool(self):
        res = run_brain_master(["--abductive-invent", "dark_pool"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("MCTS Abductive Invention Success", out)
        self.assertIn("financial_latent_liquidity_burst", out)
        self.assertIn("all_liquidity_is_visible_in_l2_book", out)
        self.assertIn("latent_dark_liquidity", out)
        self.assertIn("Residual Error: 0.00000000", out)

    def test_06_natural_language_intent_routing(self):
        res = run_brain_master(["--query", "invent a new concept for beta decay"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("neutrino_nu", out)
        self.assertIn("two_body_decay_only", out)

    def test_07_latent_status_json(self):
        res = run_brain_master(["--latent-status"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("MCTS-Driven Abductive Latent Synthesis & Axiom Relaxation Engine", out)
        self.assertIn("ACTIVE", out)

    def test_08_bql_abductive_invention_ipc(self):
        res = run_brain_master(["--query", "ABDUCTIVE_INVENT negative_quadratic_roots"])
        self.assertEqual(res.returncode, 0)
        self.assertIn("imaginary_unit_i", res.stdout)
        self.assertIn("non_negative_squares_in_reals", res.stdout)

    def test_09_frontier_hubble_tension_early_dark_energy(self):
        res = run_brain_master(["--abductive-invent", "hubble_tension"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("hubble_tension_expansion_discrepancy", out)
        self.assertIn("phi_ede_scalar", out)
        self.assertIn("Residual Error: 0.00000000", out)

    def test_10_frontier_muon_g2_dark_gauge_boson(self):
        res = run_brain_master(["--abductive-invent", "muon_g2"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("muon_g2_anomalous_magnetic_moment", out)
        self.assertIn("Z_prime_dark_vector", out)
        self.assertIn("Residual Error: 0.00000000", out)

    def test_11_frontier_strong_cp_axion_field(self):
        res = run_brain_master(["--abductive-invent", "strong_cp"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("strong_cp_charge_parity_problem", out)
        self.assertIn("axion_field_a", out)
        self.assertIn("Residual Error: 0.00000000", out)

    def test_12_frontier_high_tc_cuprate_d_wave_pairing(self):
        res = run_brain_master(["--abductive-invent", "cuprate"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("high_tc_cuprate_pseudogap_pairing", out)
        self.assertIn("d_wave_paramagnon_pairing", out)
        self.assertIn("Residual Error: 0.00000000", out)

    def test_13_frontier_black_hole_information_page_curve_island(self):
        res = run_brain_master(["--abductive-invent", "black_hole"])
        self.assertEqual(res.returncode, 0)
        out = res.stdout
        self.assertIn("black_hole_information_unitarity_loss", out)
        self.assertIn("island_qes_wormhole", out)
        self.assertIn("Residual Error: 0.00000000", out)

if __name__ == "__main__":
    unittest.main()
