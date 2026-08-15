#!/usr/bin/env python3
"""
Comprehensive Unit & Integration Test Suite for The Brain Quantitative Finance & Survival Instinct Branch
"""

import unittest
import subprocess
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FINANCE_DIR = REPO_ROOT / "brain3" / "finance"
BIN_PATH = FINANCE_DIR / "brain_finance"

class TestFinanceSurvivalBranch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-Icore", "-I.",
            "-o", str(BIN_PATH),
            "finance_orchestrator.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(FINANCE_DIR), capture_output=True, text=True)
        assert res.returncode == 0, f"Compilation failed: {res.stderr}"

    def _query_finance_json(self, command: str) -> dict:
        proc = subprocess.Popen(
            [str(BIN_PATH), "--json-stream"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, _ = proc.communicate(input=f"{command}\nQUIT\n")
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if line and line.startswith("{"):
                try:
                    return json.loads(line)
                except Exception:
                    pass
        return {}

    def test_01_initial_survival_status(self):
        """Test initial life force is 50% (at ₹1000 baseline) and state is SURVIVING."""
        res = self._query_finance_json("FINANCE_STATUS")
        self.assertEqual(res.get("currency"), "INR")
        self.assertEqual(res.get("survival_state"), "SURVIVING")
        self.assertTrue(res.get("is_alive"))
        self.assertEqual(res.get("life_force_pct"), 50.0)
        self.assertEqual(res.get("current_equity"), 1000.0)
        self.assertEqual(res.get("ruin_floor"), -1000.0)
        self.assertEqual(res.get("cap_limit"), 100000.0)

    def test_02_limit_order_book_depth_and_spread(self):
        """Test L2 Order Book depth, bids/asks ladder, and spread."""
        res = self._query_finance_json("ORDER_BOOK NIFTY50/INR")
        self.assertEqual(res.get("symbol"), "NIFTY50/INR")
        self.assertGreater(res.get("mid_price"), 0.0)
        self.assertGreater(res.get("best_bid"), 0.0)
        self.assertGreater(res.get("best_ask"), 0.0)
        self.assertGreater(res.get("best_ask"), res.get("best_bid"))
        self.assertTrue(len(res.get("bids", [])) > 0)
        self.assertTrue(len(res.get("asks", [])) > 0)

    def test_03_sample_survival_trade_execution(self):
        """Test executing a sample survival trade on NIFTY50/INR."""
        res = self._query_finance_json("SAMPLE_SURVIVAL_TRADE NIFTY50/INR")
        self.assertEqual(res.get("symbol"), "NIFTY50/INR")
        self.assertIn(res.get("side"), ["BUY", "SELL"])
        self.assertGreater(res.get("entry_price"), 0.0)
        self.assertGreater(res.get("exit_price"), 0.0)
        self.assertGreater(res.get("quantity"), 0.0)
        self.assertIn("strategy", res)
        self.assertGreater(res.get("capital_after"), -1000.0)

    def test_04_microstructure_telemetry(self):
        """Test real-time VWAP, OFI, and realized volatility calculation."""
        res = self._query_finance_json("MICROSTRUCTURE NIFTY50/INR")
        self.assertEqual(res.get("symbol"), "NIFTY50/INR")
        self.assertIn("vwap", res)
        self.assertIn("ofi", res)
        self.assertIn("realized_vol", res)
        self.assertIn("effective_spread_bps", res)

    def test_05_kelly_position_sizing_safe_allocation(self):
        """Test mathematical Kelly allocation with survival dampening and ruin avoidance."""
        res = self._query_finance_json("KELLY_SIZE 0.60 1.8")
        self.assertEqual(res.get("win_probability"), 0.60)
        self.assertEqual(res.get("win_loss_ratio"), 1.8)
        self.assertGreater(res.get("safe_allocation_inr"), 0.0)
        # Margin above -1000 is 2000; allocation must respect max risk cap
        self.assertLessEqual(res.get("safe_allocation_inr"), 600.0)

    def test_06_statistical_arbitrage_scan(self):
        """Test cross-asset cointegration and Ornstein-Uhlenbeck mean-reversion scanner."""
        res = self._query_finance_json("STAT_ARB_SCAN BTC/USDT ETH/USDT")
        self.assertEqual(res.get("asset_a"), "BTC/USDT")
        self.assertEqual(res.get("asset_b"), "ETH/USDT")
        self.assertIn("hedge_ratio_beta", res)
        self.assertIn("z_score", res)
        self.assertIn("action", res)

    def test_07_metabolic_tick_burn(self):
        """Test biological metabolic upkeep burn reduces equity if no trades are made."""
        s0 = self._query_finance_json("FINANCE_STATUS")
        eq0 = s0["current_equity"]
        s1 = self._query_finance_json("METABOLIC_TICK")
        self.assertLess(s1["current_equity"], eq0)

    def test_08_autonomous_survival_cycle(self):
        """Test multi-tick autonomous trading cycle grows capital without breaching -₹1000 ruin floor."""
        res = self._query_finance_json("AUTONOMOUS_SURVIVAL_CYCLE 200")
        self.assertEqual(res.get("initial_capital"), 1000.0)
        self.assertTrue(res.get("survived_without_ruin"))
        self.assertGreater(res.get("total_trades"), 0)
        self.assertGreater(res.get("win_rate_pct"), 50.0)
        self.assertGreater(res.get("profit_factor"), 1.0)
        self.assertGreater(res.get("final_capital"), -1000.0)

if __name__ == "__main__":
    unittest.main()
