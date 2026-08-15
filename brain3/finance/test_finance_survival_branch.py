#!/usr/bin/env python3
"""
Comprehensive Unit & Integration Test Suite for The Brain Quantitative Finance & Survival Instinct Branch
"""

import unittest
import subprocess
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FINANCE_DIR = REPO_ROOT / "brain3" / "finance"
BIN_PATH = FINANCE_DIR / "brain_finance"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brain3.finance.adapters.real_market_feed import RealMarketFeedAdapter, LiveMarketTick

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

    def _query_multi_commands(self, commands: list) -> list:
        proc = subprocess.Popen(
            [str(BIN_PATH), "--json-stream"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        input_data = "\n".join(commands) + "\nQUIT\n"
        stdout, _ = proc.communicate(input=input_data)
        results = []
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if line and line.startswith("{"):
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
        return results

    def _query_finance_json(self, command: str) -> dict:
        res = self._query_multi_commands([command])
        return res[0] if res else {}

    def test_01_initial_survival_status(self):
        """Test initial life force is 50% (at ₹1000 baseline) and state is SURVIVING."""
        res = self._query_finance_json("FINANCE_STATUS")
        self.assertEqual(res.get("currency"), "INR")
        self.assertEqual(res.get("survival_state"), "SURVIVING")
        self.assertTrue(res.get("is_alive"))
        self.assertEqual(res.get("life_force_pct"), 50.0)
        self.assertEqual(res.get("current_equity"), 1000.0)
        self.assertEqual(res.get("ruin_floor"), -100.0)
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
        self.assertGreater(res.get("capital_after"), -100.0)

    def test_04_microstructure_telemetry(self):
        """Test real-time VWAP, OFI, and realized volatility calculation."""
        res = self._query_finance_json("MICROSTRUCTURE NIFTY50/INR")
        self.assertEqual(res.get("symbol"), "NIFTY50/INR")
        self.assertIn("vwap", res)
        self.assertIn("ofi", res)
        self.assertIn("realized_vol", res)
        self.assertIn("effective_spread_bps", res)

    def test_05_kelly_position_sizing_safe_allocation(self):
        """Test mathematical Kelly allocation with survival dampening and strict -₹100 ruin avoidance."""
        res = self._query_finance_json("KELLY_SIZE 0.60 1.8")
        self.assertEqual(res.get("win_probability"), 0.60)
        self.assertEqual(res.get("win_loss_ratio"), 1.8)
        self.assertGreater(res.get("safe_allocation_inr"), 0.0)
        # Margin above -100 is 1100; allocation must respect max risk cap
        self.assertLessEqual(res.get("safe_allocation_inr"), 500.0)

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

    def test_08_live_tick_execution_with_real_market_feed(self):
        """Test ingesting live tick data and executing against real market spreads."""
        feed = RealMarketFeedAdapter()
        tick = feed.get_live_tick("USD/INR")
        self.assertIsNotNone(tick)
        self.assertGreater(tick.price, 0.0)

        cmd = f"LIVE_TICK_EXEC {tick.symbol} {tick.price:.4f} {tick.best_bid:.4f} {tick.best_ask:.4f} {tick.volume:.2f}"
        res = self._query_finance_json(cmd)
        self.assertIn("status", res)
        self.assertIn(res["status"], ["TRADE_EXECUTED", "NO_TRADE", "DEFENSIVE_HOLD"])

    def test_09_strict_ruin_floor_killswitch(self):
        """Test that dropping below strict -₹100 ruin floor triggers immediate BRAIN_DEAD state."""
        # Inject severe loss to drop below -100 in the same session
        res = self._query_multi_commands([
            "INJECT_DRAWDOWN_PAIN 1200",
            "TRADE_ORDER BTC/INR BUY MARKET 6000000 0.01"
        ])
        self.assertEqual(len(res), 2)
        s_dead = res[0]
        self.assertEqual(s_dead.get("survival_state"), "BRAIN_DEAD")
        self.assertFalse(s_dead.get("is_alive"))
        self.assertEqual(s_dead.get("life_force_pct"), 0.0)

        # Attempt trade when dead - must be rejected
        tr = res[1]
        self.assertEqual(tr.get("status"), "REJECTED")
        self.assertEqual(tr.get("reason"), "AGENT_IS_BRAIN_DEAD")

    def test_10_autonomous_survival_cycle(self):
        """Test multi-tick autonomous trading cycle grows capital without breaching -₹100 ruin floor."""
        # Reset engine to fresh ₹1000
        self._query_finance_json("RESET_LIFE_FORCE 1000.0")
        res = self._query_finance_json("AUTONOMOUS_SURVIVAL_CYCLE 200")
        self.assertEqual(res.get("initial_capital"), 1000.0)
        self.assertTrue(res.get("survived_without_ruin"))
        self.assertGreater(res.get("total_trades"), 0)
        self.assertGreater(res.get("win_rate_pct"), 50.0)
        self.assertGreater(res.get("profit_factor"), 1.0)
        self.assertGreater(res.get("final_capital"), -100.0)

    def test_11_multi_stream_market_feed_and_alpha_scanner(self):
        """Test concurrent multi-asset ingestion and high-conviction alpha trade execution."""
        from brain3.finance.adapters.multi_stream_market_feed import MultiStreamMarketFeed
        feed = MultiStreamMarketFeed()
        feed.start()
        time.sleep(1.0)
        snapshot = feed.get_market_snapshot()
        self.assertGreater(len(snapshot), 5)

        # Test MULTI_ASSET_TICK command in C++ engine
        item = list(snapshot.values())[0]
        cmd = f"MULTI_ASSET_TICK {item.symbol} {item.price:.4f} {item.best_bid:.4f} {item.best_ask:.4f} {item.volume:.2f} {item.change_24h_pct:.2f}"
        res = self._query_finance_json(cmd)
        self.assertIn("status", res)
        self.assertIn(res["status"], ["MULTI_TRADE_EXECUTED", "MONITORING"])
        feed.stop()

if __name__ == "__main__":
    unittest.main()
