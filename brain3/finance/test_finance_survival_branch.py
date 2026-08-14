#!/usr/bin/env python3
"""
Comprehensive Unit & Integration Test Suite for The Brain 3 Quantitative Finance & Survival Instinct Branch
"""

import unittest
import subprocess
import json
import os
import sys
import time

class TestFinanceSurvivalBranch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.brain_finance_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "brain_finance"))
        cls.brain_master_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "../brain_master"))

        # Verify binaries exist
        if not os.path.exists(cls.brain_finance_bin):
            cmd = f"clang++ -std=c++17 -O3 -I{os.path.dirname(cls.brain_finance_bin)} {os.path.dirname(cls.brain_finance_bin)}/finance_orchestrator.cpp -o {cls.brain_finance_bin}"
            subprocess.run(cmd, shell=True, check=True)

    def _query_finance_json(self, command: str) -> dict:
        return self._query_session([command])[0]

    def _query_session(self, commands: list) -> list:
        proc = subprocess.Popen(
            [self.brain_finance_bin, "--json-stream"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        payload = "\n".join(commands) + "\nQUIT\n"
        stdout, _ = proc.communicate(input=payload)
        lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
        results = []
        for l in lines:
            try:
                results.append(json.loads(l))
            except Exception:
                pass
        return results

    def test_01_initial_survival_status(self):
        """Test initial life force is 100% and state is THRIVING."""
        res = self._query_finance_json("FINANCE_STATUS")
        self.assertEqual(res.get("survival_state"), "THRIVING")
        self.assertTrue(res.get("is_alive"))
        self.assertEqual(res.get("life_force_pct"), 100.0)
        self.assertEqual(res.get("current_equity"), 10000.0)
        self.assertEqual(res.get("peak_equity"), 10000.0)
        self.assertEqual(res.get("max_drawdown_pct"), 0.0)

    def test_02_limit_order_book_depth_and_spread(self):
        """Test L2 Order Book depth, bids/asks ladder, and spread."""
        res = self._query_finance_json("ORDER_BOOK BTC/USDT")
        self.assertEqual(res.get("symbol"), "BTC/USDT")
        self.assertGreater(res.get("mid_price"), 0.0)
        self.assertGreater(res.get("best_bid"), 0.0)
        self.assertGreater(res.get("best_ask"), 0.0)
        self.assertGreater(res.get("best_ask"), res.get("best_bid"))
        self.assertTrue(len(res.get("bids", [])) > 0)
        self.assertTrue(len(res.get("asks", [])) > 0)

    def test_03_market_order_execution_and_slippage(self):
        """Test submitting market buy and sell orders with slippage & fee accounting."""
        # 1. Market BUY order
        buy_res = self._query_finance_json("TRADE_ORDER BTC/USDT BUY MARKET 65000.0 0.5")
        self.assertEqual(buy_res.get("status"), "FILLED")
        self.assertEqual(buy_res.get("side"), "BUY")
        self.assertAlmostEqual(buy_res.get("executed_qty"), 0.5, delta=1e-5)
        self.assertGreater(buy_res.get("avg_fill_price"), 0.0)
        self.assertGreater(buy_res.get("fee"), 0.0)

        # 2. Market SELL order
        sell_res = self._query_finance_json("TRADE_ORDER BTC/USDT SELL MARKET 65000.0 0.5")
        self.assertEqual(sell_res.get("status"), "FILLED")
        self.assertEqual(sell_res.get("side"), "SELL")
        self.assertAlmostEqual(sell_res.get("executed_qty"), 0.5, delta=1e-5)

    def test_04_microstructure_telemetry(self):
        """Test real-time VWAP, OFI, and realized volatility calculation."""
        res = self._query_finance_json("MICROSTRUCTURE BTC/USDT")
        self.assertEqual(res.get("symbol"), "BTC/USDT")
        self.assertIn("vwap", res)
        self.assertIn("ofi", res)
        self.assertIn("realized_vol", res)
        self.assertIn("effective_spread_bps", res)

    def test_05_kelly_position_sizing_safe_allocation(self):
        """Test mathematical Kelly allocation with survival dampening."""
        res = self._query_finance_json("KELLY_SIZE 0.60 1.8")
        self.assertEqual(res.get("win_probability"), 0.60)
        self.assertEqual(res.get("win_loss_ratio"), 1.8)
        self.assertGreater(res.get("safe_allocation_dollars"), 0.0)
        self.assertLessEqual(res.get("safe_allocation_dollars"), 1000.0) # Within 10% max allocation limit

    def test_06_statistical_arbitrage_and_ou_drift(self):
        """Test cross-asset cointegration OLS beta and Ornstein-Uhlenbeck mean-reversion analysis."""
        res = self._query_finance_json("STAT_ARB_SCAN BTC/USDT ETH/USDT")
        self.assertEqual(res.get("asset_a"), "BTC/USDT")
        self.assertEqual(res.get("asset_b"), "ETH/USDT")
        self.assertIn("hedge_ratio_beta", res)
        self.assertIn("z_score", res)
        self.assertIn("ou_theta", res)
        self.assertIn("half_life_periods", res)
        self.assertIn(res.get("action"), ["BUY_A_SELL_B", "SELL_A_BUY_B", "CLOSE", "NONE"])

    def test_07_monte_carlo_market_simulation_cycle(self):
        """Test 50-tick Monte Carlo simulation with continuous order book matching and survival tracking."""
        res = self._query_finance_json("SIMULATE_MARKET_CYCLE BTC/USDT 50 0.0005 0.012")
        self.assertEqual(res.get("symbol"), "BTC/USDT")
        self.assertEqual(res.get("simulated_ticks"), 50)
        self.assertGreater(res.get("trades_executed"), 0)
        
        status = res.get("survival_status", {})
        self.assertTrue(status.get("is_alive"))
        self.assertGreater(status.get("total_trades"), 0)

    def test_08_acute_pain_reflex_and_brain_death_liquidation(self):
        """Test that severe drawdown triggers acute pain reflex, and dropping below ruin threshold causes Brain Death."""
        commands = [
            "RESET_LIFE_FORCE 10000.0",
            "INJECT_DRAWDOWN_PAIN 3000.0",
            "INJECT_DRAWDOWN_PAIN 3500.0",
            "INJECT_DRAWDOWN_PAIN 2000.0",
            "TRADE_ORDER BTC/USDT BUY MARKET 65000.0 1.0"
        ]
        responses = self._query_session(commands)
        self.assertEqual(len(responses), 5)

        # Step 1: Fresh State
        fresh = responses[0]
        self.assertEqual(fresh.get("survival_state"), "THRIVING")
        self.assertEqual(fresh.get("life_force_pct"), 100.0)

        # Step 2: Moderate Drawdown ($3,000 loss => Capital = $7,000)
        pain1 = responses[1]
        self.assertAlmostEqual(pain1.get("life_force_pct"), 62.5, delta=1.0)
        self.assertGreater(pain1.get("stress_level"), 0.2)

        # Step 3: Severe Drawdown ($3,500 additional loss => Capital = $3,500)
        pain2 = responses[2]
        self.assertAlmostEqual(pain2.get("life_force_pct"), 18.75, delta=1.0)
        self.assertEqual(pain2.get("survival_state"), "CRITICAL")

        # Step 4: Terminal Drawdown ($2,000 additional loss => Capital = $1,500 < $2,000 Ruin Threshold)
        pain3 = responses[3]
        self.assertEqual(pain3.get("survival_state"), "BRAIN_DEAD")
        self.assertFalse(pain3.get("is_alive"))
        self.assertEqual(pain3.get("life_force_pct"), 0.0)

        # Step 5: Verify subsequent trade is strictly REJECTED
        rej = responses[4]
        self.assertEqual(rej.get("status"), "REJECTED")
        self.assertEqual(rej.get("reason"), "AGENT_IS_BRAIN_DEAD")

    def test_09_master_cognitive_core_bql_integration(self):
        """Test that Master Cognitive Orchestrator (brain_master) correctly dispatches financial BQL queries."""
        if not os.path.exists(self.brain_master_bin):
            self.skipTest("brain_master binary not found")

        proc = subprocess.Popen(
            [self.brain_master_bin, "--json-stream"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            test_queries = [
                "FINANCE_STATUS",
                "ORDER_BOOK BTC/USDT",
                "KELLY_SIZE 0.58 1.6",
                "SIMULATE_MARKET_CYCLE BTC/USDT 20 0.0002 0.01"
            ]

            for q in test_queries:
                proc.stdin.write(f"{q}\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                self.assertTrue(len(line.strip()) > 0, f"No response from brain_master for query: {q}")
                resp_obj = json.loads(line)
                self.assertEqual(resp_obj.get("engine_used"), "finance_survival_branch")
                self.assertTrue(resp_obj.get("verified"))
                self.assertIn("raw_output", resp_obj)
        finally:
            try:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.communicate(timeout=1.0)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    unittest.main(verbosity=2)

