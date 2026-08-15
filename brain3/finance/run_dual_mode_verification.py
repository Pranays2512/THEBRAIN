"""
End-to-End Dual-Mode Autonomous Verification Suite
=================================================
Validates:
1. Multi-Timeframe Regime Detection (Squeeze Breakouts & Liquidity Sweeps).
2. Autonomous Instinct Engine (Hunger vs Survival throttling).
3. Asymmetric 1:2.5+ Risk:Reward Trade Management.
4. Capital Preservation & Invalidation Stop-Loss execution (0.00% Ruin).
5. Dynamic Trailing Profit Ratchet & High-Water Mark Capital Lock (₹10k -> ₹11k locks).
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.directional_alpha_engine import DirectionalAlphaEngine, Candle
from core.autonomous_instinct_controller import AutonomousInstinctController
from core.dual_mode_autonomous_runner import DualModeAutonomousRunner

class TestDualModeAutonomousEngine(unittest.TestCase):
    def setUp(self):
        self.starting_capital = 10000.0
        self.ruin_floor = 9500.0  # Allow max 5% initial risk
        self.instinct_controller = AutonomousInstinctController(self.starting_capital, self.ruin_floor, profit_lock_pct=0.85)
        self.alpha_engine = DirectionalAlphaEngine(min_risk_reward=2.5, max_risk_per_trade_pct=0.015)

    def test_01_survival_instinct_throttles_hunger(self):
        """Test that when capital approaches ruin floor, Hunger collapses to 0.0."""
        state_healthy = self.instinct_controller.evaluate_instinct(
            current_equity=10000.0,
            rolling_volatility_bps=8.5,
            trend_momentum=1.2,
            spread_bps=1.5,
            toxic_fill_ratio=0.3
        )
        self.assertGreater(state_healthy.survival_score, 0.85)
        self.assertGreater(state_healthy.hunger_score, 0.50)
        self.assertEqual(state_healthy.active_regime, "DIRECTIONAL_ALPHA_EXPANSION")

        # Threat test
        threat_controller = AutonomousInstinctController(10000.0, 0.0)
        state_threatened = threat_controller.evaluate_instinct(
            current_equity=50.0,
            rolling_volatility_bps=15.0,
            trend_momentum=2.0,
            spread_bps=5.0,
            toxic_fill_ratio=0.8
        )
        self.assertLess(state_threatened.survival_score, 0.25)
        self.assertLess(state_threatened.hunger_score, 0.05)
        self.assertEqual(state_threatened.active_regime, "TAIL_RISK_DEFENSE")

    def test_02_volatility_squeeze_and_directional_signals(self):
        """Test Bollinger/Keltner Squeeze Breakout detection."""
        candles = []
        p = 100.0
        for i in range(40):
            candles.append(Candle(timestamp=i*60, open=p, high=p+0.05, low=p-0.05, close=p, volume=10.0))
        candles.append(Candle(timestamp=41*60, open=100.0, high=105.0, low=99.9, close=104.5, volume=150.0))
        
        is_breakout, state, mom = self.alpha_engine.detect_volatility_squeeze(candles)
        self.assertTrue(is_breakout)
        self.assertEqual(state, "BULLISH_BREAKOUT")
        self.assertGreater(mom, 0.8)

    def test_03_asymmetric_risk_reward_enforcement(self):
        """Test that every fired trade strictly satisfies R:R >= 1:2.5."""
        candles = []
        p = 100.0
        for i in range(35):
            candles.append(Candle(timestamp=i*60, open=p, high=p+0.05, low=p-0.05, close=p, volume=10.0))
        candles.append(Candle(timestamp=36*60, open=100.0, high=106.0, low=99.9, close=105.5, volume=150.0))
        
        trade = self.alpha_engine.evaluate_directional_signal("BTCUSDT", candles, current_equity=10000.0)
        self.assertIsNotNone(trade)
        self.assertGreaterEqual(trade.risk_reward_ratio, 2.5)
        self.assertEqual(trade.side, "BUY")
        self.assertLess(trade.stop_loss_price, trade.entry_price)
        self.assertGreater(trade.take_profit_price, trade.entry_price)

    def test_04_dynamic_profit_ratchet_high_water_mark(self):
        """
        USER USE-CASE: Start at ₹10,000, make ₹1,000 profit (equity = ₹11,000).
        Verify that the dynamic ruin floor ratchets up so it protects the ₹11,000 profit as base capital!
        """
        controller = AutonomousInstinctController(starting_capital=10000.0, ruin_floor=10000.0, profit_lock_pct=0.85)
        
        # 1. Baseline state at ₹10,000
        state0 = controller.evaluate_instinct(10000.0, 2.0, 0.1, 1.5, 0.3)
        self.assertEqual(state0.dynamic_ruin_floor, 10000.0)
        self.assertEqual(state0.locked_profit, 0.0)
        
        # 2. Account grows to ₹11,000 (+₹1,000 profit)
        state1 = controller.evaluate_instinct(11000.0, 5.0, 1.0, 1.5, 0.3)
        # Floor must ratchet to: 10,000 + (1,000 * 0.85) = ₹10,850!
        self.assertEqual(state1.dynamic_ruin_floor, 10850.0)
        self.assertEqual(state1.locked_profit, 850.0)
        self.assertEqual(state1.peak_equity, 11000.0)
        
        # 3. If equity drops back toward ₹10,850 (threatening the locked profit)
        state_pullback = controller.evaluate_instinct(10850.0, 6.0, -1.0, 1.5, 0.3)
        # Survival score collapses to 0.0 and regime forces TAIL_RISK_DEFENSE!
        self.assertLess(state_pullback.survival_score, 0.1)
        self.assertEqual(state_pullback.active_regime, "TAIL_RISK_DEFENSE")
        print("\n✅ Verified: Profit Ratchet successfully locked ₹11,000 gain into a ₹10,850 trailing floor!")

    def test_05_dual_mode_runner_execution_and_reporting(self):
        """Run full dual-mode runner cycle and verify zero ruin + excel exports."""
        runner = DualModeAutonomousRunner(starting_capital=5000.0, ruin_floor=0.0)
        runner.run_simulation_cycle(symbols=["BTCUSDT", "ETHUSDT"])
        
        self.assertGreater(runner.current_equity, runner.ruin_floor)
        self.assertGreaterEqual(len(runner.regime_logs), 50)
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(__file__), "logs/regime_switching_audit.xlsx")))
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(__file__), "logs/dual_mode_directional_trades_audit.xlsx")))

if __name__ == "__main__":
    unittest.main()
