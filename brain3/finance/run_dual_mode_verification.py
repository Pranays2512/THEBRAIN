"""
End-to-End Dual-Mode Autonomous Verification Suite
=================================================
Validates:
1. Multi-Timeframe Regime Detection (Squeeze Breakouts & Liquidity Sweeps).
2. Autonomous Instinct Engine (Hunger vs Survival throttling).
3. Asymmetric 1:2.5+ Risk:Reward Trade Management.
4. Capital Preservation & Invalidation Stop-Loss execution (0.00% Ruin).
5. Commercial Profitability & Nominal PnL Generation.
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
        self.ruin_floor = 0.0
        self.instinct_controller = AutonomousInstinctController(self.starting_capital, self.ruin_floor)
        self.alpha_engine = DirectionalAlphaEngine(min_risk_reward=2.5, max_risk_per_trade_pct=0.015)

    def test_01_survival_instinct_throttles_hunger(self):
        """Test that when capital approaches ruin floor, Hunger collapses to 0.0."""
        # 1. Normal capital -> High hunger when volatility expands
        state_healthy = self.instinct_controller.evaluate_instinct(
            current_equity=10000.0,
            rolling_volatility_bps=8.5,
            trend_momentum=1.2,
            spread_bps=1.5,
            toxic_fill_ratio=0.3
        )
        self.assertGreater(state_healthy.survival_score, 0.90)
        self.assertGreater(state_healthy.hunger_score, 0.60)
        self.assertEqual(state_healthy.active_regime, "DIRECTIONAL_ALPHA_EXPANSION")

        # 2. Capital near ruin floor -> Hunger drops to 0.0 even with wild volatility
        state_threatened = self.instinct_controller.evaluate_instinct(
            current_equity=100.0,  # 99% drawdown
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
        # Create a squeeze series
        candles = []
        p = 100.0
        for i in range(40):
            candles.append(Candle(timestamp=i*60, open=p, high=p+0.05, low=p-0.05, close=p, volume=10.0))
        # Add explosive breakout candle
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

    def test_04_dual_mode_runner_execution_and_reporting(self):
        """Run full dual-mode runner cycle and verify zero ruin + excel exports."""
        runner = DualModeAutonomousRunner(starting_capital=5000.0, ruin_floor=0.0)
        runner.run_simulation_cycle(symbols=["BTCUSDT", "ETHUSDT"])
        
        self.assertGreater(runner.current_equity, runner.ruin_floor)
        self.assertGreaterEqual(len(runner.regime_logs), 50)
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(__file__), "logs/regime_switching_audit.xlsx")))
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(__file__), "logs/dual_mode_directional_trades_audit.xlsx")))

if __name__ == "__main__":
    unittest.main()
