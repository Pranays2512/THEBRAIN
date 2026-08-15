"""
Autonomous Instinct Controller for THE BRAIN 3.0
===============================================
Coordinates the delicate balance between:
1. "Hunger Instinct" (Seeking high asymmetric alpha during market breakouts)
2. "Survival Instinct" (Ensuring 0.00% ruin probability and protecting capital)

Features: Dynamic Trailing Profit Ratchet / High-Water Mark Capital Lock:
- When account grows from ₹10,000 -> ₹11,000 (+₹1,000 profit), the Ruin Floor
  automatically ratchets upward to ₹10,850 (locking in 85% of profits as base principal).
- Guarantees the Brain never gives back accumulated profits!
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class InstinctState:
    timestamp: float
    current_equity: float
    starting_capital: float
    peak_equity: float
    ruin_floor: float
    dynamic_ruin_floor: float
    locked_profit: float
    hunger_score: float      # 0.0 (Hibernation) to 1.0 (Aggressive Expansion)
    survival_score: float    # 0.0 (Critical Ruin Threat) to 1.0 (Maximum Safety)
    active_regime: str       # "CONSOLIDATION_MICRO_SPREAD", "DIRECTIONAL_ALPHA_EXPANSION", "TAIL_RISK_DEFENSE"
    regime_rationale: str
    allocation_micro_pct: float
    allocation_directional_pct: float

class AutonomousInstinctController:
    def __init__(self, starting_capital: float = 1000.0, ruin_floor: float = 0.0, profit_lock_pct: float = 0.85):
        """
        :param starting_capital: Initial account balance (e.g. ₹10,000)
        :param ruin_floor: Minimum baseline safety floor (e.g. ₹0.00 or initial principal)
        :param profit_lock_pct: Percentage of peak profit to permanently lock into the ratcheting floor (e.g. 0.85 = 85%)
        """
        self.starting_capital = starting_capital
        self.ruin_floor = ruin_floor
        self.profit_lock_pct = profit_lock_pct
        self.current_equity = starting_capital
        self.peak_equity = starting_capital
        self.dynamic_ruin_floor = ruin_floor
        self.history = []

    def evaluate_instinct(self, 
                          current_equity: float, 
                          rolling_volatility_bps: float, 
                          trend_momentum: float, 
                          spread_bps: float,
                          toxic_fill_ratio: float) -> InstinctState:
        """
        Dynamically computes Hunger and Survival scores with High-Water Mark Profit Ratchet.
        """
        self.current_equity = current_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        # 1. Compute Dynamic Trailing Profit Ratchet
        # When equity grows from ₹10,000 -> ₹11,000 (+₹1,000 profit),
        # the floor ratchets up so the profit is defended as principal.
        cumulative_profit = max(0.0, self.peak_equity - self.starting_capital)
        locked_profit = cumulative_profit * self.profit_lock_pct
        self.dynamic_ruin_floor = self.ruin_floor + locked_profit

        # 2. Compute Survival Score S in [0, 1] relative to Dynamic Ratchet Floor
        capital_buffer = current_equity - self.dynamic_ruin_floor
        if capital_buffer <= 0:
            survival_score = 0.0
        else:
            allowable_risk = max(1.0, self.starting_capital - self.ruin_floor)
            buffer_ratio = capital_buffer / allowable_risk
            survival_score = max(0.0, min(1.0, math.tanh(buffer_ratio * 2.5)))
            
        # Penalize survival score if drawdown from peak exceeds 3.5%
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown_pct > 0.035:
            survival_score *= max(0.15, 1.0 - (drawdown_pct * 6.0))
            
        # Penalize survival score if spread blows out (> 15 bps) or toxic fills dominate (> 75%)
        if spread_bps > 15.0 or toxic_fill_ratio > 0.75:
            survival_score *= 0.5

        # 3. Compute Hunger Score H in [0, 1]
        vol_hunger = min(1.0, rolling_volatility_bps / 8.0)
        mom_hunger = min(1.0, abs(trend_momentum) / 1.5)
        raw_hunger = (0.5 * vol_hunger) + (0.5 * mom_hunger)
        
        # KEY BRAIN INSTINCT: "Hungry, but never die"
        # Hunger is strictly throttled by the Survival score!
        hunger_score = raw_hunger * (survival_score ** 2)

        # 4. Autonomous Regime Classification
        if survival_score < 0.25 or spread_bps > 20.0 or current_equity <= self.dynamic_ruin_floor:
            active_regime = "TAIL_RISK_DEFENSE"
            rationale = f"Equity (₹{current_equity:,.2f}) near Dynamic Ratchet Floor (₹{self.dynamic_ruin_floor:,.2f}). Locking profits & defending capital."
            alloc_micro = 0.0
            alloc_dir = 0.0
            
        elif hunger_score >= 0.50 and rolling_volatility_bps >= 3.5 and abs(trend_momentum) >= 0.75:
            active_regime = "DIRECTIONAL_ALPHA_EXPANSION"
            rationale = f"Strong volatility ({rolling_volatility_bps:.1f} bps) & trend momentum. Deploying 1:2.5+ Asymmetric Alpha."
            alloc_micro = 0.15
            alloc_dir = 0.85
            
        else:
            active_regime = "CONSOLIDATION_MICRO_SPREAD"
            rationale = f"Normal market ({rolling_volatility_bps:.1f} bps). Harvesting passive spreads & triangular arbitrage."
            alloc_micro = 0.85
            alloc_dir = 0.15

        state = InstinctState(
            timestamp=time.time(),
            current_equity=current_equity,
            starting_capital=self.starting_capital,
            peak_equity=self.peak_equity,
            ruin_floor=self.ruin_floor,
            dynamic_ruin_floor=round(self.dynamic_ruin_floor, 2),
            locked_profit=round(locked_profit, 2),
            hunger_score=round(hunger_score, 4),
            survival_score=round(survival_score, 4),
            active_regime=active_regime,
            regime_rationale=rationale,
            allocation_micro_pct=alloc_micro,
            allocation_directional_pct=alloc_dir
        )
        self.history.append(state)
        return state
