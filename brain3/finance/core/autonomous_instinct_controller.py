"""
Autonomous Instinct Controller for THE BRAIN 3.0
===============================================
Coordinates the delicate balance between:
1. "Hunger Instinct" (Seeking high asymmetric alpha during market breakouts)
2. "Survival Instinct" (Ensuring 0.00% ruin probability and protecting capital)

Autonomously classifies market regimes and routes capital between:
- Strategy Mode A: Micro-Spread Maker Spread Capture & Triangular Arbitrage (Chop/Consolidation)
- Strategy Mode B: Directional Alpha Volatility Breakouts & Liquidity Sweeps (Trend/Breakout)
- Strategy Mode C: Emergency Capital Defense & Circuit Breaker (Tail Risk)
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
    ruin_floor: float
    hunger_score: float      # 0.0 (Hibernation) to 1.0 (Aggressive Expansion)
    survival_score: float    # 0.0 (Critical Ruin Threat) to 1.0 (Maximum Safety)
    active_regime: str       # "CONSOLIDATION_MICRO_SPREAD", "DIRECTIONAL_ALPHA_EXPANSION", "TAIL_RISK_DEFENSE"
    regime_rationale: str
    allocation_micro_pct: float
    allocation_directional_pct: float

class AutonomousInstinctController:
    def __init__(self, starting_capital: float = 1000.0, ruin_floor: float = 0.0):
        self.starting_capital = starting_capital
        self.ruin_floor = ruin_floor
        self.current_equity = starting_capital
        self.peak_equity = starting_capital
        self.history = []

    def evaluate_instinct(self, 
                          current_equity: float, 
                          rolling_volatility_bps: float, 
                          trend_momentum: float, 
                          spread_bps: float,
                          toxic_fill_ratio: float) -> InstinctState:
        """
        Dynamically computes Hunger and Survival scores, determining which strategy executes.
        """
        self.current_equity = current_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        # 1. Compute Survival Score S in [0, 1]
        # Distance from ruin floor
        capital_buffer = current_equity - self.ruin_floor
        if capital_buffer <= 0:
            survival_score = 0.0
        else:
            # Buffer relative to starting capital
            buffer_ratio = capital_buffer / self.starting_capital
            survival_score = max(0.0, min(1.0, math.tanh(buffer_ratio * 2.0)))
            
        # Penalize survival score if drawdown from peak exceeds 5%
        drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown_pct > 0.05:
            survival_score *= max(0.2, 1.0 - (drawdown_pct * 5.0))
            
        # Penalize survival score if spread blows out (> 15 bps) or toxic fills dominate (> 75%)
        if spread_bps > 15.0 or toxic_fill_ratio > 0.75:
            survival_score *= 0.5

        # 2. Compute Hunger Score H in [0, 1]
        # Base hunger driven by volatility expansion and trend momentum
        vol_hunger = min(1.0, rolling_volatility_bps / 8.0)  # Volatility > 8 bps fuels hunger
        mom_hunger = min(1.0, abs(trend_momentum) / 1.5)    # Strong trend momentum fuels hunger
        
        raw_hunger = (0.5 * vol_hunger) + (0.5 * mom_hunger)
        
        # KEY BRAIN INSTINCT: "Hungry, but never die"
        # Hunger is strictly throttled by the Survival score!
        # If survival score drops, hunger collapses to 0 regardless of market excitement.
        hunger_score = raw_hunger * (survival_score ** 2)

        # 3. Autonomous Regime Classification
        if survival_score < 0.25 or spread_bps > 20.0:
            active_regime = "TAIL_RISK_DEFENSE"
            rationale = "High market toxicity / Capital near ruin floor. Defending equity."
            alloc_micro = 0.0
            alloc_dir = 0.0
            
        elif hunger_score >= 0.50 and rolling_volatility_bps >= 3.5 and abs(trend_momentum) >= 0.75:
            active_regime = "DIRECTIONAL_ALPHA_EXPANSION"
            rationale = f"Strong volatility expansion ({rolling_volatility_bps:.1f} bps) & trend momentum. Deploying 1:2.5+ Asymmetric Alpha."
            alloc_micro = 0.15
            alloc_directional = 0.85
            alloc_dir = alloc_directional
            
        else:
            active_regime = "CONSOLIDATION_MICRO_SPREAD"
            rationale = f"Low/Normal volatility ({rolling_volatility_bps:.1f} bps). Harvesting passive spreads & triangular arbitrage."
            alloc_micro = 0.85
            alloc_dir = 0.15

        state = InstinctState(
            timestamp=time.time(),
            current_equity=current_equity,
            starting_capital=self.starting_capital,
            ruin_floor=self.ruin_floor,
            hunger_score=round(hunger_score, 4),
            survival_score=round(survival_score, 4),
            active_regime=active_regime,
            regime_rationale=rationale,
            allocation_micro_pct=alloc_micro,
            allocation_directional_pct=alloc_dir
        )
        self.history.append(state)
        return state
