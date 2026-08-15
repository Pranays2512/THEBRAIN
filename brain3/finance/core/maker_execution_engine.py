#!/usr/bin/env python3
"""
brain3/finance/core/maker_execution_engine.py

Maker / Post-Only Limit Order Execution Engine for THE BRAIN 3.0
- Simulates realistic resting limit order placement on the inside book (Best Bid / Best Ask).
- Models queue depth and order priority based on live bookTicker volumes.
- Implements Time-In-Force (TIF) and stale quote cancellation when mid price diverges.
- Calculates true maker economics (zero fee / maker rebate vs taker spread drag).
"""

import sys
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FINANCE_DIR = REPO_ROOT / "brain3" / "finance"
LOGS_DIR = FINANCE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brain3.finance.adapters.real_exchange_feed import RealMarketTick

@dataclass
class MakerOrder:
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    placed_time_ms: float
    limit_price: float
    limit_price_inr: float
    initial_mid_price: float
    initial_spread_bps: float
    initial_queue_ahead: float  # Quantity ahead in queue at limit price
    remaining_queue_ahead: float
    order_qty: float
    allocated_capital_inr: float
    time_to_live_ms: float = 4000.0  # Max resting TTL (4 seconds)
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED, EXPIRED
    filled_time_ms: Optional[float] = None
    filled_price: Optional[float] = None
    filled_price_inr: Optional[float] = None
    cancel_reason: Optional[str] = None
    queue_wait_duration_ms: float = 0.0

@dataclass
class MakerTradeResult:
    trade_id: int
    order_id: str
    symbol: str
    side: str
    placed_time_ms: float
    filled_time_ms: float
    queue_wait_ms: float
    limit_price_inr: float
    filled_price_inr: float
    exit_price_inr: float
    spread_captured_bps: float
    allocated_capital_inr: float
    maker_rebate_fee_inr: float
    gross_pnl_inr: float
    net_pnl_inr: float
    account_equity_inr: float
    status: str
    fill_efficiency_pct: float

class MakerExecutionEngine:
    def __init__(self, initial_capital: float = 1.0, ruin_floor: float = 0.0, maker_rebate_bps: float = -0.5):
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.ruin_floor = ruin_floor
        self.maker_rebate_bps = maker_rebate_bps  # -0.5 bps = maker rebate (positive income)
        self.usd_inr_rate = 87.25
        
        self.active_orders: Dict[str, MakerOrder] = {}
        self.completed_trades: List[MakerTradeResult] = []
        self.cancelled_orders_count = 0
        self.filled_orders_count = 0
        self.trade_counter = 0

    def place_post_only_limit_order(self, tick: RealMarketTick, side: str, capital_allocation_inr: float) -> MakerOrder:
        """Place a resting limit order at the inside book (Best Bid for BUY, Best Ask for SELL)."""
        order_id = f"MKR-{int(time.time()*1000)}-{random.randint(1000, 9999)}"
        
        if side == "BUY":
            limit_price = tick.bid_price_usd
            queue_ahead = tick.bid_qty  # Volume ahead of us in the bid queue
        else:
            limit_price = tick.ask_price_usd
            queue_ahead = tick.ask_qty  # Volume ahead of us in the ask queue
            
        limit_price_inr = limit_price * self.usd_inr_rate
        order_qty = (capital_allocation_inr / self.usd_inr_rate) / max(limit_price, 1e-6)
        
        order = MakerOrder(
            order_id=order_id,
            symbol=tick.symbol,
            side=side,
            placed_time_ms=tick.local_received_timestamp * 1000.0,
            limit_price=limit_price,
            limit_price_inr=limit_price_inr,
            initial_mid_price=tick.mid_price_usd,
            initial_spread_bps=tick.spread_bps,
            initial_queue_ahead=queue_ahead,
            remaining_queue_ahead=queue_ahead,
            order_qty=order_qty,
            allocated_capital_inr=capital_allocation_inr,
            time_to_live_ms=4000.0,
            status="PENDING"
        )
        self.active_orders[order_id] = order
        return order

    def update_order_book_tick(self, tick: RealMarketTick) -> List[MakerTradeResult]:
        """Update active resting orders with incoming market tick and simulate realistic queue clearing."""
        newly_filled_trades = []
        orders_to_remove = []
        now_ms = tick.local_received_timestamp * 1000.0
        
        for order_id, order in self.active_orders.items():
            if order.symbol != tick.symbol:
                continue
                
            elapsed_time_ms = now_ms - order.placed_time_ms
            order.queue_wait_duration_ms = elapsed_time_ms
            
            # Check 1: Time-In-Force TTL Expiry
            if elapsed_time_ms >= order.time_to_live_ms:
                order.status = "EXPIRED"
                order.cancel_reason = "TTL_EXPIRED (4000ms limit reached without fill)"
                self.cancelled_orders_count += 1
                orders_to_remove.append(order_id)
                continue
                
            # Check 2: Adverse Mid-Price Divergence (Price moving away from our quote)
            mid_move_bps = ((tick.mid_price_usd - order.initial_mid_price) / order.initial_mid_price) * 10000.0
            if order.side == "BUY" and mid_move_bps < -2.5:
                order.status = "CANCELLED"
                order.cancel_reason = f"ADVERSE_DIVERGENCE_PROTECT ({mid_move_bps:.1f} bps drop)"
                self.cancelled_orders_count += 1
                orders_to_remove.append(order_id)
                continue
            elif order.side == "SELL" and mid_move_bps > 2.5:
                order.status = "CANCELLED"
                order.cancel_reason = f"ADVERSE_DIVERGENCE_PROTECT ({mid_move_bps:.1f} bps rally)"
                self.cancelled_orders_count += 1
                orders_to_remove.append(order_id)
                continue
                
            # Check 3: Queue Depletion / Fill Condition
            if order.side == "BUY":
                if tick.ask_price_usd <= order.limit_price:
                    order.remaining_queue_ahead -= tick.ask_qty
                else:
                    decay = order.initial_queue_ahead * (elapsed_time_ms / order.time_to_live_ms) * 0.5
                    order.remaining_queue_ahead = max(0.0, order.initial_queue_ahead - decay)
            else:
                if tick.bid_price_usd >= order.limit_price:
                    order.remaining_queue_ahead -= tick.bid_qty
                else:
                    decay = order.initial_queue_ahead * (elapsed_time_ms / order.time_to_live_ms) * 0.5
                    order.remaining_queue_ahead = max(0.0, order.initial_queue_ahead - decay)
                    
            if order.remaining_queue_ahead <= 0.0:
                # ORDER IS FILLED AS A MAKER!
                order.status = "FILLED"
                order.filled_time_ms = now_ms
                order.filled_price = order.limit_price
                order.filled_price_inr = order.limit_price_inr
                self.filled_orders_count += 1
                orders_to_remove.append(order_id)
                
                self.trade_counter += 1
                spread_captured_bps = order.initial_spread_bps / 2.0
                
                if order.side == "BUY":
                    exit_price_inr = tick.ask_price_usd * self.usd_inr_rate
                    gross_ret = (exit_price_inr - order.filled_price_inr) / order.filled_price_inr
                else:
                    exit_price_inr = tick.bid_price_usd * self.usd_inr_rate
                    gross_ret = (order.filled_price_inr - exit_price_inr) / order.filled_price_inr
                    
                gross_pnl_inr = round(order.allocated_capital_inr * gross_ret, 6)
                # Maker rebate: earning 0.5 bps rather than paying 2-4 bps taker fee!
                rebate_inr = round(order.allocated_capital_inr * (abs(self.maker_rebate_bps) / 10000.0), 6)
                net_pnl_inr = round(gross_pnl_inr + rebate_inr, 6)
                
                self.current_equity = round(self.current_equity + net_pnl_inr, 6)
                
                trade_res = MakerTradeResult(
                    trade_id=self.trade_counter,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    placed_time_ms=order.placed_time_ms,
                    filled_time_ms=order.filled_time_ms,
                    queue_wait_ms=round(elapsed_time_ms, 2),
                    limit_price_inr=order.limit_price_inr,
                    filled_price_inr=order.filled_price_inr,
                    exit_price_inr=exit_price_inr,
                    spread_captured_bps=round(spread_captured_bps, 2),
                    allocated_capital_inr=order.allocated_capital_inr,
                    maker_rebate_fee_inr=rebate_inr,
                    gross_pnl_inr=gross_pnl_inr,
                    net_pnl_inr=net_pnl_inr,
                    account_equity_inr=self.current_equity,
                    status="MAKER_FILL",
                    fill_efficiency_pct=round((1.0 - max(0.0, order.remaining_queue_ahead)/max(order.initial_queue_ahead, 1e-6))*100.0, 1)
                )
                self.completed_trades.append(trade_res)
                newly_filled_trades.append(trade_res)
                
        for oid in orders_to_remove:
            self.active_orders.pop(oid, None)
            
        return newly_filled_trades

    def export_maker_trades_spreadsheet(self):
        """Export maker execution audit log to CSV and Excel."""
        if not self.completed_trades:
            return
            
        df = pd.DataFrame([asdict(t) for t in self.completed_trades])
        csv_path = LOGS_DIR / "maker_execution_trades_audit.csv"
        df.to_csv(csv_path, index=False)
        print(f"📊 Exported Maker Trades CSV: {csv_path}")
        
        xlsx_path = LOGS_DIR / "maker_execution_trades_audit.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Maker Limit Fills", index=False)
        print(f"📑 Exported Maker Trades Excel: {xlsx_path}")

if __name__ == "__main__":
    engine = MakerExecutionEngine()
    print("MakerExecutionEngine initialized and ready for live queue routing.")
