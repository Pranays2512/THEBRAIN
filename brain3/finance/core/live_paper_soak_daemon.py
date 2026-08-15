#!/usr/bin/env python3
"""
brain3/finance/core/live_paper_soak_daemon.py

48-Hour Background Live Paper Soak Engine for THE BRAIN 3.0 (Step 1)
- Connects to unauthenticated live Binance WebSocket feeds (wss://stream.binance.com:9443/ws/).
- Executes Maker / Post-Only Limit Orders with 2.0 bps Volatility + 0.70 Imbalance Gate.
- Tracks real-time Adverse Selection markout curves (T+500ms, T+2s, T+10s).
- Persists session state (equity, drawdown, trades, uptime) to soak_session_state.json every 30s.
- Supports continuous unattended execution across Asian, European, and US market sessions.
"""

import sys
import os
import json
import time
import signal
import threading
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FINANCE_DIR = REPO_ROOT / "brain3" / "finance"
LOGS_DIR = FINANCE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = LOGS_DIR / "soak_session_state.json"
PID_FILE = LOGS_DIR / "soak_daemon.pid"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brain3.finance.adapters.real_exchange_feed import RealExchangeFeed, RealMarketTick
from brain3.finance.core.maker_execution_engine import MakerExecutionEngine, MakerOrder, MakerTradeResult
from brain3.finance.core.adverse_selection_analyzer import AdverseSelectionAnalyzer

class LivePaperSoakDaemon:
    def __init__(self, initial_capital: float = 1.0, ruin_floor: float = 0.0):
        self.initial_capital = initial_capital
        self.ruin_floor = ruin_floor
        self.running = False
        self.start_time = time.time()
        self.ticks_processed = 0
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
        
        # Load or initialize state
        self.maker_engine = MakerExecutionEngine(initial_capital=initial_capital, ruin_floor=ruin_floor)
        self.adverse_analyzer = AdverseSelectionAnalyzer()
        self.feed = RealExchangeFeed()
        
        self.recent_ticks_window: Dict[str, List[RealMarketTick]] = {s: [] for s in self.symbols}
        self.post_fill_buffers: Dict[int, List[Dict[str, Any]]] = {} # trade_id -> list of tick snapshots
        
        self.load_state()

    def load_state(self):
        """Restore daemon state from soak_session_state.json if available."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    st = json.load(f)
                self.initial_capital = st.get("initial_capital", 1.0)
                self.maker_engine.current_equity = st.get("current_equity", 1.0)
                self.ticks_processed = st.get("ticks_processed", 0)
                self.start_time = st.get("start_time", time.time())
                print(f"🔄 Resumed existing soak session (Uptime: {((time.time()-self.start_time)/3600):.2f}h, Equity: ₹{self.maker_engine.current_equity:.4f})")
            except Exception as e:
                print(f"⚠️ Failed to parse state file: {e}. Starting fresh session.")

    def save_state(self):
        """Persist daemon state to soak_session_state.json."""
        uptime_s = time.time() - self.start_time
        summary = self.adverse_analyzer.compute_adverse_selection_summary()
        
        state = {
            "session_status": "RUNNING" if self.running else "STOPPED",
            "start_time": self.start_time,
            "last_updated_time": time.time(),
            "uptime_hours": round(uptime_s / 3600.0, 3),
            "ticks_processed": self.ticks_processed,
            "initial_capital": self.initial_capital,
            "current_equity": self.maker_engine.current_equity,
            "realized_return_pct": round(((self.maker_engine.current_equity - self.initial_capital) / self.initial_capital) * 100.0, 4),
            "total_maker_orders_placed": len(self.maker_engine.active_orders) + self.maker_engine.filled_orders_count + self.maker_engine.cancelled_orders_count,
            "maker_fills_count": self.maker_engine.filled_orders_count,
            "maker_cancellations_count": self.maker_engine.cancelled_orders_count,
            "adverse_selection_summary": summary,
            "ruin_floor_status": "SAFE (> ₹0.00)" if self.maker_engine.current_equity > self.ruin_floor else "BREACHED"
        }
        
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def run_live_soak(self, max_duration_seconds: Optional[float] = None):
        """Main daemon loop running continuously against live WebSocket ticks."""
        print(f"🚀 Starting 48-Hour Live Paper Soak Daemon...")
        print(f"   Feed: Binance Public WebSocket wss://stream.binance.com:9443/ws/")
        print(f"   Symbols: {', '.join(self.symbols)}")
        print(f"   Execution Mode: Maker Limit Orders with 2.0 bps Volatility + 0.70 Imbalance Gate")
        
        self.running = True
        self.feed.start()
        time.sleep(1.0)
        
        last_save_t = time.time()
        last_order_t: Dict[str, float] = {s: 0.0 for s in self.symbols}
        
        try:
            for tick in self.feed.stream_ticks():
                if not self.running:
                    break
                    
                self.ticks_processed += 1
                sym = tick.symbol
                
                # Maintain rolling window of last 10 ticks for volatility & markout tracking
                win = self.recent_ticks_window.get(sym, [])
                win.append(tick)
                if len(win) > 30:
                    win.pop(0)
                self.recent_ticks_window[sym] = win
                
                # 1. Update existing resting maker orders
                new_fills = self.maker_engine.update_order_book_tick(tick)
                
                # 2. For newly filled orders, register for markout tracking
                now_ms = tick.local_received_timestamp * 1000.0
                for fill in new_fills:
                    self.post_fill_buffers[fill.trade_id] = [{
                        'timestamp_ms': now_ms,
                        'mid_price_inr': tick.mid_price_inr
                    }]
                    print(f"  ⚡ [MAKER FILL #{fill.trade_id:03d}] {fill.symbol} {fill.side} @ ₹{fill.filled_price_inr:,.2f} | Captured: {fill.spread_captured_bps} bps | Eq: ₹{fill.account_equity_inr:.4f}")
                    
                # 3. Update markout buffers for active post-fill tracking
                for trade_id, buf in list(self.post_fill_buffers.items()):
                    buf.append({
                        'timestamp_ms': now_ms,
                        'mid_price_inr': tick.mid_price_inr
                    })
                    # If buffer has accumulated 10 seconds of data, evaluate adverse selection
                    first_t = buf[0]['timestamp_ms']
                    if (now_ms - first_t) >= 10000.0 or len(buf) >= 40:
                        trade = next((t for t in self.maker_engine.completed_trades if t.trade_id == trade_id), None)
                        if trade:
                            record = self.adverse_analyzer.evaluate_maker_trade_markouts(trade, buf)
                            print(f"  📊 [MARKOUT #{trade_id:03d}] T+500ms: {record.markout_t500ms_bps:+.1f}bps | T+2s: {record.markout_t2s_bps:+.1f}bps | Type: {record.fill_classification}")
                        self.post_fill_buffers.pop(trade_id, None)

                # 4. Check Stand-Down Filter Gate & Place New Maker Order if qualified
                now_t = time.time()
                if (now_t - last_order_t.get(sym, 0.0)) >= 3.0 and len(win) >= 5:
                    # Calculate rolling range in bps
                    recent_ranges = [((t.ask_price_usd - t.bid_price_usd) / t.mid_price_usd) * 10000.0 for t in win[-5:]]
                    rolling_vol_bps = float(np.mean(recent_ranges))
                    
                    # Imbalance
                    imb = (tick.bid_qty - tick.ask_qty) / max(tick.bid_qty + tick.ask_qty, 1e-6)
                    
                    # Gate Rule: Rolling Vol >= 2.0 bps AND |Imbalance| >= 0.70
                    if rolling_vol_bps >= 2.0 and abs(imb) >= 0.70:
                        side = "BUY" if imb > 0 else "SELL"
                        # Half-Kelly sizing
                        alloc = min(self.maker_engine.current_equity * 0.10, 0.12)
                        order = self.maker_engine.place_post_only_limit_order(tick, side=side, capital_allocation_inr=alloc)
                        last_order_t[sym] = now_t
                        
                # 5. Periodic State Save
                if now_t - last_save_t >= 10.0:
                    self.save_state()
                    last_save_t = now_t
                    
                # 6. Check duration limit if set
                if max_duration_seconds and (now_t - self.start_time) >= max_duration_seconds:
                    print(f"⏱️ Reached duration limit ({max_duration_seconds}s). Stopping soak run.")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal. Safely shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Clean shutdown and flush all logs."""
        self.running = False
        self.feed.stop()
        self.save_state()
        self.maker_engine.export_maker_trades_spreadsheet()
        self.adverse_analyzer.export_adverse_selection_spreadsheets()
        print("✅ Soak daemon cleanly stopped. All state flushed to logs.")

def start_daemon_background():
    """Fork and start daemon in background."""
    daemon = LivePaperSoakDaemon(initial_capital=1.0, ruin_floor=0.0)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    daemon.run_live_soak()

def print_daemon_status():
    """Print current daemon status."""
    if not STATE_FILE.exists():
        print("ℹ️ No active soak session found. Run `python3 live_paper_soak_daemon.py start` to launch.")
        return
        
    with open(STATE_FILE, "r") as f:
        st = json.load(f)
        
    print("\n" + "=" * 75)
    print("📡 LIVE PAPER SOAK DAEMON STATUS")
    print("=" * 75)
    print(f"  • Session Status             : {st.get('session_status')}")
    print(f"  • Total Uptime               : {st.get('uptime_hours', 0):.2f} hours")
    print(f"  • Live Market Ticks Processed: {st.get('ticks_processed', 0):,}")
    print(f"  • Starting Capital           : ₹{st.get('initial_capital', 1.0):.2f}")
    print(f"  • Current Equity             : ₹{st.get('current_equity', 1.0):.4f}")
    print(f"  • Realized Return            : {st.get('realized_return_pct', 0):+.2f}%")
    print(f"  • Maker Fills Executed       : {st.get('maker_fills_count', 0)}")
    print(f"  • Maker TTL Cancellations    : {st.get('maker_cancellations_count', 0)}")
    print(f"  • Ruin Floor Status          : {st.get('ruin_floor_status')}")
    
    adv = st.get("adverse_selection_summary", {})
    if adv:
        print("\n  [ADVERSE SELECTION METRICS]")
        for k, v in adv.items():
            print(f"    • {k:30s}: {v}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="48-Hour Live Paper Soak Daemon for THE BRAIN 3.0")
    parser.add_argument("command", choices=["start", "status", "stop", "once"], help="Daemon command")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds for 'once' mode")
    
    args = parser.parse_args()
    
    if args.command == "start":
        daemon = LivePaperSoakDaemon()
        daemon.run_live_soak()
    elif args.command == "once":
        daemon = LivePaperSoakDaemon()
        daemon.run_live_soak(max_duration_seconds=args.duration)
    elif args.command == "status":
        print_daemon_status()
    elif args.command == "stop":
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                st = json.load(f)
            st["session_status"] = "STOPPED"
            with open(STATE_FILE, "w") as f:
                json.dump(st, f, indent=2)
        print("🛑 Soak daemon marked as STOPPED.")
