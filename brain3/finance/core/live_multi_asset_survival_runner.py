#!/usr/bin/env python3
"""
brain3/finance/core/live_multi_asset_survival_runner.py

Massive Multi-Stream Live Market Survival Engine
Monitors hundreds of stocks, cryptos, indices, and global assets simultaneously:
- Indian Large Caps: NIFTY50, BANKNIFTY, RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, TATAMOTORS, SBIN, AIRTEL, etc.
- Global Tech: NVDA, AAPL, MSFT, GOOGL, TSLA, AMZN
- Crypto Spot: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, PEPE, + dozens of live pairs
- Rules: Starting Mock Cash = ₹1,000 | Strict Ruin Floor = -₹100 | Target Abundance = ₹100,000 (1 Lakh)
"""

import sys
import os
import json
import time
import signal
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FINANCE_DIR = REPO_ROOT / "brain3" / "finance"
BIN_PATH = FINANCE_DIR / "brain_finance"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brain3.finance.adapters.multi_stream_market_feed import MultiStreamMarketFeed, MultiAssetTick

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.columns import Columns

console = Console()

class LiveMultiAssetSurvivalRunner:
    def __init__(self, initial_capital: float = 1000.0,
                 ruin_floor: float = -100.0,
                 cap_limit: float = 100000.0,
                 metabolic_burn: float = 0.02):
        self.initial_capital = initial_capital
        self.ruin_floor = ruin_floor
        self.cap_limit = cap_limit
        self.metabolic_burn = metabolic_burn

        self.feed = MultiStreamMarketFeed()
        self.proc: Optional[subprocess.Popen] = None
        self.running = True
        self.trades_executed: List[Dict[str, Any]] = []
        self.recent_ticks: deque = deque(maxlen=20)

        self.last_status: Dict[str, Any] = {
            "current_equity": initial_capital,
            "peak_equity": initial_capital,
            "life_force_pct": 50.0,
            "survival_state": "SURVIVING",
            "is_alive": True,
            "ticks_survived": 0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 1.0
        }

        self.setup_engine_process()

    def setup_engine_process(self):
        """Start the persistent C++ json-stream subprocess."""
        if not BIN_PATH.exists():
            cmd = ["clang++", "-std=c++17", "-O3", "-Icore", "-I.", "-o", str(BIN_PATH), "finance_orchestrator.cpp"]
            subprocess.run(cmd, cwd=str(FINANCE_DIR), check=True)

        self.proc = subprocess.Popen(
            [str(BIN_PATH), "--json-stream"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Configure capital parameters
        self.send_command(f"SET_CAPITAL_PARAMETERS {self.initial_capital} {self.ruin_floor} {self.cap_limit} {self.metabolic_burn}")
        status_resp = self.send_command("FINANCE_STATUS")
        if status_resp:
            self.last_status.update(status_resp)

    def send_command(self, cmd: str) -> Optional[Dict[str, Any]]:
        """Send a single BrainQL command and parse JSON output."""
        if not self.proc or self.proc.poll() is not None:
            return None
        try:
            self.proc.stdin.write(f"{cmd}\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if line:
                return json.loads(line.strip())
        except Exception:
            pass
        return None

    def close(self):
        self.running = False
        self.feed.stop()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.terminate()
            except Exception:
                pass

    def generate_ui_panel(self, latest_tick: Optional[MultiAssetTick] = None, last_action: str = "SCANNING_MULTI_STREAM") -> Panel:
        """Render multi-stream live trading matrix and telemetry."""
        eq = self.last_status.get("current_equity", self.initial_capital)
        lf = self.last_status.get("life_force_pct", 50.0)
        state = self.last_status.get("survival_state", "SURVIVING")
        peak = self.last_status.get("peak_equity", self.initial_capital)
        ticks = self.last_status.get("ticks_survived", 0)

        state_color = "green" if lf >= 75.0 else ("cyan" if lf >= 40.0 else ("yellow" if lf >= 15.0 else "red"))

        # Top Metric Cards
        metrics_table = Table.grid(padding=(0, 2))
        metrics_table.add_column(style="bold white", width=22)
        metrics_table.add_column(style="bold yellow", width=26)
        metrics_table.add_column(style="bold white", width=22)
        metrics_table.add_column(style="bold yellow", width=26)

        metrics_table.add_row("Current Mock Equity:", f"₹{eq:,.2f}", "Starting Capital:", f"₹{self.initial_capital:,.2f}")
        metrics_table.add_row("Ruin Floor (Hard Stop):", f"[bold red]₹{self.ruin_floor:,.2f}[/bold red]", "Apex Target (1 Lakh):", f"[bold green]₹{self.cap_limit:,.2f}[/bold green]")
        metrics_table.add_row("Life Force Homeostasis:", f"[{state_color}]{lf:.2f}% [{state}][/{state_color}]", "Peak Capital:", f"₹{peak:,.2f}")
        
        snapshot = self.feed.get_market_snapshot()
        metrics_table.add_row("Active Market Universe:", f"[bold cyan]{len(snapshot)} Live Instruments[/bold cyan]", "Total Trades Executed:", f"{len(self.trades_executed)}")

        # Live Market Universe Matrix Table (Sample top 8 active instruments)
        matrix_table = Table(title="Live Multi-Stream Market Matrix (NSE Equities + Global Crypto Spot)", show_header=True, header_style="bold magenta")
        matrix_table.add_column("Asset Class", width=14)
        matrix_table.add_column("Symbol", width=15)
        matrix_table.add_column("Live Price (₹)", width=16)
        matrix_table.add_column("24h %", width=10)
        matrix_table.add_column("Feed Source", width=16)

        items = list(snapshot.values())
        for t in items[:8]:
            chg_style = "green" if t.change_24h_pct >= 0 else "red"
            matrix_table.add_row(
                t.asset_class,
                t.symbol,
                f"₹{t.price:,.2f}",
                f"[{chg_style}]{t.change_24h_pct:+.2f}%[/{chg_style}]",
                t.source
            )

        # Recent Multi-Asset Real Executions
        trade_table = Table(title="Real-Time Alpha Trade Executions Across All Streams", show_header=True, header_style="bold blue")
        trade_table.add_column("ID", width=4)
        trade_table.add_column("Symbol", width=14)
        trade_table.add_column("Side", width=6)
        trade_table.add_column("Fill (₹)", width=14)
        trade_table.add_column("PnL (₹)", width=12)
        trade_table.add_column("Strategy Alpha", width=22)

        for tr in reversed(self.trades_executed[-5:]):
            pnl = tr.get("realized_pnl", 0.0)
            pnl_str = f"+₹{pnl:,.2f}" if pnl >= 0 else f"-₹{abs(pnl):,.2f}"
            pnl_style = "bold green" if pnl >= 0 else "bold red"
            trade_table.add_row(
                str(tr.get("trade_id", 0)),
                tr.get("symbol", ""),
                tr.get("side", ""),
                f"₹{tr.get('entry_price', 0):,.2f}",
                f"[{pnl_style}]{pnl_str}[/{pnl_style}]",
                tr.get("strategy", "")
            )

        layout = Layout()
        layout.split_column(
            Layout(metrics_table, size=5),
            Layout(matrix_table, size=11),
            Layout(trade_table, size=8)
        )

        return Panel(
            layout,
            title="🧠 [bold cyan]THE BRAIN 3.0 — MASSIVE MULTI-STREAM AUTONOMOUS SURVIVAL ENGINE[/bold cyan]",
            subtitle="[dim]Hundreds of Concurrent Streams | NSE 50 + Global Equities + Crypto Spot | Target: ₹100,000.00[/dim]",
            border_style=state_color
        )

    def run_live_loop(self, max_ticks: int = 10000):
        """Run continuous multi-stream execution loop until ₹100,000 cap or -₹100 ruin floor."""
        console.clear()
        self.feed.start()
        time.sleep(1.5)  # Allow WebSocket and worker threads to populate initial stream cache

        LOGS_DIR = FINANCE_DIR / "logs"
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.state_file = LOGS_DIR / "live_session_state.json"
        
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            tick_count = 0
            while self.running:
                tick = self.feed.get_next_tick(timeout=0.2)
                if not tick:
                    continue

                tick_count += 1
                cmd = f"MULTI_ASSET_TICK {tick.symbol} {tick.price:.4f} {tick.best_bid:.4f} {tick.best_ask:.4f} {tick.volume:.2f} {tick.change_24h_pct:.2f}"
                resp = self.send_command(cmd)

                action_desc = "SCANNING_STREAMS"
                if resp:
                    status_type = resp.get("status", "")
                    if status_type == "MULTI_TRADE_EXECUTED":
                        self.trades_executed.append(resp)
                        action_desc = f"FILLED {resp.get('side')} {resp.get('symbol')} ({resp.get('strategy')})"
                    elif status_type == "MONITORING":
                        action_desc = f"SCANNED {tick.symbol} (Alpha={resp.get('alpha_score', 0):.2f})"
                    elif status_type == "BRAIN_DEAD":
                        action_desc = "TERMINAL RUIN REACHED (-₹100 Floor Breached)"
                        self.running = False

                # Periodically update status and persist state
                if tick_count % 5 == 0:
                    st = self.send_command("FINANCE_STATUS")
                    if st:
                        self.last_status.update(st)
                    self.persist_session_state()

                current_eq = self.last_status.get("current_equity", self.initial_capital)

                # Check stopping conditions:
                if current_eq <= self.ruin_floor:
                    live.update(self.generate_ui_panel(tick, "[bold red]STRICT -₹100 RUIN FLOOR BREACHED — HALTING[/bold red]"))
                    console.print(f"\n[bold red]💀 TERMINAL STOP: Capital reached ₹{current_eq:,.2f} <= -₹100.00 ruin floor.[/bold red]\n")
                    break

                if current_eq >= self.cap_limit:
                    live.update(self.generate_ui_panel(tick, "[bold green]1 LAKH (₹100,000) CAP ACHIEVED — APEX SUCCESS[/bold green]"))
                    console.print(f"\n[bold green]🏆 APEX ABUNDANCE ACHIEVED: Capital reached ₹{current_eq:,.2f} >= ₹100,000.00![/bold green]\n")
                    break

                live.update(self.generate_ui_panel(tick, action_desc))

                if max_ticks > 0 and tick_count >= max_ticks:
                    break

        self.persist_session_state()
        self.close()

    def persist_session_state(self):
        """Save clean JSON session state for telemetry dashboards."""
        try:
            snapshot = self.feed.get_market_snapshot()
            data = {
                "timestamp": time.time(),
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "initial_capital": self.initial_capital,
                "current_equity": self.last_status.get("current_equity", self.initial_capital),
                "peak_equity": self.last_status.get("peak_equity", self.initial_capital),
                "ruin_floor": self.ruin_floor,
                "cap_limit": self.cap_limit,
                "life_force_pct": self.last_status.get("life_force_pct", 50.0),
                "survival_state": self.last_status.get("survival_state", "SURVIVING"),
                "is_alive": self.last_status.get("is_alive", True),
                "total_trades": len(self.trades_executed),
                "active_universe_count": len(snapshot),
                "recent_trades": self.trades_executed[-10:],
                "active_market_sample": [
                    {
                        "symbol": t.symbol,
                        "price": t.price,
                        "change_24h_pct": t.change_24h_pct,
                        "asset_class": t.asset_class,
                        "source": t.source
                    }
                    for t in list(snapshot.values())[:10]
                ]
            }
            tmp_path = self.state_file.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self.state_file)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Multi-Stream Real-Market Survival Trading Runner for The Brain")
    parser.add_argument("--initial", type=float, default=1000.0, help="Initial mock capital in INR (default: 1000.0)")
    parser.add_argument("--floor", type=float, default=-100.0, help="Strict ruin floor in INR (default: -100.0)")
    parser.add_argument("--cap", type=float, default=100000.0, help="Cap limit in INR (default: 100000.0)")
    parser.add_argument("--ticks", type=int, default=500, help="Max ticks to process (default: 500)")
    args = parser.parse_args()

    runner = LiveMultiAssetSurvivalRunner(
        initial_capital=args.initial,
        ruin_floor=args.floor,
        cap_limit=args.cap
    )

    def sig_handler(sig, frame):
        print("\nStopping multi-stream live trading runner...")
        runner.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    runner.run_live_loop(max_ticks=args.ticks)

if __name__ == "__main__":
    main()
