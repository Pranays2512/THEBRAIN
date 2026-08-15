#!/usr/bin/env python3
"""
brain3/finance/core/live_broker_survival_runner.py

Live Broker Autonomous Survival Trading Engine
Executes paper trading with mock capital against real-time live market prices.

Parameters:
- Initial Capital : ₹1,000.00 INR (Mock Money)
- Strict Ruin Floor: -₹100.00 INR (Existential killswitch threshold)
- Target Abundance : ₹100,000.00 INR (1 Lakh INR Cap)
- Engine: Sub-millisecond C++17 L2 Order Book & Survival Instinct Alpha
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FINANCE_DIR = REPO_ROOT / "brain3" / "finance"
BIN_PATH = FINANCE_DIR / "brain_finance"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brain3.finance.adapters.real_market_feed import RealMarketFeedAdapter, LiveMarketTick

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import ProgressBar

console = Console()

class LiveBrokerSurvivalRunner:
    def __init__(self, initial_capital: float = 1000.0,
                 ruin_floor: float = -100.0,
                 cap_limit: float = 100000.0,
                 metabolic_burn: float = 0.02,
                 poll_interval: float = 0.3):
        self.initial_capital = initial_capital
        self.ruin_floor = ruin_floor
        self.cap_limit = cap_limit
        self.metabolic_burn = metabolic_burn
        self.poll_interval = poll_interval

        self.feed = RealMarketFeedAdapter()
        self.proc: Optional[subprocess.Popen] = None
        self.running = True
        self.trades_executed: List[Dict[str, Any]] = []

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
        """Start the persistent C++ json-stream subprocess with exact parameters."""
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
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.terminate()
            except Exception:
                pass

    def generate_ui_panel(self, latest_tick: Optional[LiveMarketTick] = None, last_action: str = "WAITING_FOR_TICK") -> Panel:
        """Render rich live status dashboard."""
        eq = self.last_status.get("current_equity", self.initial_capital)
        lf = self.last_status.get("life_force_pct", 50.0)
        state = self.last_status.get("survival_state", "SURVIVING")
        peak = self.last_status.get("peak_equity", self.initial_capital)
        ticks = self.last_status.get("ticks_survived", 0)

        # Color coding state
        state_color = "green" if lf >= 75.0 else ("cyan" if lf >= 40.0 else ("yellow" if lf >= 15.0 else "red"))
        if state == "BRAIN_DEAD":
            state_color = "bold white on red"
        elif state == "APEX_ABUNDANCE":
            state_color = "bold black on green"

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold white", width=24)
        table.add_column(style="bold yellow", width=28)
        table.add_column(style="bold white", width=24)
        table.add_column(style="bold yellow", width=28)

        table.add_row("Current Equity (INR):", f"₹{eq:,.2f}", "Starting Capital:", f"₹{self.initial_capital:,.2f}")
        table.add_row("Ruin Floor (Death):", f"[bold red]₹{self.ruin_floor:,.2f}[/bold red]", "Apex Target (1 Lakh):", f"[bold green]₹{self.cap_limit:,.2f}[/bold green]")
        table.add_row("Survival Life Force:", f"[{state_color}]{lf:.2f}% [{state}][/{state_color}]", "Peak Equity Achieved:", f"₹{peak:,.2f}")
        table.add_row("Metabolic Ticks:", f"{ticks:,}", "Total Trades Executed:", f"{len(self.trades_executed)}")

        if latest_tick:
            table.add_row(
                "Live Market Feed:",
                f"[bold magenta]{latest_tick.symbol}[/bold magenta] @ ₹{latest_tick.price:,.2f}",
                "Bid / Ask Spread:",
                f"₹{latest_tick.best_bid:,.2f} / ₹{latest_tick.best_ask:,.2f}"
            )
            table.add_row("Last Action / Signal:", f"[bold cyan]{last_action}[/bold cyan]", "Feed Source:", f"{latest_tick.source}")

        # Recent Trades Table
        trade_table = Table(title="Recent Live Real-Price Executions", show_header=True, header_style="bold blue")
        trade_table.add_column("ID", width=4)
        trade_table.add_column("Symbol", width=12)
        trade_table.add_column("Side", width=6)
        trade_table.add_column("Fill Price", width=14)
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
            Layout(table, size=6),
            Layout(trade_table, size=8)
        )

        return Panel(
            layout,
            title="🧠 [bold cyan]THE BRAIN 3.0 — AUTONOMOUS REAL-MARKET SURVIVAL ENGINE[/bold cyan]",
            subtitle="[dim]Target: ₹100,000.00 (1 Lakh) | Hard Floor: -₹100.00 | Real Live Ticker Quotes[/dim]",
            border_style=state_color
        )

    def run_live_loop(self, max_ticks: int = 10000):
        """Run continuous real-time market execution loop until ₹100,000 cap or -₹100 ruin floor."""
        console.clear()
        instruments = ["BTC/INR", "ETH/INR", "SOL/INR", "NIFTY50/INR", "RELIANCE/INR"]

        with Live(console=console, refresh_per_second=4, screen=False) as live:
            tick_count = 0
            for tick in self.feed.stream_live_ticks(instruments, interval_sec=self.poll_interval):
                if not self.running:
                    break

                tick_count += 1
                cmd = f"LIVE_TICK_EXEC {tick.symbol} {tick.price:.4f} {tick.best_bid:.4f} {tick.best_ask:.4f} {tick.volume:.2f}"
                resp = self.send_command(cmd)

                action_desc = "TICK_INGESTION"
                if resp:
                    status_type = resp.get("status", "")
                    if status_type == "TRADE_EXECUTED":
                        self.trades_executed.append(resp)
                        action_desc = f"FILLED {resp.get('side')} {resp.get('symbol')} ({resp.get('strategy')})"
                    elif status_type == "DEFENSIVE_HOLD":
                        action_desc = "DEFENSIVE_HOLD (Risk Dampened)"
                    elif status_type == "NO_TRADE":
                        action_desc = f"MONITORING ({tick.symbol} OFI={resp.get('ofi', 0):.2f})"
                    elif status_type == "BRAIN_DEAD":
                        action_desc = "TERMINAL RUIN REACHED (-₹100 Floor Breached)"
                        self.running = False

                # Refresh status
                st = self.send_command("FINANCE_STATUS")
                if st:
                    self.last_status.update(st)

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

                if tick_count >= max_ticks:
                    break

        self.close()

def main():
    parser = argparse.ArgumentParser(description="Live Real-Market Survival Trading Runner for The Brain")
    parser.add_argument("--initial", type=float, default=1000.0, help="Initial mock capital in INR (default: 1000.0)")
    parser.add_argument("--floor", type=float, default=-100.0, help="Strict ruin floor in INR (default: -100.0)")
    parser.add_argument("--cap", type=float, default=100000.0, help="Cap limit in INR (default: 100000.0)")
    parser.add_argument("--speed", type=float, default=0.2, help="Tick polling interval in seconds (default: 0.2)")
    parser.add_argument("--ticks", type=int, default=500, help="Max ticks to run (default: 500)")
    args = parser.parse_args()

    runner = LiveBrokerSurvivalRunner(
        initial_capital=args.initial,
        ruin_floor=args.floor,
        cap_limit=args.cap,
        poll_interval=args.speed
    )

    def sig_handler(sig, frame):
        print("\nStopping live trading runner...")
        runner.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    runner.run_live_loop(max_ticks=args.ticks)

if __name__ == "__main__":
    main()
