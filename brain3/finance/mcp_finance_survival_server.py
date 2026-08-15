#!/usr/bin/env python3
"""
brain3/finance/mcp_finance_survival_server.py

Model Context Protocol (MCP) Interface & Tool Server for The Brain's Financial Survival Instinct Branch.
Exposes autonomous trading tools, live orderbook streaming, and survival state management.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

FINANCE_DIR = Path(__file__).resolve().parent
BIN_PATH = FINANCE_DIR / "brain_finance"

def ensure_bin():
    if not BIN_PATH.exists():
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-Icore", "-I.",
            "-o", str(BIN_PATH),
            "finance_orchestrator.cpp"
        ]
        subprocess.run(cmd, cwd=str(FINANCE_DIR), check=True)

def execute_finance_command(command: str) -> Dict[str, Any]:
    ensure_bin()
    proc = subprocess.Popen(
        [str(BIN_PATH), "--json-stream"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, _ = proc.communicate(input=f"{command}\nQUIT\n")
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if line and line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {"error": "FAILED_TO_PARSE_RESPONSE", "raw": stdout}

# ── MCP Tool Implementations ───────────────────────────────────────────────────

def get_survival_status() -> Dict[str, Any]:
    """Retrieve current homeostatic survival state, life force %, equity, and metabolic stats."""
    return execute_finance_command("FINANCE_STATUS")

def execute_sample_trade(symbol: str = "NIFTY50/INR") -> Dict[str, Any]:
    """Execute a single sample survival trade with real-time L2 order matching and PnL realization."""
    return execute_finance_command(f"SAMPLE_SURVIVAL_TRADE {symbol}")

def run_autonomous_survival_cycle(ticks: int = 300) -> Dict[str, Any]:
    """Run an autonomous high-frequency multi-strategy survival trading cycle over N market ticks."""
    return execute_finance_command(f"AUTONOMOUS_SURVIVAL_CYCLE {ticks}")

def get_order_book(symbol: str = "BTC/INR") -> Dict[str, Any]:
    """Inspect the limit order book bids, asks, mid price, and depth for a given symbol."""
    return execute_finance_command(f"ORDER_BOOK {symbol}")

def get_microstructure(symbol: str = "NIFTY50/INR") -> Dict[str, Any]:
    """Query high-frequency microstructure telemetry (OFI, VWAP, Realized Volatility)."""
    return execute_finance_command(f"MICROSTRUCTURE {symbol}")

def scan_statistical_arbitrage(asset_a: str = "BTC/INR", asset_b: str = "ETH/INR") -> Dict[str, Any]:
    """Run cross-asset Engle-Granger cointegration and Ornstein-Uhlenbeck mean-reversion analysis."""
    return execute_finance_command(f"STAT_ARB_SCAN {asset_a} {asset_b}")

def calculate_kelly_position_size(win_prob: float = 0.58, win_loss_ratio: float = 1.6) -> Dict[str, Any]:
    """Compute mathematical Kelly position allocation attenuated by distance to the -₹1,000 ruin floor."""
    return execute_finance_command(f"KELLY_SIZE {win_prob} {win_loss_ratio}")

def set_capital_parameters(initial_capital: float = 1000.0,
                           ruin_floor: float = -1000.0,
                           cap_limit: float = 100000.0,
                           metabolic_burn: float = 0.02) -> Dict[str, Any]:
    """Configure or reset the financial instinct parameters."""
    return execute_finance_command(f"SET_CAPITAL_PARAMETERS {initial_capital} {ruin_floor} {cap_limit} {metabolic_burn}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
        print(json.dumps(execute_finance_command(cmd), indent=2))
    else:
        print(json.dumps(get_survival_status(), indent=2))
