#!/usr/bin/env python3
"""
brain3/finance/run_sample_trade_simulation.py

High-Speed Autonomous Financial Survival Instinct Demonstration
Executes sample trades and multi-tick survival cycles for The Brain.

Parameters:
- Initial Capital : ₹1,000.00 INR
- Ruin Floor      : -₹1,000.00 INR (Terminal Death Threshold)
- Cap Limit       : ₹100,000.00 INR (1 Lakh Target Abundance)
- Biological Drive: Trading to Live (Metabolic ATP upkeep per tick)
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FINANCE_DIR = REPO_ROOT / "brain3" / "finance"
BIN_PATH = FINANCE_DIR / "brain_finance"

def ensure_binary():
    if not BIN_PATH.exists():
        cmd = [
            "clang++", "-std=c++17", "-O3",
            "-Icore", "-I.",
            "-o", str(BIN_PATH),
            "finance_orchestrator.cpp"
        ]
        res = subprocess.run(cmd, cwd=str(FINANCE_DIR), capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Compilation error: {res.stderr}")
            sys.exit(1)

def query_brain(command: str) -> dict:
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
    return {}

def print_banner():
    print("=" * 80)
    print("🧠  THE BRAIN — QUANTITATIVE FINANCIAL SURVIVAL INSTINCT ENGINE")
    print("    Biological Principle: Finance as an Instinct — Trade to Live")
    print("    Capital: ₹1,000.00 | Ruin Floor: -₹1,000.00 | Cap Limit: ₹100,000.00")
    print("=" * 80)

def run_sample_trade():
    ensure_binary()
    print_banner()
    
    print("\n[STEP 1] Querying Baseline Homeostatic Survival State...")
    status = query_brain("FINANCE_STATUS")
    print(f"  • Currency           : {status.get('currency', 'INR')}")
    print(f"  • Current Equity     : ₹{status.get('current_equity', 1000):,.2f}")
    print(f"  • Ruin Floor (Death) : ₹{status.get('ruin_floor', -1000):,.2f}")
    print(f"  • Cap Limit (Apex)   : ₹{status.get('cap_limit', 100000):,.2f}")
    print(f"  • Life Force         : {status.get('life_force_pct', 50):.2f}% [{status.get('survival_state', 'SURVIVING')}]")
    print(f"  • Hunger Urgency     : {status.get('hunger_urgency', 1.0):.2f}x (Metabolic drive active)")

    print("\n[STEP 2] Executing Single Sample Survival Trade (NIFTY50/INR)...")
    t0 = time.perf_counter()
    trade = query_brain("SAMPLE_SURVIVAL_TRADE NIFTY50/INR")
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  • Trade ID           : #{trade.get('trade_id', 1)}")
    print(f"  • Instrument         : {trade.get('symbol', 'NIFTY50/INR')}")
    print(f"  • Action / Side      : {trade.get('side', 'BUY')}")
    print(f"  • Entry Fill Price   : ₹{trade.get('entry_price', 0):,.2f}")
    print(f"  • Exit Realized Price: ₹{trade.get('exit_price', 0):,.2f}")
    print(f"  • Quantity Traded    : {trade.get('quantity', 0):.4f}")
    print(f"  • Realized PnL       : {'+' if trade.get('realized_pnl', 0) >= 0 else ''}₹{trade.get('realized_pnl', 0):,.2f}")
    print(f"  • Strategy Alpha     : {trade.get('strategy', 'OFI_MOMENTUM_SCALP')}")
    print(f"  • Post-Trade Capital : ₹{trade.get('capital_after', 1000):,.2f}")
    print(f"  • Life Force After   : {trade.get('life_force_pct', 50):.2f}%")
    print(f"  • Execution Latency  : {dt_ms:.3f} ms")

    print("\n[STEP 3] Running High-Frequency Autonomous Survival Loop (300 Ticks)...")
    t0 = time.perf_counter()
    cycle = query_brain("AUTONOMOUS_SURVIVAL_CYCLE 300")
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  • Initial Capital    : ₹{cycle.get('initial_capital', 1000):,.2f}")
    print(f"  • Final Capital      : ₹{cycle.get('final_capital', 0):,.2f}")
    print(f"  • Peak Capital       : ₹{cycle.get('peak_capital', 0):,.2f}")
    print(f"  • Net Profit Realized: {'+' if cycle.get('net_profit', 0) >= 0 else ''}₹{cycle.get('net_profit', 0):,.2f} ({cycle.get('return_pct', 0):+.2f}%)")
    print(f"  • Total Trades Exec  : {cycle.get('total_trades', 0)} (Wins: {cycle.get('winning_trades', 0)} | Losses: {cycle.get('losing_trades', 0)})")
    print(f"  • Win Rate           : {cycle.get('win_rate_pct', 0):.2f}%")
    print(f"  • Profit Factor      : {cycle.get('profit_factor', 0):.2f}")
    print(f"  • High-Freq Sharpe   : {cycle.get('sharpe_ratio', 0):.2f}")
    print(f"  • Max Drawdown       : {cycle.get('max_drawdown_pct', 0):.2f}%")
    print(f"  • Survived w/o Ruin  : {'YES (Zero breach of -₹1,000 floor)' if cycle.get('survived_without_ruin') else 'NO'}")
    print(f"  • Simulation Speed   : {cycle.get('simulation_time_ms', 0):.2f} ms for 300 ticks")

    print("\n" + "=" * 80)
    print("🏁 SAMPLE FINANCIAL SURVIVAL DEMONSTRATION COMPLETE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_sample_trade()
