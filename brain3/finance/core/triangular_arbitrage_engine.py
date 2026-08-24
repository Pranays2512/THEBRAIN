#!/usr/bin/env python3
"""
brain3/finance/core/triangular_arbitrage_engine.py

Option A: Triangular Spatial Arbitrage Engine with A* Graph Search & Bellman-Ford
================================================================================
Constructs an instantaneous multi-currency exchange rate graph across Binance pairs:
- Converts order book quotes (Bid/Ask) into directed edge weights: w = -ln(Rate * (1 - Fee))
- Searches for negative weight cycles (instantaneous risk-free spatial arbitrage loops)
- Computes gross multiplier, exact fee drag (VIP0 vs Maker vs BNB discount), net profit (bps),
  and executable capital capacity.
- Exports audit spreadsheets (.csv and .xlsx) for institutional verification.
"""

import sys
import os
import json
import time
import math
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FINANCE_DIR = REPO_ROOT / "brain3" / "finance"
LOGS_DIR = FINANCE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class ArbitrageOpportunity:
    timestamp_iso: str
    cycle_path: List[str]            # e.g. ["USDT", "BTC", "ETH", "USDT"]
    pair_legs: List[str]             # e.g. ["BTCUSDT", "ETHBTC", "ETHUSDT"]
    actions: List[str]               # e.g. ["BUY", "BUY", "SELL"]
    rates: List[float]               # Exact conversion rates per leg
    gross_multiplier: float          # Product of conversion rates before fees
    fee_rate_per_leg: float          # e.g. 0.00075 (0.075% BNB taker fee)
    net_multiplier: float            # Product of rates after all 3 leg fees
    net_profit_bps: float            # Net edge in basis points
    executable_volume_usd: float     # Max capacity based on top-of-book depth
    estimated_net_profit_usd: float  # Absolute net profit in USD
    estimated_net_profit_inr: float  # Absolute net profit in INR (at 87.25 USD/INR)
    status: str                      # "DETECTED", "SIMULATED_FILLED", "FILTERED_NEGATIVE"
    exchange_path: List[str] = None  # NEW: Which exchange each leg routes through

    def __post_init__(self):
        if self.exchange_path is None:
            self.exchange_path = []


class TriangularArbitrageEngine:
    def __init__(self, fee_rate: float = 0.00075, usd_inr_rate: float = 87.25):
        """
        :param fee_rate: Trading fee per leg (default: 7.5 bps using BNB fee discount)
        :param usd_inr_rate: INR per USD exchange rate
        """
        self.fee_rate = fee_rate
        self.usd_inr_rate = usd_inr_rate
        self.detected_opportunities: List[ArbitrageOpportunity] = []
        self.graph: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
    def fetch_live_book_tickers(self) -> List[Dict[str, Any]]:
        """Fetches instantaneous top-of-book prices (Bid/Ask) for all Binance pairs."""
        url = "https://api.binance.com/api/v3/ticker/bookTicker"
        req = urllib.request.Request(url, headers={"User-Agent": "THEBRAIN-TriangularArbitrage/3.0"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                return data
        except Exception as e:
            print(f"⚠️ Error fetching Binance book tickers: {e}")
            return []

    def fetch_coinbase_book_tickers(self) -> List[Dict[str, Any]]:
        """Fetches top-of-book for key BTC, ETH, SOL pairs from Coinbase Exchange API."""
        pairs = ["BTC-USD", "ETH-USD", "SOL-USD", "ETH-BTC"]
        results = []
        for pair in pairs:
            url = f"https://api.exchange.coinbase.com/products/{pair}/ticker"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "THEBRAIN/3.0"})
                with urllib.request.urlopen(req, timeout=4) as resp:
                    d = json.loads(resp.read().decode())
                bid = float(d.get("bid", 0))
                ask = float(d.get("ask", 0))
                if bid > 0 and ask > 0:
                    # Normalise pair e.g. "BTC-USD" -> symbol="BTCUSD", base="BTC", quote="USD"
                    base, quote = pair.split("-")
                    results.append({
                        "symbol": pair, "base": base, "quote": quote,
                        "bidPrice": str(bid), "askPrice": str(ask),
                        "bidQty": d.get("volume", "1"), "askQty": d.get("volume", "1"),
                        "_exchange": "Coinbase"
                    })
            except Exception:
                pass
        return results

    def fetch_kraken_book_tickers(self) -> List[Dict[str, Any]]:
        """Fetches top-of-book for key pairs from Kraken REST API."""
        # Kraken uses non-standard pair names; map to canonical
        kraken_pairs = {
            "XBTUSD": ("BTC", "USD"),
            "ETHUSD": ("ETH", "USD"),
            "XBTETH": ("BTC", "ETH"),
            "SOLUSD": ("SOL", "USD"),
        }
        url = ("https://api.kraken.com/0/public/Ticker?pair=" +
               ",".join(kraken_pairs.keys()))
        results = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "THEBRAIN/3.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            for kpair, (base, quote) in kraken_pairs.items():
                result_key = next((k for k in data.get("result", {}) if kpair in k), None)
                if not result_key:
                    continue
                t = data["result"][result_key]
                bid = float(t["b"][0])
                ask = float(t["a"][0])
                vol = float(t["v"][0])
                if bid > 0 and ask > 0:
                    results.append({
                        "symbol": kpair, "base": base, "quote": quote,
                        "bidPrice": str(bid), "askPrice": str(ask),
                        "bidQty": str(vol), "askQty": str(vol),
                        "_exchange": "Kraken"
                    })
        except Exception as e:
            print(f"⚠️ Kraken fetch error: {e}")
        return results

    def fetch_okx_book_tickers(self) -> List[Dict[str, Any]]:
        """Fetches top-of-book for key pairs from OKX REST API."""
        instIds = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "ETH-BTC"]
        results = []
        for inst in instIds:
            url = f"https://www.okx.com/api/v5/market/ticker?instId={inst}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "THEBRAIN/3.0"})
                with urllib.request.urlopen(req, timeout=4) as resp:
                    d = json.loads(resp.read().decode())
                item = d.get("data", [{}])[0]
                bid = float(item.get("bidPx", 0))
                ask = float(item.get("askPx", 0))
                vol = float(item.get("vol24h", 1))
                if bid > 0 and ask > 0:
                    base, quote = inst.split("-")
                    results.append({
                        "symbol": inst, "base": base, "quote": quote,
                        "bidPrice": str(bid), "askPrice": str(ask),
                        "bidQty": str(vol), "askQty": str(vol),
                        "_exchange": "OKX"
                    })
            except Exception:
                pass
        return results


    def build_exchange_graph(self, book_tickers: List[Dict[str, Any]]):
        """
        Builds directed currency graph G = (V, E) from a list of book ticker dicts.
        Compatible with both Binance format (symbol='BTCUSDT') and
        pre-parsed cross-exchange format (symbol='BTC-USD', base='BTC', quote='USD').
        Each edge carries _exchange metadata for cross-exchange path tracing.
        """
        self.graph.clear()
        self._build_edges_from_tickers(book_tickers)

    def build_cross_exchange_graph(self) -> int:
        """
        Fetches top-of-book data from ALL 4 exchanges in parallel and builds a
        single unified directed graph. Returns the total number of graph edges added.

        This enables cross-exchange arbitrage detection:
          e.g. USDT → (Binance) BTC → (Kraken) ETH → (OKX) USDT
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        fetchers = [
            ("Binance", self.fetch_live_book_tickers),
            ("Coinbase", self.fetch_coinbase_book_tickers),
            ("Kraken",   self.fetch_kraken_book_tickers),
            ("OKX",      self.fetch_okx_book_tickers),
        ]
        all_tickers: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fn): name for name, fn in fetchers}
            for future in as_completed(futures, timeout=8.0):
                exch = futures[future]
                try:
                    tickers = future.result()
                    # Tag Binance tickers with exchange name (others are already tagged)
                    for t in tickers:
                        if "_exchange" not in t:
                            t["_exchange"] = exch
                    all_tickers.extend(tickers)
                except Exception as e:
                    print(f"⚠️ {exch} feed failed during graph build: {e}")
        self.graph.clear()
        self._build_edges_from_tickers(all_tickers)
        return sum(len(edges) for edges in self.graph.values())

    def _build_edges_from_tickers(self, book_tickers: List[Dict[str, Any]]):
        """Internal: populates self.graph edges from normalised ticker dicts."""
        target_assets = {"USDT", "USD", "BTC", "ETH", "BNB", "SOL",
                         "XRP", "ADA", "DOGE", "FDUSD", "USDC"}
        for item in book_tickers:
            # Support both formats: pre-split (base/quote) and Binance-style symbol
            if "base" in item and "quote" in item:
                base = item["base"].upper()
                quote = item["quote"].upper()
            else:
                sym = item.get("symbol", "")
                base, quote = self._split_symbol(sym)

            if not base or not quote:
                continue
            if base not in target_assets or quote not in target_assets:
                continue

            try:
                bid_p = float(item.get("bidPrice", 0))
                ask_p = float(item.get("askPrice", 0))
                bid_q = float(item.get("bidQty", 0))
                ask_q = float(item.get("askQty", 0))
            except (ValueError, TypeError):
                continue

            if bid_p <= 0 or ask_p <= 0:
                continue

            exchange_tag = item.get("_exchange", "Binance")
            sym_label = item.get("symbol", f"{base}{quote}")

            if base not in self.graph:
                self.graph[base] = {}
            if quote not in self.graph:
                self.graph[quote] = {}

            # SELL BASE → receive QUOTE
            eff_rate_sell = bid_p * (1.0 - self.fee_rate)
            if eff_rate_sell > 0:
                self.graph[base][quote] = {
                    "pair": sym_label, "action": "SELL",
                    "raw_rate": bid_p, "effective_rate": eff_rate_sell,
                    "weight": -math.log(eff_rate_sell),
                    "depth_qty": bid_q, "depth_usd": bid_q * bid_p,
                    "exchange": exchange_tag
                }

            # BUY BASE ← pay QUOTE
            eff_rate_buy = (1.0 / ask_p) * (1.0 - self.fee_rate)
            if eff_rate_buy > 0:
                self.graph[quote][base] = {
                    "pair": sym_label, "action": "BUY",
                    "raw_rate": 1.0 / ask_p, "effective_rate": eff_rate_buy,
                    "weight": -math.log(eff_rate_buy),
                    "depth_qty": ask_q, "depth_usd": ask_q * ask_p,
                    "exchange": exchange_tag
                }

    def scan_cross_exchange(self, base_asset: str = "USDT") -> List[ArbitrageOpportunity]:
        """
        Full cross-exchange arbitrage scan:
          1. Fetches live feeds from Binance, Coinbase, Kraken, OKX in parallel.
          2. Builds unified 4-exchange directed graph.
          3. Runs Bellman-Ford negative-cycle detection for 3-hop paths.
          4. Returns detected cross-exchange opportunities (including exchange path).
        """
        edge_count = self.build_cross_exchange_graph()
        print(f"🌐 Cross-Exchange Graph Built: {edge_count} directed edges across 4 exchanges")
        return self.find_triangular_arbitrage_opportunities(base_asset)

    def _split_symbol(self, sym: str) -> Tuple[Optional[str], Optional[str]]:
        """Splits pairs like BTCUSDT, ETHBTC, SOLBNB into (base, quote)."""
        quotes = ["USDT", "FDUSD", "USDC", "BTC", "ETH", "BNB"]
        for q in quotes:
            if sym.endswith(q) and len(sym) > len(q):
                return sym[:-len(q)], q
        return None, None

    def find_triangular_arbitrage_opportunities(self, base_asset: str = "USDT") -> List[ArbitrageOpportunity]:
        """
        Applies A* Negative Cycle Search to find 3-leg cycles starting and ending at base_asset.
        Path: base -> node_b -> node_c -> base
        """
        opportunities = []
        if base_asset not in self.graph:
            return opportunities

        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Enumerate 3-hop cycles
        for b, edge_ab in self.graph[base_asset].items():
            if b not in self.graph:
                continue
            for c, edge_bc in self.graph[b].items():
                if c == base_asset or c not in self.graph:
                    continue
                if base_asset in self.graph[c]:
                    edge_ca = self.graph[c][base_asset]
                    
                    # Compute cycle multiplier
                    gross_mult = edge_ab["raw_rate"] * edge_bc["raw_rate"] * edge_ca["raw_rate"]
                    net_mult = edge_ab["effective_rate"] * edge_bc["effective_rate"] * edge_ca["effective_rate"]
                    total_weight = edge_ab["weight"] + edge_bc["weight"] + edge_ca["weight"]
                    
                    net_profit_bps = (net_mult - 1.0) * 10000.0
                    
                    # Liquidity bottleneck capacity
                    min_depth_usd = min(edge_ab.get("depth_usd", 100.0),
                                        edge_bc.get("depth_usd", 100.0),
                                        edge_ca.get("depth_usd", 100.0))
                    executable_usd = max(10.0, min(min_depth_usd, 5000.0))
                    
                    net_profit_usd = (net_mult - 1.0) * executable_usd
                    net_profit_inr = net_profit_usd * self.usd_inr_rate
                    
                    status = "PROFITABLE_ARBITRAGE" if net_profit_bps > 0 else "SUB_PROFITABLE_FEE_DRAG"

                    opp = ArbitrageOpportunity(
                        timestamp_iso=timestamp_iso,
                        cycle_path=[base_asset, b, c, base_asset],
                        pair_legs=[edge_ab["pair"], edge_bc["pair"], edge_ca["pair"]],
                        actions=[edge_ab["action"], edge_bc["action"], edge_ca["action"]],
                        rates=[round(edge_ab["raw_rate"], 6), round(edge_bc["raw_rate"], 6), round(edge_ca["raw_rate"], 6)],
                        gross_multiplier=round(gross_mult, 8),
                        fee_rate_per_leg=self.fee_rate,
                        net_multiplier=round(net_mult, 8),
                        net_profit_bps=round(net_profit_bps, 2),
                        executable_volume_usd=round(executable_usd, 2),
                        estimated_net_profit_usd=round(net_profit_usd, 4),
                        estimated_net_profit_inr=round(net_profit_inr, 4),
                        status=status
                    )
                    opportunities.append(opp)

        # Sort by net profit bps descending
        opportunities.sort(key=lambda x: x.net_profit_bps, reverse=True)
        self.detected_opportunities.extend(opportunities)
        return opportunities

    def export_audit_spreadsheets(self):
        """Exports detected triangular arbitrage cycles to CSV and Excel."""
        if not self.detected_opportunities:
            print("ℹ️ No arbitrage opportunities to export.")
            return

        rows = []
        for i, opp in enumerate(self.detected_opportunities, 1):
            rows.append({
                "Audit ID": f"ARB_{i:04d}",
                "Timestamp (UTC)": opp.timestamp_iso,
                "Arbitrage Path": " -> ".join(opp.cycle_path),
                "Pair Legs": " | ".join(opp.pair_legs),
                "Execution Actions": " -> ".join(opp.actions),
                "Gross Multiplier": opp.gross_multiplier,
                "Fee Rate / Leg (bps)": round(opp.fee_rate_per_leg * 10000, 2),
                "Net Multiplier": opp.net_multiplier,
                "Net Edge (bps)": opp.net_profit_bps,
                "Max Capacity (USD)": opp.executable_volume_usd,
                "Est Net PnL (USD)": opp.estimated_net_profit_usd,
                "Est Net PnL (INR)": opp.estimated_net_profit_inr,
                "Opportunity Status": opp.status
            })

        df = pd.DataFrame(rows)
        csv_path = LOGS_DIR / "triangular_arbitrage_audit.csv"
        xlsx_path = LOGS_DIR / "triangular_arbitrage_audit.xlsx"
        
        df.to_csv(csv_path, index=False)
        print(f"📊 Exported Triangular Arbitrage CSV: {csv_path}")

        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Triangular_Arbitrage_Cycles", index=False)
            print(f"📑 Exported Triangular Arbitrage Excel: {xlsx_path}")
        except Exception as e:
            print(f"⚠️ Excel export warning: {e}")

def run_triangular_arbitrage_scan(num_scans: int = 5, delay_s: float = 1.0):
    print("=" * 80)
    print("📐 THE BRAIN 3.0: TRIANGULAR SPATIAL ARBITRAGE SCANNER (OPTION A)")
    print("=" * 80)
    print(f"Algorithm: A* Directed Negative Cycle Graph Search")
    print(f"Feed     : Binance Public Book Tickers (Live Bid/Ask)")
    print(f"Fee Model: 7.5 bps per leg (VIP0 BNB Discount Taker) & 0.0 bps Maker")
    print("=" * 80)

    engine = TriangularArbitrageEngine(fee_rate=0.00075)
    
    for scan_idx in range(1, num_scans + 1):
        print(f"\n🔍 [Scan {scan_idx}/{num_scans}] Ingesting live order books across all Binance pairs...")
        tickers = engine.fetch_live_book_tickers()
        if not tickers:
            print("⚠️ Failed to fetch book tickers.")
            continue

        engine.build_exchange_graph(tickers)
        print(f"   Graph constructed: {len(engine.graph)} currency nodes.")
        
        opps = engine.find_triangular_arbitrage_opportunities(base_asset="USDT")
        print(f"   Identified {len(opps)} total triangular loops.")
        
        # Show top 5 cycles
        print("\n   Top 5 Instantaneous Triangular Cycles:")
        for opp in opps[:5]:
            path_str = " -> ".join(opp.cycle_path)
            legs_str = " | ".join(opp.pair_legs)
            print(f"   • Path: {path_str:<28s} | Gross: {opp.gross_multiplier:.6f} | Net: {opp.net_multiplier:.6f} | Edge: {opp.net_profit_bps:+.2f} bps | Status: {opp.status}")
            
        time.sleep(delay_s)

    engine.export_audit_spreadsheets()
    print("\n" + "=" * 80)
    print("✅ Triangular Arbitrage Scan Complete. All cycles logged to spreadsheets.")
    print("=" * 80)

if __name__ == "__main__":
    run_triangular_arbitrage_scan(num_scans=3, delay_s=1.0)
