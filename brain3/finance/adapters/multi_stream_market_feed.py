#!/usr/bin/env python3
"""
brain3/finance/adapters/multi_stream_market_feed.py

High-Throughput Multi-Stream Market Data Aggregator
Streams hundreds of live instruments concurrently across:
1. Indian Equities & Indices (NSE / BSE Top 30)
2. Global Crypto Spot (Binance Multi-Ticker WebSocket)
3. Global Tech Equities (US Top 10)
4. Foreign Exchange & Commodities (USD/INR, Gold, Crude)
"""

import sys
import os
import json
import time
import ssl
import threading
import queue
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Generator
from concurrent.futures import ThreadPoolExecutor

import websockets
import asyncio

@dataclass
class MultiAssetTick:
    symbol: str
    asset_class: str  # "INDIAN_EQUITY", "CRYPTO_INR", "GLOBAL_EQUITY", "COMMODITY_FX"
    price: float
    best_bid: float
    best_ask: float
    volume: float
    change_24h_pct: float
    timestamp_ns: int
    source: str

class MultiStreamMarketFeed:
    # 30 Top Indian Companies & Indices
    INDIAN_UNIVERSE = [
        ("^NSEI", "NIFTY50/INR", "INDEX"),
        ("^NSEBANK", "BANKNIFTY/INR", "INDEX"),
        ("RELIANCE.NS", "RELIANCE/INR", "ENERGY"),
        ("TCS.NS", "TCS/INR", "IT"),
        ("HDFCBANK.NS", "HDFCBANK/INR", "BANK"),
        ("INFY.NS", "INFOSYS/INR", "IT"),
        ("ICICIBANK.NS", "ICICIBANK/INR", "BANK"),
        ("TATAMOTORS.NS", "TATAMOTORS/INR", "AUTO"),
        ("SBIN.NS", "SBIN/INR", "BANK"),
        ("BHARTIARTL.NS", "AIRTEL/INR", "TELECOM"),
        ("ITC.NS", "ITC/INR", "FMCG"),
        ("LT.NS", "LT/INR", "INFRA"),
        ("BAJFINANCE.NS", "BAJFINANCE/INR", "FINANCE"),
        ("ADANIENT.NS", "ADANIENT/INR", "CONGLOMERATE"),
        ("MARUTI.NS", "MARUTI/INR", "AUTO"),
        ("SUNPHARMA.NS", "SUNPHARMA/INR", "PHARMA"),
        ("KOTAKBANK.NS", "KOTAKBANK/INR", "BANK"),
        ("AXISBANK.NS", "AXISBANK/INR", "BANK"),
        ("WIPRO.NS", "WIPRO/INR", "IT"),
        ("HCLTECH.NS", "HCLTECH/INR", "IT"),
        ("TITAN.NS", "TITAN/INR", "CONSUMER"),
        ("TATASTEEL.NS", "TATASTEEL/INR", "METALS"),
        ("NTPC.NS", "NTPC/INR", "POWER"),
        ("POWERGRID.NS", "POWERGRID/INR", "POWER"),
        ("ONGC.NS", "ONGC/INR", "ENERGY")
    ]

    # Global Tech Equities
    GLOBAL_UNIVERSE = [
        ("AAPL", "AAPL/INR", "TECH"),
        ("MSFT", "MSFT/INR", "TECH"),
        ("NVDA", "NVDA/INR", "AI_CHIPS"),
        ("GOOGL", "GOOGL/INR", "TECH"),
        ("AMZN", "AMZN/INR", "ECOMMERCE"),
        ("TSLA", "TSLA/INR", "AUTO_AI")
    ]

    def __init__(self, usd_inr_rate: float = 83.95):
        self.usd_inr = usd_inr_rate
        self.tick_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.running = True
        self.latest_prices: Dict[str, MultiAssetTick] = {}

        # TLS: certificate verification is ON by default (CERT_REQUIRED via the
        # default SSL context). Insecure mode is a deliberate escape hatch for
        # environments where a feed genuinely breaks under verified TLS — it must
        # be opted into EXPLICITLY via BRAIN_INSECURE_FEEDS=1 and is never silent:
        # enabling it DISABLES ALL CERTIFICATE VERIFICATION (MITM risk).
        self.ssl_ctx = ssl.create_default_context()
        if os.environ.get("BRAIN_INSECURE_FEEDS") == "1":
            # WARNING: insecure — do not enable in production.
            self.ssl_ctx.check_hostname = False
            self.ssl_ctx.verify_mode = ssl.CERT_NONE

        # Threads
        self.ws_thread: Optional[threading.Thread] = None
        self.equity_thread: Optional[threading.Thread] = None

    def start(self):
        """Start asynchronous ingestion workers across crypto WebSocket and equity thread pool."""
        self.running = True
        self.ws_thread = threading.Thread(target=self._run_crypto_websocket, daemon=True)
        self.ws_thread.start()

        self.equity_thread = threading.Thread(target=self._run_equities_polling_loop, daemon=True)
        self.equity_thread.start()

    def stop(self):
        self.running = False

    def _run_crypto_websocket(self):
        """Connect to Binance real-time all-ticker WebSocket and stream 50+ crypto pairs."""
        async def _ws_loop():
            uri = "wss://stream.binance.com:9443/ws/!miniTicker@arr"
            while self.running:
                try:
                    async with websockets.connect(uri, ping_interval=20, ping_timeout=10, ssl=self.ssl_ctx) as ws:
                        while self.running:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            ts = int(time.time() * 1e9)
                            for item in data:
                                raw_sym = item.get("s", "")
                                if raw_sym.endswith("USDT"):
                                    sym_clean = raw_sym[:-4] + "/INR"
                                    close_usd = float(item.get("c", 0.0))
                                    high_usd = float(item.get("h", close_usd))
                                    low_usd = float(item.get("l", close_usd))
                                    vol = float(item.get("v", 0.0))

                                    price_inr = close_usd * self.usd_inr
                                    spread = price_inr * 0.0004
                                    bid_inr = price_inr - spread / 2.0
                                    ask_inr = price_inr + spread / 2.0
                                    
                                    open_usd = float(item.get("o", close_usd))
                                    change_pct = ((close_usd - open_usd) / open_usd * 100.0) if open_usd > 0 else 0.0

                                    tick = MultiAssetTick(
                                        symbol=sym_clean,
                                        asset_class="CRYPTO_INR",
                                        price=price_inr,
                                        best_bid=bid_inr,
                                        best_ask=ask_inr,
                                        volume=vol,
                                        change_24h_pct=change_pct,
                                        timestamp_ns=ts,
                                        source="BINANCE_WS"
                                    )
                                    self.latest_prices[sym_clean] = tick
                                    try:
                                        self.tick_queue.put_nowait(tick)
                                    except queue.Full:
                                        pass
                except Exception:
                    await asyncio.sleep(2.0)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_ws_loop())

    def _fetch_single_equity(self, item) -> Optional[MultiAssetTick]:
        raw_sym, display_sym, sector = item
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(raw_sym)}?interval=1m&range=1d"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (THEBRAIN/3.0)"})
            with urllib.request.urlopen(req, context=self.ssl_ctx, timeout=3.0) as resp:
                d = json.loads(resp.read().decode())
                meta = d["chart"]["result"][0]["meta"]
                p = float(meta.get("regularMarketPrice") or meta.get("chartPreviousClose") or 100.0)
                prev_c = float(meta.get("chartPreviousClose", p))
                chg_pct = ((p - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
                vol = float(meta.get("regularMarketVolume", 1000.0))

                # If global USD equity, convert to INR
                if not raw_sym.endswith(".NS") and not raw_sym.startswith("^"):
                    p = p * self.usd_inr

                spread = p * 0.0003
                tick = MultiAssetTick(
                    symbol=display_sym,
                    asset_class="INDIAN_EQUITY" if raw_sym.endswith(".NS") or raw_sym.startswith("^") else "GLOBAL_EQUITY",
                    price=p,
                    best_bid=p - spread / 2.0,
                    best_ask=p + spread / 2.0,
                    volume=vol,
                    change_24h_pct=chg_pct,
                    timestamp_ns=int(time.time() * 1e9),
                    source="YAHOO_NSE_PARALLEL"
                )
                return tick
        except Exception:
            return None

    def _run_equities_polling_loop(self):
        """Poll 30+ Indian & Global equities in parallel worker threads."""
        all_equities = self.INDIAN_UNIVERSE + self.GLOBAL_UNIVERSE
        while self.running:
            try:
                with ThreadPoolExecutor(max_workers=12) as ex:
                    results = list(ex.map(self._fetch_single_equity, all_equities))
                for t in results:
                    if t:
                        self.latest_prices[t.symbol] = t
                        try:
                            self.tick_queue.put_nowait(t)
                        except queue.Full:
                            pass
            except Exception:
                pass
            time.sleep(1.0)

    def get_market_snapshot(self) -> Dict[str, MultiAssetTick]:
        """Return instantaneous snapshot of all monitored symbols."""
        return dict(self.latest_prices)

    def get_next_tick(self, timeout: float = 0.1) -> Optional[MultiAssetTick]:
        """Get next tick from real-time multiplexed queue."""
        try:
            return self.tick_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stream_ticks(self) -> Generator[MultiAssetTick, None, None]:
        """Yield ticks continuously while feed is running."""
        while self.running:
            tick = self.get_next_tick(timeout=0.05)
            if tick:
                yield tick

if __name__ == "__main__":
    feed = MultiStreamMarketFeed()
    print("Starting Multi-Stream Market Ingestion across Crypto, Indian Equities, and Global Stocks...")
    feed.start()
    time.sleep(2.0)
    snap = feed.get_market_snapshot()
    print(f"\nActively streaming {len(snap)} instruments simultaneously:")
    for sym, tick in list(snap.items())[:15]:
        print(f"  [{tick.asset_class:14s}] {tick.symbol:16s} : ₹{tick.price:>12,.2f} ({tick.change_24h_pct:+.2f}%) | Source: {tick.source}")
    feed.stop()
