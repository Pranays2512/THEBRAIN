#!/usr/bin/env python3
"""
brain3/finance/adapters/real_market_feed.py

Real Market Data Feed Adapter
Streams real-time live prices, bid/ask spreads, and volumes for real financial assets:
- Indian Indices: NIFTY 50 (^NSEI), BANK NIFTY (^NSEBANK)
- Indian Equities: RELIANCE.NS, TCS.NS, TATAMOTORS.NS
- Crypto Spot (INR converted): BTC/INR, ETH/INR, SOL/INR
- FX: USD/INR (USDINR=X)
"""

import json
import time
import urllib.request
import urllib.parse
import ssl
import threading
from typing import Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass

@dataclass
class LiveMarketTick:
    symbol: str
    price: float
    best_bid: float
    best_ask: float
    volume: float
    timestamp_ns: int
    source: str

class RealMarketFeedAdapter:
    def __init__(self, usd_inr_rate: float = 83.95):
        self.usd_inr = usd_inr_rate
        self.last_update_time = 0.0
        self.ssl_ctx = ssl.create_default_context()
        self.ssl_ctx.check_hostname = False
        self.ssl_ctx.verify_mode = ssl.CERT_NONE
        self._cached_prices: Dict[str, float] = {}

    def fetch_live_usdinr_rate(self) -> float:
        """Fetch real-time USD/INR rate from Yahoo Finance query."""
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/USDINR=X?interval=1m&range=1d"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (THEBRAIN/3.0)"})
            with urllib.request.urlopen(req, context=self.ssl_ctx, timeout=3.0) as resp:
                data = json.loads(resp.read().decode())
                meta = data["chart"]["result"][0]["meta"]
                rate = float(meta.get("regularMarketPrice", 83.95))
                if rate > 50.0:
                    self.usd_inr = rate
                    return rate
        except Exception:
            pass
        return self.usd_inr

    def fetch_crypto_ticker_binance(self, symbol: str = "BTCUSDT") -> Optional[LiveMarketTick]:
        """Fetch live best bid/ask and price from Binance public ticker API."""
        try:
            url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=self.ssl_ctx, timeout=2.5) as resp:
                data = json.loads(resp.read().decode())
                bid_usd = float(data["bidPrice"])
                ask_usd = float(data["askPrice"])
                bid_qty = float(data.get("bidQty", 1.0))
                ask_qty = float(data.get("askQty", 1.0))
                mid_usd = (bid_usd + ask_usd) / 2.0

                # Convert to INR
                sym_inr = symbol.replace("USDT", "/INR")
                price_inr = mid_usd * self.usd_inr
                bid_inr = bid_usd * self.usd_inr
                ask_inr = ask_usd * self.usd_inr

                self._cached_prices[sym_inr] = price_inr

                return LiveMarketTick(
                    symbol=sym_inr,
                    price=price_inr,
                    best_bid=bid_inr,
                    best_ask=ask_inr,
                    volume=(bid_qty + ask_qty) / 2.0,
                    timestamp_ns=int(time.time() * 1e9),
                    source="BINANCE_LIVE"
                )
        except Exception:
            return None

    def fetch_yahoo_live_quote(self, yf_symbol: str, target_symbol: str) -> Optional[LiveMarketTick]:
        """Fetch real-time stock/index quote from Yahoo Finance API."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yf_symbol)}?interval=1m&range=1d"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (THEBRAIN/3.0)"})
            with urllib.request.urlopen(req, context=self.ssl_ctx, timeout=3.0) as resp:
                data = json.loads(resp.read().decode())
                meta = data["chart"]["result"][0]["meta"]
                price = float(meta.get("regularMarketPrice") or meta.get("chartPreviousClose") or 100.0)
                
                # Synthetic tight spread around real live price
                spread = price * 0.0003
                best_bid = price - spread / 2.0
                best_ask = price + spread / 2.0
                vol = float(meta.get("regularMarketVolume", 1000.0))

                self._cached_prices[target_symbol] = price

                return LiveMarketTick(
                    symbol=target_symbol,
                    price=price,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    volume=vol,
                    timestamp_ns=int(time.time() * 1e9),
                    source="YAHOO_LIVE"
                )
        except Exception:
            return None

    def get_live_tick(self, instrument: str) -> Optional[LiveMarketTick]:
        """Get the latest real live market tick for any requested instrument."""
        if instrument in ["BTC/INR", "BTCUSDT"]:
            return self.fetch_crypto_ticker_binance("BTCUSDT")
        elif instrument in ["ETH/INR", "ETHUSDT"]:
            return self.fetch_crypto_ticker_binance("ETHUSDT")
        elif instrument in ["SOL/INR", "SOLUSDT"]:
            return self.fetch_crypto_ticker_binance("SOLUSDT")
        elif instrument in ["NIFTY50/INR", "NIFTY"]:
            return self.fetch_yahoo_live_quote("^NSEI", "NIFTY50/INR")
        elif instrument in ["BANKNIFTY/INR", "BANKNIFTY"]:
            return self.fetch_yahoo_live_quote("^NSEBANK", "BANKNIFTY/INR")
        elif instrument in ["RELIANCE/INR", "RELIANCE.NS"]:
            return self.fetch_yahoo_live_quote("RELIANCE.NS", "RELIANCE/INR")
        elif instrument in ["TCS/INR", "TCS.NS"]:
            return self.fetch_yahoo_live_quote("TCS.NS", "TCS/INR")
        elif instrument in ["USD/INR", "USDINR=X"]:
            rate = self.fetch_live_usdinr_rate()
            return LiveMarketTick(
                symbol="USD/INR",
                price=rate,
                best_bid=rate - 0.01,
                best_ask=rate + 0.01,
                volume=10000.0,
                timestamp_ns=int(time.time() * 1e9),
                source="FX_LIVE"
            )
        else:
            return self.fetch_crypto_ticker_binance("BTCUSDT")

    def stream_live_ticks(self, instruments=None, interval_sec: float = 0.5) -> Generator[LiveMarketTick, None, None]:
        """Continuous generator streaming live real-time market ticks across selected universe."""
        if instruments is None:
            instruments = ["BTC/INR", "ETH/INR", "SOL/INR", "NIFTY50/INR", "RELIANCE/INR"]

        # Initial USD/INR sync
        self.fetch_live_usdinr_rate()

        idx = 0
        while True:
            sym = instruments[idx % len(instruments)]
            tick = self.get_live_tick(sym)
            if tick:
                yield tick
            idx += 1
            time.sleep(interval_sec)

if __name__ == "__main__":
    feed = RealMarketFeedAdapter()
    print("Fetching live real-time quotes...")
    for sym in ["USD/INR", "BTC/INR", "ETH/INR", "NIFTY50/INR", "RELIANCE/INR"]:
        t = feed.get_live_tick(sym)
        if t:
            print(f"[{t.source}] {t.symbol}: Price = ₹{t.price:,.2f} | Bid = ₹{t.best_bid:,.2f} | Ask = ₹{t.best_ask:,.2f}")
        else:
            print(f"Failed to fetch live quote for {sym}")
