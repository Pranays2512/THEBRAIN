"""
Directional Alpha Engine for THE BRAIN 3.0
=========================================
Implements high-conviction directional quantitative strategies designed for
asymmetric returns (1:2.5 to 1:3.5 Risk:Reward) on 15m to 4h timeframes.

Key Features:
1. Multi-Timeframe Regime & Trend Analysis — HTF 1h EMA Confluence REQUIRED.
2. Volatility Compression Squeeze Scanner (Bollinger Bands vs ATR Keltner Channels)
   with mandatory Volume Expansion confirmation (breakout candle vol > 1.5x avg).
3. Smart Money Liquidity Sweep & Stop Hunt Detector.
4. Asymmetric Trade Manager with Hard Invalidation Stops & Fixed Take-Profit Targets
   (TRAILING_STOP exits are NOT implemented — only fixed TAKE_PROFIT / STOP_LOSS).
5. Fractional Kelly Position Sizer — adaptive sizing based on rolling 20-trade stats.
6. Per-symbol cooldown (3-bar minimum) after a stop-loss to prevent back-to-back losses.
"""

import math
import time
import json
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Deque

@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class DirectionalTrade:
    trade_id: str
    symbol: str
    side: str  # "BUY" (Long) or "SELL" (Short)
    entry_time: float
    entry_price: float
    position_size_usd: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    setup_type: str  # "VOLATILITY_SQUEEZE_BREAKOUT" or "LIQUIDITY_SWEEP_REVERSAL"
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    # Only "TAKE_PROFIT" and "STOP_LOSS" are produced; "TRAILING_STOP"/"TIMEOUT" are reserved but not implemented
    exit_reason: Optional[str] = None
    realized_pnl_usd: float = 0.0
    realized_pnl_pct: float = 0.0
    is_closed: bool = False

class DirectionalAlphaEngine:
    def __init__(self, min_risk_reward: float = 2.5, max_risk_per_trade_pct: float = 0.015):
        self.min_risk_reward = min_risk_reward
        self.max_risk_per_trade_pct = max_risk_per_trade_pct  # Fallback: 1.5% of equity per trade
        self.open_trades: List[DirectionalTrade] = []
        self.closed_trades: List[DirectionalTrade] = []
        self.trade_counter = 0

        # ── Kelly Criterion State (rolling 20-trade window) ──────────────────────
        # Stores (win: bool, rr: float) tuples for the last 20 closed trades
        self._kelly_window: Deque[Tuple[bool, float]] = deque(maxlen=20)
        self._kelly_fraction: float = 0.015  # Starts at fallback, updates after each fill

        # ── Per-symbol cooldown after stop-loss (keyed by symbol) ────────────────
        # Stores candle-count remaining before accepting a new signal
        self._cooldown_ticks: Dict[str, int] = {}
        self._cooldown_bars: int = 3  # Minimum bars after stop-loss

    # ── Kelly & Cooldown helpers ──────────────────────────────────────────────

    def update_kelly_stats(self, is_win: bool, rr_achieved: float) -> None:
        """Call this after every closed trade to keep the Kelly fraction current."""
        self._kelly_window.append((is_win, max(0.01, rr_achieved)))
        if len(self._kelly_window) < 5:
            return  # Need at least 5 trades before trusting Kelly
        win_rate = sum(1 for w, _ in self._kelly_window if w) / len(self._kelly_window)
        avg_rr = sum(r for _, r in self._kelly_window) / len(self._kelly_window)
        # Full Kelly: f* = (W*R - (1-W)) / R
        full_kelly = (win_rate * avg_rr - (1.0 - win_rate)) / avg_rr
        # Half-Kelly for safety; floor at 0.5%, cap at 4% of equity
        self._kelly_fraction = max(0.005, min(0.04, full_kelly * 0.5))

    def _realized_rr(self, trade: DirectionalTrade) -> float:
        """Realized reward/risk ratio from actual entry/exit/stop prices.

        Falls back to the configured risk_reward_ratio only when the realized
        risk distance is degenerate (zero/negative)."""
        exit_price = trade.exit_price if trade.exit_price is not None else trade.entry_price
        risk_per_unit = abs(trade.entry_price - trade.stop_loss_price)
        reward_per_unit = abs(exit_price - trade.entry_price)
        if risk_per_unit <= 0:
            return trade.risk_reward_ratio
        return reward_per_unit / risk_per_unit

    def _tick_cooldown(self, symbol: str) -> None:
        """Decrements cooldown counter for a symbol each time a candle is evaluated."""
        if symbol in self._cooldown_ticks and self._cooldown_ticks[symbol] > 0:
            self._cooldown_ticks[symbol] -= 1

    def _is_in_cooldown(self, symbol: str) -> bool:
        return self._cooldown_ticks.get(symbol, 0) > 0

    def _trigger_cooldown(self, symbol: str) -> None:
        self._cooldown_ticks[symbol] = self._cooldown_bars

    # ── EMA helper ────────────────────────────────────────────────────────────

    def calculate_ema(self, candles: List[Candle], period: int) -> float:
        """Exponential moving average of close prices."""
        if len(candles) < period:
            return candles[-1].close if candles else 0.0
        closes = [c.close for c in candles]
        k = 2.0 / (period + 1)
        ema = sum(closes[:period]) / period  # SMA seed
        for price in closes[period:]:
            ema = price * k + ema * (1.0 - k)
        return ema

    # ── HTF Confluence Filter ─────────────────────────────────────────────────

    def fetch_htf_1h_candles(self, symbol: str, limit: int = 55) -> List[Candle]:
        """
        Fetches the last `limit` 1-hour klines from Binance REST.
        Returns empty list on failure — caller treats that as NO_DATA (skip signal).
        """
        binance_symbol = (symbol.replace("-", "").replace("/", "")
                          .replace("XBT", "BTC").upper())
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={binance_symbol}&interval=1h&limit={limit}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "THEBRAIN/3.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = json.loads(resp.read().decode())
            return [
                Candle(
                    timestamp=float(row[0]), open=float(row[1]),
                    high=float(row[2]), low=float(row[3]),
                    close=float(row[4]), volume=float(row[5])
                )
                for row in raw
            ]
        except Exception:
            return []

    def check_htf_ema_confluence(self, htf_candles: List[Candle], signal_side: str) -> bool:
        """
        Returns True only when the 1h trend aligns with the trade direction.
        BUY: EMA21 > EMA50 on 1h | SELL: EMA21 < EMA50 on 1h.
        """
        if len(htf_candles) < 55:
            return False  # Insufficient data — skip rather than risk false signal
        ema21 = self.calculate_ema(htf_candles, 21)
        ema50 = self.calculate_ema(htf_candles, 50)
        if signal_side == "BUY":
            return ema21 > ema50
        elif signal_side == "SELL":
            return ema21 < ema50
        return False

    # ── Volume Expansion Filter ───────────────────────────────────────────────

    def check_volume_expansion(self, candles: List[Candle], lookback: int = 20) -> bool:
        """
        Returns True when the breakout candle has volume >= 1.5x the 20-bar average.
        Filters low-conviction, false breakouts in sideways chop.
        """
        if len(candles) < lookback + 1:
            return False
        recent_vol = candles[-1].volume
        avg_vol = sum(c.volume for c in candles[-(lookback + 1):-1]) / lookback
        return avg_vol > 0 and recent_vol >= 1.5 * avg_vol

    def calculate_atr(self, candles: List[Candle], period: int = 14) -> float:

        if len(candles) < period + 1:
            return candles[-1].close * 0.01 if candles else 1.0
        trs = []
        for i in range(1, len(candles)):
            c_curr = candles[i]
            c_prev = candles[i-1]
            tr = max(c_curr.high - c_curr.low, abs(c_curr.high - c_prev.close), abs(c_curr.low - c_prev.close))
            trs.append(tr)
        return sum(trs[-period:]) / period

    def calculate_bollinger_bands(self, candles: List[Candle], period: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        if len(candles) < period:
            c = candles[-1].close if candles else 100.0
            return c * 1.02, c, c * 0.98
        closes = [c.close for c in candles[-period:]]
        mean = sum(closes) / period
        variance = sum((x - mean) ** 2 for x in closes) / period
        std_dev = math.sqrt(variance)
        return mean + (num_std * std_dev), mean, mean - (num_std * std_dev)

    def calculate_keltner_channels(self, candles: List[Candle], period: int = 20, multiplier: float = 1.5) -> Tuple[float, float, float]:
        atr = self.calculate_atr(candles, period)
        if len(candles) < period:
            c = candles[-1].close if candles else 100.0
            return c + atr * multiplier, c, c - atr * multiplier
        closes = [c.close for c in candles[-period:]]
        ema = sum(closes) / period
        return ema + (multiplier * atr), ema, ema - (multiplier * atr)

    def detect_volatility_squeeze(self, candles: List[Candle]) -> Tuple[bool, str, float]:
        """
        Detects if Bollinger Bands are inside Keltner Channels (Squeeze in effect)
        and if a breakout is currently firing.
        """
        if len(candles) < 25:
            return False, "NONE", 0.0
        
        bb_upper, bb_mid, bb_lower = self.calculate_bollinger_bands(candles, 20, 2.0)
        kc_upper, kc_mid, kc_lower = self.calculate_keltner_channels(candles, 20, 1.5)
        
        # Squeeze condition: BB inside KC
        in_squeeze = (bb_upper < kc_upper) and (bb_lower > kc_lower)
        
        current_close = candles[-1].close
        prev_close = candles[-2].close
        atr = self.calculate_atr(candles, 14)
        
        # Momentum check
        momentum = (current_close - prev_close) / (atr if atr > 0 else 1.0)
        
        # Breakout condition: Price closes outside BB upper or lower with strong momentum
        if current_close > bb_upper and momentum > 0.8:
            return True, "BULLISH_BREAKOUT", momentum
        elif current_close < bb_lower and momentum < -0.8:
            return True, "BEARISH_BREAKOUT", momentum
            
        return False, "IN_SQUEEZE" if in_squeeze else "NO_SQUEEZE", momentum

    def detect_liquidity_sweep(self, candles: List[Candle], lookback: int = 20) -> Tuple[bool, str, float]:
        """
        Detects smart money stop hunts: price pierces a key high/low but closes back inside with a long wick.
        """
        if len(candles) < lookback + 1:
            return False, "NONE", 0.0
        
        recent = candles[-1]
        prior_high = max(c.high for c in candles[-lookback-1:-1])
        prior_low = min(c.low for c in candles[-lookback-1:-1])
        
        body = abs(recent.close - recent.open)
        total_range = recent.high - recent.low
        if total_range == 0:
            return False, "NONE", 0.0
            
        upper_wick = recent.high - max(recent.open, recent.close)
        lower_wick = min(recent.open, recent.close) - recent.low
        
        # Bearish Liquidity Sweep: pierced prior high, rejected with massive upper wick
        if recent.high > prior_high and recent.close < prior_high and (upper_wick / total_range) > 0.55:
            return True, "BEARISH_SWEEP_REVERSAL", recent.high
            
        # Bullish Liquidity Sweep: pierced prior low, rejected with massive lower wick
        if recent.low < prior_low and recent.close > prior_low and (lower_wick / total_range) > 0.55:
            return True, "BULLISH_SWEEP_REVERSAL", recent.low
            
        return False, "NONE", 0.0

    def evaluate_directional_signal(
        self,
        symbol: str,
        candles: List[Candle],
        current_equity: float,
        htf_candles: Optional[List[Candle]] = None
    ) -> Optional[DirectionalTrade]:
        """
        Evaluates multi-indicator setup for a high-asymmetry directional entry.

        Gating filters (ALL must pass before a trade fires):
          1. Volatility Squeeze Breakout OR Liquidity Sweep pattern detected on LTF.
          2. HTF 1h EMA21 > EMA50 for BUY / EMA21 < EMA50 for SELL (trend alignment).
          3. Breakout candle volume >= 1.5x the 20-bar average (expansion confirmation).
          4. No active cooldown on this symbol (3-bar minimum gap after a stop-loss).
        
        Position sizing uses fractional Kelly (adaptive), falling back to max_risk_per_trade_pct.
        """
        # Decrement cooldown counter each bar evaluated
        self._tick_cooldown(symbol)

        if len(candles) < 30:
            return None

        # ── Gate 4: Cooldown check (stop-chase prevention) ───────────────────
        if self._is_in_cooldown(symbol):
            return None

        current_price = candles[-1].close
        atr = self.calculate_atr(candles, 14)

        # ── Step 1: Detect LTF pattern ───────────────────────────────────────
        is_breakout, squeeze_state, momentum = self.detect_volatility_squeeze(candles)
        is_sweep, sweep_state, level = self.detect_liquidity_sweep(candles)

        side = None
        setup = None
        stop_loss = None
        take_profit = None

        if is_breakout and squeeze_state == "BULLISH_BREAKOUT":
            side = "BUY"
            setup = "VOLATILITY_SQUEEZE_BREAKOUT"
            stop_loss = current_price - (1.2 * atr)
            risk_dist = current_price - stop_loss
            take_profit = current_price + (risk_dist * self.min_risk_reward)

        elif is_breakout and squeeze_state == "BEARISH_BREAKOUT":
            side = "SELL"
            setup = "VOLATILITY_SQUEEZE_BREAKOUT"
            stop_loss = current_price + (1.2 * atr)
            risk_dist = stop_loss - current_price
            take_profit = current_price - (risk_dist * self.min_risk_reward)

        elif is_sweep and sweep_state == "BULLISH_SWEEP_REVERSAL":
            side = "BUY"
            setup = "LIQUIDITY_SWEEP_REVERSAL"
            stop_loss = level - (0.2 * atr)
            risk_dist = current_price - stop_loss
            take_profit = current_price + (risk_dist * 3.0)

        elif is_sweep and sweep_state == "BEARISH_SWEEP_REVERSAL":
            side = "SELL"
            setup = "LIQUIDITY_SWEEP_REVERSAL"
            stop_loss = level + (0.2 * atr)
            risk_dist = stop_loss - current_price
            take_profit = current_price - (risk_dist * 3.0)

        if not (side and stop_loss and take_profit):
            return None

        # ── Gate 2: HTF 1h EMA Confluence ───────────────────────────────────
        # Fetch live 1h candles if not pre-supplied (allows callers to pre-fetch)
        if htf_candles is None:
            htf_candles = self.fetch_htf_1h_candles(symbol)

        if not self.check_htf_ema_confluence(htf_candles, side):
            return None  # HTF trend opposes LTF signal — skip

        # ── Gate 3: Volume Expansion on breakout candle ─────────────────────
        if not self.check_volume_expansion(candles):
            return None  # Low-volume chop breakout — skip

        # ── All gates passed → size and create trade ─────────────────────────
        risk_per_unit = abs(current_price - stop_loss)
        target_per_unit = abs(take_profit - current_price)
        rr = target_per_unit / risk_per_unit if risk_per_unit > 0 else 0.0

        if rr < self.min_risk_reward:
            return None

        self.trade_counter += 1

        # Fractional Kelly position sizing (adaptive; falls back to max_risk_per_trade_pct)
        kelly_pct = self._kelly_fraction  # Updated by update_kelly_stats() after each close
        risk_budget_usd = current_equity * kelly_pct
        position_size_units = risk_budget_usd / risk_per_unit if risk_per_unit > 0 else 0.0
        position_size_usd = position_size_units * current_price

        trade = DirectionalTrade(
            trade_id=f"DIR-{symbol}-{self.trade_counter:04d}",
            symbol=symbol,
            side=side,
            entry_time=time.time(),
            entry_price=current_price,
            position_size_usd=position_size_usd,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            risk_reward_ratio=round(rr, 2),
            setup_type=setup
        )
        self.open_trades.append(trade)
        return trade

    def update_open_trades(self, current_prices: Dict[str, float], fee_bps: float = 4.0) -> List[DirectionalTrade]:
        """
        Monitors open positions and triggers fixed take profits and stop losses.
        NOTE: TRAILING_STOP exits are documented in the trade schema but are NOT
        implemented here — only TAKE_PROFIT and STOP_LOSS can close a trade.
        Every exit path records (is_win, rr_achieved) into the Kelly window so
        the adaptive sizing never decays on a single outcome type.
        """
        closed_this_tick = []
        fee_mult = (fee_bps / 10000.0) * 2.0  # Round-trip taker fee
        
        for trade in list(self.open_trades):
            price = current_prices.get(trade.symbol)
            if not price:
                continue
                
            if trade.side == "BUY":
                # Take profit hit
                if price >= trade.take_profit_price:
                    gross_pnl_pct = (price - trade.entry_price) / trade.entry_price
                    net_pnl_pct = gross_pnl_pct - fee_mult
                    trade.realized_pnl_usd = trade.position_size_usd * net_pnl_pct
                    trade.realized_pnl_pct = net_pnl_pct
                    trade.exit_price = price
                    trade.exit_time = time.time()
                    trade.exit_reason = "TAKE_PROFIT"
                    trade.is_closed = True
                    self.open_trades.remove(trade)
                    self.closed_trades.append(trade)
                    closed_this_tick.append(trade)
                    # Record the win in Kelly stats (realized RR from actual entry/exit/stop)
                    self.update_kelly_stats(is_win=True, rr_achieved=self._realized_rr(trade))
                    
                # Stop loss hit
                elif price <= trade.stop_loss_price:
                    gross_pnl_pct = (price - trade.entry_price) / trade.entry_price
                    net_pnl_pct = gross_pnl_pct - fee_mult
                    trade.realized_pnl_usd = trade.position_size_usd * net_pnl_pct
                    trade.realized_pnl_pct = net_pnl_pct
                    trade.exit_price = price
                    trade.exit_time = time.time()
                    trade.exit_reason = "STOP_LOSS"
                    trade.is_closed = True
                    self.open_trades.remove(trade)
                    self.closed_trades.append(trade)
                    closed_this_tick.append(trade)
                    # Trigger per-symbol cooldown & update Kelly stats
                    self._trigger_cooldown(trade.symbol)
                    self.update_kelly_stats(is_win=False, rr_achieved=0.0)

            elif trade.side == "SELL":
                # Take profit hit (price fell to target)
                if price <= trade.take_profit_price:
                    gross_pnl_pct = (trade.entry_price - price) / trade.entry_price
                    net_pnl_pct = gross_pnl_pct - fee_mult
                    trade.realized_pnl_usd = trade.position_size_usd * net_pnl_pct
                    trade.realized_pnl_pct = net_pnl_pct
                    trade.exit_price = price
                    trade.exit_time = time.time()
                    trade.exit_reason = "TAKE_PROFIT"
                    trade.is_closed = True
                    self.open_trades.remove(trade)
                    self.closed_trades.append(trade)
                    closed_this_tick.append(trade)
                    # Record the win in Kelly stats (realized RR from actual entry/exit/stop)
                    self.update_kelly_stats(is_win=True, rr_achieved=self._realized_rr(trade))
                    
                # Stop loss hit (price rose above stop)
                elif price >= trade.stop_loss_price:
                    gross_pnl_pct = (trade.entry_price - price) / trade.entry_price
                    net_pnl_pct = gross_pnl_pct - fee_mult
                    trade.realized_pnl_usd = trade.position_size_usd * net_pnl_pct
                    trade.realized_pnl_pct = net_pnl_pct
                    trade.exit_price = price
                    trade.exit_time = time.time()
                    trade.exit_reason = "STOP_LOSS"
                    trade.is_closed = True
                    self.open_trades.remove(trade)
                    self.closed_trades.append(trade)
                    closed_this_tick.append(trade)
                    # Record the loss in Kelly stats (mirrors the BUY stop-loss path;
                    # per-symbol cooldown intentionally left unchanged)
                    self.update_kelly_stats(is_win=False, rr_achieved=0.0)
                    
        return closed_this_tick
