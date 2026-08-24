#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "order_book.hpp"

namespace brain3 {
namespace finance {

struct MarketTick {
    uint64_t timestamp_ns{0};
    double price{0.0};
    double volume{0.0};
    double best_bid{0.0};
    double best_ask{0.0};
    double bid_vol{0.0};
    double ask_vol{0.0};
    int trade_sign{0}; // +1 Buy, -1 Sell, 0 Unknown
};

class MicrostructureAnalyzer {
private:
    std::string symbol_;
    std::deque<MarketTick> tick_history_;
    size_t max_history_size_{1000};

    double cumulative_dollar_volume_{0.0};
    double cumulative_volume_{0.0};
    double prev_bid_{0.0};
    double prev_ask_{0.0};
    double prev_bid_vol_{0.0};
    double prev_ask_vol_{0.0};
    double rolling_ofi_{0.0};

public:
    explicit MicrostructureAnalyzer(std::string symbol = "BTC/USDT", size_t max_history = 1000)
        : symbol_(std::move(symbol)), max_history_size_(max_history) {}

    // Ingest a tick and calculate OFI, VWAP, and trade direction
    void on_tick(double price, double volume, double best_bid, double best_ask, double bid_vol, double ask_vol, uint64_t timestamp_ns = 0) {
        MarketTick tick;
        tick.timestamp_ns = timestamp_ns;
        tick.price = price;
        tick.volume = volume;
        tick.best_bid = best_bid;
        tick.best_ask = best_ask;
        tick.bid_vol = bid_vol;
        tick.ask_vol = ask_vol;

        // Lee-Ready Trade Direction Classification.
        // Guard the mid: with crossed/zero/garbage quotes the mid is meaningless
        // (a 0 mid would classify every trade as buyer-initiated), so mark the
        // tick unknown instead of classifying it.
        const bool quotes_valid = (best_bid > 1e-9 && best_ask > 1e-9 && best_ask >= best_bid);
        if (!quotes_valid) {
            tick.trade_sign = 0; // Unknown — skip classification for this tick
        } else {
            double mid = (best_bid + best_ask) * 0.5;
            if (price > mid + 1e-9) {
                tick.trade_sign = 1; // Buyer initiated
            } else if (price < mid - 1e-9) {
                tick.trade_sign = -1; // Seller initiated
            } else {
                // Tick rule fallback
                if (!tick_history_.empty()) {
                    double last_p = tick_history_.back().price;
                    if (price > last_p) tick.trade_sign = 1;
                    else if (price < last_p) tick.trade_sign = -1;
                    else tick.trade_sign = tick_history_.back().trade_sign;
                } else {
                    tick.trade_sign = 0;
                }
            }
        }

        // Order Flow Imbalance (OFI) Calculation:
        // Delta V_bid: if bid > prev_bid: bid_vol; if bid == prev_bid: bid_vol - prev_bid_vol; else: 0
        // Delta V_ask: if ask < prev_ask: ask_vol; if ask == prev_ask: ask_vol - prev_ask_vol; else: 0
        // Both sides require a valid previous quote so the FIRST tick does not
        // inject a phantom flow spike (e.g. delta_bid = full bid_vol vs prev=0).
        double delta_bid = 0.0;
        if (prev_bid_ > 0.0 && best_bid > prev_bid_) delta_bid = bid_vol;
        else if (prev_bid_ > 0.0 && std::abs(best_bid - prev_bid_) < 1e-9) delta_bid = bid_vol - prev_bid_vol_;

        double delta_ask = 0.0;
        if (prev_ask_ > 0.0 && best_ask < prev_ask_) delta_ask = ask_vol;
        else if (prev_ask_ > 0.0 && std::abs(best_ask - prev_ask_) < 1e-9) delta_ask = ask_vol - prev_ask_vol_;

        double tick_ofi = delta_bid - delta_ask;
        rolling_ofi_ = 0.9 * rolling_ofi_ + 0.1 * tick_ofi;

        prev_bid_ = best_bid;
        prev_ask_ = best_ask;
        prev_bid_vol_ = bid_vol;
        prev_ask_vol_ = ask_vol;

        cumulative_dollar_volume_ += (price * volume);
        cumulative_volume_ += volume;

        tick_history_.push_back(tick);
        if (tick_history_.size() > max_history_size_) {
            tick_history_.pop_front();
        }
    }

    double vwap() const {
        if (cumulative_volume_ <= 1e-9) return 0.0;
        return cumulative_dollar_volume_ / cumulative_volume_;
    }

    double ofi() const {
        return rolling_ofi_;
    }

    // Realized Volatility over recent N ticks (annualized in bps)
    double realized_volatility(size_t window = 50) const {
        if (tick_history_.size() < 2) return 0.0;
        size_t n = std::min(window, tick_history_.size());
        
        std::vector<double> log_returns;
        log_returns.reserve(n);

        for (size_t i = tick_history_.size() - n + 1; i < tick_history_.size(); ++i) {
            double p_curr = tick_history_[i].price;
            double p_prev = tick_history_[i - 1].price;
            if (p_prev > 1e-9 && p_curr > 1e-9) {
                log_returns.push_back(std::log(p_curr / p_prev));
            }
        }

        if (log_returns.size() < 2) return 0.0;

        double mean = std::accumulate(log_returns.begin(), log_returns.end(), 0.0) / log_returns.size();
        double var = 0.0;
        for (double r : log_returns) {
            var += (r - mean) * (r - mean);
        }
        var /= (log_returns.size() - 1);
        
        // Standard deviation per tick
        return std::sqrt(var);
    }

    // Effective Spread = 2 * |Trade Price - Midpoint|
    double effective_spread_bps() const {
        if (tick_history_.empty()) return 0.0;
        const auto& t = tick_history_.back();
        double mid = (t.best_bid + t.best_ask) * 0.5;
        if (mid <= 1e-9) return 0.0;
        return (2.0 * std::abs(t.price - mid) / mid) * 10000.0;
    }

    // Kyle's Lambda estimated price impact regression
    double estimate_kyle_lambda() const {
        if (tick_history_.size() < 10) return 0.0001;
        double sum_xy = 0.0;
        double sum_x2 = 0.0;

        for (size_t i = 1; i < tick_history_.size(); ++i) {
            double dp = tick_history_[i].price - tick_history_[i - 1].price;
            double signed_v = tick_history_[i].volume * tick_history_[i].trade_sign;
            sum_xy += dp * signed_v;
            sum_x2 += signed_v * signed_v;
        }

        if (sum_x2 < 1e-9) return 0.0001;
        return std::max(1e-6, sum_xy / sum_x2);
    }

    std::string to_json_summary() const {
        std::ostringstream oss;
        oss << "{"
            << "\"symbol\":\"" << symbol_ << "\","
            << "\"vwap\":" << std::fixed << std::setprecision(4) << vwap() << ","
            << "\"ofi\":" << ofi() << ","
            << "\"realized_vol\":" << std::setprecision(6) << realized_volatility(50) << ","
            << "\"effective_spread_bps\":" << std::setprecision(2) << effective_spread_bps() << ","
            << "\"kyle_lambda\":" << std::setprecision(8) << estimate_kyle_lambda() << ","
            << "\"tick_count\":" << tick_history_.size()
            << "}";
        return oss.str();
    }
};

} // namespace finance
} // namespace brain3
