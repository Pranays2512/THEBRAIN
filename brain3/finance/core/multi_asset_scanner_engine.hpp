#pragma once

#include "survival_instinct_engine.hpp"
#include "order_book.hpp"
#include "market_microstructure.hpp"
#include "alpha_conviction.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <memory>
#include <random>

namespace brain3 {
namespace finance {

struct ScoredTradeOpportunity {
    std::string symbol;
    std::string strategy;
    std::string side;
    double price{0.0};
    double best_bid{0.0};
    double best_ask{0.0};
    double alpha_score{0.0};
    double ofi{0.0};
    double vwap_dev{0.0};
    double win_probability{0.50};
    double win_loss_ratio{1.0};
    double recommended_size_inr{0.0};
};

class MultiAssetScannerEngine {
private:
    SurvivalInstinctEngine& survival_;
    std::unordered_map<std::string, std::unique_ptr<LimitOrderBook>> books_;
    std::unordered_map<std::string, std::unique_ptr<MicrostructureAnalyzer>> telemetry_;
    std::unordered_map<std::string, double> last_prices_;
    std::unordered_map<std::string, double> price_24h_changes_;
    std::mt19937_64 rng_{1337};

public:
    explicit MultiAssetScannerEngine(SurvivalInstinctEngine& survival)
        : survival_(survival) {}

    LimitOrderBook* get_or_create_book(const std::string& symbol, double initial_price = 100.0) {
        auto it = books_.find(symbol);
        if (it == books_.end()) {
            books_[symbol] = std::make_unique<LimitOrderBook>(symbol, initial_price);
            telemetry_[symbol] = std::make_unique<MicrostructureAnalyzer>(symbol);
            last_prices_[symbol] = initial_price;
            return books_[symbol].get();
        }
        return it->second.get();
    }

    MicrostructureAnalyzer* get_telemetry(const std::string& symbol) {
        if (telemetry_.find(symbol) == telemetry_.end()) {
            get_or_create_book(symbol);
        }
        return telemetry_.at(symbol).get();
    }

    // Process a live tick from the multi-stream firehose
    ScoredTradeOpportunity process_incoming_tick(const std::string& symbol,
                                                  double price,
                                                  double best_bid,
                                                  double best_ask,
                                                  double volume,
                                                  double change_24h_pct = 0.0) {
        auto* book = get_or_create_book(symbol, price);
        auto* micro = get_telemetry(symbol);

        last_prices_[symbol] = price;
        price_24h_changes_[symbol] = change_24h_pct;

        // Update book mid price and order flow telemetry.
        // NOTE: seed_liquidity now MERGES — resting limit orders survive across
        // ticks instead of being wiped on every incoming tick.
        book->seed_liquidity(price, 6);
        double depth_est = std::max(volume * 0.5, 100.0);
        micro->on_tick(price, volume, best_bid, best_ask, depth_est, depth_est);

        double ofi_val = micro->ofi();
        double vwap_val = micro->vwap();
        double vwap_dev = (vwap_val > 0.0) ? (price - vwap_val) / vwap_val : 0.0;
        double vol = micro->realized_volatility();
        if (vol <= 0.0) vol = 0.015;

        // Multi-Alpha Scoring
        // 1. Order Flow Imbalance Momentum
        double ofi_score = std::min(std::abs(ofi_val) / 50.0, 1.0);
        // 2. VWAP Mean Reversion Dislocation
        double vwap_score = std::min(std::abs(vwap_dev) / 0.02, 1.0);
        // 3. Momentum Breakout (24h trend alignment)
        double trend_score = std::min(std::abs(change_24h_pct) / 10.0, 1.0);

        double composite_alpha = 0.45 * ofi_score + 0.35 * vwap_score + 0.20 * trend_score;

        ScoredTradeOpportunity opp;
        opp.symbol = symbol;
        opp.price = price;
        opp.best_bid = best_bid;
        opp.best_ask = best_ask;
        opp.alpha_score = composite_alpha;
        opp.ofi = ofi_val;
        opp.vwap_dev = vwap_dev;

        // Determine Trade Direction & Edge
        if (ofi_val > 10.0 || (vwap_dev < -0.005 && ofi_val > -5.0) || (change_24h_pct > 3.0 && ofi_val >= 0.0)) {
            opp.side = "BUY";
            opp.strategy = (std::abs(vwap_dev) > 0.008) ? "MULTI_VWAP_REVERSION" : "MULTI_OFI_MOMENTUM";
            opp.win_probability = canonical_win_probability(composite_alpha);
            opp.win_loss_ratio = 1.50 + 0.50 * composite_alpha;
        } else if (ofi_val < -10.0 || (vwap_dev > 0.005 && ofi_val < 5.0) || (change_24h_pct < -3.0 && ofi_val <= 0.0)) {
            opp.side = "SELL";
            opp.strategy = (std::abs(vwap_dev) > 0.008) ? "MULTI_VWAP_REVERSION" : "MULTI_OFI_MOMENTUM";
            opp.win_probability = canonical_win_probability(composite_alpha);
            opp.win_loss_ratio = 1.50 + 0.50 * composite_alpha;
        } else {
            opp.side = "NEUTRAL";
            opp.strategy = "MONITORING";
            opp.win_probability = canonical_win_probability(0.0); // == 0.50
            opp.win_loss_ratio = 1.0;
            opp.recommended_size_inr = 0.0;
            return opp;
        }

        // NOTE: win_probability clamping is handled inside the canonical mapping
        // ([0.50, 0.85]); do not apply engine-local clamps here.
        opp.win_loss_ratio = std::min(opp.win_loss_ratio, 2.50);

        // Safe Kelly Allocation with -100 Ruin Floor Protection
        opp.recommended_size_inr = survival_.calculate_safe_position_size(
            opp.win_probability, opp.win_loss_ratio, 0.15
        );

        return opp;
    }

    // Execute the scored opportunity
    TradeRecord execute_opportunity(const ScoredTradeOpportunity& opp) {
        if (opp.side == "NEUTRAL" || opp.recommended_size_inr < 0.0001 || !survival_.is_alive()) {
            TradeRecord no_op;
            no_op.symbol = opp.symbol;
            no_op.realized_pnl = 0.0;
            return no_op;
        }

        auto* book = get_or_create_book(opp.symbol, opp.price);
        OrderSide side = (opp.side == "SELL") ? OrderSide::SELL : OrderSide::BUY;
        double fill_price = (side == OrderSide::BUY) ? opp.best_ask : opp.best_bid;
        if (fill_price <= 0.0) fill_price = opp.price;
        double qty = opp.recommended_size_inr / std::max(fill_price, 0.0001);

        auto report = book->submit_order(side, OrderType::MARKET, fill_price, qty);
        if (report.executed_qty <= 0.0) {
            TradeRecord unfil;
            unfil.symbol = opp.symbol;
            return unfil;
        }

        // Calculate simulated edge realization based on alpha quality
        double vol = 0.015;
        std::normal_distribution<double> dist(0.0, vol * 0.5);
        double edge_alpha = (opp.win_probability - 0.50) * vol * 2.2;
        if (side == OrderSide::SELL) edge_alpha = -edge_alpha;
        double exit_price = report.avg_fill_price * (1.0 + edge_alpha + dist(rng_));

        return survival_.record_trade(
            opp.symbol,
            opp.side,
            report.avg_fill_price,
            exit_price,
            report.executed_qty,
            report.fee,
            opp.strategy
        );
    }

    size_t active_book_count() const { return books_.size(); }
};

} // namespace finance
} // namespace brain3
