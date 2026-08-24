#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>

#include "order_book.hpp"
#include "market_microstructure.hpp"
#include "survival_instinct_engine.hpp"
#include "cross_asset_arbitrage_hunter.hpp"
#include "alpha_conviction.hpp"

namespace brain3 {
namespace finance {

struct SimulationMetrics {
    double initial_capital{1000.0};
    double final_capital{0.0};
    double peak_capital{0.0};
    double net_profit{0.0};
    double return_pct{0.0};
    double max_drawdown_pct{0.0};
    double sharpe_ratio{0.0};
    double win_rate_pct{0.0};
    double profit_factor{0.0};
    int total_trades{0};
    int winning_trades{0};
    int losing_trades{0};
    uint64_t ticks_survived{0};
    double total_metabolic_cost{0.0};
    std::string final_survival_state;
    bool survived_without_ruin{true};
    double simulation_time_ms{0.0};
};

class AutonomousTradingInstinctEngine {
private:
    SurvivalInstinctEngine survival_;
    std::unordered_map<std::string, std::unique_ptr<LimitOrderBook>> books_;
    std::unordered_map<std::string, std::unique_ptr<MicrostructureAnalyzer>> micros_;
    std::mt19937_64 rng_{1337};

public:
    explicit AutonomousTradingInstinctEngine(double initial_capital = 1000.0,
                                            double ruin_floor = -1000.0,
                                            double cap_limit = 100000.0,
                                            double metabolic_burn = 0.02)
        : survival_(initial_capital, ruin_floor, cap_limit, metabolic_burn) {
        init_instruments();
    }

    void init_instruments() {
        get_or_create_book("NIFTY50/INR", 24500.0);
        get_or_create_book("BANKNIFTY/INR", 51200.0);
        get_or_create_book("BTC/INR", 5850000.0);
        get_or_create_book("ETH/INR", 315000.0);
        get_or_create_book("USD/INR", 83.90);
    }

    LimitOrderBook* get_or_create_book(const std::string& symbol, double default_price = 100.0) {
        auto it = books_.find(symbol);
        if (it == books_.end()) {
            books_[symbol] = std::make_unique<LimitOrderBook>(symbol, default_price);
            micros_[symbol] = std::make_unique<MicrostructureAnalyzer>(symbol);
            return books_[symbol].get();
        }
        return it->second.get();
    }

    MicrostructureAnalyzer* get_micro(const std::string& symbol) {
        get_or_create_book(symbol);
        return micros_[symbol].get();
    }

    SurvivalInstinctEngine& survival() { return survival_; }
    const SurvivalInstinctEngine& survival() const { return survival_; }

    // ── Execute Single Autonomous Trading Decision Cycle ──────────────────────
    // Fired every market tick (or sub-millisecond event)
    TradeRecord process_tick_and_trade(const std::string& symbol, double market_drift = 0.0, double market_vol = 0.008) {
        survival_.metabolic_tick(); // Burn metabolic ATP

        if (!survival_.is_alive()) {
            TradeRecord dead_rec;
            dead_rec.symbol = symbol;
            dead_rec.realized_pnl = 0.0;
            return dead_rec;
        }

        auto* book = get_or_create_book(symbol);
        auto* micro = get_micro(symbol);

        // 1. Advance synthetic market microstructure & order flow
        std::normal_distribution<double> ret_dist(market_drift, market_vol);
        std::uniform_real_distribution<double> qty_dist(1.0, 10.0);

        double price_change = ret_dist(rng_);
        double new_mid = book->mid_price() * (1.0 + price_change);
        // NOTE: seed_liquidity now MERGES — resting limit orders survive across
        // ticks instead of being wiped on every incoming tick.
        book->seed_liquidity(new_mid, 10);

        micro->on_tick(new_mid, qty_dist(rng_), book->best_bid(), book->best_ask(),
                       book->total_bid_depth(), book->total_ask_depth());

        // 2. Extract Microstructure Alpha Signals
        double ofi = micro->ofi();
        double vwap_dev = (micro->vwap() > 0) ? (new_mid - micro->vwap()) / micro->vwap() : 0.0;
        double hunger = survival_.hunger_urgency_factor();

        // 3. Multi-Strategy Alpha Synthesis
        // All win probabilities flow through the canonical conviction mapping in
        // alpha_conviction.hpp: raw signals are normalized to an alpha in [0,1],
        // then canonical_win_probability(alpha) = clamp(0.55 + 0.20*alpha, 0.5, 0.85).
        std::string chosen_strategy = "PASSIVE_MARKET_MAKING";
        OrderSide side = OrderSide::BUY;
        double win_prob = canonical_win_probability(0.0); // == 0.55
        double win_loss_ratio = 1.5;
        bool should_trade = false;

        // Alpha A: Order Flow Imbalance (OFI) Trend Scalper.
        // Raw |ofi| is mapped to [0,1] with saturation at |ofi| = 0.75 (the point
        // where the legacy gain min(0.15, |ofi|*0.2) capped out).
        if (std::abs(ofi) > 0.25) {
            chosen_strategy = "OFI_MOMENTUM_SCALP";
            side = (ofi > 0.0) ? OrderSide::BUY : OrderSide::SELL;
            const double ofi_alpha = std::min(std::abs(ofi) / 0.75, 1.0);
            win_prob = canonical_win_probability(ofi_alpha);
            win_loss_ratio = 1.75;
            should_trade = true;
        }
        // Alpha B: VWAP Mean Reversion Scalper
        else if (std::abs(vwap_dev) > 0.005) {
            chosen_strategy = "VWAP_MEAN_REVERSION";
            side = (vwap_dev > 0.0) ? OrderSide::SELL : OrderSide::BUY; // Fade the extreme
            win_prob = canonical_win_probability(0.25); // == legacy 0.60
            win_loss_ratio = 1.60;
            should_trade = true;
        }
        // Alpha C: Instinctual Survival Harvest (When hungry, take high-probability spread edges)
        else if (hunger > 1.2) {
            chosen_strategy = "SURVIVAL_SPREAD_HARVEST";
            side = (price_change >= 0.0) ? OrderSide::BUY : OrderSide::SELL;
            win_prob = canonical_win_probability(0.15); // == legacy 0.58
            win_loss_ratio = 1.45;
            should_trade = true;
        }

        if (!should_trade) {
            TradeRecord no_trade;
            no_trade.symbol = symbol;
            no_trade.realized_pnl = 0.0;
            return no_trade;
        }

        // 4. Instinctual Position Sizing with Ruin-Floor Protection (-1000 INR)
        double safe_allocation = survival_.calculate_safe_position_size(win_prob, win_loss_ratio, 0.20);
        if (safe_allocation < 1.0) {
            TradeRecord tiny_rec;
            tiny_rec.symbol = symbol;
            tiny_rec.realized_pnl = 0.0;
            return tiny_rec;
        }

        double trade_qty = safe_allocation / new_mid;
        auto order_rep = book->submit_order(side, OrderType::MARKET, new_mid, trade_qty);

        if (order_rep.executed_qty <= 0.0) {
            TradeRecord unfil_rec;
            unfil_rec.symbol = symbol;
            return unfil_rec;
        }

        // 5. Simulate Realized Exit with Alpha Edge
        std::normal_distribution<double> edge_dist(0.0, market_vol * 0.7);
        double edge_realization = edge_dist(rng_);
        
        // Strategy alpha tilt
        double strategy_edge = (win_prob - 0.50) * market_vol * 2.0;
        if (side == OrderSide::SELL) strategy_edge = -strategy_edge;

        double exit_price = order_rep.avg_fill_price * (1.0 + strategy_edge + edge_realization);

        return survival_.record_trade(symbol, (side == OrderSide::BUY ? "BUY" : "SELL"),
                                      order_rep.avg_fill_price, exit_price,
                                      order_rep.executed_qty, order_rep.fee,
                                      chosen_strategy);
    }

    // ── Run Complete Multi-Tick Autonomous Survival Simulation ────────────────
    SimulationMetrics run_simulation(int num_ticks = 500,
                                     const std::vector<std::string>& symbols = {"NIFTY50/INR", "BTC/INR", "ETH/INR"}) {
        auto t0 = std::chrono::high_resolution_clock::now();
        SimulationMetrics m;
        m.initial_capital = survival_.starting_equity();

        std::vector<double> daily_returns;
        double prev_equity = survival_.current_equity();

        for (int t = 0; t < num_ticks; ++t) {
            if (!survival_.is_alive()) break;
            if (survival_.current_equity() >= survival_.cap_limit()) break;

            const auto& sym = symbols[t % symbols.size()];
            process_tick_and_trade(sym, 0.0002, 0.012);

            double current_eq = survival_.current_equity();
            if (prev_equity > 0.0) {
                daily_returns.push_back((current_eq - prev_equity) / prev_equity);
            }
            prev_equity = current_eq;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        m.simulation_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        m.final_capital = survival_.current_equity();
        m.peak_capital = survival_.peak_equity();
        m.net_profit = m.final_capital - m.initial_capital;
        m.return_pct = (m.initial_capital > 0.0) ? (m.net_profit / m.initial_capital) * 100.0 : 0.0;
        m.max_drawdown_pct = survival_.max_drawdown_pct();
        m.win_rate_pct = survival_.win_rate_pct();
        m.profit_factor = survival_.profit_factor();
        m.total_trades = static_cast<int>(survival_.history().size());
        m.ticks_survived = survival_.ticks_survived();
        m.final_survival_state = survival_.state_string();
        m.survived_without_ruin = survival_.is_alive() && (survival_.current_equity() > survival_.ruin_floor());

        for (const auto& tr : survival_.history()) {
            if (tr.is_winner) m.winning_trades++;
            else m.losing_trades++;
        }

        // Calculate Sharpe Ratio
        if (daily_returns.size() > 1) {
            double sum = 0.0;
            for (double r : daily_returns) sum += r;
            double mean = sum / daily_returns.size();

            double sq_sum = 0.0;
            for (double r : daily_returns) sq_sum += (r - mean) * (r - mean);
            double std_dev = std::sqrt(sq_sum / (daily_returns.size() - 1));

            if (std_dev > 1e-8) {
                m.sharpe_ratio = (mean / std_dev) * std::sqrt(252.0 * 100.0); // Annualized high-freq Sharpe
            }
        }

        return m;
    }
};

} // namespace finance
} // namespace brain3
