#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <random>
#include <iomanip>

#include "core/order_book.hpp"
#include "core/market_microstructure.hpp"
#include "core/survival_instinct_engine.hpp"
#include "core/cross_asset_arbitrage_hunter.hpp"
#include "core/autonomous_trading_instinct_engine.hpp"

namespace brain3 {
namespace finance {

class FinanceOrchestrator {
private:
    std::unordered_map<std::string, std::unique_ptr<LimitOrderBook>> order_books_;
    std::unordered_map<std::string, std::unique_ptr<MicrostructureAnalyzer>> micro_analyzers_;
    SurvivalInstinctEngine survival_engine_;
    CrossAssetArbitrageHunter stat_arb_hunter_;
    AutonomousTradingInstinctEngine autonomous_engine_;

    std::mt19937_64 rng_{42};

public:
    explicit FinanceOrchestrator(double initial_capital = 1000.0,
                                 double ruin_floor = -1000.0,
                                 double cap_limit = 100000.0,
                                 double metabolic_burn = 0.02)
        : survival_engine_(initial_capital, ruin_floor, cap_limit, metabolic_burn),
          autonomous_engine_(initial_capital, ruin_floor, cap_limit, metabolic_burn) {
        // Initialize default books
        get_or_create_book("NIFTY50/INR", 24500.0);
        get_or_create_book("BANKNIFTY/INR", 51200.0);
        get_or_create_book("BTC/INR", 5850000.0);
        get_or_create_book("ETH/INR", 315000.0);
        get_or_create_book("BTC/USDT", 65000.0);
        get_or_create_book("ETH/USDT", 3500.0);
        get_or_create_book("SOL/USDT", 150.0);
    }

    LimitOrderBook* get_or_create_book(const std::string& symbol, double default_price = 100.0) {
        auto it = order_books_.find(symbol);
        if (it == order_books_.end()) {
            order_books_[symbol] = std::make_unique<LimitOrderBook>(symbol, default_price);
            micro_analyzers_[symbol] = std::make_unique<MicrostructureAnalyzer>(symbol);
            return order_books_[symbol].get();
        }
        return it->second.get();
    }

    MicrostructureAnalyzer* get_micro_analyzer(const std::string& symbol) {
        auto it = micro_analyzers_.find(symbol);
        if (it == micro_analyzers_.end()) {
            get_or_create_book(symbol);
            return micro_analyzers_[symbol].get();
        }
        return it->second.get();
    }

    SurvivalInstinctEngine& survival() { return survival_engine_; }
    const SurvivalInstinctEngine& survival() const { return survival_engine_; }

    AutonomousTradingInstinctEngine& autonomous() { return autonomous_engine_; }
    const AutonomousTradingInstinctEngine& autonomous() const { return autonomous_engine_; }

    // Execute BrainQL Financial Command
    std::string execute_command(const std::string& command_line) {
        std::istringstream iss(command_line);
        std::string opcode;
        iss >> opcode;

        if (opcode == "FINANCE_STATUS" || opcode == "SURVIVAL_STATUS") {
            return survival_engine_.to_json_summary();
        }

        if (opcode == "METABOLIC_TICK") {
            survival_engine_.metabolic_tick();
            return survival_engine_.to_json_summary();
        }

        if (opcode == "SET_CAPITAL_PARAMETERS") {
            double init_cap = 1000.0, ruin_fl = -1000.0, cap_lim = 100000.0, burn = 0.02;
            iss >> init_cap >> ruin_fl >> cap_lim >> burn;
            survival_engine_ = SurvivalInstinctEngine(init_cap, ruin_fl, cap_lim, burn);
            autonomous_engine_ = AutonomousTradingInstinctEngine(init_cap, ruin_fl, cap_lim, burn);
            return survival_engine_.to_json_summary();
        }
        
        if (opcode == "ORDER_BOOK") {
            std::string sym = "BTC/USDT";
            iss >> sym;
            auto* book = get_or_create_book(sym);
            return book->to_json_summary();
        }

        if (opcode == "MICROSTRUCTURE") {
            std::string sym = "BTC/USDT";
            iss >> sym;
            auto* micro = get_micro_analyzer(sym);
            return micro->to_json_summary();
        }

        if (opcode == "KELLY_SIZE") {
            double win_p = 0.58;
            double win_loss = 1.6;
            iss >> win_p >> win_loss;
            double safe_alloc = survival_engine_.calculate_safe_position_size(win_p, win_loss);
            std::ostringstream oss;
            oss << "{"
                << "\"win_probability\":" << win_p << ","
                << "\"win_loss_ratio\":" << win_loss << ","
                << "\"safe_allocation_inr\":" << std::fixed << std::setprecision(2) << safe_alloc << ","
                << "\"life_force_pct\":" << survival_engine_.life_force() << ","
                << "\"survival_state\":\"" << survival_engine_.state_string() << "\""
                << "}";
            return oss.str();
        }

        if (opcode == "TRADE_ORDER") {
            std::string sym, side_str, type_str;
            double price = 0.0, qty = 0.0;
            iss >> sym >> side_str >> type_str >> price >> qty;

            if (!survival_engine_.is_alive()) {
                return "{\"status\":\"REJECTED\",\"reason\":\"AGENT_IS_BRAIN_DEAD\"}";
            }

            auto* book = get_or_create_book(sym);
            OrderSide side = (side_str == "SELL" || side_str == "sell") ? OrderSide::SELL : OrderSide::BUY;
            OrderType type = (type_str == "MARKET" || type_str == "market") ? OrderType::MARKET : OrderType::LIMIT;

            auto report = book->submit_order(side, type, price, qty);

            // Feed to microstructure
            auto* micro = get_micro_analyzer(sym);
            micro->on_tick(report.avg_fill_price > 0 ? report.avg_fill_price : book->mid_price(),
                           report.executed_qty, book->best_bid(), book->best_ask(),
                           book->total_bid_depth(), book->total_ask_depth());

            std::ostringstream oss;
            oss << "{"
                << "\"order_id\":" << report.order_id << ","
                << "\"symbol\":\"" << report.symbol << "\","
                << "\"status\":\"" << report.status << "\","
                << "\"side\":\"" << side_str << "\","
                << "\"requested_qty\":" << report.requested_qty << ","
                << "\"executed_qty\":" << report.executed_qty << ","
                << "\"avg_fill_price\":" << std::fixed << std::setprecision(4) << report.avg_fill_price << ","
                << "\"slippage\":" << report.slippage << ","
                << "\"fee\":" << report.fee
                << "}";
            return oss.str();
        }

        if (opcode == "STAT_ARB_SCAN") {
            std::string symA = "BTC/USDT", symB = "ETH/USDT";
            iss >> symA >> symB;

            // Generate synthetic cointegrated walk for scan
            std::vector<double> pA, pB;
            double pA_curr = 65000.0, pB_curr = 3500.0;
            std::normal_distribution<double> dist(0.0, 1.0);
            
            for (int i = 0; i < 50; ++i) {
                pA_curr += dist(rng_) * 50.0;
                pB_curr += dist(rng_) * 5.0 + (pA_curr * 0.05 - pB_curr) * 0.1;
                pA.push_back(pA_curr);
                pB.push_back(pB_curr);
            }

            auto sig = CrossAssetArbitrageHunter::analyze_pair(symA, symB, pA, pB);
            std::ostringstream oss;
            oss << "{"
                << "\"asset_a\":\"" << sig.asset_a << "\","
                << "\"asset_b\":\"" << sig.asset_b << "\","
                << "\"hedge_ratio_beta\":" << std::fixed << std::setprecision(4) << sig.hedge_ratio_beta << ","
                << "\"current_spread\":" << sig.current_spread << ","
                << "\"z_score\":" << sig.z_score << ","
                << "\"ou_theta\":" << sig.ou_theta << ","
                << "\"half_life_periods\":" << sig.half_life_periods << ","
                << "\"action\":\"" << sig.action << "\","
                << "\"expected_edge_bps\":" << sig.expected_edge_bps
                << "}";
            return oss.str();
        }

        if (opcode == "SAMPLE_SURVIVAL_TRADE") {
            std::string sym = "NIFTY50/INR";
            iss >> sym;
            auto tr = autonomous_engine_.process_tick_and_trade(sym, 0.001, 0.009);
            std::ostringstream oss;
            oss << "{"
                << "\"trade_id\":" << tr.trade_id << ","
                << "\"symbol\":\"" << tr.symbol << "\","
                << "\"side\":\"" << tr.side << "\","
                << "\"entry_price\":" << std::fixed << std::setprecision(2) << tr.entry_price << ","
                << "\"exit_price\":" << tr.exit_price << ","
                << "\"quantity\":" << std::setprecision(4) << tr.quantity << ","
                << "\"realized_pnl\":" << std::setprecision(2) << tr.realized_pnl << ","
                << "\"capital_after\":" << tr.capital_after << ","
                << "\"life_force_pct\":" << tr.life_force_after << ","
                << "\"strategy\":\"" << tr.strategy_used << "\","
                << "\"is_winner\":" << (tr.is_winner ? "true" : "false") << ","
                << "\"survival_state\":\"" << autonomous_engine_.survival().state_string() << "\""
                << "}";
            return oss.str();
        }

        if (opcode == "AUTONOMOUS_SURVIVAL_CYCLE") {
            int ticks = 200;
            iss >> ticks;
            auto m = autonomous_engine_.run_simulation(ticks);
            std::ostringstream oss;
            oss << "{"
                << "\"initial_capital\":" << std::fixed << std::setprecision(2) << m.initial_capital << ","
                << "\"final_capital\":" << m.final_capital << ","
                << "\"peak_capital\":" << m.peak_capital << ","
                << "\"net_profit\":" << m.net_profit << ","
                << "\"return_pct\":" << m.return_pct << ","
                << "\"max_drawdown_pct\":" << m.max_drawdown_pct << ","
                << "\"sharpe_ratio\":" << m.sharpe_ratio << ","
                << "\"win_rate_pct\":" << m.win_rate_pct << ","
                << "\"profit_factor\":" << m.profit_factor << ","
                << "\"total_trades\":" << m.total_trades << ","
                << "\"winning_trades\":" << m.winning_trades << ","
                << "\"losing_trades\":" << m.losing_trades << ","
                << "\"ticks_survived\":" << m.ticks_survived << ","
                << "\"final_survival_state\":\"" << m.final_survival_state << "\","
                << "\"survived_without_ruin\":" << (m.survived_without_ruin ? "true" : "false") << ","
                << "\"simulation_time_ms\":" << m.simulation_time_ms
                << "}";
            return oss.str();
        }

        if (opcode == "SIMULATE_MARKET_CYCLE") {
            std::string sym = "BTC/USDT";
            int ticks = 50;
            double drift = 0.0;
            double vol = 0.01;
            iss >> sym >> ticks >> drift >> vol;

            auto* book = get_or_create_book(sym);
            auto* micro = get_micro_analyzer(sym);

            std::normal_distribution<double> norm(drift, vol);
            std::uniform_real_distribution<double> unif_qty(0.5, 5.0);

            int executed_trades = 0;
            for (int t = 0; t < ticks; ++t) {
                if (!survival_engine_.is_alive()) break;

                double current_mid = book->mid_price();
                double ret = norm(rng_);
                double new_mid = current_mid * (1.0 + ret);
                book->seed_liquidity(new_mid, 8);

                // Agent decides to trade using safe Kelly allocation
                double safe_dollars = survival_engine_.calculate_safe_position_size(0.58, 1.6, 0.05);
                if (safe_dollars > 10.0) {
                    double trade_qty = safe_dollars / new_mid;
                    OrderSide side = (ret >= 0.0) ? OrderSide::BUY : OrderSide::SELL;
                    auto rep = book->submit_order(side, OrderType::MARKET, new_mid, trade_qty);
                    
                    if (rep.executed_qty > 0.0) {
                        double exit_p = new_mid * (1.0 + norm(rng_) * 0.5);
                        survival_engine_.record_trade(sym, (side == OrderSide::BUY ? "BUY" : "SELL"),
                                                     rep.avg_fill_price, exit_p, rep.executed_qty, rep.fee);
                        executed_trades++;
                    }
                }

                micro->on_tick(book->last_trade_price(), unif_qty(rng_),
                               book->best_bid(), book->best_ask(),
                               book->total_bid_depth(), book->total_ask_depth());
            }

            std::ostringstream oss;
            oss << "{"
                << "\"symbol\":\"" << sym << "\","
                << "\"simulated_ticks\":" << ticks << ","
                << "\"trades_executed\":" << executed_trades << ","
                << "\"survival_status\":" << survival_engine_.to_json_summary()
                << "}";
            return oss.str();
        }

        if (opcode == "INJECT_DRAWDOWN_PAIN") {
            double loss_amount = 500.0;
            iss >> loss_amount;
            survival_engine_.record_trade("TEST/INR", "BUY", 100.0, 100.0 - (loss_amount / 10.0), 10.0, 0.0, "DRAWDOWN_TEST");
            return survival_engine_.to_json_summary();
        }

        if (opcode == "RESET_LIFE_FORCE") {
            double init_cap = 1000.0;
            iss >> init_cap;
            survival_engine_ = SurvivalInstinctEngine(init_cap, -1000.0, 100000.0);
            autonomous_engine_ = AutonomousTradingInstinctEngine(init_cap, -1000.0, 100000.0);
            return survival_engine_.to_json_summary();
        }

        return "{\"error\":\"UNKNOWN_FINANCE_OPCODE\",\"opcode\":\"" + opcode + "\"}";
    }
};

} // namespace finance
} // namespace brain3
