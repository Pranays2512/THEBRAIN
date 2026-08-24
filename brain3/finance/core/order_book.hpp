#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <mutex>

namespace brain3 {
namespace finance {

enum class OrderSide { BUY, SELL };
enum class OrderType { LIMIT, MARKET };

struct Order {
    uint64_t id{0};
    std::string symbol;
    OrderSide side{OrderSide::BUY};
    OrderType type{OrderType::LIMIT};
    double price{0.0};
    double quantity{0.0};
    double filled_qty{0.0};
    uint64_t timestamp_ns{0};

    bool is_filled() const { return filled_qty >= quantity - 1e-9; }
    double remaining_qty() const { return std::max(0.0, quantity - filled_qty); }
};

struct ExecutionReport {
    uint64_t order_id{0};
    std::string symbol;
    OrderSide side{OrderSide::BUY};
    double requested_qty{0.0};
    double executed_qty{0.0};
    double avg_fill_price{0.0};
    double slippage{0.0};
    double price_impact{0.0};
    double fee{0.0};
    bool fully_filled{false};
    std::string status; // "FILLED", "PARTIAL", "REJECTED", "RESTING"
};

struct Level2Quote {
    double price{0.0};
    double volume{0.0};
    uint32_t order_count{0};
};

// Thread-safety: LimitOrderBook is internally thread-safe for single-operation
// atomicity — every public method takes an internal mutex (mutable, so const
// accessors lock too). Callers composing multiple operations into one logical
// step must still provide their own external synchronization.
class LimitOrderBook {
private:
    std::string symbol_;
    uint64_t next_order_id_{1};

    // Price -> Queue of Orders (Price-Time Priority)
    // Bids sorted descending (highest bid first)
    std::map<double, std::deque<Order>, std::greater<double>> bids_;
    // Asks sorted ascending (lowest ask first)
    std::map<double, std::deque<Order>, std::less<double>> asks_;

    // Synthetic market-maker order id per seeded price level (merge bookkeeping)
    std::map<double, uint64_t> seeded_bid_ids_;
    std::map<double, uint64_t> seeded_ask_ids_;

    double last_trade_price_{100.0};
    double last_trade_volume_{0.0};
    double total_volume_traded_{0.0};
    double fee_rate_{0.0004}; // 4 bps fee
    double kyle_lambda_{0.00005}; // Price impact parameter: Delta P = lambda * Q

    mutable std::mutex mtx_; // Guards all state above

public:
    explicit LimitOrderBook(std::string symbol = "BTC/USDT", double initial_mid = 100.0)
        : symbol_(std::move(symbol)), last_trade_price_(initial_mid) {
        seed_liquidity(initial_mid, 10);
    }

    // MERGE seeding (deterministic): refresh the synthetic market-maker quotes at
    // the levels this seed grid touches instead of clearing the whole book.
    // Resting orders — user limit orders and quotes at untouched levels — now
    // SURVIVE across ticks. A touched level's seeded quote is refreshed to the
    // target quantity; if it was consumed, a fresh seeded quote is placed there.
    void seed_liquidity(double mid_price, int depth = 10, double spread_bps = 5.0, double base_qty = 10.0) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!(mid_price > 1e-9) || depth <= 0) return;
        double half_spread = mid_price * (spread_bps / 20000.0);

        for (int i = 1; i <= depth; ++i) {
            double bid_p = mid_price - half_spread - (i - 1) * (mid_price * 0.0005);
            double ask_p = mid_price + half_spread + (i - 1) * (mid_price * 0.0005);
            double qty = base_qty * (1.0 + i * 0.2);

            upsert_seeded_order(bids_, seeded_bid_ids_, bid_p, qty, OrderSide::BUY);
            upsert_seeded_order(asks_, seeded_ask_ids_, ask_p, qty, OrderSide::SELL);
        }
    }

    double best_bid() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return best_bid_unlocked();
    }

    double best_ask() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return best_ask_unlocked();
    }

    double mid_price() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return mid_price_unlocked();
    }

    double spread() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return spread_unlocked();
    }

    double spread_bps() const {
        std::lock_guard<std::mutex> lock(mtx_);
        double mid = mid_price_unlocked();
        if (mid <= 0.0) return 0.0;
        return (spread_unlocked() / mid) * 10000.0;
    }

    double total_bid_depth() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return depth_unlocked(bids_);
    }

    double total_ask_depth() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return depth_unlocked(asks_);
    }

    // Submit Market or Limit Order
    ExecutionReport submit_order(OrderSide side, OrderType type, double price, double quantity) {
        std::lock_guard<std::mutex> lock(mtx_);
        ExecutionReport report;
        report.order_id = next_order_id_++;
        report.symbol = symbol_;
        report.side = side;
        report.requested_qty = quantity;

        if (quantity <= 1e-9) {
            report.status = "REJECTED";
            return report;
        }

        double initial_mid = mid_price_unlocked();
        double remaining = quantity;
        double filled_cost = 0.0;

        if (side == OrderSide::BUY) {
            // Match against Asks (lowest ask first)
            auto it = asks_.begin();
            while (it != asks_.end() && remaining > 1e-9) {
                double level_price = it->first;
                if (type == OrderType::LIMIT && level_price > price) {
                    break; // Limit price exceeded
                }

                auto& queue = it->second;
                while (!queue.empty() && remaining > 1e-9) {
                    auto& resting = queue.front();
                    double match_qty = std::min(remaining, resting.remaining_qty());
                    
                    resting.filled_qty += match_qty;
                    remaining -= match_qty;
                    filled_cost += match_qty * level_price;
                    report.executed_qty += match_qty;
                    last_trade_price_ = level_price;
                    last_trade_volume_ = match_qty;
                    total_volume_traded_ += match_qty;

                    if (resting.is_filled()) {
                        queue.pop_front();
                    }
                }

                if (queue.empty()) {
                    it = asks_.erase(it);
                } else {
                    ++it;
                }
            }

            // If limit order has remaining qty, add to bids
            if (type == OrderType::LIMIT && remaining > 1e-9) {
                Order rest_ord{report.order_id, symbol_, side, type, price, remaining, 0.0, get_now_ns()};
                bids_[price].push_back(rest_ord);
            }
        } else {
            // Match against Bids (highest bid first)
            auto it = bids_.begin();
            while (it != bids_.end() && remaining > 1e-9) {
                double level_price = it->first;
                if (type == OrderType::LIMIT && level_price < price) {
                    break; // Limit price under-bid
                }

                auto& queue = it->second;
                while (!queue.empty() && remaining > 1e-9) {
                    auto& resting = queue.front();
                    double match_qty = std::min(remaining, resting.remaining_qty());
                    
                    resting.filled_qty += match_qty;
                    remaining -= match_qty;
                    filled_cost += match_qty * level_price;
                    report.executed_qty += match_qty;
                    last_trade_price_ = level_price;
                    last_trade_volume_ = match_qty;
                    total_volume_traded_ += match_qty;

                    if (resting.is_filled()) {
                        queue.pop_front();
                    }
                }

                if (queue.empty()) {
                    it = bids_.erase(it);
                } else {
                    ++it;
                }
            }

            // If limit order has remaining qty, add to asks
            if (type == OrderType::LIMIT && remaining > 1e-9) {
                Order rest_ord{report.order_id, symbol_, side, type, price, remaining, 0.0, get_now_ns()};
                asks_[price].push_back(rest_ord);
            }
        }

        if (report.executed_qty > 1e-9) {
            report.avg_fill_price = filled_cost / report.executed_qty;
            report.price_impact = std::abs(report.avg_fill_price - initial_mid);
            report.slippage = (side == OrderSide::BUY)
                                  ? (report.avg_fill_price - initial_mid)
                                  : (initial_mid - report.avg_fill_price);
            report.fee = filled_cost * fee_rate_;
        } else {
            report.avg_fill_price = 0.0;
        }

        report.fully_filled = (report.executed_qty >= quantity - 1e-9);
        if (report.fully_filled) {
            report.status = "FILLED";
        } else if (report.executed_qty > 1e-9) {
            report.status = "PARTIAL";
        } else {
            report.status = (type == OrderType::LIMIT) ? "RESTING" : "REJECTED";
        }

        return report;
    }

    std::vector<Level2Quote> get_l2_bids(size_t max_levels = 5) const {
        std::lock_guard<std::mutex> lock(mtx_);
        return l2_unlocked(bids_, max_levels);
    }

    std::vector<Level2Quote> get_l2_asks(size_t max_levels = 5) const {
        std::lock_guard<std::mutex> lock(mtx_);
        return l2_unlocked(asks_, max_levels);
    }

    std::string to_json_summary(size_t depth = 5) const {
        std::lock_guard<std::mutex> lock(mtx_);
        std::ostringstream oss;
        oss << "{"
            << "\"symbol\":\"" << symbol_ << "\","
            << "\"mid_price\":" << std::fixed << std::setprecision(4) << mid_price_unlocked() << ","
            << "\"best_bid\":" << best_bid_unlocked() << ","
            << "\"best_ask\":" << best_ask_unlocked() << ","
            << "\"spread\":" << spread_unlocked() << ",";

        double mid = mid_price_unlocked();
        double sp = spread_unlocked();
        oss << "\"spread_bps\":" << ((mid > 0.0) ? (sp / mid) * 10000.0 : 0.0) << ","
            << "\"total_bid_depth\":" << depth_unlocked(bids_) << ","
            << "\"total_ask_depth\":" << depth_unlocked(asks_) << ","
            << "\"last_price\":" << last_trade_price_ << ","
            << "\"bids\":[";

        auto b_list = l2_unlocked(bids_, depth);
        for (size_t i = 0; i < b_list.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "{\"price\":" << b_list[i].price << ",\"volume\":" << b_list[i].volume << "}";
        }
        oss << "],\"asks\":[";
        auto a_list = l2_unlocked(asks_, depth);
        for (size_t i = 0; i < a_list.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "{\"price\":" << a_list[i].price << ",\"volume\":" << a_list[i].volume << "}";
        }
        oss << "]}";
        return oss.str();
    }

    const std::string& symbol() const { return symbol_; } // Immutable after construction
    double last_trade_price() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return last_trade_price_;
    }

private:
    // ── Unlocked internals: callers must hold mtx_ ────────────────────────────
    double best_bid_unlocked() const {
        if (bids_.empty()) return 0.0;
        return bids_.begin()->first;
    }

    double best_ask_unlocked() const {
        if (asks_.empty()) return 0.0;
        return asks_.begin()->first;
    }

    double mid_price_unlocked() const {
        double bb = best_bid_unlocked();
        double ba = best_ask_unlocked();
        if (bb > 0.0 && ba > 0.0) return (bb + ba) * 0.5;
        if (bb > 0.0) return bb;
        if (ba > 0.0) return ba;
        return last_trade_price_;
    }

    double spread_unlocked() const {
        double bb = best_bid_unlocked();
        double ba = best_ask_unlocked();
        if (bb > 0.0 && ba > 0.0) return std::max(0.0, ba - bb);
        return 0.0;
    }

    template <typename BookMap>
    double depth_unlocked(const BookMap& book) const {
        double sum = 0.0;
        for (const auto& [price, queue] : book) {
            for (const auto& ord : queue) sum += ord.remaining_qty();
        }
        return sum;
    }

    template <typename BookMap>
    std::vector<Level2Quote> l2_unlocked(const BookMap& book, size_t max_levels) const {
        std::vector<Level2Quote> out;
        size_t count = 0;
        for (const auto& [p, queue] : book) {
            if (count++ >= max_levels) break;
            double vol = 0.0;
            for (const auto& o : queue) vol += o.remaining_qty();
            out.push_back({p, vol, static_cast<uint32_t>(queue.size())});
        }
        return out;
    }

    // Upsert one synthetic market-maker quote at `price`: refresh quantity in
    // place if our seeded order still rests there; otherwise place a fresh one.
    // Never touches other orders resting at the same or different levels.
    template <typename BookMap>
    void upsert_seeded_order(BookMap& book, std::map<double, uint64_t>& seeded_ids,
                             double price, double qty, OrderSide side) {
        auto id_it = seeded_ids.find(price);
        if (id_it != seeded_ids.end()) {
            auto lvl = book.find(price);
            if (lvl != book.end()) {
                for (auto& ord : lvl->second) {
                    if (ord.id == id_it->second) {
                        ord.quantity = qty;
                        ord.filled_qty = 0.0;
                        return;
                    }
                }
            }
        }
        uint64_t oid = next_order_id_++;
        seeded_ids[price] = oid;
        book[price].push_back(Order{oid, symbol_, side, OrderType::LIMIT, price, qty, 0.0, get_now_ns()});
    }

    static uint64_t get_now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
};

} // namespace finance
} // namespace brain3
