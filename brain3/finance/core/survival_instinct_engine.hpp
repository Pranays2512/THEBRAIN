#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace brain3 {
namespace finance {

enum class BrainSurvivalState {
    THRIVING,       // Life Force > 80%
    ALERT,          // Life Force 50% - 80%
    ACUTE_PAIN,     // Severe drawdown / loss spike (> 5% single loss or Life Force 20% - 50%)
    CRITICAL,       // Life Force 5% - 20% (Emergency defensive mode, minimal sizing)
    BRAIN_DEAD      // Life Force <= 0% (Capital Ruin, Trading Halted)
};

struct TradeRecord {
    uint64_t trade_id{0};
    std::string symbol;
    std::string side; // BUY or SELL
    double entry_price{0.0};
    double exit_price{0.0};
    double quantity{0.0};
    double realized_pnl{0.0};
    double return_pct{0.0};
    double capital_after{0.0};
    double life_force_after{0.0};
    bool is_winner{false};
    uint64_t timestamp_ns{0};
};

class SurvivalInstinctEngine {
private:
    double starting_capital_{10000.0};
    double current_capital_{10000.0};
    double peak_capital_{10000.0};
    double ruin_threshold_{2000.0}; // 20% of starting capital is existential death
    
    double stress_level_{0.0}; // 0.0 (calm) to 1.0 (panicked)
    int consecutive_losses_{0};
    int consecutive_wins_{0};

    std::vector<TradeRecord> trade_history_;
    uint64_t next_trade_id_{1};

    double total_profit_{0.0};
    double total_loss_{0.0};
    int win_count_{0};
    int loss_count_{0};

public:
    explicit SurvivalInstinctEngine(double initial_capital = 10000.0, double ruin_ratio = 0.20)
        : starting_capital_(initial_capital),
          current_capital_(initial_capital),
          peak_capital_(initial_capital),
          ruin_threshold_(initial_capital * ruin_ratio) {}

    // Life Force Metric [0.0% to 100.0%]
    double life_force() const {
        if (current_capital_ <= ruin_threshold_) return 0.0;
        double range = starting_capital_ - ruin_threshold_;
        if (range <= 0.0) return 0.0;
        double lf = ((current_capital_ - ruin_threshold_) / range) * 100.0;
        return std::min(100.0, std::max(0.0, lf));
    }

    double current_equity() const { return current_capital_; }
    double peak_equity() const { return peak_capital_; }
    double max_drawdown_pct() const {
        if (peak_capital_ <= 0.0) return 0.0;
        return ((peak_capital_ - current_capital_) / peak_capital_) * 100.0;
    }

    BrainSurvivalState state() const {
        double lf = life_force();
        if (current_capital_ <= ruin_threshold_ || lf <= 0.0) {
            return BrainSurvivalState::BRAIN_DEAD;
        }
        if (lf < 20.0) {
            return BrainSurvivalState::CRITICAL;
        }
        if (stress_level_ > 0.6 || lf < 50.0) {
            return BrainSurvivalState::ACUTE_PAIN;
        }
        if (lf < 80.0 || stress_level_ > 0.3) {
            return BrainSurvivalState::ALERT;
        }
        return BrainSurvivalState::THRIVING;
    }

    std::string state_string() const {
        switch (state()) {
            case BrainSurvivalState::THRIVING: return "THRIVING";
            case BrainSurvivalState::ALERT: return "ALERT";
            case BrainSurvivalState::ACUTE_PAIN: return "ACUTE_PAIN";
            case BrainSurvivalState::CRITICAL: return "CRITICAL";
            case BrainSurvivalState::BRAIN_DEAD: return "BRAIN_DEAD";
        }
        return "UNKNOWN";
    }

    bool is_alive() const {
        return state() != BrainSurvivalState::BRAIN_DEAD;
    }

    // Mathematical Kelly Position Sizing with Survival Attenuation
    // f* = (p * b - q) / b
    double calculate_safe_position_size(double win_prob = 0.55, double win_loss_ratio = 1.5, double max_risk_pct = 0.10) const {
        if (!is_alive()) return 0.0;

        double q = 1.0 - win_prob;
        double b = std::max(0.01, win_loss_ratio);
        double unconstrained_kelly = (win_prob * b - q) / b;

        if (unconstrained_kelly <= 0.0) return 0.0; // Negative edge, do not trade

        // Survival Attenuation Factor
        // When life force drops or stress rises, position size aggressively contracts
        double lf_factor = std::pow(life_force() / 100.0, 2.0);
        double stress_dampening = std::max(0.1, 1.0 - stress_level_);
        
        // Half-Kelly safety multiplier
        double safe_fraction = unconstrained_kelly * 0.5 * lf_factor * stress_dampening;
        safe_fraction = std::min(safe_fraction, max_risk_pct);

        // Emergency critical cap
        if (state() == BrainSurvivalState::CRITICAL) {
            safe_fraction = std::min(safe_fraction, 0.01); // Max 1% allocation in critical mode
        }

        return current_capital_ * safe_fraction;
    }

    // Record trade result and trigger somatic / pain reflexes
    TradeRecord record_trade(const std::string& symbol, const std::string& side,
                             double entry_p, double exit_p, double qty, double fee = 0.0) {
        TradeRecord rec;
        rec.trade_id = next_trade_id_++;
        rec.symbol = symbol;
        rec.side = side;
        rec.entry_price = entry_p;
        rec.exit_price = exit_p;
        rec.quantity = qty;
        rec.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        double gross_pnl = 0.0;
        if (side == "BUY" || side == "buy") {
            gross_pnl = (exit_p - entry_p) * qty;
            rec.return_pct = (entry_p > 0.0) ? ((exit_p - entry_p) / entry_p) * 100.0 : 0.0;
        } else {
            gross_pnl = (entry_p - exit_p) * qty;
            rec.return_pct = (entry_p > 0.0) ? ((entry_p - exit_p) / entry_p) * 100.0 : 0.0;
        }

        rec.realized_pnl = gross_pnl - fee;
        current_capital_ += rec.realized_pnl;
        if (current_capital_ > peak_capital_) {
            peak_capital_ = current_capital_;
        }

        rec.is_winner = (rec.realized_pnl > 0.0);
        rec.capital_after = current_capital_;
        rec.life_force_after = life_force();

        // Update Instinct State & Somatic Pain Reflex
        if (rec.is_winner) {
            win_count_++;
            total_profit_ += rec.realized_pnl;
            consecutive_wins_++;
            consecutive_losses_ = 0;
            stress_level_ = std::max(0.0, stress_level_ - 0.15); // Relieve stress
        } else {
            loss_count_++;
            total_loss_ += std::abs(rec.realized_pnl);
            consecutive_losses_++;
            consecutive_wins_ = 0;
            
            // Acute Pain Reflex: Stress increases non-linearly with loss severity
            double loss_severity = std::abs(rec.realized_pnl) / std::max(100.0, current_capital_);
            stress_level_ = std::min(1.0, stress_level_ + 0.25 + (loss_severity * 2.0));
        }

        trade_history_.push_back(rec);
        return rec;
    }

    double win_rate_pct() const {
        int total = win_count_ + loss_count_;
        if (total == 0) return 0.0;
        return (static_cast<double>(win_count_) / total) * 100.0;
    }

    double profit_factor() const {
        if (total_loss_ <= 1e-9) return (total_profit_ > 0.0) ? 99.9 : 1.0;
        return total_profit_ / total_loss_;
    }

    std::string to_json_summary() const {
        std::ostringstream oss;
        oss << "{"
            << "\"survival_state\":\"" << state_string() << "\","
            << "\"is_alive\":" << (is_alive() ? "true" : "false") << ","
            << "\"life_force_pct\":" << std::fixed << std::setprecision(2) << life_force() << ","
            << "\"current_equity\":" << current_capital_ << ","
            << "\"peak_equity\":" << peak_capital_ << ","
            << "\"max_drawdown_pct\":" << max_drawdown_pct() << ","
            << "\"stress_level\":" << std::setprecision(3) << stress_level_ << ","
            << "\"consecutive_losses\":" << consecutive_losses_ << ","
            << "\"consecutive_wins\":" << consecutive_wins_ << ","
            << "\"win_rate_pct\":" << std::setprecision(2) << win_rate_pct() << ","
            << "\"profit_factor\":" << std::setprecision(2) << profit_factor() << ","
            << "\"total_trades\":" << trade_history_.size() << ","
            << "\"total_profit\":" << total_profit_ << ","
            << "\"total_loss\":" << total_loss_
            << "}";
        return oss.str();
    }

    const std::vector<TradeRecord>& history() const { return trade_history_; }
};

} // namespace finance
} // namespace brain3
