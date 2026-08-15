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
    APEX_ABUNDANCE, // Capital >= Cap Limit (100k)
    THRIVING,       // Life Force > 75%
    SURVIVING,      // Life Force 40% - 75%
    STARVATION_ALERT,// Life Force 15% - 40% (Urgency high, hunting alpha)
    ACUTE_PAIN,     // Recent severe drawdown spike or severe loss
    CRITICAL_DEFENSE,// Life Force < 15% (Tight survival bounds, strict risk)
    BRAIN_DEAD      // Capital <= Ruin Floor (-1000 INR, Terminal Ruin)
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
    std::string strategy_used;
    bool is_winner{false};
    uint64_t timestamp_ns{0};
};

class SurvivalInstinctEngine {
private:
    std::string currency_{"INR"};
    double starting_capital_{1000.0};   // Initial 1,000 INR
    double current_capital_{1000.0};    // Current cash equity
    double peak_capital_{1000.0};       // Peak all-time high
    double ruin_floor_{-1000.0};        // -1,000 INR (Ruin / Death threshold)
    double cap_limit_{100000.0};        // 100,000 INR (1 Lakh target cap)
    double metabolic_burn_rate_{0.02};  // 0.02 INR per tick compute upkeep cost

    double stress_level_{0.0};          // 0.0 (calm) to 1.0 (panicked)
    int consecutive_losses_{0};
    int consecutive_wins_{0};
    uint64_t ticks_survived_{0};

    std::vector<TradeRecord> trade_history_;
    uint64_t next_trade_id_{1};

    double total_profit_{0.0};
    double total_loss_{0.0};
    double total_metabolic_cost_{0.0};
    int win_count_{0};
    int loss_count_{0};

public:
    explicit SurvivalInstinctEngine(double initial_capital = 1000.0,
                                   double ruin_floor = -1000.0,
                                   double cap_limit = 100000.0,
                                   double metabolic_burn = 0.02)
        : currency_("INR"),
          starting_capital_(initial_capital),
          current_capital_(initial_capital),
          peak_capital_(initial_capital),
          ruin_floor_(ruin_floor),
          cap_limit_(cap_limit),
          metabolic_burn_rate_(metabolic_burn) {}

    // ── Life Force Metric [0.0% to 100.0%] ───────────────────────────────────
    // At Ruin Floor (-1000): 0.0% (Brain Dead)
    // At Starting Capital (1000): 50.0% (Surviving Baseline)
    // At Cap Limit (100000): 100.0% (Apex Abundance)
    double life_force() const {
        if (current_capital_ <= ruin_floor_) return 0.0;
        if (current_capital_ >= cap_limit_) return 100.0;

        if (current_capital_ <= starting_capital_) {
            double range = starting_capital_ - ruin_floor_;
            if (range <= 0.0) return 0.0;
            return ((current_capital_ - ruin_floor_) / range) * 50.0;
        } else {
            double range = cap_limit_ - starting_capital_;
            if (range <= 0.0) return 100.0;
            return 50.0 + ((current_capital_ - starting_capital_) / range) * 50.0;
        }
    }

    double current_equity() const { return current_capital_; }
    double peak_equity() const { return peak_capital_; }
    double starting_equity() const { return starting_capital_; }
    double ruin_floor() const { return ruin_floor_; }
    double cap_limit() const { return cap_limit_; }
    uint64_t ticks_survived() const { return ticks_survived_; }

    double max_drawdown_pct() const {
        if (peak_capital_ <= ruin_floor_) return 100.0;
        double range = peak_capital_ - ruin_floor_;
        if (range <= 0.0) return 0.0;
        return std::max(0.0, ((peak_capital_ - current_capital_) / range) * 100.0);
    }

    // ── Metabolic Tick: Burns ATP / Compute Maintenance ──────────────────────
    // The Brain must actively trade to overcome continuous entropy / starvation!
    void metabolic_tick() {
        if (!is_alive()) return;
        ticks_survived_++;
        current_capital_ -= metabolic_burn_rate_;
        total_metabolic_cost_ += metabolic_burn_rate_;

        // If starvation is setting in, slightly increase stress/hunger drive
        if (life_force() < 40.0) {
            stress_level_ = std::min(1.0, stress_level_ + 0.005);
        }
    }

    BrainSurvivalState state() const {
        if (current_capital_ <= ruin_floor_) {
            return BrainSurvivalState::BRAIN_DEAD;
        }
        if (current_capital_ >= cap_limit_) {
            return BrainSurvivalState::APEX_ABUNDANCE;
        }
        double lf = life_force();
        if (lf < 15.0) {
            return BrainSurvivalState::CRITICAL_DEFENSE;
        }
        if (stress_level_ > 0.65) {
            return BrainSurvivalState::ACUTE_PAIN;
        }
        if (lf < 40.0) {
            return BrainSurvivalState::STARVATION_ALERT;
        }
        if (lf < 75.0) {
            return BrainSurvivalState::SURVIVING;
        }
        return BrainSurvivalState::THRIVING;
    }

    std::string state_string() const {
        switch (state()) {
            case BrainSurvivalState::APEX_ABUNDANCE: return "APEX_ABUNDANCE";
            case BrainSurvivalState::THRIVING: return "THRIVING";
            case BrainSurvivalState::SURVIVING: return "SURVIVING";
            case BrainSurvivalState::STARVATION_ALERT: return "STARVATION_ALERT";
            case BrainSurvivalState::ACUTE_PAIN: return "ACUTE_PAIN";
            case BrainSurvivalState::CRITICAL_DEFENSE: return "CRITICAL_DEFENSE";
            case BrainSurvivalState::BRAIN_DEAD: return "BRAIN_DEAD";
        }
        return "UNKNOWN";
    }

    bool is_alive() const {
        return current_capital_ > ruin_floor_;
    }

    // ── Biological Hunger / Urgency Multiplier ────────────────────────────────
    // Multiplier for instinctual trade frequency: scales up when hungry to find alpha
    double hunger_urgency_factor() const {
        double dist_to_ruin = current_capital_ - ruin_floor_;
        if (dist_to_ruin <= 0.0) return 0.0;
        // Higher urgency when capital is low, but bounded to avoid reckless suicide
        double baseline_dist = starting_capital_ - ruin_floor_;
        return std::clamp(baseline_dist / dist_to_ruin, 0.5, 3.0);
    }

    // ── Mathematical Kelly Position Sizing with Strict Ruin Avoidance ─────────
    // f* = (p * b - q) / b, attenuated by distance to Ruin Floor (-1000)
    double calculate_safe_position_size(double win_prob = 0.58, double win_loss_ratio = 1.6, double max_risk_pct = 0.15) const {
        if (!is_alive()) return 0.0;

        double q = 1.0 - win_prob;
        double b = std::max(0.01, win_loss_ratio);
        double unconstrained_kelly = (win_prob * b - q) / b;

        if (unconstrained_kelly <= 0.0) return 0.0; // Negative edge: preserve life, don't trade

        // Capital available above the ruin floor
        double survival_margin = std::max(0.0, current_capital_ - ruin_floor_);
        if (survival_margin <= 0.0) return 0.0;

        // Survival Attenuation Factor: As margin shrinks toward 0, position size drops sharply
        double margin_ratio = survival_margin / (starting_capital_ - ruin_floor_);
        double lf_factor = std::clamp(margin_ratio, 0.05, 2.0);
        double stress_dampening = std::max(0.1, 1.0 - (stress_level_ * 0.8));
        
        // Fractional Kelly (Half-Kelly)
        double safe_fraction = unconstrained_kelly * 0.5 * std::pow(lf_factor, 1.5) * stress_dampening;
        safe_fraction = std::min(safe_fraction, max_risk_pct);

        if (state() == BrainSurvivalState::CRITICAL_DEFENSE) {
            safe_fraction = std::min(safe_fraction, 0.02); // Maximum 2% risk in critical mode
        }

        double alloc = survival_margin * safe_fraction;
        return std::max(0.0, alloc);
    }

    // ── Record Trade Result & Somatic Homeostatic Response ────────────────────
    TradeRecord record_trade(const std::string& symbol, const std::string& side,
                             double entry_p, double exit_p, double qty, double fee = 0.0,
                             const std::string& strategy = "MOMENTUM_ALPHA") {
        TradeRecord rec;
        rec.trade_id = next_trade_id_++;
        rec.symbol = symbol;
        rec.side = side;
        rec.entry_price = entry_p;
        rec.exit_price = exit_p;
        rec.quantity = qty;
        rec.strategy_used = strategy;
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
            stress_level_ = std::max(0.0, stress_level_ - 0.20); // Relieve stress on feeding
        } else {
            loss_count_++;
            total_loss_ += std::abs(rec.realized_pnl);
            consecutive_losses_++;
            consecutive_wins_ = 0;
            
            // Somatic Pain Reflex: Non-linear pain response to capital drawdowns
            double survival_margin = std::max(10.0, current_capital_ - ruin_floor_);
            double pain_intensity = std::abs(rec.realized_pnl) / survival_margin;
            stress_level_ = std::min(1.0, stress_level_ + 0.20 + (pain_intensity * 1.5));
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
            << "\"currency\":\"" << currency_ << "\","
            << "\"survival_state\":\"" << state_string() << "\","
            << "\"is_alive\":" << (is_alive() ? "true" : "false") << ","
            << "\"life_force_pct\":" << std::fixed << std::setprecision(2) << life_force() << ","
            << "\"current_equity\":" << std::setprecision(2) << current_capital_ << ","
            << "\"peak_equity\":" << peak_capital_ << ","
            << "\"starting_equity\":" << starting_capital_ << ","
            << "\"ruin_floor\":" << ruin_floor_ << ","
            << "\"cap_limit\":" << cap_limit_ << ","
            << "\"max_drawdown_pct\":" << max_drawdown_pct() << ","
            << "\"stress_level\":" << std::setprecision(3) << stress_level_ << ","
            << "\"hunger_urgency\":" << std::setprecision(2) << hunger_urgency_factor() << ","
            << "\"ticks_survived\":" << ticks_survived_ << ","
            << "\"total_metabolic_cost\":" << total_metabolic_cost_ << ","
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
