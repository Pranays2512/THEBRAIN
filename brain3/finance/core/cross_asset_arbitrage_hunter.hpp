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

namespace brain3 {
namespace finance {

struct StatArbSignal {
    std::string asset_a;
    std::string asset_b;
    double price_a{0.0};
    double price_b{0.0};
    double hedge_ratio_beta{1.0};
    double current_spread{0.0};
    double spread_mean{0.0};
    double spread_std{0.0};
    double z_score{0.0};
    double ou_theta{0.0}; // Mean reversion speed
    double half_life_periods{0.0}; // Half-life = ln(2) / theta
    std::string action; // "BUY_A_SELL_B", "SELL_A_BUY_B", "CLOSE", "NONE"
    double expected_edge_bps{0.0};
};

struct TriangularArbOpportunity {
    std::string pair_ab;
    std::string pair_bc;
    std::string pair_ac;
    double price_ab{0.0};
    double price_bc{0.0};
    double price_ac{0.0};
    double implied_ac{0.0};
    double discrepancy_bps{0.0};
    double total_fee_bps{0.0}; // Round-trip cost: one taker fee per leg (3 legs)
    double net_edge_bps{0.0};  // Signed edge AFTER fees; gross edge = discrepancy_bps
    bool is_executable{false};
};

class CrossAssetArbitrageHunter {
private:
    std::string name_{"CrossAssetArbitrageHunter"};

public:
    CrossAssetArbitrageHunter() = default;

    // Estimate Cointegration Hedge Ratio Beta via Ordinary Least Squares (OLS)
    // Spread S_t = P_A - Beta * P_B
    static double calculate_ols_beta(const std::vector<double>& series_a, const std::vector<double>& series_b) {
        if (series_a.size() != series_b.size() || series_a.size() < 10) return 1.0;
        size_t n = series_a.size();

        double mean_a = std::accumulate(series_a.begin(), series_a.end(), 0.0) / n;
        double mean_b = std::accumulate(series_b.begin(), series_b.end(), 0.0) / n;

        double cov_ab = 0.0;
        double var_b = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double da = series_a[i] - mean_a;
            double db = series_b[i] - mean_b;
            cov_ab += da * db;
            var_b += db * db;
        }

        if (var_b < 1e-9) return 1.0;
        return cov_ab / var_b;
    }

    // Fit Ornstein-Uhlenbeck parameters to the spread series
    // dS_t = theta * (mu - S_t) dt + sigma * dW_t
    // Regressing (S_t - S_{t-1}) on S_{t-1}:
    // Delta S_t = a + b * S_{t-1} + e_t => theta = -b / dt, mu = -a / b
    static StatArbSignal analyze_pair(const std::string& asset_a, const std::string& asset_b,
                                      const std::vector<double>& prices_a, const std::vector<double>& prices_b,
                                      double entry_z_threshold = 2.0, double exit_z_threshold = 0.5) {
        StatArbSignal sig;
        sig.asset_a = asset_a;
        sig.asset_b = asset_b;
        sig.action = "NONE";

        if (prices_a.empty() || prices_b.empty() || prices_a.size() != prices_b.size() || prices_a.size() < 20) {
            return sig;
        }

        size_t n = prices_a.size();
        sig.price_a = prices_a.back();
        sig.price_b = prices_b.back();

        // 1. Calculate Hedge Ratio Beta
        sig.hedge_ratio_beta = calculate_ols_beta(prices_a, prices_b);

        // 2. Compute Spread Series
        std::vector<double> spreads(n);
        for (size_t i = 0; i < n; ++i) {
            spreads[i] = prices_a[i] - sig.hedge_ratio_beta * prices_b[i];
        }

        sig.current_spread = spreads.back();
        sig.spread_mean = std::accumulate(spreads.begin(), spreads.end(), 0.0) / n;

        double sum_sq = 0.0;
        for (double s : spreads) {
            sum_sq += (s - sig.spread_mean) * (s - sig.spread_mean);
        }
        sig.spread_std = std::sqrt(sum_sq / (n - 1));

        if (sig.spread_std < 1e-9) {
            sig.z_score = 0.0;
            return sig;
        }

        // 3. Compute Current Z-Score
        sig.z_score = (sig.current_spread - sig.spread_mean) / sig.spread_std;

        // 4. Fit Ornstein-Uhlenbeck Drift Parameter (Mean Reversion Speed)
        std::vector<double> delta_s(n - 1);
        std::vector<double> lag_s(n - 1);
        for (size_t i = 1; i < n; ++i) {
            delta_s[i - 1] = spreads[i] - spreads[i - 1];
            lag_s[i - 1] = spreads[i - 1];
        }

        double mean_delta = std::accumulate(delta_s.begin(), delta_s.end(), 0.0) / (n - 1);
        double mean_lag = std::accumulate(lag_s.begin(), lag_s.end(), 0.0) / (n - 1);

        double cov_lag_delta = 0.0;
        double var_lag = 0.0;
        for (size_t i = 0; i < n - 1; ++i) {
            cov_lag_delta += (lag_s[i] - mean_lag) * (delta_s[i] - mean_delta);
            var_lag += (lag_s[i] - mean_lag) * (lag_s[i] - mean_lag);
        }

        double b_coeff = (var_lag > 1e-9) ? (cov_lag_delta / var_lag) : -0.1;
        if (b_coeff < 0.0) {
            sig.ou_theta = -b_coeff;
            sig.half_life_periods = std::log(2.0) / sig.ou_theta;
        } else {
            sig.ou_theta = 0.01;
            sig.half_life_periods = 999.0; // Non-mean-reverting
        }

        // 5. Generate Statistical Arbitrage Action
        if (sig.z_score >= entry_z_threshold && sig.half_life_periods < 50.0) {
            sig.action = "SELL_A_BUY_B"; // Spread is overextended to the upside -> short spread
            sig.expected_edge_bps = (sig.z_score * sig.spread_std / sig.price_a) * 10000.0;
        } else if (sig.z_score <= -entry_z_threshold && sig.half_life_periods < 50.0) {
            sig.action = "BUY_A_SELL_B"; // Spread is overextended to the downside -> long spread
            sig.expected_edge_bps = (std::abs(sig.z_score) * sig.spread_std / sig.price_a) * 10000.0;
        } else if (std::abs(sig.z_score) <= exit_z_threshold) {
            sig.action = "CLOSE";
            sig.expected_edge_bps = 0.0;
        }

        return sig;
    }

    // Triangular Synthetic Basis Scanner (e.g. BTC/USDT, ETH/BTC, ETH/USDT)
    static TriangularArbOpportunity scan_triangular(const std::string& pair_ab, double price_ab,
                                                     const std::string& pair_bc, double price_bc,
                                                     const std::string& pair_ac, double price_ac,
                                                     double fee_bps = 8.0) {
        TriangularArbOpportunity opp;
        opp.pair_ab = pair_ab;
        opp.pair_bc = pair_bc;
        opp.pair_ac = pair_ac;
        opp.price_ab = price_ab;
        opp.price_bc = price_bc;
        opp.price_ac = price_ac;

        // A triangular cycle fills three legs (AB, BC, AC) and pays one taker fee
        // per leg, so total round-trip cost is 3 * fee_bps — not a single fee.
        const double total_fee_bps = 3.0 * fee_bps;
        opp.total_fee_bps = total_fee_bps;

        // Guard ALL THREE legs against zero/garbage prints before forming the
        // implied product; otherwise a zero price fabricates a fake -10000 bps edge.
        if (price_ab <= 1e-9 || price_bc <= 1e-9 || price_ac <= 1e-9) return opp;

        opp.implied_ac = price_ab * price_bc;
        opp.discrepancy_bps = ((opp.implied_ac - price_ac) / price_ac) * 10000.0;

        // Net edge keeps the sign of the gross discrepancy but subtracts full
        // costs. NOTE: only meaningful when is_executable — below-cost cycles
        // flip this sign; always gate on is_executable, not on net > 0.
        const double direction = (opp.discrepancy_bps >= 0.0) ? 1.0 : -1.0;
        opp.net_edge_bps = direction * (std::abs(opp.discrepancy_bps) - total_fee_bps);

        // Executable only if the gross discrepancy exceeds ALL three taker fees.
        opp.is_executable = ((std::abs(opp.discrepancy_bps) - total_fee_bps) > 0.0);
        return opp;
    }
};

} // namespace finance
} // namespace brain3
