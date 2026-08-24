#pragma once

#include <algorithm>

namespace brain3 {
namespace finance {

// ============================================================================
// CANONICAL conviction -> win-probability mapping for ALL brain3::finance
// engines (multi_asset_scanner_engine, autonomous_trading_instinct_engine,
// and any future engine or backtest harness).
//
// THIS IS THE MAPPING ALL BACKTESTS MUST USE. Do not invent per-engine
// formulas like "0.58 + 0.15*alpha" or "0.62 + min(0.15, |ofi|*0.2)":
// normalize the raw signal into an alpha score in [0,1] first, then call
// canonical_win_probability(alpha).
//
//     win_probability = clamp(kWinProbBase + kWinProbAlphaGain * alpha, 0.5, 0.85)
//
// Rationale: base 0.55 encodes a mild structural edge; each unit of
// normalized alpha adds up to kWinProbAlphaGain; the [0.5, 0.85] band keeps
// Kelly sizing sane (never coin-flip-negative, never overconfident).
// ============================================================================

inline constexpr double kWinProbBase{0.55};
inline constexpr double kWinProbAlphaGain{0.20};
inline constexpr double kWinProbMin{0.50};
inline constexpr double kWinProbMax{0.85};

// alpha_score_01: composite alpha strength in [0,1] (values outside are clamped).
inline double canonical_win_probability(double alpha_score_01) {
    const double a = std::min(std::max(alpha_score_01, 0.0), 1.0);
    const double p = kWinProbBase + kWinProbAlphaGain * a;
    return std::min(std::max(p, kWinProbMin), kWinProbMax);
}

} // namespace finance
} // namespace brain3
