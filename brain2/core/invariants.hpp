#pragma once
// invariants.hpp — native port of the proven invariant checker (invariant_miner.py).
// Self-minted necessary-condition verifiers: mine the invariants that hold across examples,
// then reject any candidate output that violates one — a fast pre-filter before the
// expensive oracle. Stable + verified in Python; ported here for the synthesis hot path.
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace brain2 {

struct Example {
    std::vector<double> args;
    double              y;
};

inline const std::vector<std::string>& invariant_names() {
    static const std::vector<std::string> N = {
        "out_nonneg", "out_positive", "out_ge_max_arg", "out_le_min_arg",
        "out_ge_first", "out_le_first", "out_divides_args"};
    return N;
}

inline bool invariant_holds(const std::string& name, const Example& e) {
    const auto& a = e.args;
    double y = e.y;
    if (a.empty()) return true;
    if (name == "out_nonneg")      return y >= 0;
    if (name == "out_positive")    return y > 0;
    if (name == "out_ge_max_arg")  return y >= *std::max_element(a.begin(), a.end());
    if (name == "out_le_min_arg")  return y <= *std::min_element(a.begin(), a.end());
    if (name == "out_ge_first")    return y >= a[0];
    if (name == "out_le_first")    return y <= a[0];
    if (name == "out_divides_args") {
        if (y <= 0) return false;
        for (double v : a)
            if (std::fmod(v, y) > 1e-9 && std::fabs(std::fmod(v, y) - y) > 1e-9) return false;
        return true;
    }
    return false;
}

inline bool invariant_holds_all(const std::string& name, const std::vector<Example>& ex) {
    for (const auto& e : ex)
        if (!invariant_holds(name, e)) return false;
    return true;
}

// Mine: the invariants that hold across ALL examples.
inline std::vector<std::string> mine_invariants(const std::vector<Example>& ex) {
    std::vector<std::string> out;
    for (const auto& n : invariant_names())
        if (invariant_holds_all(n, ex)) out.push_back(n);
    return out;
}

// Check: a candidate's outputs vs admitted invariants. Returns the violated name, or "".
inline std::string check_invariants(const std::vector<Example>& cand,
                                    const std::vector<std::string>& admitted) {
    for (const auto& n : admitted)
        if (!invariant_holds_all(n, cand)) return n;
    return "";
}

}  // namespace brain2
