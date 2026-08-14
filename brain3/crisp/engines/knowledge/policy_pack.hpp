#pragma once
/**
 * policy_pack.hpp — hardcoded physics + chemistry policy axioms.
 * Port of brain2/engines/knowledge/policy_pack.py to C++.
 *
 * Provides PACK: the full set of multi-domain composition policies
 * (physics: force, momentum, KE, work, power, pressure; chemistry:
 *  moles, molarity, density, conc_mass).
 *
 * Usage:
 *   auto policies = brain2::knowledge::get_policy_pack();
 *   // Each entry: {target, {input1, input2, ...}, formula_string}
 */

#include <string>
#include <vector>
#include <tuple>
#include "crisp/engines/reasoning/means_ends.hpp"

namespace brain2 {
namespace knowledge {

/**
 * Returns the full pre-seeded policy pack as a vector of Policy objects.
 * Mirrors brain2's PACK list exactly, including dead routes to test proposer.
 *
 * (target, inputs, formula_expr_string)
 * The formula strings use the same tuple-notation as means_ends Policy.
 */
struct PolicyEntry {
    std::string target;
    std::vector<std::string> inputs;
    std::string formula;   // s-expr string: "(* mass accel)" etc.
};

inline std::vector<PolicyEntry> POLICY_PACK_ENTRIES() {
    return {
        // ── Physics ─────────────────────────────────────────────────────────
        // F = m * a
        {"force",    {"mass", "accel"},           "(* mass accel)"},
        // weight = m * g  [dead route — no gravity in sample facts]
        {"weight",   {"mass", "gravity"},         "(* mass gravity)"},
        // momentum route1: p = F * t  [dead — no time]
        {"momentum", {"force", "time"},           "(* force time)"},
        // momentum route2: p = m * v  [live]
        {"momentum", {"mass", "speed"},           "(* mass speed)"},
        // KE = 0.5 * m * v^2
        {"ke",       {"mass", "speed"},           "(* 0.5 (* mass (^ speed 2)))"},
        // PE = m * g * h  [dead — no gravity/height]
        {"pe",       {"mass", "gravity", "height"}, "(* (* mass gravity) height)"},
        // work route1: W = P * t  [dead — no time]
        {"work",     {"power", "time"},           "(* power time)"},
        // work route2: W = F * d  [live]
        {"work",     {"force", "distance"},       "(* force distance)"},
        // power route1: P = W / t  [dead — needs time]
        {"power",    {"work", "time"},            "(/ work time)"},
        // power route2: P = F * v  [live]
        {"power",    {"force", "speed"},          "(* force speed)"},
        // pressure route1: p = W / A  [dead — weight dead]
        {"pressure", {"weight", "area"},          "(/ weight area)"},
        // pressure route2: p = F / A  [live]
        {"pressure", {"force", "area"},           "(/ force area)"},
        // impulse = F * t  [dead if no time]
        {"impulse",  {"force", "time"},           "(* force time)"},

        // ── Chemistry ────────────────────────────────────────────────────────
        // n = m / M
        {"moles",     {"mass", "molar_mass"},     "(/ mass molar_mass)"},
        // c = n / V
        {"molarity",  {"moles", "volume"},        "(/ moles volume)"},
        // ρ = m / V
        {"density",   {"mass", "volume"},         "(/ mass volume)"},
        // c_m = c * M
        {"conc_mass", {"molarity", "molar_mass"}, "(* molarity molar_mass)"},
    };
}

/**
 * Build Policy objects (compatible with MeansEndsSolver) from the pack.
 * Each formula string is converted to a means_ends ExprPtr via a mini-parser.
 */
inline reasoning::ExprPtr _str_to_expr(const std::string& s) {
    // Minimal s-expr parser: "(op a b)" or "symbol" or "number"
    std::string t = s;
    // Trim
    while (!t.empty() && t.front() == ' ') t = t.substr(1);
    while (!t.empty() && t.back() == ' ')  t.pop_back();
    if (t.empty()) return reasoning::lit(0.0);

    if (t.front() == '(') {
        // strip outer parens
        t = t.substr(1, t.size() - 2);
        // extract op
        size_t i = 0;
        while (i < t.size() && t[i] != ' ') ++i;
        std::string op_name = t.substr(0, i);
        // split remaining into args
        std::vector<std::string> arg_strs;
        int depth = 0;
        std::string cur;
        for (size_t j = i + 1; j <= t.size(); ++j) {
            char c = (j < t.size()) ? t[j] : ' ';
            if (c == '(') { ++depth; cur += c; }
            else if (c == ')') { --depth; cur += c; }
            else if (c == ' ' && depth == 0) {
                if (!cur.empty()) { arg_strs.push_back(cur); cur.clear(); }
            } else {
                cur += c;
            }
        }
        if (!cur.empty()) arg_strs.push_back(cur);
        std::vector<reasoning::ExprPtr> args;
        for (const auto& a : arg_strs) args.push_back(_str_to_expr(a));
        return reasoning::op(op_name, args);
    }

    // number?
    try {
        size_t sz;
        double v = std::stod(t, &sz);
        if (sz == t.size()) return reasoning::lit(v);
    } catch (...) {}

    // variable
    return reasoning::var(t);
}

inline std::vector<reasoning::Policy> get_policy_pack() {
    std::vector<reasoning::Policy> policies;
    for (const auto& e : POLICY_PACK_ENTRIES()) {
        reasoning::Policy p;
        p.target = e.target;
        p.inputs = e.inputs;
        p.expr   = _str_to_expr(e.formula);
        policies.push_back(p);
    }
    return policies;
}

/**
 * Load policy pack facts into a PolicyMemory.
 * Returns the number of policies loaded.
 */
inline int load_policy_pack(reasoning::PolicyMemory& mem) {
    auto policies = get_policy_pack();
    for (auto& p : policies) mem.add(p);
    return (int)policies.size();
}

} // namespace knowledge
} // namespace brain2
