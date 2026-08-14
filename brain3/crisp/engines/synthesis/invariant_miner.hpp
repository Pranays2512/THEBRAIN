#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace brain3 {
namespace engines {
namespace synthesis {

class InvariantMiner {
public:
    // ── Predicate table: (args_tuple, output) -> bool ───────────────────────────
    using Args = std::vector<int>;
    using Predicate = std::function<bool(const Args&, int)>;

    std::map<std::string, Predicate> PREDICATES = {
        {"out_nonneg",       [](const Args& a, int y){ return y >= 0; }},
        {"out_positive",     [](const Args& a, int y){ return y > 0; }},
        {"out_ge_max_arg",   [](const Args& a, int y){ return !a.empty() && y >= *std::max_element(a.begin(), a.end()); }},
        {"out_le_min_arg",   [](const Args& a, int y){ return !a.empty() && y <= *std::min_element(a.begin(), a.end()); }},
        {"out_ge_first",     [](const Args& a, int y){ return !a.empty() && y >= a[0]; }},
        {"out_le_first",     [](const Args& a, int y){ return !a.empty() && y <= a[0]; }},
        {"out_le_prod",      [](const Args& a, int y) -> bool {
            int p = 1;
            for (int v : a) {
                if (v == 0) return true;
                p *= v;
            }
            return y <= std::abs(p);
        }},
        {"out_divides_args", [](const Args& a, int y) -> bool {
            if (y <= 0) return false;
            for (int v : a) { if (v % y != 0) return false; }
            return true;
        }},
        {"out_even",         [](const Args& a, int y){ return y % 2 == 0; }},
    };

    // Extra predicates can be injected at runtime
    void add_predicate(const std::string& name, Predicate p) {
        PREDICATES[name] = p;
    }

private:
    bool holds(const std::string& name, const std::vector<std::pair<Args, int>>& examples) {
        auto it = PREDICATES.find(name);
        if (it == PREDICATES.end()) return false;
        try {
            return std::all_of(examples.begin(), examples.end(), [&](const auto& ex) {
                return it->second(ex.first, ex.second);
            });
        } catch (...) { return false; }
    }

    bool is_monotone(const std::vector<std::pair<Args, int>>& examples) {
        // 1-arg only: sorted by input, output non-decreasing
        for (const auto& [a, y] : examples) if (a.size() != 1) return false;
        std::vector<std::pair<int, int>> sv;
        for (const auto& [a, y] : examples) sv.push_back({a[0], y});
        std::sort(sv.begin(), sv.end());
        for (size_t i = 0; i + 1 < sv.size(); i++)
            if (sv[i].second > sv[i+1].second) return false;
        return true;
    }

public:
    std::set<std::string> mine(const std::vector<std::pair<Args, int>>& train) {
        std::set<std::string> cands;
        for (const auto& [name, pred] : PREDICATES) {
            if (holds(name, train)) cands.insert(name);
        }
        if (is_monotone(train)) cands.insert("monotonic_increasing");
        return cands;
    }

    std::set<std::string> validate(const std::set<std::string>& cands,
                                   const std::vector<std::pair<Args, int>>& holdout) {
        std::set<std::string> admitted;
        for (const auto& n : cands) {
            if (n == "monotonic_increasing") {
                if (is_monotone(holdout)) admitted.insert(n);
            } else if (holds(n, holdout)) {
                admitted.insert(n);
            }
        }
        return admitted;
    }

    std::pair<bool, std::string> check(std::function<int(Args)> fn,
                                       const std::vector<Args>& probe_inputs,
                                       const std::set<std::string>& admitted) {
        std::vector<std::pair<Args, int>> pairs;
        for (const auto& x : probe_inputs) {
            try { pairs.push_back({x, fn(x)}); }
            catch (...) { return {false, "raised on probe"}; }
        }
        for (const auto& n : admitted) {
            if (n == "monotonic_increasing") {
                if (!is_monotone(pairs)) return {false, "violates monotonic_increasing"};
            } else {
                auto it = PREDICATES.find(n);
                if (it == PREDICATES.end()) continue;
                bool ok = std::all_of(pairs.begin(), pairs.end(), [&](const auto& ex) {
                    return it->second(ex.first, ex.second);
                });
                if (!ok) return {false, "violates " + n};
            }
        }
        return {true, "passes all " + std::to_string(admitted.size()) + " admitted invariants"};
    }

    std::pair<std::set<std::string>, std::vector<std::string>>
    demote(const std::set<std::string>& admitted,
           const std::vector<std::pair<Args, int>>& correct_examples) {
        std::set<std::string> keep;
        std::vector<std::string> dropped;
        for (const auto& n : admitted) {
            bool ok = (n == "monotonic_increasing") ? is_monotone(correct_examples) : holds(n, correct_examples);
            if (ok) keep.insert(n);
            else dropped.push_back(n);
        }
        return {keep, dropped};
    }

    // ── Functional invariants ──────────────────────────────────────────────────
    struct FunctionalPred {
        int arity;
        std::function<bool(std::function<int(int)>, int, int)> pred1;
        std::function<bool(std::function<int(int, int)>, int, int)> pred2;
    };

    std::set<std::string> mine_functional_1arg(std::function<int(int)> f, int seed = 0) {
        // Probes for 1-arg: 0..20
        std::vector<int> probes;
        for (int i = 0; i <= 20; i++) probes.push_back(i);
        std::set<std::string> out;
        // idempotent: f(f(x)) == f(x)
        try {
            bool idem = std::all_of(probes.begin(), probes.end(), [&](int x) {
                return f(f(x)) == f(x);
            });
            if (idem) out.insert("idempotent");
        } catch (...) {}
        // involutive: f(f(x)) == x
        try {
            bool inv = std::all_of(probes.begin(), probes.end(), [&](int x) {
                return f(f(x)) == x;
            });
            if (inv) out.insert("involutive");
        } catch (...) {}
        return out;
    }

    std::set<std::string> mine_functional_2arg(std::function<int(int, int)> f, int seed = 0) {
        std::vector<std::pair<int,int>> probes;
        for (int i = 1; i <= 12; i++)
            for (int j = 1; j <= 12; j += 3)
                probes.push_back({i, j});
        std::set<std::string> out;
        // commutative: f(a,b) == f(b,a)
        try {
            bool comm = std::all_of(probes.begin(), probes.end(), [&](const auto& p) {
                return f(p.first, p.second) == f(p.second, p.first);
            });
            if (comm) out.insert("commutative");
        } catch (...) {}
        return out;
    }
};

}}}
