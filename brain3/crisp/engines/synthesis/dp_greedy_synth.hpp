#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>
#include <algorithm>

namespace brain3 {
namespace engines {
namespace synthesis {

struct DPGreedySpec {
    std::string cur, best;
};

class DPGreedySynth {
private:
    std::map<std::string, std::function<int(int, int)>> CUR = {
        {"std::max(x, cur + x)", [](int c, int x){ return std::max(x, c + x); }},
        {"cur + x", [](int c, int x){ return c + x; }},
        {"std::max(cur, x)", [](int c, int x){ return std::max(c, x); }}
    };

    std::map<std::string, std::function<int(int, int)>> BEST = {
        {"std::max(best, cur)", [](int b, int c){ return std::max(b, c); }},
        {"best + cur", [](int b, int c){ return b + c; }}
    };

public:
    int run_dp(std::function<int(int, int)> cf, std::function<int(int, int)> bf, const std::vector<int>& arr) {
        if (arr.empty()) return 0;
        int cur = arr[0], best = arr[0];
        for (size_t i = 1; i < arr.size(); i++) {
            cur = cf(cur, arr[i]);
            best = bf(best, cur);
        }
        return best;
    }

    int brute_max_subarray(const std::vector<int>& arr) {
        if (arr.empty()) return 0;
        int max_sum = -1000000;
        for (size_t i = 0; i < arr.size(); i++) {
            int sum = 0;
            for (size_t j = i; j < arr.size(); j++) {
                sum += arr[j];
                max_sum = std::max(max_sum, sum);
            }
        }
        return max_sum;
    }

    DPGreedySpec* synth_dp(const std::vector<std::vector<int>>& pos) {
        for (const auto& [cc, cf] : CUR) {
            for (const auto& [bc, bf] : BEST) {
                bool ok = true;
                for (const auto& a : pos) {
                    if (run_dp(cf, bf, a) != brute_max_subarray(a)) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    auto spec = new DPGreedySpec();
                    spec->cur = cc;
                    spec->best = bc;
                    return spec;
                }
            }
        }
        return nullptr;
    }

    // ── GREEDY: coin change (fixed strategy; oracle = DP-optimal) ────────────────
    int greedy_coins(std::vector<int> coins, int amount) {
        std::sort(coins.begin(), coins.end(), std::greater<int>());
        int n = 0, amt = amount;
        for (int c : coins) {
            while (amt >= c) {
                amt -= c;
                n++;
            }
        }
        return amt == 0 ? n : -1;
    }

    int dp_min_coins(const std::vector<int>& coins, int amount) {
        const int INF = 1000000;
        std::vector<int> dp(amount + 1, INF);
        dp[0] = 0;
        for (int a = 1; a <= amount; a++) {
            for (int c : coins) {
                if (c <= a && dp[a - c] + 1 < dp[a]) {
                    dp[a] = dp[a - c] + 1;
                }
            }
        }
        return dp[amount] < INF ? dp[amount] : -1;
    }

    int stress_greedy(const std::vector<int>& coins, const std::vector<int>& random_amounts) {
        for (int amt : random_amounts) {
            if (greedy_coins(coins, amt) != dp_min_coins(coins, amt)) {
                return amt;
            }
        }
        return -1;
    }
};

}}}
