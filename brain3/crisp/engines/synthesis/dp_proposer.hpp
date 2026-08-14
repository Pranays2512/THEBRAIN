#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>
#include <cmath>
#include <numeric>

namespace brain3 {
namespace engines {
namespace synthesis {

struct DPRecurrence {
    std::string init, cur, best;
    std::function<int(int, int)> cf;
    std::function<int(int, int)> bf;
};

class DPProposer {
public:
    std::vector<std::string> IK = {"first", "zero"};
    
    std::map<std::string, std::function<int(int, int)>> CUR = {
        {"std::max(x, cur + x)", [](int c, int x){ return std::max(x, c + x); }},
        {"std::min(x, cur + x)", [](int c, int x){ return std::min(x, c + x); }},
        {"cur + x", [](int c, int x){ return c + x; }},
        {"std::max(cur, x)", [](int c, int x){ return std::max(c, x); }},
        {"std::min(cur, x)", [](int c, int x){ return std::min(c, x); }},
        {"x", [](int c, int x){ return x; }},
        {"cur * x", [](int c, int x){ return c * x; }}
    };

    std::map<std::string, std::function<int(int, int)>> BEST = {
        {"std::max(best, cur)", [](int b, int c){ return std::max(b, c); }},
        {"std::min(best, cur)", [](int b, int c){ return std::min(b, c); }},
        {"best + cur", [](int b, int c){ return b + c; }},
        {"cur", [](int b, int c){ return c; }}
    };

    std::vector<DPRecurrence> ALL;

    DPProposer() {
        for (const std::string& ik : IK) {
            for (const auto& [ck, cf] : CUR) {
                for (const auto& [bk, bf] : BEST) {
                    ALL.push_back({ik, ck, bk, cf, bf});
                }
            }
        }
    }

    int run_dp(const DPRecurrence& rec, const std::vector<int>& arr) {
        if (arr.empty()) return 0;
        int cur = 0, best = 0;
        size_t start = 0;
        if (rec.init == "first") {
            cur = best = arr[0];
            start = 1;
        }
        for (size_t i = start; i < arr.size(); i++) {
            cur = rec.cf(cur, arr[i]);
            best = rec.bf(best, cur);
        }
        return best;
    }

    // Dummy feature extraction since we don't have the full tree_reason implemented yet.
    // In a real port, we'd use Pearson correlation and decision trees.
    std::vector<double> feats(const std::vector<std::pair<std::vector<int>, int>>& ex) {
        return std::vector<double>(9, 0.0); 
    }

    std::pair<DPRecurrence*, int> search(const std::vector<std::pair<std::vector<int>, int>>& ex, const std::vector<DPRecurrence>& cands) {
        int cut = std::max(3, (int)(ex.size() * 0.6));
        
        for (size_t k = 0; k < cands.size(); k++) {
            const auto& rec = cands[k];
            bool ok = true;
            for (int i = 0; i < cut; i++) {
                if (run_dp(rec, ex[i].first) != ex[i].second) { ok = false; break; }
            }
            if (!ok) continue;
            for (size_t i = cut; i < ex.size(); i++) {
                if (run_dp(rec, ex[i].first) != ex[i].second) { ok = false; break; }
            }
            if (ok) {
                return {new DPRecurrence(rec), k + 1};
            }
        }
        return {nullptr, cands.size()};
    }
};

}}}
