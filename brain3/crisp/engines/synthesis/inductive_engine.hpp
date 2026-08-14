#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <algorithm>
#include <iostream>

namespace brain3 {
namespace engines {
namespace synthesis {

// ── Rule: A -> B with confidence scores ──────────────────────────────────────
struct InductiveRule {
    std::string a;
    std::string b;
    double conf_train;
    double conf_test;
    int support;
};

class InductiveLearner {
private:
    bool has_event(const std::vector<std::string>& ep, const std::string& a) const {
        return std::find(ep.begin(), ep.end(), a) != ep.end();
    }

    bool before(const std::vector<std::string>& ep, const std::string& a, const std::string& b) const {
        auto ia = std::find(ep.begin(), ep.end(), a);
        auto ib = std::find(ep.begin(), ep.end(), b);
        return ia != ep.end() && ib != ep.end() && ia < ib;
    }

public:
    std::map<std::pair<std::string,std::string>, std::pair<int,double>>
    candidates(const std::vector<std::vector<std::string>>& train,
               int min_support, double min_conf) const
    {
        std::set<std::pair<std::string,std::string>> pairs;
        for (const auto& ep : train) {
            for (size_t i = 0; i < ep.size(); i++) {
                for (size_t j = i + 1; j < ep.size(); j++) {
                    if (ep[i] != ep[j])
                        pairs.insert({ep[i], ep[j]});
                }
            }
        }

        std::map<std::pair<std::string,std::string>, std::pair<int,double>> out;
        for (const auto& [a, b] : pairs) {
            int support = 0;
            int seen_a = 0;
            for (const auto& ep : train) {
                if (has_event(ep, a)) {
                    seen_a++;
                    if (before(ep, a, b)) support++;
                }
            }
            double conf = seen_a > 0 ? (double)support / seen_a : 0.0;
            if (support >= min_support && conf >= min_conf)
                out[{a, b}] = {support, conf};
        }
        return out;
    }

    struct MineResult {
        std::vector<InductiveRule> promoted;
        std::vector<std::tuple<std::string, std::string, std::string>> rejected;
    };

    MineResult mine(
        const std::vector<std::vector<std::string>>& train,
        const std::vector<std::vector<std::string>>& test,
        int min_support = 2, double min_conf = 0.8,
        double verify_conf = 0.7, int min_test = 2) const
    {
        MineResult result;
        auto cand = candidates(train, min_support, min_conf);

        for (auto& [ab, sc] : cand) {
            const auto& [a, b] = ab;
            auto [support, conf] = sc;

            int seen_a = 0, holds = 0;
            for (const auto& ep : test) {
                if (has_event(ep, a)) {
                    seen_a++;
                    if (before(ep, a, b)) holds++;
                }
            }
            if (seen_a < min_test) {
                result.rejected.push_back({a, b, "untested (only " + std::to_string(seen_a) + " hold-out cases)"});
                continue;
            }
            double conf_test = (double)holds / seen_a;
            if (conf_test >= verify_conf) {
                result.promoted.push_back({a, b, (float)conf, (float)conf_test, support});
            } else {
                result.rejected.push_back({a, b,
                    "spurious — train " + std::to_string((int)(conf*100)) + "% but hold-out "
                    + std::to_string((int)(conf_test*100)) + "%"});
            }
        }
        return result;
    }

    // Install verified rules into a simple fact store
    void promote_into(
        std::map<std::pair<std::string,std::string>, std::string>& kb,
        const std::vector<InductiveRule>& promoted,
        const std::string& relation = "leads_to") const
    {
        for (const auto& r : promoted)
            kb[{r.a, relation}] = r.b;
    }
};

}}}
