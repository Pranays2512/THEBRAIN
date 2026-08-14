#pragma once
#include "composable_synth.hpp"
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

class ComposableProposer {
private:
    std::vector<std::pair<int, int>> IK = {{0, 1}, {0, 0}, {1, 1}, {1, 0}, {0, -1}};
    std::vector<std::pair<std::string, std::string>> RK = {
        {"1", "n + 1"}, {"2", "n"}, {"0", "n"}, {"1", "n"}, {"2", "n + 1"}
    };
    std::vector<std::string> GK = {
        "None", "n % i == 0", "i % 2 == 0", "i % 2 != 0", "i * i <= n", "n % i != 0", "i <= n / 2"
    };
    std::vector<std::pair<std::string, std::string>> UK = {
        {"a + i", "b"}, {"a + 1", "b"}, {"a + i * i", "b"}, {"a + i * i * i", "b"},
        {"a * i", "b"}, {"a * b", "b + 1"}, {"a * i * i", "b"},
        {"b", "a + b"}, {"b", "a * b"}, {"b", "a + i"},
        {"a + 1", "b + i"}, {"std::max(a,i)", "b"}, {"a>=0 ? std::min(a,i) : i", "b"}
    };
    std::vector<std::string> EK = {
        "None", "a >= n", "a > n", "a == n", "n % i == 0", "a * a > n", "b == 0"
    };
    std::vector<std::string> FK = {
        "a", "b", "-1", "i", "a + b", "a - 1", "b - 1"
    };

public:
    std::vector<ComposableSpec> ALL;

    ComposableProposer() {
        for (const auto& i : IK) {
            for (const auto& r : RK) {
                for (const auto& g : GK) {
                    for (const auto& u : UK) {
                        for (const auto& e : EK) {
                            for (const auto& f : FK) {
                                ALL.push_back({i, r.first, r.second, g, u, e, f});
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<double> feats(const std::vector<std::pair<int, int>>& ex) {
        // dummy feature extraction
        return std::vector<double>(9, 0.0);
    }

    std::vector<ComposableSpec> order(const std::vector<double>& target_feats) {
        // dummy ordering - in production would evaluate decision trees
        return ALL; 
    }

    std::pair<ComposableSpec*, int> search(const std::vector<std::pair<int, int>>& examples, const std::vector<ComposableSpec>& candidates) {
        ComposableSynth synth;
        int cut = std::max(3, (int)(examples.size() * 0.6));
        
        for (size_t k = 0; k < candidates.size(); k++) {
            const auto& p = candidates[k];
            bool ok = true;
            for (int i = 0; i < cut; i++) {
                bool early;
                try {
                    if (synth.run(p, examples[i].first, early) != examples[i].second) { ok = false; break; }
                } catch (...) { ok = false; break; }
            }
            if (!ok) continue;
            for (size_t i = cut; i < examples.size(); i++) {
                bool early;
                try {
                    if (synth.run(p, examples[i].first, early) != examples[i].second) { ok = false; break; }
                } catch (...) { ok = false; break; }
            }
            if (ok) {
                return {new ComposableSpec(p), k + 1};
            }
        }
        return {nullptr, candidates.size()};
    }
};

}}}
