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

// ── Feature signature: cheap structural flags that separate task families ─────
using TaskSig = std::set<std::string>;

inline TaskSig task_features(
    const std::string& kind,
    const std::vector<std::pair<std::vector<int>, int>>& scalar_examples)
{
    TaskSig sig;
    if (kind == "int1" || kind == "int2") sig.insert("scalar");
    else sig.insert("listish");

    sig.insert("out_scalar");

    std::vector<int> in_vals;
    for (const auto& [a, y] : scalar_examples)
        for (int v : a) in_vals.push_back(v);

    if (std::any_of(in_vals.begin(), in_vals.end(), [](int v){ return v < 0; }))
        sig.insert("neg_in");

    for (const auto& [a, y] : scalar_examples) {
        if (!a.empty() && y > *std::max_element(a.begin(), a.end())) {
            sig.insert("out_exceeds_max");
            break;
        }
    }
    return sig;
}

// ── FeatureProposer: keyed by task signature, not just kind ───────────────────
class OnlineProposer2 {
private:
    // signature -> {space_name: weight}
    std::map<TaskSig, std::map<std::string, double>> prior;

    // Available space names per kind (mirrors synth_engine ROUTES)
    std::map<std::string, std::vector<std::string>> ROUTES = {
        {"int1",   {"loop_synth", "math_synth", "dp_proposer"}},
        {"int2",   {"loop_synth_v3", "math_synth"}},
        {"list",   {"loop_synth_v4", "dp_proposer", "graph_synth"}}
    };

public:
    std::vector<std::string> order(const std::string& kind, const TaskSig& sig) {
        auto it = ROUTES.find(kind);
        if (it == ROUTES.end()) return {};
        const auto& backs = it->second;

        auto& w = prior[sig];
        for (const auto& n : backs) w.emplace(n, 1.0);

        std::vector<std::string> ordered = backs;
        std::sort(ordered.begin(), ordered.end(), [&](const std::string& a, const std::string& b) {
            return w[a] > w[b];
        });
        return ordered;
    }

    // Reward a space that produced a verified result
    void reward(const TaskSig& sig, const std::string& space, double amount = 2.0) {
        prior[sig][space] += amount;
    }

    // Penalize a space whose candidate broke under stress
    void penalize(const TaskSig& sig, const std::string& space, double amount = 1.0) {
        prior[sig][space] -= amount;
    }

    // Attempt solve: returns (space_name, attempts)
    std::pair<std::string, int> solve(
        const std::string& kind,
        const TaskSig& sig,
        std::function<bool(const std::string&)> try_space)
    {
        int attempts = 0;
        for (const auto& space : order(kind, sig)) {
            attempts++;
            bool ok = try_space(space);
            if (ok) {
                reward(sig, space);
                return {space, attempts};
            }
            penalize(sig, space);
        }
        return {"", attempts};
    }

    void print_learned() const {
        for (const auto& [sig, weights] : prior) {
            std::string fam = (sig.count("out_list") ? "sort-family" : "subarray-family");
            std::cout << "  " << fam << " -> {";
            bool first = true;
            for (const auto& [k, v] : weights) {
                if (!first) std::cout << ", ";
                std::cout << k << ": " << v;
                first = false;
            }
            std::cout << "}\n";
        }
    }
};

}}}
