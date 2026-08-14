#pragma once
#include "crisp/engines/synthesis/invariant_miner.hpp"
#include <string>
#include <vector>
#include <set>
#include <functional>
#include <iostream>

namespace brain3 {
namespace engines {
namespace synthesis {

class SynthInvariant {
public:
    static const int STRESS_N = 1000;

    // Compute admitted invariants from oracle examples
    std::set<std::string> task_invariants(
        std::function<int(int)> oracle,
        const std::vector<int>& mine_inputs,
        const std::vector<int>& holdout_inputs)
    {
        InvariantMiner im;
        std::vector<std::pair<InvariantMiner::Args, int>> train, hold;
        for (int x : mine_inputs)  train.push_back({{x}, oracle(x)});
        for (int x : holdout_inputs) hold.push_back({{x}, oracle(x)});
        return im.validate(im.mine(train), hold);
    }

    struct TriageResult {
        std::string name;
        std::string verdict;
        std::string why;
    };

    // Triage: cheap invariant check first, then full stress vs oracle
    std::pair<std::vector<TriageResult>, int> triage(
        const std::vector<std::pair<std::string, std::function<int(int)>>>& candidates,
        std::function<int(int)> oracle,
        const std::set<std::string>& invariants,
        const std::vector<int>& probe,
        int seed = 0)
    {
        InvariantMiner im;
        // Generate stress inputs (0..20 repeated to STRESS_N)
        std::vector<InvariantMiner::Args> stress_inputs;
        for (int i = 0; i < STRESS_N; i++) {
            stress_inputs.push_back({i % 21});
        }

        // Convert probe to Args format
        std::vector<InvariantMiner::Args> probe_args;
        for (int x : probe) probe_args.push_back({x});

        int calls = 0;
        std::vector<TriageResult> results;

        for (const auto& [name, fn] : candidates) {
            // Wrap fn as Args->int
            auto fn_args = [&fn](InvariantMiner::Args a) -> int {
                return fn(a[0]);
            };

            // Cheap invariant check
            auto [ok, why] = im.check(fn_args, probe_args, invariants);
            if (!ok) {
                results.push_back({name, "rejected cheaply", why});
                continue;
            }

            // Survivor: full stress vs oracle
            bool passed = true;
            for (const auto& xi : stress_inputs) {
                calls++;
                try {
                    if (fn(xi[0]) != oracle(xi[0])) { passed = false; break; }
                } catch (...) { passed = false; break; }
            }
            results.push_back({name, std::string("STRESS ") + (passed ? "pass" : "fail"), ""});
        }

        return {results, calls};
    }
};

}}}
