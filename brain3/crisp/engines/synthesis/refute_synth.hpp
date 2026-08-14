#pragma once
#include "crisp/engines/synthesis/refuter.hpp"
#include <vector>
#include <string>
#include <functional>
#include <optional>

namespace brain3 {
namespace engines {
namespace synthesis {

class RefuteSynth {
public:
    struct RepairResult {
        std::function<int(const std::vector<int>&)> final_fn;
        std::vector<std::string> log;
    };

    // Attempt to synthesize, refute, and self-correct on lists
    RepairResult synth_self_correct_list(
        std::function<int(const std::vector<int>&)> oracle,
        std::vector<std::vector<int>> inputs,
        std::function<std::function<int(const std::vector<int>&)>(const std::vector<std::pair<std::vector<int>, int>>&)> synth_fn,
        int max_iters = 5)
    {
        std::vector<std::string> log;
        std::vector<std::pair<std::vector<int>, int>> examples;
        
        for (const auto& a : inputs) {
            try { examples.push_back({a, oracle(a)}); } catch (...) {}
        }

        std::function<int(const std::vector<int>&)> current_best = nullptr;
        Refuter refuter;

        for (int it = 0; it < max_iters; it++) {
            auto code = synth_fn(examples);
            if (!code) {
                log.push_back("  iter " + std::to_string(it) + ": no candidate");
                return {nullptr, log};
            }
            current_best = code;

            auto rep = refuter.refute_list(code, oracle);
            if (rep.robust) {
                log.push_back("  iter " + std::to_string(it) + ": ROBUST  (scope: " + rep.scope + ")");
                return {code, log};
            }
            
            auto bad = rep.breaks_at.value();
            log.push_back("  iter " + std::to_string(it) + ": breaks at [some input]  [" + rep.scope + "]  -> add counterexample, re-synth");
            
            try {
                examples.push_back({bad, oracle(bad)});
            } catch (...) {
                return {code, log};
            }
        }
        log.push_back("  gave up after " + std::to_string(max_iters) + " iters");
        return {current_best, log};
    }
};

}}}
