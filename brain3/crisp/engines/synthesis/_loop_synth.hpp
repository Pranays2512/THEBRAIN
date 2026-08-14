#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>

namespace brain3 { 
namespace engines { 
namespace synthesis {

struct LoopSpec {
    int init;
    std::string lo;
    std::string hi;
    std::string upd;
    std::function<int(int)> rng_lo;
    std::function<int(int)> rng_hi;
    std::function<int(int, int)> upd_fn;
};

class LoopSynth {
private:
    std::vector<int> INITS = {0, 1};
    
    struct RangeDef {
        std::string lo_str, hi_str;
        std::function<int(int)> r_lo;
        std::function<int(int)> r_hi;
    };
    
    std::vector<RangeDef> RANGES = {
        {"1", "n + 1", [](int n){ return 1; }, [](int n){ return n + 1; }},
        {"2", "n + 1", [](int n){ return 2; }, [](int n){ return n + 1; }},
        {"0", "n",     [](int n){ return 0; }, [](int n){ return n; }},
        {"1", "n",     [](int n){ return 1; }, [](int n){ return n; }}
    };
    
    std::map<std::string, std::function<int(int, int)>> UPDATES = {
        {"acc + i",      [](int acc, int i){ return acc + i; }},
        {"acc * i",      [](int acc, int i){ return acc * i; }},
        {"acc + i * i",  [](int acc, int i){ return acc + i * i; }},
        {"std::max(acc, i)", [](int acc, int i){ return std::max(acc, i); }},
        {"acc + 1",      [](int acc, int i){ return acc + 1; }}
    };

    int run(int init, std::function<int(int)> r_lo, std::function<int(int)> r_hi, std::function<int(int, int)> upd_fn, int n) {
        int acc = init;
        int start = r_lo(n);
        int end = r_hi(n);
        for (int i = start; i < end; i++) {
            acc = upd_fn(acc, i);
        }
        return acc;
    }

public:
    LoopSpec* synthesize(const std::vector<std::pair<int, int>>& examples) {
        if (examples.size() < 4) return nullptr;
        
        int cut = std::max(3, (int)(examples.size() * 0.6));
        
        for (int init : INITS) {
            for (const auto& r : RANGES) {
                for (const auto& [ucode, ufn] : UPDATES) {
                    bool ok = true;
                    // train
                    for (int i = 0; i < cut; i++) {
                        if (run(init, r.r_lo, r.r_hi, ufn, examples[i].first) != examples[i].second) {
                            ok = false; break;
                        }
                    }
                    if (!ok) continue;
                    // holdout
                    for (size_t i = cut; i < examples.size(); i++) {
                        if (run(init, r.r_lo, r.r_hi, ufn, examples[i].first) != examples[i].second) {
                            ok = false; break;
                        }
                    }
                    if (ok) {
                        return new LoopSpec{init, r.lo_str, r.hi_str, ucode, r.r_lo, r.r_hi, ufn};
                    }
                }
            }
        }
        return nullptr;
    }

    std::string render(const std::string& fn, const LoopSpec& spec) {
        return "int " + fn + "(int n) {\n" +
               "    int acc = " + std::to_string(spec.init) + ";\n" +
               "    for (int i = " + spec.lo + "; i < " + spec.hi + "; i++) {\n" +
               "        acc = " + spec.upd + ";\n" +
               "    }\n" +
               "    return acc;\n" +
               "}\n";
    }
};

}}}
