#pragma once
#include <string>
#include <vector>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <algorithm>

namespace brain3 {
namespace engines {
namespace synthesis {

class Refuter {
private:
    std::string int1_scope(std::function<int(int)> f, std::function<int(int)> oracle, int lo = 0, int hi = 60) const {
        std::vector<int> holds;
        for (int n = lo; n <= hi; n++) {
            try {
                if (f(n) == oracle(n)) holds.push_back(n);
            } catch (...) {}
        }
        if (holds.empty()) return "holds nowhere in [" + std::to_string(lo) + "," + std::to_string(hi) + "]";
        
        std::vector<std::pair<int, int>> runs;
        int start = holds[0], prev = holds[0];
        for (size_t i = 1; i < holds.size(); i++) {
            if (holds[i] == prev + 1) prev = holds[i];
            else { runs.push_back({start, prev}); start = prev = holds[i]; }
        }
        runs.push_back({start, prev});
        
        std::ostringstream txt;
        bool first = true;
        for (const auto& r : runs) {
            if (!first) txt << " U ";
            if (r.first != r.second) txt << "[" << r.first << "," << r.second << "]";
            else txt << "{" << r.first << "}";
            first = false;
        }
        if (runs.size() == 1 && runs[0].first == lo && runs[0].second == hi)
            return "holds for ALL n in [" + std::to_string(lo) + "," + std::to_string(hi) + "]";
        return "holds for n in " + txt.str();
    }

    std::string list_probes(std::function<int(const std::vector<int>&)> f, 
                            std::function<int(const std::vector<int>&)> oracle) const {
        std::map<std::string, std::vector<std::vector<int>>> cases = {
            {"empty", {{}}},
            {"singleton", {{7}, {-3}}},
            {"negatives", {{-3, -5, -1}, {-2, -9, -4}}},
            {"duplicates", {{5, 5, 5}, {2, 2, 3, 3}}},
            {"sorted", {{1, 2, 3, 4}, {0, 5, 9}}},
            {"reverse", {{9, 5, 1}, {4, 3, 2, 1}}},
            {"mixed", {{3, -1, 4, -1, 5}, {-2, 8, -6, 7}}}
        };
        std::vector<std::string> broken;
        for (const auto& [name, inputs] : cases) {
            for (const auto& lst : inputs) {
                try {
                    // if it throws or gives wrong answer, it breaks
                    if (f(lst) != oracle(lst)) {
                        broken.push_back(name); break;
                    }
                } catch (...) {
                    // f threw. Did oracle throw?
                    try {
                        oracle(lst);
                        broken.push_back(name); break;  // oracle fine, f threw -> breaks
                    } catch (...) {} // both threw, OK test
                }
            }
        }
        if (broken.empty()) return "holds on all probed list classes";
        std::string res = "BREAKS on: ";
        for (size_t i = 0; i < broken.size(); i++) {
            if (i > 0) res += ", ";
            res += broken[i];
        }
        return res;
    }

public:
    struct RefuteIntResult {
        bool robust;
        std::optional<int> breaks_at;
        double fail_rate;
        std::string scope;
    };

    RefuteIntResult refute_int1(std::function<int(int)> f, std::function<int(int)> oracle, int n_tests = 2000) const {
        std::optional<int> breaks_at;
        int fails = 0, fair = 0;
        for (int i = 0; i < n_tests; i++) {
            int arg = i % 26; // simple generic test range
            try {
                int exp = oracle(arg);
                fair++;
                try {
                    if (f(arg) != exp) throw std::runtime_error("wrong");
                } catch (...) {
                    fails++;
                    if (!breaks_at.has_value()) breaks_at = arg;
                }
            } catch (...) {}
        }
        return {
            !breaks_at.has_value(),
            breaks_at,
            fair ? (double)fails / fair : 0.0,
            int1_scope(f, oracle)
        };
    }

    struct RefuteListResult {
        bool robust;
        std::optional<std::vector<int>> breaks_at;
        double fail_rate;
        std::string scope;
    };

    RefuteListResult refute_list(std::function<int(const std::vector<int>&)> f, 
                                 std::function<int(const std::vector<int>&)> oracle, 
                                 int n_tests = 2000) const {
        std::optional<std::vector<int>> breaks_at;
        int fails = 0, fair = 0;
        
        auto gen_list = [](int i) {
            std::vector<int> L;
            int sz = (i % 8) + 1;
            for (int j = 0; j < sz; j++) L.push_back((i + j) % 19 - 9);
            return L;
        };

        for (int i = 0; i < n_tests; i++) {
            auto arg = gen_list(i);
            try {
                int exp = oracle(arg);
                fair++;
                try {
                    if (f(arg) != exp) throw std::runtime_error("wrong");
                } catch (...) {
                    fails++;
                    if (!breaks_at.has_value()) breaks_at = arg;
                }
            } catch (...) {}
        }
        return {
            !breaks_at.has_value(),
            breaks_at,
            fair ? (double)fails / fair : 0.0,
            list_probes(f, oracle)
        };
    }
};

}}}
