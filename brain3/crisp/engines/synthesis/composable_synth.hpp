#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>

namespace brain3 { 
namespace engines { 
namespace synthesis {

struct ComposableSpec {
    std::pair<int, int> init;
    std::string lo, hi, guard;
    std::pair<std::string, std::string> upd;
    std::string early, final_ret;
};

class ComposableSynth {
private:
    std::vector<std::pair<int, int>> INITS = {
        {0, 1}, {0, 0}, {1, 1}, {1, 0}, {0, -1}
    };

    struct RangeDef {
        std::string lo_str, hi_str;
        std::function<int(int)> r_lo;
        std::function<int(int)> r_hi;
    };
    
    std::vector<RangeDef> RANGES = {
        {"1", "n + 1", [](int n){ return 1; }, [](int n){ return n + 1; }},
        {"2", "n",     [](int n){ return 2; }, [](int n){ return n; }},
        {"0", "n",     [](int n){ return 0; }, [](int n){ return n; }},
        {"1", "n",     [](int n){ return 1; }, [](int n){ return n; }},
        {"2", "n + 1", [](int n){ return 2; }, [](int n){ return n + 1; }}
    };

    std::map<std::string, std::function<bool(int, int)>> GUARDS = {
        {"None",         nullptr},
        {"n % i == 0",   [](int i, int n){ return i != 0 && n % i == 0; }},
        {"i % 2 == 0",   [](int i, int n){ return i % 2 == 0; }},
        {"i % 2 != 0",   [](int i, int n){ return i % 2 != 0; }},
        {"i * i <= n",   [](int i, int n){ return i * i <= n; }},
        {"n % i != 0",   [](int i, int n){ return i != 0 && n % i != 0; }},
        {"i <= n / 2",   [](int i, int n){ return i <= n / 2; }}
    };

    struct UpdDef {
        std::function<int(int, int, int)> ua;
        std::function<int(int, int, int)> ub;
    };
    std::map<std::pair<std::string, std::string>, UpdDef> UPDATES = {
        {{"a + i", "b"}, {[](int a, int b, int i){ return a + i; }, [](int a, int b, int i){ return b; }}},
        {{"a + 1", "b"}, {[](int a, int b, int i){ return a + 1; }, [](int a, int b, int i){ return b; }}},
        {{"a + i * i", "b"}, {[](int a, int b, int i){ return a + i * i; }, [](int a, int b, int i){ return b; }}},
        {{"a + i * i * i", "b"}, {[](int a, int b, int i){ return a + i * i * i; }, [](int a, int b, int i){ return b; }}},
        {{"a * i", "b"}, {[](int a, int b, int i){ return a * i; }, [](int a, int b, int i){ return b; }}},
        {{"a * b", "b + 1"}, {[](int a, int b, int i){ return a * b; }, [](int a, int b, int i){ return b + 1; }}},
        {{"a * i * i", "b"}, {[](int a, int b, int i){ return a * i * i; }, [](int a, int b, int i){ return b; }}},
        {{"b", "a + b"}, {[](int a, int b, int i){ return b; }, [](int a, int b, int i){ return a + b; }}},
        {{"b", "a * b"}, {[](int a, int b, int i){ return b; }, [](int a, int b, int i){ return a * b; }}},
        {{"b", "a + i"}, {[](int a, int b, int i){ return b; }, [](int a, int b, int i){ return a + i; }}},
        {{"a + 1", "b + i"}, {[](int a, int b, int i){ return a + 1; }, [](int a, int b, int i){ return b + i; }}},
        {{"std::max(a,i)", "b"}, {[](int a, int b, int i){ return std::max(a, i); }, [](int a, int b, int i){ return b; }}},
        {{"a>=0 ? std::min(a,i) : i", "b"}, {[](int a, int b, int i){ return (a >= 0) ? std::min(a, i) : i; }, [](int a, int b, int i){ return b; }}}
    };

    std::map<std::string, std::function<bool(int, int, int, int)>> EARLIES = {
        {"None", nullptr},
        {"a >= n",     [](int a, int b, int i, int n){ return a >= n; }},
        {"a > n",      [](int a, int b, int i, int n){ return a > n; }},
        {"a == n",     [](int a, int b, int i, int n){ return a == n; }},
        {"n % i == 0", [](int a, int b, int i, int n){ return i != 0 && n % i == 0; }},
        {"a * a > n",  [](int a, int b, int i, int n){ return a * a > n; }},
        {"b == 0",     [](int a, int b, int i, int n){ return b == 0; }}
    };

    std::map<std::string, std::function<int(int, int, int)>> FINALS = {
        {"a", [](int a, int b, int last){ return a; }},
        {"b", [](int a, int b, int last){ return b; }},
        {"-1", [](int a, int b, int last){ return -1; }},
        {"i", [](int a, int b, int last){ return last; }},
        {"a + b", [](int a, int b, int last){ return a + b; }},
        {"a - 1", [](int a, int b, int last){ return a - 1; }},
        {"b - 1", [](int a, int b, int last){ return b - 1; }}
    };

public:
    int run(const ComposableSpec& p, int n, bool& early_returned) {
        int a = p.init.first, b = p.init.second;
        
        RangeDef rd;
        for (const auto& r : RANGES) {
            if (r.lo_str == p.lo && r.hi_str == p.hi) { rd = r; break; }
        }
        auto gf = GUARDS[p.guard];
        auto uf = UPDATES[p.upd];
        auto ef = EARLIES[p.early];
        
        int last = 0;
        for (int i = rd.r_lo(n); i < rd.r_hi(n); i++) {
            last = i;
            if (!gf || gf(i, n)) {
                int next_a = uf.ua(a, b, i);
                int next_b = uf.ub(a, b, i);
                a = next_a;
                b = next_b;
            }
            if (ef && ef(a, b, i, n)) {
                early_returned = true;
                return i;
            }
        }
        early_returned = false;
        return FINALS[p.final_ret](a, b, last);
    }

public:
    ComposableSpec* synthesize(const std::vector<std::pair<int, int>>& examples) {
        int cut = std::max(3, (int)(examples.size() * 0.6));
        
        for (auto init : INITS) {
            for (const auto& r : RANGES) {
                for (const auto& [g_name, g_fn] : GUARDS) {
                    for (const auto& [u_name, u_fn] : UPDATES) {
                        for (const auto& [e_name, e_fn] : EARLIES) {
                            for (const auto& [f_name, f_fn] : FINALS) {
                                
                                ComposableSpec spec{init, r.lo_str, r.hi_str, g_name, u_name, e_name, f_name};
                                bool ok = true;
                                for (const auto& ex : examples) {
                                    bool early;
                                    try {
                                        if (run(spec, ex.first, early) != ex.second) { ok = false; break; }
                                    } catch (...) { ok = false; break; }
                                }
                                if (ok) {
                                    return new ComposableSpec(spec);
                                }
                            }
                        }
                    }
                }
            }
        }
        return nullptr;
    }

    std::string render(const std::string& fn, const ComposableSpec& p) {
        std::string body = "";
        if (p.guard == "None") {
            body = "        int next_a = " + p.upd.first + ";\n"
                   "        int next_b = " + p.upd.second + ";\n"
                   "        a = next_a; b = next_b;\n";
        } else {
            body = "        if (" + p.guard + ") {\n"
                   "            int next_a = " + p.upd.first + ";\n"
                   "            int next_b = " + p.upd.second + ";\n"
                   "            a = next_a; b = next_b;\n"
                   "        }\n";
        }
        std::string early = "";
        if (p.early != "None") {
            early = "        if (" + p.early + ") return i;\n";
        }
        return "int " + fn + "(int n) {\n"
               "    int a = " + std::to_string(p.init.first) + ", b = " + std::to_string(p.init.second) + ";\n"
               "    int last = 0;\n"
               "    for (int i = " + p.lo + "; i < " + p.hi + "; i++) {\n"
               "        last = i;\n"
               + body + early +
               "    }\n"
               "    return " + p.final_ret + ";\n"
               "}\n";
    }
};

}}}
