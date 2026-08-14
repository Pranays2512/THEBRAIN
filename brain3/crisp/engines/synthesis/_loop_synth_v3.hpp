#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>

namespace brain3 { 
namespace engines { 
namespace synthesis {

struct LoopSpecV3 {
    std::string kind;
    
    // while params
    std::string cond, na, nb, ret;
    std::function<bool(int, int)> cf;
    std::function<int(int, int)> naf;
    std::function<int(int, int)> nbf;

    // early-return params
    std::string pre_cond;
    std::function<bool(int)> pre_f;
    bool pre_ret;
    
    std::string lo, hi, econd, r1, r2;
    std::function<int(int)> r_lo;
    std::function<int(int)> r_hi;
    std::function<bool(int, int)> ecf;
};

class LoopSynthV3 {
private:
    struct RangeDef {
        std::string lo_str, hi_str;
        std::function<int(int)> r_lo;
        std::function<int(int)> r_hi;
    };
    
    std::vector<RangeDef> RANGES = {
        {"2", "n",     [](int n){ return 2; }, [](int n){ return n; }},
        {"1", "n + 1", [](int n){ return 1; }, [](int n){ return n + 1; }},
        {"2", "n + 1", [](int n){ return 2; }, [](int n){ return n + 1; }},
        {"1", "n",     [](int n){ return 1; }, [](int n){ return n; }}
    };

    std::map<std::string, std::function<int(int, int)>> WSTATE = {
        {"a", [](int a, int b){ return a; }},
        {"b", [](int a, int b){ return b; }},
        {"a % b", [](int a, int b){ return a % b; }},
        {"b % a", [](int a, int b){ return b % a; }},
        {"a - b", [](int a, int b){ return a - b; }},
        {"b - a", [](int a, int b){ return b - a; }}
    };

    std::map<std::string, std::function<bool(int, int)>> WCOND = {
        {"b != 0", [](int a, int b){ return b != 0; }},
        {"a != 0", [](int a, int b){ return a != 0; }},
        {"a != b", [](int a, int b){ return a != b; }},
        {"a > 0",  [](int a, int b){ return a > 0; }}
    };

    struct PreDef {
        std::string cond;
        std::function<bool(int)> f;
        bool ret;
    };
    std::vector<PreDef> PRES = {
        {"", nullptr, false},
        {"n < 2", [](int n){ return n < 2; }, false},
        {"n < 1", [](int n){ return n < 1; }, false}
    };

    std::map<std::string, std::function<bool(int, int)>> ECOND = {
        {"n % i == 0", [](int i, int n){ return i != 0 && n % i == 0; }},
        {"i * i > n",  [](int i, int n){ return i * i > n; }},
        {"n % i != 0", [](int i, int n){ return i != 0 && n % i != 0; }}
    };

    int rv(const std::string& tag, int i, int n) {
        if (tag == "True") return 1;
        if (tag == "False") return 0;
        if (tag == "i") return i;
        if (tag == "n") return n;
        if (tag == "-1") return -1;
        if (tag == "0") return 0;
        if (tag == "1") return 1;
        return 0;
    }

public:
    LoopSpecV3* synth_while(const std::vector<std::pair<std::pair<int, int>, int>>& examples) {
        for (const auto& [cc, cf] : WCOND) {
            for (const auto& [na, naf] : WSTATE) {
                for (const auto& [nb, nbf] : WSTATE) {
                    for (std::string ret : {"a", "b"}) {
                        auto f = [&](int a, int b) {
                            for (int i = 0; i < 100000; i++) {
                                if (!cf(a, b)) break;
                                int next_a = naf(a, b);
                                int next_b = nbf(a, b);
                                a = next_a;
                                b = next_b;
                            }
                            return (ret == "a") ? a : b;
                        };
                        bool ok = true;
                        for (const auto& ex : examples) {
                            if (f(ex.first.first, ex.first.second) != ex.second) { ok = false; break; }
                        }
                        if (ok) {
                            auto spec = new LoopSpecV3();
                            spec->kind = "while";
                            spec->cond = cc; spec->na = na; spec->nb = nb; spec->ret = ret;
                            return spec;
                        }
                    }
                }
            }
        }
        return nullptr;
    }

    LoopSpecV3* synth_search(const std::vector<std::pair<int, int>>& examples) {
        int cut = std::max(3, (int)(examples.size() * 0.6));
        
        for (const auto& pre : PRES) {
            for (const auto& r : RANGES) {
                for (const auto& [cc, cf] : ECOND) {
                    for (std::string r1 : {"True", "False", "i", "n", "-1", "0", "1"}) {
                        for (std::string r2 : {"True", "False", "n", "-1", "0", "1"}) {
                            
                            auto f = [&](int n) {
                                if (pre.f && pre.f(n)) return pre.ret ? 1 : 0;
                                for (int i = r.r_lo(n); i < r.r_hi(n); i++) {
                                    if (cf(i, n)) return rv(r1, i, n);
                                }
                                return rv(r2, 0, n);
                            };

                            bool ok = true;
                            for (const auto& ex : examples) {
                                if (f(ex.first) != ex.second) { ok = false; break; }
                            }
                            if (ok) {
                                auto spec = new LoopSpecV3();
                                spec->kind = "early";
                                spec->pre_cond = pre.cond;
                                spec->pre_ret = pre.ret;
                                spec->lo = r.lo_str; spec->hi = r.hi_str;
                                spec->econd = cc;
                                spec->r1 = r1; spec->r2 = r2;
                                return spec;
                            }
                        }
                    }
                }
            }
        }
        return nullptr;
    }

    std::string render(const std::string& fn, const LoopSpecV3& s) {
        if (s.kind == "while") {
            return "int " + fn + "(int a, int b) {\n"
                   "    while (" + s.cond + ") {\n"
                   "        int next_a = " + s.na + ";\n"
                   "        int next_b = " + s.nb + ";\n"
                   "        a = next_a; b = next_b;\n"
                   "    }\n"
                   "    return " + s.ret + ";\n"
                   "}\n";
        } else {
            std::string pre = "";
            if (!s.pre_cond.empty()) {
                pre = "    if (" + s.pre_cond + ") return " + (s.pre_ret ? "1" : "0") + ";\n";
            }
            return "int " + fn + "(int n) {\n" + pre +
                   "    for (int i = " + s.lo + "; i < " + s.hi + "; i++) {\n"
                   "        if (" + s.econd + ") return " + (s.r1 == "True" ? "1" : (s.r1 == "False" ? "0" : s.r1)) + ";\n"
                   "    }\n"
                   "    return " + (s.r2 == "True" ? "1" : (s.r2 == "False" ? "0" : s.r2)) + ";\n"
                   "}\n";
        }
    }
};

}}}
